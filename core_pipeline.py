# core_pipeline.py
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from scipy.stats import chi2
import joblib
from config_sci import *


# ==================== 1. MVTec AD 数据集加载 ====================
class MVTecDataset(Dataset):
    def __init__(self, class_name, is_train=True):
        self.class_name = class_name
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        base = os.path.join(DATASET_ROOT, self.class_name)
        if self.is_train:
            train_dir = os.path.join(base, "train", "good")
            for img_name in sorted(os.listdir(train_dir)):
                if img_name.endswith(".png"):
                    samples.append((os.path.join(train_dir, img_name), None, 0))
        else:
            test_dir = os.path.join(base, "test")
            gt_dir = os.path.join(base, "ground_truth")
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                for img_name in sorted(os.listdir(defect_dir)):
                    if img_name.endswith(".png"):
                        img_path = os.path.join(defect_dir, img_name)
                        if defect_type == "good":
                            samples.append((img_path, None, 0))
                        else:
                            gt_name = img_name.replace(".png", "_mask.png")
                            gt_path = os.path.join(gt_dir, defect_type, gt_name)
                            samples.append((img_path, gt_path, 1))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        if gt_path and os.path.exists(gt_path):
            gt = Image.open(gt_path).convert("L")
            gt = transforms.Resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)(gt)
            gt = transforms.CenterCrop(CROP_SIZE)(gt)
            gt_tensor = transforms.ToTensor()(gt).squeeze()
        else:
            gt_tensor = torch.zeros((CROP_SIZE, CROP_SIZE))
        return img_tensor, gt_tensor, torch.tensor(label), img_path


# ==================== 2. PaDiM 无监督检测 ====================
class PaDiM:
    def __init__(self, backbone="resnet18", reduction_dim=100):
        self.backbone_name = backbone
        self.reduction_dim = reduction_dim
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
            self.feature_layers = ["layer1", "layer2", "layer3"]
            self.feature_channels = [64, 128, 256]
        else:
            raise ValueError("Only resnet18 supported")
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad = False
        total_ch = sum(self.feature_channels)
        self.random_indices = np.random.choice(total_ch, reduction_dim, replace=False)
        self.mean = None
        self.inv_cov = None
        self.feature_size = None
        self.eps = 0.01

    def _extract_features(self, x):
        features = []

        def hook(module, input, output): features.append(output)

        handles = [getattr(self.backbone, n).register_forward_hook(hook) for n in self.feature_layers]
        with torch.no_grad(): self.backbone(x.to(DEVICE))
        for h in handles: h.remove()
        max_h, max_w = features[0].shape[2], features[0].shape[3]
        aligned = [F.interpolate(f, (max_h, max_w), mode='bilinear', align_corners=False) for f in features]
        concat = torch.cat(aligned, dim=1)
        reduced = concat[:, self.random_indices, :, :]
        return reduced.permute(0, 2, 3, 1).cpu().numpy()

    def fit(self, loader):
        all_emb = []
        for imgs, _, _, _ in tqdm(loader, desc="Training PaDiM", leave=False):
            all_emb.append(self._extract_features(imgs))
        all_emb = np.concatenate(all_emb, axis=0)
        N, H, W, C = all_emb.shape
        self.feature_size = (H, W)
        self.mean = np.mean(all_emb, axis=0)
        x_centered = all_emb - self.mean
        self.cov = np.zeros((H, W, C, C), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                cov_ij = np.dot(x_centered[:, i, j, :].T, x_centered[:, i, j, :]) / (N - 1) + self.eps * np.eye(C)
                self.cov[i, j] = cov_ij
        self.inv_cov = np.linalg.inv(self.cov)

    def predict(self, imgs):
        B = imgs.shape[0]
        H, W = self.feature_size
        patch_emb = self._extract_features(imgs)
        x_centered = patch_emb - self.mean
        dist = np.zeros((B, H, W), dtype=np.float32)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    x = x_centered[b, i, j, :]
                    inv_c = self.inv_cov[i, j, :, :]
                    dist[b, i, j] = np.sqrt(np.dot(np.dot(x.T, inv_c), x))
        resized = []
        for b in range(B):
            am = cv2.resize(dist[b], (CROP_SIZE, CROP_SIZE))
            am = cv2.GaussianBlur(am, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
            resized.append(am)
        return np.array(resized), np.max(resized, axis=(1, 2))


# ==================== 3. 空间耦合特征提取 ====================
def extract_features(anomaly_map, threshold):
    binary = (anomaly_map > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {m: 0.0 for m in METRICS} | {'has_defect': 0}

    total_pixels = anomaly_map.size
    area = np.sum(binary)
    area_ratio = area / total_pixels
    max_area = max(cv2.contourArea(c) for c in contours)
    max_area_ratio = max_area / total_pixels
    max_resp = np.max(anomaly_map[binary == 1])
    mean_resp = np.mean(anomaly_map[binary == 1])
    cum_resp = np.sum(anomaly_map[binary == 1])
    num_regions = len(contours)

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            centers.append((cx, cy))
    dispersion = 0.0
    if len(centers) > 1:
        dists = [np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                 for i in range(len(centers)) for j in range(i + 1, len(centers))]
        dispersion = np.std(dists) if dists else 0.0

    return {
        'area_ratio': area_ratio,
        'max_area_ratio': max_area_ratio,
        'max_response': max_resp,
        'mean_response': mean_resp,
        'cum_response': cum_resp,
        'num_regions': num_regions,
        'dispersion': dispersion,
        'has_defect': 1
    }


# ==================== 4. 熵权法 ====================
def entropy_weight(data_df):
    data = data_df.values
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    p = data / (data.sum(axis=0, keepdims=True) + 1e-10)
    k = 1 / np.log(len(data))
    e = -k * np.sum(p * np.log(p + 1e-10), axis=0)
    w = (1 - e) / np.sum(1 - e)
    return w


# ==================== 5. 新增：提取区域列表用于GNN ====================
def extract_regions(anomaly_map, threshold):
    """
    从异常热图中提取每个连通区域的详细属性，用于构建图。
    返回区域列表，每个区域是一个字典。
    """
    binary = (anomaly_map > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    total_pixels = anomaly_map.size

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        mean_resp = np.mean(anomaly_map[mask == 1])
        cum_resp = np.sum(anomaly_map[mask == 1])
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
        regions.append({
            'area': area,
            'area_ratio': area / total_pixels,
            'mean_response': mean_resp,
            'cum_response': cum_resp,
            'center_x': cx,
            'center_y': cy,
            'contour': cnt
        })
    return regions
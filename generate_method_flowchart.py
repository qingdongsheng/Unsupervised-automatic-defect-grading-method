"""
生成方法流程图所需的极小尺寸子图（原4x4英寸的1%面积，边长约0.4英寸）。
输出：results_sci/method_figures/ 下的 PNG/PDF/TIFF 文件。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from skimage import measure
from scipy.stats import chi2
from torchvision import transforms
from config_sci import *
from core_pipeline import PaDiM, extract_features
from sklearn.preprocessing import MinMaxScaler

# ==================== 配置 ====================
TARGET_CLASS = "leather"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "method_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型和规则
model_path = os.path.join(MODEL_DIR, f"{TARGET_CLASS}_padim.pkl")
rule_path = os.path.join(OUTPUT_BASE, "results", f"{TARGET_CLASS}_rule.pkl")
if not os.path.exists(model_path) or not os.path.exists(rule_path):
    raise FileNotFoundError(f"请先运行主实验，生成 {TARGET_CLASS} 的模型和规则文件。")

params = joblib.load(model_path)
padim = PaDiM(backbone=params['backbone'], reduction_dim=params['reduction_dim'])
padim.mean = params['mean']
padim.inv_cov = params['inv_cov']
padim.feature_size = params['feature_size']
padim.random_indices = params['random_indices']

rule = joblib.load(rule_path)
scaler = rule['scaler']
weights = rule['weights']
metrics = rule['metrics']
boundaries = rule['boundaries']

chi2_thresh = chi2.ppf(CHI2_PERCENTILE, padim.reduction_dim)
threshold_map = np.sqrt(chi2_thresh)

# 加载测试集，挑选一个中等评分的缺陷样本
from core_pipeline import MVTecDataset
from torch.utils.data import DataLoader

test_set = MVTecDataset(TARGET_CLASS, is_train=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

samples = []
for imgs, gt, label, path in test_loader:
    if label.item() == 0:
        continue
    anomaly_map, _ = padim.predict(imgs)
    anomaly_map = anomaly_map[0]
    mask = (anomaly_map > threshold_map).astype(np.uint8)
    feat_dict = extract_features(anomaly_map, threshold_map)
    feat_array = np.array([feat_dict[m] for m in metrics]).reshape(1, -1)
    feat_norm = scaler.transform(pd.DataFrame(feat_array, columns=metrics))[0]
    score = np.dot(feat_norm, weights)
    samples.append({
        'img': imgs[0],
        'img_path': path[0],
        'anomaly_map': anomaly_map,
        'mask': mask,
        'feat_norm': feat_norm,
        'score': score,
    })
    if len(samples) >= 30:
        break

scores = [s['score'] for s in samples]
median_score = np.median(scores)
idx = np.argmin(np.abs(np.array(scores) - median_score))
sample = samples[idx]
print(f"选中缺陷样本: {sample['img_path']}, 评分 = {sample['score']:.4f}")

# 图像反归一化
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def tensor_to_numpy(tensor):
    img = tensor.permute(1,2,0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

img_defect = tensor_to_numpy(sample['img'])
amap = sample['anomaly_map']
amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
mask = sample['mask']

# 正常图像
normal_dir = os.path.join(DATASET_ROOT, TARGET_CLASS, "train", "good")
normal_imgs = [f for f in os.listdir(normal_dir) if f.endswith('.png')]
if normal_imgs:
    normal_path = os.path.join(normal_dir, normal_imgs[0])
    normal_img = Image.open(normal_path).convert("RGB")
    normal_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    normal_tensor = normal_transform(normal_img)
    img_normal = tensor_to_numpy(normal_tensor)
else:
    img_normal = np.zeros((CROP_SIZE, CROP_SIZE, 3))

# 连通域信息
labeled = measure.label(mask, connectivity=2)
props = measure.regionprops(labeled)
centroids = np.array([p.centroid for p in props])
K = len(props)
lines = []
if K >= 2:
    from itertools import combinations
    for (i, j) in combinations(range(K), 2):
        lines.append((centroids[i], centroids[j]))
max_loc = np.unravel_index(np.argmax(amap), amap.shape)

# 保存函数
def save_fig(fig, name):
    for fmt in ['png', 'pdf', 'tiff']:
        out_path = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        if fmt == 'tiff':
            fig.savefig(out_path, dpi=600, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(out_path, dpi=300, bbox_inches='tight', format=fmt)
    plt.close(fig)
    print(f"  已保存: {name}")

# ========== Stage 1 (0.4x0.4英寸) ==========
for name, img, title in [
    ("Stage1_normal", img_normal, "Normal"),
    ("Stage1_defective", img_defect, "Defective"),
    ("Stage1_heatmap", amap_norm, "Heatmap"),
    ("Stage1_mask", mask, "Mask")
]:
    fig, ax = plt.subplots(figsize=(0.4, 0.4))
    if name == "Stage1_heatmap":
        ax.imshow(img, cmap='jet', vmin=0, vmax=1)
    elif name == "Stage1_mask":
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=3)
    ax.axis('off')
    save_fig(fig, name)

# ========== Stage 2 (0.4x0.4英寸) ==========
# 连通域彩色图
fig, ax = plt.subplots(figsize=(0.4, 0.4))
colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
for i, p in enumerate(props, start=1):
    color = (np.array(plt.cm.tab10(i % 10)[:3]) * 255).astype(np.uint8)
    colored[labeled == i] = color
ax.imshow(colored)
ax.set_title("Components", fontsize=3)
ax.axis('off')
save_fig(fig, "Stage2_connected_components")

# 离散度
fig, ax = plt.subplots(figsize=(0.4, 0.4))
ax.imshow(img_defect)
if K >= 2:
    for (c1, c2) in lines:
        ax.plot([c1[1], c2[1]], [c1[0], c2[0]], 'r-', linewidth=0.3)
    ax.scatter(centroids[:, 1], centroids[:, 0], c='red', s=2, marker='o')
ax.set_title("Dispersion", fontsize=3)
ax.axis('off')
save_fig(fig, "Stage2_dispersion")

# 强度叠加
fig, ax = plt.subplots(figsize=(0.4, 0.4))
ax.imshow(img_defect)
ax.imshow(amap_norm, cmap='jet', alpha=0.5)
ax.scatter(max_loc[1], max_loc[0], c='white', s=4, marker='x', linewidths=0.5)
ax.set_title("Intensity", fontsize=3)
ax.axis('off')
save_fig(fig, "Stage2_intensity_overlay")

# ========== Stage 3 (柱状图 0.6x0.4英寸) ==========
feature_names = [SHORT_NAMES[m] for m in metrics]

# 全局权重
fig, ax = plt.subplots(figsize=(0.6, 0.4))
ax.bar(feature_names, weights, width=0.6, color='#1f77b4')
ax.set_ylabel("W", fontsize=3)
ax.set_title("Global", fontsize=4)
ax.tick_params(axis='x', labelrotation=45, labelsize=2.5)
ax.tick_params(axis='y', labelsize=2.5)
plt.tight_layout(pad=0.05)
save_fig(fig, "Stage3_global_weights")

# 贡献分解
fig, ax = plt.subplots(figsize=(0.6, 0.4))
contrib = sample['feat_norm'] * weights
ax.bar(feature_names, contrib, width=0.6, color='#ff7f0e')
ax.set_ylabel("C", fontsize=3)
ax.set_title("Contrib", fontsize=4)
ax.tick_params(axis='x', labelrotation=45, labelsize=2.5)
ax.tick_params(axis='y', labelsize=2.5)
plt.tight_layout(pad=0.05)
save_fig(fig, "Stage3_contribution")

# 标定集边界
df_defect = pd.read_csv(os.path.join(OUTPUT_BASE, "results", f"{TARGET_CLASS}_test.csv"))
df_defect = df_defect[df_defect['label'] == 1].reset_index(drop=True)
np.random.seed(RANDOM_SEED)
calib_idx = np.random.choice(df_defect.index, size=int(len(df_defect) * CALIBRATION_RATIO), replace=False)
calib = df_defect.loc[calib_idx].copy()
calib_scaled = scaler.transform(pd.DataFrame(calib[metrics].values, columns=metrics))
calib_scores = calib_scaled @ weights

fig, ax = plt.subplots(figsize=(0.6, 0.4))
ax.hist(calib_scores, bins=20, color='gray', alpha=0.7, edgecolor='black')
for b in boundaries:
    ax.axvline(b, color='red', linestyle='--', linewidth=0.3)
ax.set_xlabel("S", fontsize=3)
ax.set_ylabel("F", fontsize=3)
ax.set_title("Boundary", fontsize=4)
ax.tick_params(axis='both', labelsize=2.5)
plt.tight_layout(pad=0.05)
save_fig(fig, "Stage3_boundaries")

print(f"\n✅ 所有子图已生成（1%大小），保存至: {OUTPUT_DIR}")
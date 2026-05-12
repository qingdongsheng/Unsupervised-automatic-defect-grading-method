# infer_defect_grade_sci.py
"""
单张图像缺陷检测与分级推理脚本
使用方法：
    python infer_defect_grade_sci.py --class_name leather --image_path ./test.png
    python infer_defect_grade_sci.py --all   # 测试所有类别的推理时间（不含网络前向）
"""
import os
import argparse
import cv2
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from scipy.stats import chi2
from PIL import Image
from torchvision import transforms
import pandas as pd   # 用于保存 CSV

from config_sci import *
from core_pipeline import PaDiM, extract_features


def measure_inference_time(class_name, padim, scaler, weights, boundaries, metrics):
    """测量单个类别的推理时间（后处理+分级，不含网络前向）"""
    # 找一张缺陷图像
    test_dir = os.path.join(DATASET_ROOT, class_name, "test")
    if not os.path.exists(test_dir):
        print(f"  跳过 {class_name}: 测试目录不存在")
        return None
    defect_dirs = [d for d in os.listdir(test_dir) if d != "good"]
    if not defect_dirs:
        print(f"  跳过 {class_name}: 无缺陷样本")
        return None
    img_name = os.listdir(os.path.join(test_dir, defect_dirs[0]))[0]
    img_path = os.path.join(test_dir, defect_dirs[0], img_name)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # 网络前向（不计时）
    anomaly_map, _ = padim.predict(img_tensor)
    anomaly_map = anomaly_map[0]

    # 后处理+分级 计时开始
    import time
    start = time.perf_counter()

    chi2_thresh = chi2.ppf(CHI2_PERCENTILE, padim.reduction_dim)
    threshold = np.sqrt(chi2_thresh)
    features = extract_features(anomaly_map, threshold)

    if features['has_defect'] == 0:
        grade_num = 0
        grade_name = "Normal"
        score = 0.0
    else:
        feat_array = np.array([features[m] for m in metrics]).reshape(1, -1)
        feat_norm = scaler.transform(feat_array)[0]
        score = np.dot(feat_norm, weights)
        grade_num = 1 if score < boundaries[0] else (2 if score < boundaries[1] else 3)
        grade_name = ["Mild", "Moderate", "Severe"][grade_num - 1]

    end = time.perf_counter()
    elapsed = end - start
    return elapsed


def run_all_classes_inference():
    """遍历所有类别，输出后处理+分级时间（不含网络前向），并保存到 CSV"""
    print("=" * 60)
    print("  测量所有类别的推理时间（后处理+分级）")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    times = {}

    for class_name in ALL_CLASSES:
        print(f"\n>>> 处理类别: {class_name}")
        model_path = os.path.join(MODEL_DIR, f"{class_name}_padim.pkl")
        rule_path = os.path.join(OUTPUT_BASE, "results", f"{class_name}_rule.pkl")
        if not os.path.exists(model_path) or not os.path.exists(rule_path):
            print(f"  ⚠️ 模型或规则文件缺失，跳过")
            continue

        # 加载模型
        params = joblib.load(model_path)
        padim = PaDiM(backbone=params['backbone'], reduction_dim=params['reduction_dim'])
        padim.mean = params['mean']
        padim.inv_cov = params['inv_cov']
        padim.feature_size = params['feature_size']
        padim.random_indices = params['random_indices']

        # 加载规则
        rule = joblib.load(rule_path)
        scaler = rule['scaler']
        weights = rule['weights']
        boundaries = rule['boundaries']
        metrics = rule['metrics']

        # 测量时间（多次取平均）
        n_repeats = 5
        t_list = []
        for _ in range(n_repeats):
            t = measure_inference_time(class_name, padim, scaler, weights, boundaries, metrics)
            if t is not None:
                t_list.append(t)
        if t_list:
            avg_t = np.mean(t_list)
            times[class_name] = avg_t
            print(f"  ✅ 后处理+分级平均用时: {avg_t:.6f} 秒 (基于{n_repeats}次)")
        else:
            print(f"  ❌ 无法测量")

    # 输出汇总
    print("\n" + "=" * 60)
    print("所有类别推理时间汇总（后处理+分级，不含网络前向）:")
    for cls, t in times.items():
        print(f"  {cls:15s}: {t:.6f} 秒")
    if times:
        avg_all = np.mean(list(times.values()))
        print(f"\n  平均时间: {avg_all:.6f} 秒")
    print("=" * 60)

    # 保存到 CSV
    if times:
        df_times = pd.DataFrame(list(times.items()), columns=["class", "inference_time_sec"])
        csv_path = os.path.join(OUTPUT_BASE, "inference_times.csv")
        df_times.to_csv(csv_path, index=False)
        print(f"\n✅ 推理时间已保存至 {csv_path}")


def single_image_inference(class_name, image_path):
    """原有的单张图像推理+可视化"""
    # 加载模型
    model_path = os.path.join(MODEL_DIR, f"{class_name}_padim.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PaDiM模型不存在: {model_path}")
    print("✅ 加载PaDiM模型...")
    params = joblib.load(model_path)
    padim = PaDiM(backbone=params['backbone'], reduction_dim=params['reduction_dim'])
    padim.mean = params['mean']
    padim.inv_cov = params['inv_cov']
    padim.feature_size = params['feature_size']
    padim.random_indices = params['random_indices']

    # 加载分级规则
    rule_path = os.path.join(OUTPUT_BASE, "results", f"{class_name}_rule.pkl")
    if not os.path.exists(rule_path):
        raise FileNotFoundError(f"分级规则不存在: {rule_path}")
    rule = joblib.load(rule_path)
    scaler = rule['scaler']
    weights = rule['weights']
    boundaries = rule['boundaries']
    metrics = rule['metrics']
    print("✅ 加载分级规则...")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # 推理
    print("🔍 正在检测...")
    anomaly_map, image_score = padim.predict(img_tensor)
    anomaly_map = anomaly_map[0]
    chi2_thresh = chi2.ppf(CHI2_PERCENTILE, padim.reduction_dim)
    threshold = np.sqrt(chi2_thresh)
    features = extract_features(anomaly_map, threshold)

    # 分级
    if features['has_defect'] == 0:
        print("✅ 未检测到缺陷，等级：正常")
        grade_num = 0
        grade_name = "Normal"
        score = 0.0
    else:
        feat_array = np.array([features[m] for m in metrics]).reshape(1, -1)
        feat_norm = scaler.transform(feat_array)[0]
        score = np.dot(feat_norm, weights)
        grade_num = 1 if score < boundaries[0] else (2 if score < boundaries[1] else 3)
        grade_name = ["Mild", "Moderate", "Severe"][grade_num - 1]
        print(f"⚠️  检测到缺陷！综合评分: {score:.4f}，等级: {grade_name}")
        print("\n特征贡献详情：")
        contribution = feat_norm * weights
        for m, c in zip(metrics, contribution):
            print(f"  {SHORT_NAMES[m]}: {c:.4f} ({c/score*100:.1f}%)")

    # 可视化
    img_np = img_tensor[0].permute(1,2,0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    im = plt.imshow(anomaly_map, cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Anomaly Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.5)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Defect Grade: {grade_name} (Score: {score:.3f})")
    plt.axis('off')

    plt.tight_layout()
    out_path = os.path.splitext(image_path)[0] + "_result.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 结果图已保存至: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, help="类别名，如 leather")
    parser.add_argument("--image_path", type=str, help="待检测图像路径")
    parser.add_argument("--all", action="store_true", help="测试所有类别的推理时间（后处理+分级，不含网络前向）")
    args = parser.parse_args()

    if args.all:
        run_all_classes_inference()
    elif args.class_name and args.image_path:
        single_image_inference(args.class_name, args.image_path)
    else:
        # 默认运行 --all 模式
        print("未指定参数，默认运行 --all 模式")
        run_all_classes_inference()


if __name__ == "__main__":
    main()
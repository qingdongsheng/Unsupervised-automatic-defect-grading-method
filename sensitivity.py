import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
from core_pipeline import entropy_weight
from config_sci import *

# 设置全局绘图参数（SCI 风格）
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def compute_kappa_for_params(class_name, chi2_percentile, calib_ratio, quantile_cuts, use_saved_model=True):
    """计算给定超参数下的二次加权Kappa（单次随机划分，固定种子以保证可复现）"""
    from core_pipeline import MVTecDataset, PaDiM, extract_features
    from torch.utils.data import DataLoader
    from scipy.stats import chi2

    model_path = os.path.join(MODEL_DIR, f"{class_name}_padim.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型 {model_path} 不存在，请先运行主实验生成模型。")
    params = joblib.load(model_path)
    padim = PaDiM(backbone=params['backbone'], reduction_dim=params['reduction_dim'])
    padim.mean = params['mean']
    padim.inv_cov = params['inv_cov']
    padim.feature_size = params['feature_size']
    padim.random_indices = params['random_indices']

    test_set = MVTecDataset(class_name, is_train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    chi2_thresh = chi2.ppf(chi2_percentile, padim.reduction_dim)
    threshold_map = np.sqrt(chi2_thresh)

    data_list = []
    for imgs, gt, label, path in test_loader:
        anomaly_map, img_score = padim.predict(imgs)
        anomaly_map = anomaly_map[0]
        feat = extract_features(anomaly_map, threshold_map)
        gt_np = gt.numpy()[0]
        gt_area_ratio = np.sum(gt_np > 0.5) / (CROP_SIZE * CROP_SIZE)
        data_list.append({
            'label': label.numpy()[0],
            'gt_area_ratio': gt_area_ratio,
            **feat
        })
    df = pd.DataFrame(data_list)
    df_defect = df[df['label'] == 1].reset_index(drop=True)
    if len(df_defect) < 10:
        return np.nan

    np.random.seed(RANDOM_SEED)  # 固定随机种子，保证可复现
    n_calib = int(len(df_defect) * calib_ratio)
    if n_calib < 2:
        return np.nan
    calib_idx = np.random.choice(df_defect.index, size=n_calib, replace=False)
    test_idx = np.setdiff1d(df_defect.index, calib_idx)
    calib = df_defect.loc[calib_idx].copy()
    test = df_defect.loc[test_idx].copy()

    scaler = MinMaxScaler()
    calib_scaled = scaler.fit_transform(calib[METRICS])
    calib_scaled_df = pd.DataFrame(calib_scaled, columns=METRICS)
    weights = entropy_weight(calib_scaled_df)
    calib_scores = calib_scaled_df.values @ weights
    boundaries = np.quantile(calib_scores, quantile_cuts)

    test_scaled = scaler.transform(test[METRICS])
    test_scores = test_scaled @ weights
    pred_grades = np.digitize(test_scores, boundaries, right=True) + 1

    gt_bound = np.quantile(calib['gt_area_ratio'], quantile_cuts)
    true_grades = np.digitize(test['gt_area_ratio'], gt_bound, right=True) + 1
    kappa = cohen_kappa_score(true_grades, pred_grades, weights='quadratic')
    return kappa


def sensitivity_chi2(class_name, percentiles=[0.99, 0.995, 0.999, 0.9999]):
    """扫描卡方分位数，返回 (参数列表, Kappa列表)"""
    kappas = []
    for p in percentiles:
        kappa = compute_kappa_for_params(class_name, chi2_percentile=p, calib_ratio=CALIBRATION_RATIO,
                                         quantile_cuts=QUANTILE_CUTS)
        kappas.append(kappa)
    return percentiles, kappas


def sensitivity_calib_ratio(class_name, ratios=[0.2, 0.3, 0.4, 0.5]):
    """扫描标定集比例"""
    kappas = []
    for r in ratios:
        kappa = compute_kappa_for_params(class_name, chi2_percentile=CHI2_PERCENTILE, calib_ratio=r,
                                         quantile_cuts=QUANTILE_CUTS)
        kappas.append(kappa)
    return ratios, kappas


def sensitivity_quantile_cuts(class_name, cut_list=[[0.25, 0.5], [0.3, 0.6], [0.33, 0.67], [0.4, 0.7]]):
    """扫描等级分位点"""
    kappas = []
    for cuts in cut_list:
        kappa = compute_kappa_for_params(class_name, chi2_percentile=CHI2_PERCENTILE, calib_ratio=CALIBRATION_RATIO,
                                         quantile_cuts=cuts)
        kappas.append(kappa)
    return cut_list, kappas


def print_results_table(class_name, param_name, values, kappas):
    """打印表格形式的灵敏度结果"""
    print(f"\n{class_name} - {param_name}:")
    print("-" * 50)
    print(f"{param_name:20s} | Quadratic Weighted Kappa")
    print("-" * 50)
    for v, k in zip(values, kappas):
        print(f"{str(v):20s} | {k:.4f}")
    print("-" * 50)


def save_results_csv(class_name, param_name, values, kappas, save_dir):
    """保存灵敏度结果到CSV"""
    df = pd.DataFrame({
        'Parameter': values,
        'Kappa': kappas
    })
    csv_path = os.path.join(save_dir, f"sensitivity_{class_name}_{param_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"结果已保存: {csv_path}")


def plot_sensitivity(class_name, save_dir=OUTPUT_BASE):
    """生成SCI质量灵敏度分析图，并打印/保存数值"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Chi-square percentile
    x1, y1 = sensitivity_chi2(class_name)
    print_results_table(class_name, "Chi-square Percentile", x1, y1)
    save_results_csv(class_name, "chi2", x1, y1, save_dir)

    # 2. Calibration ratio
    x2, y2 = sensitivity_calib_ratio(class_name)
    print_results_table(class_name, "Calibration Set Ratio", x2, y2)
    save_results_csv(class_name, "calib_ratio", x2, y2, save_dir)

    # 3. Quantile cuts
    x3, y3 = sensitivity_quantile_cuts(class_name)
    labels = [f'{c[0]}/{c[1]}' for c in x3]
    print_results_table(class_name, "Quantile Cuts", labels, y3)
    save_results_csv(class_name, "quantile_cuts", labels, y3, save_dir)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Sensitivity Analysis (Class: {class_name})', fontweight='bold', fontsize=12)

    # 子图1: Chi2 percentile
    axes[0].plot(x1, y1, 'o-', color='#1f77b4', markersize=6, linewidth=1.5)
    axes[0].set_xlabel('Chi-square Percentile', fontsize=9)
    axes[0].set_ylabel('Quadratic Weighted Kappa', fontsize=9)
    axes[0].set_ylim([min(y1) - 0.05, max(y1) + 0.05] if len(y1) > 0 else [0, 1])
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='both', labelsize=8)

    # 子图2: Calibration ratio
    axes[1].plot(x2, y2, 's-', color='#ff7f0e', markersize=6, linewidth=1.5)
    axes[1].set_xlabel('Calibration Set Ratio', fontsize=9)
    axes[1].set_ylabel('Quadratic Weighted Kappa', fontsize=9)
    axes[1].set_ylim([min(y2) - 0.05, max(y2) + 0.05] if len(y2) > 0 else [0, 1])
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='both', labelsize=8)

    # 子图3: Quantile cuts
    axes[2].plot(range(len(labels)), y3, '^-', color='#2ca02c', markersize=6, linewidth=1.5)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2].set_xlabel('Quantile Cuts (Low/High)', fontsize=9)
    axes[2].set_ylabel('Quadratic Weighted Kappa', fontsize=9)
    axes[2].set_ylim([min(y3) - 0.05, max(y3) + 0.05] if len(y3) > 0 else [0, 1])
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, f'sensitivity_{class_name}.pdf')
    png_path = os.path.join(save_dir, f'sensitivity_{class_name}.png')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ 灵敏度分析图已保存: {pdf_path} 和 {png_path}")


# 在 sensitivity.py 末尾修改 __main__ 部分
if __name__ == "__main__":
    # 定义要分析的典型类别（5-6个，覆盖纹理和物体）
    TARGET_CLASSES = ['carpet', 'leather', 'bottle', 'metal_nut', 'screw', 'transistor']
    # 也可以加上 'cable', 'tile' 等，但注意有些类别缺陷样本极少可能导致 Kappa 为 NaN

    for cls in TARGET_CLASSES:
        print(f"\n========== 开始分析类别: {cls} ==========")
        plot_sensitivity(cls)  # 这个函数会生成每个类别的三个CSV文件
    print("\n所有灵敏度分析完成。")
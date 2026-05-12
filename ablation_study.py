# ablation_study.py
"""
消融实验：逐个移除特征组，重复随机划分多次，计算Kappa的均值和标准差，并绘制带误差棒的柱状图。
修复版：解决误差线显示不全、统计逻辑错误、绘图适配问题
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

# 导入你的配置文件（保持不变）
from config_sci import *
from core_pipeline import entropy_weight
from run_experiments import N_REPEATS

# 定义特征子集（保持不变）
GEOMETRY_METRICS = ['area_ratio', 'max_area_ratio', 'num_regions', 'dispersion']
INTENSITY_METRICS = ['max_response', 'mean_response', 'cum_response']
DISTRIBUTION_METRICS = ['num_regions', 'dispersion']
CORE_METRICS = ['area_ratio', 'max_response', 'cum_response']


def compute_subset_kappa_repeated(df_defect, subset_metrics, n_repeats=N_REPEATS):
    """
    对给定类别的所有缺陷样本，重复 n_repeats 次随机划分，计算每次的二次加权Kappa。
    返回 Kappa 列表。
    """
    kappas = []
    n = len(df_defect)
    if n < 10:
        return [np.nan] * n_repeats

    for rep in range(n_repeats):
        np.random.seed(RANDOM_SEED + rep)
        # 划分标定集和测试集
        calib_idx = np.random.choice(df_defect.index, size=int(n * CALIBRATION_RATIO), replace=False)
        test_idx = np.setdiff1d(df_defect.index, calib_idx)
        calib = df_defect.loc[calib_idx].copy()
        test = df_defect.loc[test_idx].copy()

        # 归一化
        scaler = MinMaxScaler()
        calib_scaled = scaler.fit_transform(calib[subset_metrics])
        test_scaled = scaler.transform(test[subset_metrics])
        calib_scaled_df = pd.DataFrame(calib_scaled, columns=subset_metrics)

        # 熵权法权重
        weights = entropy_weight(calib_scaled_df)
        calib_scores = calib_scaled_df.values @ weights
        boundaries = np.quantile(calib_scores, QUANTILE_CUTS)

        test_scores = test_scaled @ weights
        pred_grades = np.digitize(test_scores, boundaries, right=True) + 1

        # 真实等级（基于GT面积分位数）
        gt_bound = np.quantile(calib['gt_area_ratio'], QUANTILE_CUTS)
        true_grades = np.digitize(test['gt_area_ratio'], gt_bound, right=True) + 1

        kappa = cohen_kappa_score(true_grades, pred_grades, weights='quadratic')
        kappas.append(kappa)

    return kappas


def run_ablation():
    print("=" * 60)
    print("  消融实验：不同特征子集对分级性能的影响（重复%d次）" % N_REPEATS)
    print("=" * 60)

    results = []
    # 存储所有重复实验的Kappa值（修复统计逻辑）
    all_kappas = {
        'Geometry': [],
        'Intensity': [],
        'Distribution': [],
        'Core': []
    }

    for class_name in tqdm(ALL_CLASSES, desc="消融实验"):
        # 加载该类所有缺陷样本的量化特征和GT面积比
        test_file = os.path.join(OUTPUT_BASE, "results", f"{class_name}_test.csv")
        if not os.path.exists(test_file):
            print(f"  跳过 {class_name}：无测试文件")
            continue
        df = pd.read_csv(test_file)
        if len(df) < 10:
            print(f"  跳过 {class_name}：缺陷样本不足10")
            continue

        # 确保有 gt_area_ratio 列
        if 'gt_area_ratio' not in df.columns:
            print(f"  跳过 {class_name}：缺少 gt_area_ratio")
            continue

        # 计算各子集的 Kappa 列表
        kappas_geo = compute_subset_kappa_repeated(df, GEOMETRY_METRICS)
        kappas_int = compute_subset_kappa_repeated(df, INTENSITY_METRICS)
        kappas_dist = compute_subset_kappa_repeated(df, DISTRIBUTION_METRICS)
        kappas_core = compute_subset_kappa_repeated(df, CORE_METRICS)

        results.append({
            'class': class_name,
            'type': 'Texture' if class_name in TEXTURE_CLASSES else 'Object',
            'Geometry_mean': np.nanmean(kappas_geo),
            'Geometry_std': np.nanstd(kappas_geo),
            'Intensity_mean': np.nanmean(kappas_int),
            'Intensity_std': np.nanstd(kappas_int),
            'Distribution_mean': np.nanmean(kappas_dist),
            'Distribution_std': np.nanstd(kappas_dist),
            'Core_mean': np.nanmean(kappas_core),
            'Core_std': np.nanstd(kappas_core),
        })

        # 收集所有有效Kappa值（过滤NaN）
        all_kappas['Geometry'].extend([k for k in kappas_geo if not np.isnan(k)])
        all_kappas['Intensity'].extend([k for k in kappas_int if not np.isnan(k)])
        all_kappas['Distribution'].extend([k for k in kappas_dist if not np.isnan(k)])
        all_kappas['Core'].extend([k for k in kappas_core if not np.isnan(k)])

    # 保存类别级结果
    df_ablation = pd.DataFrame(results)
    df_ablation.to_csv(os.path.join(OUTPUT_BASE, "ablation_results.csv"), index=False)

    # ===================== 修复：计算整体统计结果 =====================
    subsets = ['Geometry', 'Intensity', 'Distribution', 'Core']
    summary_data = []
    for subset in subsets:
        k_list = all_kappas[subset]
        mean_val = np.mean(k_list)
        std_val = np.std(k_list, ddof=1)  # 无偏标准差（学术规范）
        summary_data.append({
            'Subset': subset,
            'Mean_Kappa': mean_val,
            'Std_Kappa': std_val
        })
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(OUTPUT_BASE, "ablation_summary.csv"), index=False)

    # ===================== 修复：绘制汇总柱状图（误差线完整显示） =====================
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(subsets))
    means = df_summary['Mean_Kappa'].values
    stds = df_summary['Std_Kappa'].values

    # 绘制带误差棒的柱状图
    bars = ax.bar(
        x_pos, means, width=0.6,
        yerr=stds, capsize=6,
        color='#4472C4', edgecolor='black', linewidth=1
    )

    # 柱子上方标注数值
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{mean:.3f}',
            ha='center', va='bottom', fontweight='bold'
        )

    # 坐标轴设置
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Quadratic Weighted Kappa", fontweight='bold')
    ax.set_title('Fig. 4: Ablation study results across all classes', fontweight='bold')

    # 关键修复：自动适配Y轴范围，彻底解决误差线截断
    y_max = (means + stds).max() + 0.02
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    # 保存高分辨率图表
    plt.savefig(os.path.join(OUTPUT_BASE, "Fig4_Ablation.pdf"), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_BASE, "Fig4_Ablation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ===================== 输出结果 =====================
    print("\n✅ 各特征子集整体性能（全类别+全重复实验，均值±标准差）：")
    for _, row in df_summary.iterrows():
        print(f"  {row['Subset']}: {row['Mean_Kappa']:.4f} ± {row['Std_Kappa']:.4f}")

    print(f"\n结果保存路径：")
    print(f"  - 类别详细结果：{os.path.join(OUTPUT_BASE, 'ablation_results.csv')}")
    print(f"  - 整体汇总结果：{os.path.join(OUTPUT_BASE, 'ablation_summary.csv')}")
    print(f"  - 论文图表：{os.path.join(OUTPUT_BASE, 'Fig4_Ablation.pdf/png')}")
    print("\n🎉 消融实验完成！误差线已完整显示，无截断问题。")


if __name__ == "__main__":
    run_ablation()
# config_sci.py
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 【必须修改】数据集路径 ====================
DATASET_ROOT = r"D:\yolo_jc\mvtec_anomaly_detection"  # 改为你本地的MVTec AD路径

# ==================== 路径配置 ====================
MODEL_DIR = "exp1_models"
OUTPUT_BASE = "results_sci"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ==================== 硬件与可复现性 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ==================== 核心实验参数 ====================
IMG_SIZE = 256
CROP_SIZE = 224
CHI2_PERCENTILE = 0.999
GAUSSIAN_KERNEL = (15, 15)
GAUSSIAN_SIGMA = 4
CALIBRATION_RATIO = 0.3  # 30%标定集（仅用于算权重/边界）
QUANTILE_CUTS = [0.33, 0.67]

# ==================== 指标定义 ====================
METRICS = [
    'area_ratio', 'max_area_ratio',
    'max_response', 'mean_response', 'cum_response',
    'num_regions', 'dispersion'
]
SHORT_NAMES = {
    'area_ratio': 'Area_Ratio',
    'max_area_ratio': 'Max_Area',
    'max_response': 'Max_Resp',
    'mean_response': 'Mean_Resp',
    'cum_response': 'Cum_Resp',
    'num_regions': 'Num_Reg',
    'dispersion': 'Dispersion'
}

# ==================== 【全15个MVTec AD类别】 ====================
TEXTURE_CLASSES = ["carpet", "grid", "leather", "tile", "wood"]  # 5个纹理类
OBJECT_CLASSES = [
    "bottle", "metal_nut", "capsule",
    "pill", "toothbrush", "transistor", "zipper",
    "cable", "hazelnut", "screw"
]  # 10个物体类
ALL_CLASSES = TEXTURE_CLASSES + OBJECT_CLASSES  # 全15个

# ==================== SCI顶刊级全局绘图参数 ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'normal',
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 12,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.0,
    'lines.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'image.cmap': 'viridis',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.facecolor': 'white',
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})
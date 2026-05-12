import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

def pca_fusion(calib_features, test_features, n_components=1):
    """用标定集拟合PCA，然后应用到测试集"""
    scaler = MinMaxScaler()
    calib_norm = scaler.fit_transform(calib_features)
    pca = PCA(n_components=n_components)
    pca.fit(calib_norm)
    test_norm = scaler.transform(test_features)
    test_score = pca.transform(test_norm).ravel()
    calib_score = pca.transform(calib_norm).ravel()
    min_score = calib_score.min()
    max_score = calib_score.max()
    test_score = (test_score - min_score) / (max_score - min_score + 1e-10)
    return test_score

def equal_weight_fusion(calib_features, test_features):
    """等权重融合：所有特征等权求和后归一化"""
    scaler = MinMaxScaler()
    calib_norm = scaler.fit_transform(calib_features)
    test_norm = scaler.transform(test_features)
    test_score = test_norm.mean(axis=1)
    return test_score

def single_score_baseline(calib_scores, test_scores):
    """直接使用PaDiM图像级分数，用标定集确定归一化范围"""
    min_s = calib_scores.min()
    max_s = calib_scores.max()
    test_score = (test_scores - min_s) / (max_s - min_s + 1e-10)
    return test_score

def supervised_regression(calib_df, test_df, metrics, quantile_cuts):
    """
    有监督回归基线：用线性回归预测 gt_area_ratio，然后根据分位数得到等级。
    训练集：calib_df（包含 gt_area_ratio 和 7个特征）
    测试集：test_df（同样特征）
    返回测试集预测等级（1,2,3）
    """
    X_train = calib_df[metrics].values
    y_train = calib_df['gt_area_ratio'].values
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    X_test = test_df[metrics].values
    y_pred_area = reg.predict(X_test)

    # 用标定集的真实面积分位数确定等级边界
    boundaries = np.quantile(calib_df['gt_area_ratio'], quantile_cuts)
    pred_grades = np.digitize(y_pred_area, boundaries, right=True) + 1
    return pred_grades

def supervised_classifier(calib_df, test_df, metrics, quantile_cuts):
    """
    有监督分类基线：用SVM直接预测等级。
    训练集等级由 calib_df['gt_area_ratio'] 按 quantile_cuts 划分得到。
    返回测试集预测等级（1,2,3）
    """
    # 生成训练集真实等级
    boundaries = np.quantile(calib_df['gt_area_ratio'], quantile_cuts)
    y_train = np.digitize(calib_df['gt_area_ratio'], boundaries, right=True) + 1

    X_train = calib_df[metrics].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train_scaled, y_train)

    X_test = test_df[metrics].values
    X_test_scaled = scaler.transform(X_test)
    pred_grades = clf.predict(X_test_scaled)
    return pred_grades
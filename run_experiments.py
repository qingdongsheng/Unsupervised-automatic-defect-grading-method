# run_experiments.py (修改后完整版)
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from scipy.stats import chi2, spearmanr, wilcoxon
from tqdm import tqdm

from config_sci import *
from core_pipeline import MVTecDataset, PaDiM, extract_features, entropy_weight, extract_regions
from baselines import pca_fusion, equal_weight_fusion, single_score_baseline, supervised_regression, \
    supervised_classifier
from gnn_model import DefectGNN

N_REPEATS = 10


def compute_detection_metrics(y_true, y_pred):
    if len(set(y_true)) < 2:
        return {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'accuracy': np.nan}
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def build_graph_from_regions(regions, img_w, img_h):
    if len(regions) == 0:
        return None
    node_feats = []
    centers = []
    for r in regions:
        area_ratio = r['area_ratio']
        mean_resp = r['mean_response']
        cum_resp = r['cum_response']
        cx_norm = r['center_x'] / img_w
        cy_norm = r['center_y'] / img_h
        node_feats.append([area_ratio, mean_resp, cum_resp, cx_norm, cy_norm])
        centers.append((cx_norm, cy_norm))
    x = torch.tensor(node_feats, dtype=torch.float)
    threshold_dist = 0.3
    edge_index = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if dist < threshold_dist:
                edge_index.append([i, j])
                edge_index.append([j, i])
    if len(edge_index) == 0:
        edge_index = [[i, i] for i in range(len(centers))]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def evaluate_one_split(calib, test, metrics, quantile_cuts):
    gt_bound = np.quantile(calib['gt_area_ratio'], quantile_cuts)
    true_grades = np.digitize(test['gt_area_ratio'], gt_bound, right=True) + 1

    scaler = MinMaxScaler()
    calib_scaled = scaler.fit_transform(calib[metrics])
    calib_scaled_df = pd.DataFrame(calib_scaled, columns=metrics)
    weights = entropy_weight(calib_scaled_df)
    calib_scores = calib_scaled_df.values @ weights
    boundaries = np.quantile(calib_scores, quantile_cuts)

    test_scaled = scaler.transform(test[metrics])
    test_scores_ours = test_scaled @ weights
    pred_grades_ours = np.digitize(test_scores_ours, boundaries, right=True) + 1
    kappa_ours = cohen_kappa_score(true_grades, pred_grades_ours, weights='quadratic')

    test_scores_pca = pca_fusion(calib[metrics].values, test[metrics].values)
    calib_scores_pca = pca_fusion(calib[metrics].values, calib[metrics].values)
    boundaries_pca = np.quantile(calib_scores_pca, quantile_cuts)
    pred_grades_pca = np.digitize(test_scores_pca, boundaries_pca, right=True) + 1
    kappa_pca = cohen_kappa_score(true_grades, pred_grades_pca, weights='quadratic')

    test_scores_equal = equal_weight_fusion(calib[metrics].values, test[metrics].values)
    calib_scores_equal = equal_weight_fusion(calib[metrics].values, calib[metrics].values)
    boundaries_equal = np.quantile(calib_scores_equal, quantile_cuts)
    pred_grades_equal = np.digitize(test_scores_equal, boundaries_equal, right=True) + 1
    kappa_equal = cohen_kappa_score(true_grades, pred_grades_equal, weights='quadratic')

    test_scores_single = single_score_baseline(calib['image_score'].values, test['image_score'].values)
    calib_scores_single = single_score_baseline(calib['image_score'].values, calib['image_score'].values)
    boundaries_single = np.quantile(calib_scores_single, quantile_cuts)
    pred_grades_single = np.digitize(test_scores_single, boundaries_single, right=True) + 1
    kappa_single = cohen_kappa_score(true_grades, pred_grades_single, weights='quadratic')

    pred_grades_sup_reg = supervised_regression(calib, test, metrics, quantile_cuts)
    kappa_sup_reg = cohen_kappa_score(true_grades, pred_grades_sup_reg, weights='quadratic')

    pred_grades_sup_clf = supervised_classifier(calib, test, metrics, quantile_cuts)
    kappa_sup_clf = cohen_kappa_score(true_grades, pred_grades_sup_clf, weights='quadratic')

    results = {
        'kappa_ours': kappa_ours,
        'kappa_pca': kappa_pca,
        'kappa_equal': kappa_equal,
        'kappa_single': kappa_single,
        'kappa_sup_reg': kappa_sup_reg,
        'kappa_sup_clf': kappa_sup_clf,
    }
    return results, scaler, weights, boundaries


def train_gnn_for_class(calib_graphs, calib_targets, device, epochs=200):
    if len(calib_graphs) < 5:
        return None, None
    scaler = StandardScaler()
    all_x = torch.cat([g.x for g in calib_graphs], dim=0).numpy()
    scaler.fit(all_x)
    for g in calib_graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)

    model = DefectGNN(node_feat_dim=5, hidden_dim=64, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = list(zip(calib_graphs, calib_targets))
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for g, t in dataset:
            g = g.to(device)
            pred = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            loss = criterion(pred, torch.tensor(t, dtype=torch.float, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"    GNN epoch {epoch}, loss={total_loss/len(dataset):.4f}")
    return model, scaler


def evaluate_gnn_for_class(calib, test, quantile_cuts, device):
    calib_graphs, calib_targets = [], []
    for _, row in calib.iterrows():
        regions = row['regions']
        if not regions:
            continue
        g = build_graph_from_regions(regions, CROP_SIZE, CROP_SIZE)
        if g is not None:
            calib_graphs.append(g)
            calib_targets.append(row['gt_area_ratio'])
    if len(calib_graphs) < 5:
        return np.nan
    model, scaler = train_gnn_for_class(calib_graphs, calib_targets, device, epochs=200)
    if model is None:
        return np.nan
    test_graphs, test_targets = [], []
    for _, row in test.iterrows():
        regions = row['regions']
        if not regions:
            continue
        g = build_graph_from_regions(regions, CROP_SIZE, CROP_SIZE)
        if g is not None:
            g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
            test_graphs.append(g)
            test_targets.append(row['gt_area_ratio'])
    if len(test_graphs) == 0:
        return np.nan
    model.eval()
    pred_scores = []
    with torch.no_grad():
        for g in test_graphs:
            g = g.to(device)
            pred = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            pred_scores.append(pred.cpu().item())
    gt_bound = np.quantile(calib['gt_area_ratio'], quantile_cuts)
    true_grades = np.digitize(test_targets, gt_bound, right=True) + 1
    pred_grades = np.digitize(pred_scores, np.quantile(pred_scores, quantile_cuts), right=True) + 1
    kappa = cohen_kappa_score(true_grades, pred_grades, weights='quadratic')
    return kappa


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(os.path.join(OUTPUT_BASE, "results"), exist_ok=True)

    all_results = []
    gnn_kappas_all_repeats = {}

    print("=" * 60)
    print(f"  SCI级实验流水线（重复{N_REPEATS}次随机划分）")
    print("=" * 60)

    for class_name in tqdm(ALL_CLASSES, desc="总进度"):
        print(f"\n>>> 处理类别: {class_name}")

        # -------------------- 训练/加载 PaDiM --------------------
        model_path = os.path.join(MODEL_DIR, f"{class_name}_padim.pkl")
        if not os.path.exists(model_path):
            train_set = MVTecDataset(class_name, is_train=True)
            train_loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0)
            padim = PaDiM()

            # ========== 精确计时开始（只包围 fit 方法） ==========
            import time
            start_train = time.perf_counter()
            # =================================================

            padim.fit(train_loader)

            # ========== 精确计时结束 ==========
            end_train = time.perf_counter()
            train_time = end_train - start_train
            print(f"⏱️ 训练 {class_name} 核心用时: {train_time:.2f} 秒")
            # =================================

            train_scores = []
            for imgs, _, _, _ in train_loader:
                _, img_score = padim.predict(imgs)
                train_scores.extend(img_score)
            threshold_score = np.percentile(train_scores, 99)
            joblib.dump({
                'mean': padim.mean, 'inv_cov': padim.inv_cov,
                'feature_size': padim.feature_size, 'random_indices': padim.random_indices,
                'backbone': padim.backbone_name, 'reduction_dim': padim.reduction_dim,
                'threshold_score': threshold_score
            }, model_path)
            print(f"  ✅ PaDiM模型已保存")
        else:
            params = joblib.load(model_path)
            padim = PaDiM(backbone=params['backbone'], reduction_dim=params['reduction_dim'])
            padim.mean = params['mean']
            padim.inv_cov = params['inv_cov']
            padim.feature_size = params['feature_size']
            padim.random_indices = params['random_indices']
            threshold_score = params.get('threshold_score')
            if threshold_score is None:
                train_set = MVTecDataset(class_name, is_train=True)
                train_loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0)
                train_scores = []
                for imgs, _, _, _ in train_loader:
                    _, img_score = padim.predict(imgs)
                    train_scores.extend(img_score)
                threshold_score = np.percentile(train_scores, 99)
                params['threshold_score'] = threshold_score
                joblib.dump(params, model_path)
                print(f"  ⚠️  旧模型已补充阈值: {threshold_score:.4f}")
            print(f"  ✅ PaDiM模型已加载，阈值: {threshold_score:.4f}")

        # -------------------- 推理测试集 --------------------
        test_set = MVTecDataset(class_name, is_train=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        chi2_thresh = chi2.ppf(CHI2_PERCENTILE, padim.reduction_dim)
        threshold_map = np.sqrt(chi2_thresh)

        data_list = []
        y_true, y_pred = [], []
        all_scores = []
        for imgs, gt, label, path in tqdm(test_loader, desc="推理", leave=False):
            anomaly_map, img_score = padim.predict(imgs)
            anomaly_map = anomaly_map[0]
            feat = extract_features(anomaly_map, threshold_map)
            regions = extract_regions(anomaly_map, threshold_map)
            gt_np = gt.numpy()[0]
            gt_area_ratio = np.sum(gt_np > 0.5) / (CROP_SIZE * CROP_SIZE)
            data_list.append({
                'path': path[0],
                'label': label.numpy()[0],
                'image_score': img_score[0],
                'gt_area_ratio': gt_area_ratio,
                **feat,
                'regions': regions
            })
            y_true.append(label.numpy()[0])
            y_pred.append(1 if img_score[0] >= threshold_score else 0)
            all_scores.append(img_score[0])

        detection = compute_detection_metrics(y_true, y_pred)
        if len(set(y_true)) < 2:
            image_auroc = np.nan
        else:
            image_auroc = roc_auc_score(y_true, all_scores)

        print(
            f"  检测性能: P={detection['precision']:.3f}, R={detection['recall']:.3f}, F1={detection['f1']:.3f}, AUROC={image_auroc:.3f}")

        df = pd.DataFrame(data_list)
        df_defect = df[df['label'] == 1].reset_index(drop=True)

        if len(df_defect) < 10:
            print(f"  ⚠️  缺陷样本不足，跳过分级与消融")
            all_results.append({
                'class': class_name,
                'type': 'Texture' if class_name in TEXTURE_CLASSES else 'Object',
                'n_test_total': len(df),
                'image_auroc': image_auroc,
                **detection,
                'kappa_ours_mean': np.nan, 'kappa_ours_std': np.nan,
                'kappa_pca_mean': np.nan, 'kappa_pca_std': np.nan,
                'kappa_equal_mean': np.nan, 'kappa_equal_std': np.nan,
                'kappa_single_mean': np.nan, 'kappa_single_std': np.nan,
                'kappa_sup_reg_mean': np.nan, 'kappa_sup_reg_std': np.nan,
                'kappa_sup_clf_mean': np.nan, 'kappa_sup_clf_std': np.nan,
                'kappa_gnn_mean': np.nan, 'kappa_gnn_std': np.nan,
                'w_area_ratio': np.nan, 'w_cum_response': np.nan,
                **{f'rho_{m}': np.nan for m in METRICS}
            })
            continue

        # -------------------- 重复实验（熵权法等） --------------------
        kappa_ours_list = []
        kappa_pca_list = []
        kappa_equal_list = []
        kappa_single_list = []
        kappa_sup_reg_list = []
        kappa_sup_clf_list = []
        kappa_gnn_list = []

        last_scaler = None
        last_weights = None
        last_boundaries = None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for rep in range(N_REPEATS):
            np.random.seed(RANDOM_SEED + rep)
            calib_idx = np.random.choice(df_defect.index, size=int(len(df_defect) * CALIBRATION_RATIO), replace=False)
            test_idx = np.setdiff1d(df_defect.index, calib_idx)
            calib = df_defect.loc[calib_idx].copy()
            test = df_defect.loc[test_idx].copy()

            results, scaler, weights, boundaries = evaluate_one_split(calib, test, METRICS, QUANTILE_CUTS)
            kappa_ours_list.append(results['kappa_ours'])
            kappa_pca_list.append(results['kappa_pca'])
            kappa_equal_list.append(results['kappa_equal'])
            kappa_single_list.append(results['kappa_single'])
            kappa_sup_reg_list.append(results['kappa_sup_reg'])
            kappa_sup_clf_list.append(results['kappa_sup_clf'])

            kappa_gnn = evaluate_gnn_for_class(calib, test, QUANTILE_CUTS, device)
            kappa_gnn_list.append(kappa_gnn)

            if rep == N_REPEATS - 1:
                last_scaler = scaler
                last_weights = weights
                last_boundaries = boundaries

        mean_ours = np.mean(kappa_ours_list)
        std_ours = np.std(kappa_ours_list)
        mean_pca = np.mean(kappa_pca_list)
        std_pca = np.std(kappa_pca_list)
        mean_equal = np.mean(kappa_equal_list)
        std_equal = np.std(kappa_equal_list)
        mean_single = np.mean(kappa_single_list)
        std_single = np.std(kappa_single_list)
        mean_sup_reg = np.mean(kappa_sup_reg_list)
        std_sup_reg = np.std(kappa_sup_reg_list)
        mean_sup_clf = np.mean(kappa_sup_clf_list)
        std_sup_clf = np.std(kappa_sup_clf_list)
        mean_gnn = np.nanmean(kappa_gnn_list)
        std_gnn = np.nanstd(kappa_gnn_list)

        print(f"  ✅ Ours Kappa: {mean_ours:.3f} ± {std_ours:.3f}")
        print(f"  ✅ GNN Kappa: {mean_gnn:.3f} ± {std_gnn:.3f} (based on {N_REPEATS} repeats)")

        np.random.seed(RANDOM_SEED + N_REPEATS - 1)
        calib_idx = np.random.choice(df_defect.index, size=int(len(df_defect) * CALIBRATION_RATIO), replace=False)
        test_idx = np.setdiff1d(df_defect.index, calib_idx)
        test_last = df_defect.loc[test_idx].copy()
        test_last['score'] = last_scaler.transform(test_last[METRICS]) @ last_weights
        test_last['grade'] = np.digitize(test_last['score'], last_boundaries, right=True) + 1
        test_last['class'] = class_name
        test_last.to_csv(os.path.join(OUTPUT_BASE, "results", f"{class_name}_test.csv"), index=False)

        rule_path = os.path.join(OUTPUT_BASE, "results", f"{class_name}_rule.pkl")
        joblib.dump({'scaler': last_scaler, 'weights': last_weights, 'boundaries': last_boundaries, 'metrics': METRICS},
                    rule_path)

        corr_results = {}
        for m in METRICS:
            rho, p_val = spearmanr(df_defect[m], df_defect['gt_area_ratio'])
            corr_results[f'rho_{m}'] = rho
            corr_results[f'p_{m}'] = p_val

        all_results.append({
            'class': class_name,
            'type': 'Texture' if class_name in TEXTURE_CLASSES else 'Object',
            'n_test_total': len(df),
            'n_test_defect': len(test_last),
            'image_auroc': image_auroc,
            **detection,
            'kappa_ours_mean': mean_ours,
            'kappa_ours_std': std_ours,
            'kappa_pca_mean': mean_pca,
            'kappa_pca_std': std_pca,
            'kappa_equal_mean': mean_equal,
            'kappa_equal_std': std_equal,
            'kappa_single_mean': mean_single,
            'kappa_single_std': std_single,
            'kappa_sup_reg_mean': mean_sup_reg,
            'kappa_sup_reg_std': std_sup_reg,
            'kappa_sup_clf_mean': mean_sup_clf,
            'kappa_sup_clf_std': std_sup_clf,
            'kappa_gnn_mean': mean_gnn,
            'kappa_gnn_std': std_gnn,
            'w_area_ratio': last_weights[0] if len(last_weights) > 0 else np.nan,
            'w_cum_response': last_weights[4] if len(last_weights) > 4 else np.nan,
            **corr_results
        })

        gnn_kappas_all_repeats[class_name] = kappa_gnn_list

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(OUTPUT_BASE, "summary_table.csv"), index=False)

    gnn_repeats_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gnn_kappas_all_repeats.items()]))
    gnn_repeats_df.to_csv(os.path.join(OUTPUT_BASE, "gnn_kappas_all_repeats.csv"), index=False)

    print("\n" + "=" * 60)
    print("  全类别平均检测性能:")
    print(f"  AUROC:    {final_df['image_auroc'].mean():.4f}")
    print(f"  Precision: {final_df['precision'].mean():.4f}")
    print(f"  Recall:    {final_df['recall'].mean():.4f}")
    print(f"  F1:        {final_df['f1'].mean():.4f}")
    print("\n  全类别平均分级 Kappa（均值±标准差）:")
    print(f"  Ours:          {final_df['kappa_ours_mean'].mean():.4f} ± {final_df['kappa_ours_std'].mean():.4f}")
    print(f"  GNN:           {final_df['kappa_gnn_mean'].mean():.4f} ± {final_df['kappa_gnn_std'].mean():.4f} (based on {N_REPEATS} repeats)")
    print(f"  Supervised Reg:{final_df['kappa_sup_reg_mean'].mean():.4f} ± {final_df['kappa_sup_reg_std'].mean():.4f}")
    print(f"  Supervised Clf:{final_df['kappa_sup_clf_mean'].mean():.4f} ± {final_df['kappa_sup_clf_std'].mean():.4f}")
    print(f"  PCA:           {final_df['kappa_pca_mean'].mean():.4f} ± {final_df['kappa_pca_std'].mean():.4f}")
    print(f"  Equal Weight:  {final_df['kappa_equal_mean'].mean():.4f} ± {final_df['kappa_equal_std'].mean():.4f}")
    print(f"  Single Score:  {final_df['kappa_single_mean'].mean():.4f} ± {final_df['kappa_single_std'].mean():.4f}")

    ours_means = final_df['kappa_ours_mean'].dropna().values
    for name, col in [('PCA', 'kappa_pca_mean'), ('Equal', 'kappa_equal_mean'), ('Single', 'kappa_single_mean'),
                      ('SupReg', 'kappa_sup_reg_mean'), ('SupClf', 'kappa_sup_clf_mean'),
                      ('GNN', 'kappa_gnn_mean')]:
        other_means = final_df[col].dropna().values
        if len(ours_means) == len(other_means) and len(ours_means) > 1:
            _, p = wilcoxon(ours_means, other_means)
            print(f"  Ours vs {name}: p = {p:.4f}")
        else:
            print(f"  Ours vs {name}: N/A")

    print("=" * 60)
    print("  🎉 所有实验完成！结果已保存至 results_sci/")
    print("=" * 60)


if __name__ == "__main__":
    main()
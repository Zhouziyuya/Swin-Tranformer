import matplotlib.pyplot as plt
import numpy as np
import os


def plot_robustness_dual(save_path=None, dpi=300):

    models = ["Ark+", "FoundationX", "RAD-DINO", "EVA-X",
              "Lamps", "Adam-v2", "CheXWorld"]

    results = {  # RSNA
        "Ark+": {"no_aug": 75.6, "metal": 75.27, "brightness": 75.6, "gaussian": 74.2},
        "FoundationX": {"no_aug": 74.59, "metal": 71.72, "brightness": 72.22, "gaussian": 70.26},
        "RAD-DINO": {"no_aug": 72.57, "metal": 70.51, "brightness": 72.62, "gaussian": 72.29},
        "EVA-X": {"no_aug": 71.43, "metal": 67.19, "brightness": 70.33, "gaussian": 62.76},
        "Lamps": {"no_aug": 68.06, "metal": 66.48, "brightness": 65.14, "gaussian": 61.88},
        "Adam-v2": {"no_aug": 67.44, "metal": 57.33, "brightness": 57.21, "gaussian": 55.12},
        "CheXWorld": {"no_aug": 69.83, "metal": 65.46, "brightness": 69.62, "gaussian": 68.44},
    }

    metal = np.array([results[m]["metal"] for m in models])
    brightness = np.array([results[m]["brightness"] for m in models])
    gaussian = np.array([results[m]["gaussian"] for m in models])
    no_aug = np.array([results[m]["no_aug"] for m in models])

    # Δ = augmentation - baseline
    metal_drop = metal - no_aug
    brightness_drop = brightness - no_aug
    gaussian_drop = gaussian - no_aug

    x = np.arange(len(models))
    width = 0.18

    # 论文风格
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.0,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(6, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}
    )

    # ===== 颜色 =====
    color_metal = "#A7C7E7"
    color_brightness = "#F4C7AB"
    color_gaussian = "#B8E0D2"
    baseline_color = "#4D4D4D"

    # ==================================================
    # 上图：Performance
    # ==================================================
    ax1.bar(x - width, metal, width,
            label="Metal Artifact",
            color=color_metal, edgecolor='black', linewidth=0.6)

    ax1.bar(x, brightness, width,
            label="Brightness Enhance",
            color=color_brightness, edgecolor='black', linewidth=0.6)

    ax1.bar(x + width, gaussian, width,
            label="Gaussian Noise",
            color=color_gaussian, edgecolor='black', linewidth=0.6)

    # baseline 横线
    for i in range(len(models)):
        if i == 0:
            ax1.hlines(no_aug[i], i - 0.3, i + 0.3,
                       colors=baseline_color,
                       linestyles='--',
                       linewidth=1.5,
                       label="No noise")
        else:
            ax1.hlines(no_aug[i], i - 0.3, i + 0.3,
                       colors=baseline_color,
                       linestyles='--',
                       linewidth=1.5)

    ax1.set_ylabel("RSNA / Acc (%)")

    all_vals = np.concatenate([metal, brightness, gaussian, no_aug])
    ax1.set_ylim(min(all_vals) - 3, max(all_vals) + 3)

    ax1.legend(frameon=False)

    # 🔥 隐藏上图 x 轴（真正共用）
    ax1.tick_params(axis='x', which='both',
                    bottom=False, labelbottom=False)

    # ==================================================
    # 下图：Performance Drop
    # ==================================================
    ax2.bar(x - width, metal_drop, width,
            color=color_metal, edgecolor='black', linewidth=0.6)

    ax2.bar(x, brightness_drop, width,
            color=color_brightness, edgecolor='black', linewidth=0.6)

    ax2.bar(x + width, gaussian_drop, width,
            color=color_gaussian, edgecolor='black', linewidth=0.6)

    # 0 参考线
    ax2.axhline(0, color=baseline_color, linewidth=1.2)

    ax2.set_ylabel("Δ from No Aug (%)")

    drop_vals = np.concatenate([metal_drop,
                                brightness_drop,
                                gaussian_drop])

    ax2.set_ylim(min(drop_vals) - 2, 0)


    # 只在下图显示横轴
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')


def sort_models_by_robustness(models, results):
    """
    Sort models by robustness drop:
    (no_aug - metal) + (no_aug - brightness) + (no_aug - gaussian)

    Smaller is better (more robust).
    """

    # 计算 score
    scores = {}
    for m in models:
        no_aug = results[m]["no_aug"]
        metal = results[m]["metal"]
        brightness = results[m]["brightness"]
        gaussian = results[m]["gaussian"]

        score = (no_aug - metal) + (no_aug - brightness) + (no_aug - gaussian)
        scores[m] = score

    # 从小到大排序
    sorted_models = sorted(models, key=lambda x: scores[x])

    # 重新构造排序后的 results
    sorted_results = {m: results[m] for m in sorted_models}

    return sorted_models, sorted_results, scores


def compute_group_gaps(results, groups=None):
    """
    Compute per-model gap and average gap for three groups.

    Args:
        results: dict mapping model -> dict with keys 'no_aug','metal','brightness','gaussian'
        groups: dict with keys 'group1','group2','group3' mapping to lists of model names.

    Returns:
        dict {
            'group_avgs': {'group1': float, 'group2': float, 'group3': float},
            'model_gaps': {model: gap, ...},
            'missing': {'group1': [...], ...}
        }
    """

    if groups is None:
        groups = {
            'group1': ["CARZero", "RadZero", "DeViDe", "KAD", "MAVL", "MedKLIP"],
            'group2': ["Ark+", "FoundationX"],
            'group3': ["RAD-DINO", "EVA-X", "CheXWorld", "Lamps", "Adam-v2"],
        }

    def _model_gap(r):
        return ((r["no_aug"] - r["metal"]) + (r["no_aug"] - r["brightness"]) + (r["no_aug"] - r["gaussian"])) / 3.0

    model_gaps = {}
    for m, r in results.items():
        try:
            model_gaps[m] = float(_model_gap(r))
        except Exception:
            model_gaps[m] = float('nan')

    group_avgs = {}
    missing = {}
    group_stds = {}
    for key, grp in groups.items():
        vals = []
        miss = []
        for m in grp:
            if m in model_gaps and not np.isnan(model_gaps[m]):
                vals.append(model_gaps[m])
            else:
                miss.append(m)
        group_avgs[key] = float(np.mean(vals)) if vals else float('nan')
        # 标准差：组内模型 gap 的标准差
        group_stds[key] = float(np.std(vals)) if vals else float('nan')
        missing[key] = miss

    return {'group_avgs': group_avgs, 'group_stds': group_stds, 'model_gaps': model_gaps, 'missing': missing}



import numpy as np

def compute_group_relative_gap(results, groups=None):
    """
    基于 relative drop:
        (AUC_clean - AUC_noise) / AUC_clean
    
    计算：
        1. 每个模型的平均 relative drop
        2. 三组模型的 mean 和 std

    Returns:
        {
            'group_means': {'group1': float, ...},
            'group_stds': {'group1': float, ...},
            'model_relative_means': {model: float, ...},
            'missing': {'group1': [...], ...}
        }
    """

    if groups is None:
        groups = {
            'group1': ["CARZero", "RadZero", "DeViDe", "KAD", "MAVL", "MedKLIP"],
            'group2': ["Ark+", "FoundationX"],
            'group3': ["RAD-DINO", "EVA-X", "CheXWorld", "Lamps", "Adam-v2"],
        }

    # ① 先算 relative drop
    relative_drops = compute_relative_drop(results)

    # ② 每个模型算三个噪声的平均 relative drop
    model_relative_means = {}
    for model, noise_dict in relative_drops.items():
        vals = list(noise_dict.values())
        model_relative_means[model] = float(np.mean(vals))

    # ③ 计算每组 mean + std
    group_means = {}
    group_stds = {}
    missing = {}

    for key, grp in groups.items():
        vals = []
        miss = []

        for m in grp:
            if m in model_relative_means and not np.isnan(model_relative_means[m]):
                vals.append(model_relative_means[m])
            else:
                miss.append(m)

        group_means[key] = float(np.mean(vals)) if vals else float('nan')
        group_stds[key] = float(np.std(vals)) if vals else float('nan')
        missing[key] = miss

    return {
        'group_means': group_means,
        'group_stds': group_stds,
        'model_relative_means': model_relative_means,
        'missing': missing
    }





def plot_avg_gap_per_dataset(results, dataset_name, save_path=None, dpi=300):
    # === 绘制并保存三组模型平均 gap 的直方图 ===
    
    group_gap = compute_group_gaps(results)
    print("Group average gaps:", group_gap['group_avgs'])
    print("Per-model gaps:", group_gap['model_gaps'])
    print("Missing models in groups:", group_gap['missing'])
    
    g_avgs = group_gap['group_avgs']
    labels = [
        'VLM',
        'SL',
        'SSL'
    ]
    keys = ['group1', 'group2', 'group3']
    vals = [g_avgs.get(k, np.nan) for k in keys]

    fig_g, ax_g = plt.subplots(figsize=(5, 4))
    bar_colors = ['#A7C7E7', '#F4C7AB', '#B8E0D2']
    # 让 bar 更窄
    bar_width = 0.55
    bars = ax_g.bar(labels, vals, width=bar_width, color=bar_colors, edgecolor='black', linewidth=0.6)
    ax_g.set_ylabel('Average gap (%)')
    ax_g.set_title('Group average gaps')

    # 文本标签与 y 轴范围调整
    finite_vals = [v for v in vals if np.isfinite(v)]
    if finite_vals:
        minv = min(finite_vals)
        maxv = max(finite_vals)
        # 扩大 y-range，增加 25% 的上下 padding
        padding = (maxv - minv) * 0.25 if (maxv - minv) > 0 else maxv * 0.1 if maxv != 0 else 1.0
        # 确保 y 下限至少为 0，以便 bar 底端与 x 轴对齐
        bottom = min(0.0, minv - padding)
        # ax_g.set_ylim(bottom, maxv + padding)
        ax_g.set_ylim(0, maxv + padding)
        offset = (maxv - bottom) * 0.02 if (maxv - bottom) > 0 else padding * 0.02
    else:
        offset = 0.0

    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax_g.text(i, v + offset, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()

    # 保存文件名：基于 save_path 衍生，否则使用当前目录下 group_avgs.png
    import os
    if save_path:
        base, _ = os.path.splitext(save_path)
        group_save = f"{base}_group_avgs.png"
    else:
        group_save = 'group_avgs.png'

    plt.savefig(group_save, dpi=dpi, bbox_inches='tight')
    print(f"Saved group averages bar chart to {group_save}")




def plot_avg_gap_alldatasets(results, result2, results3, save_path=None, dpi=300): # average gap for models
    

    # === 将四个数据集的 group_avgs 画在同一张图上（同类模型的四个数据集放在一起） ===
    # compute group averages and group stds for each dataset
    # cg1 = compute_group_gaps(results)
    # cg2 = compute_group_gaps(result2)
    # cg3 = compute_group_gaps(results3)

    cg1 = compute_group_relative_gap(results)
    cg2 = compute_group_relative_gap(result2)
    cg3 = compute_group_relative_gap(results3)
    
    print("Dataset 1:", cg1)
    print("Dataset 2:", cg2)
    print("Dataset 3:", cg3)

    # 提取按 group1, group2, group3 的值与组内 std
    keys = ['group1', 'group2', 'group3']
    vals1 = [cg1['group_means'].get(k, np.nan) for k in keys]
    vals2 = [cg2['group_means'].get(k, np.nan) for k in keys]
    vals3 = [cg3['group_means'].get(k, np.nan) for k in keys]

    stds1 = [cg1['group_stds'].get(k, np.nan) for k in keys]
    stds2 = [cg2['group_stds'].get(k, np.nan) for k in keys]
    stds3 = [cg3['group_stds'].get(k, np.nan) for k in keys]

    all_vals = np.array(vals1 + vals2 + vals3, dtype=float)

    groups_labels = ['VLM', 'SL', 'SSL']
    dataset_labels = ['CheXpert', 'ChestX-14', 'CovidQuEx']

    x = np.arange(len(groups_labels))
    n_sets = 3
    total_width = 0.7
    width = total_width / n_sets
    # offsets so that bars are centered per group
    offsets = (np.arange(n_sets) - (n_sets - 1) / 2.0) * width

    fig_all, ax_all = plt.subplots(figsize=(8, 5))
    # colors = ['#F6E7B4', '#E5E7EB', '#E8C3A3']
    # colors = ['#AEC6FF', '#FFCFB3', '#B5EAD7', '#D9D9D9']
    # colors = ['#A7C7E7', '#F4C7AB', '#B8E0D2']
    colors = ['#E2DAF9', '#D9F2DC', '#EFEFEF']

    # replace nan in stds with 0 to avoid plotting errors
    s1 = np.array([0.0 if not np.isfinite(s) else s for s in stds1])
    s2 = np.array([0.0 if not np.isfinite(s) else s for s in stds2])
    s3 = np.array([0.0 if not np.isfinite(s) else s for s in stds3])

    bars1 = ax_all.bar(x + offsets[0], vals1, width=width, color=colors[0], edgecolor='black', linewidth=0.6,
                       yerr=s1, capsize=4, error_kw=dict(ecolor='gray', elinewidth=1))
    bars2 = ax_all.bar(x + offsets[1], vals2, width=width, color=colors[1], edgecolor='black', linewidth=0.6,
                       yerr=s2, capsize=4, error_kw=dict(ecolor='gray', elinewidth=1))
    bars3 = ax_all.bar(x + offsets[2], vals3, width=width, color=colors[2], edgecolor='black', linewidth=0.6,
                       yerr=s3, capsize=4, error_kw=dict(ecolor='gray', elinewidth=1))
    # ax_all.bar(x + offsets[3], vals4, width=width, color=colors[3], edgecolor='black', linewidth=0.6)

    ax_all.set_xticks(x)
    ax_all.set_xticklabels(groups_labels)
    ax_all.set_ylabel('Average gap (%)')
    ax_all.set_title('Group average gaps across datasets')

    # ensure bottoms align with x-axis (include 0)
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size:
        minv = 0.0
        maxv = float(np.max(finite))
        padding = (maxv - minv) * 0.1 if (maxv - minv) > 0 else maxv * 0.1 if maxv != 0 else 1.0
        # 强制 y 轴从 0 开始
        # ax_all.set_ylim(0, maxv + padding)
        ax_all.set_ylim(0, maxv + 4)
        label_offset = (maxv - minv) * 0.02 if (maxv - minv) > 0 else padding * 0.02
    else:
        ax_all.set_ylim(0, 1)
        label_offset = 0.02

    # # 为每个 bar 添加数值与组内 std 注记
    # for i_set, (vals, s_arr) in enumerate(zip([vals1, vals2, vals3], [s1, s2, s3])):
    #     for i, v in enumerate(vals):
    #         if np.isfinite(v):
    #             std_val = float(s_arr[i]) if np.isfinite(s_arr[i]) else 0.0
    #             if std_val > 0:
    #                 ax_all.text(x[i] + offsets[i_set], v + label_offset, f"{v:.2f}\n±{std_val:.2f}",
    #                              ha='center', va='bottom', fontsize=9)
    #             else:
    #                 ax_all.text(x[i] + offsets[i_set], v + label_offset, f"{v:.2f}",
    #                              ha='center', va='bottom', fontsize=9)

    # 绘制三组跨数据集的均值灰色横线，以及均值上下的虚线表示跨数据集 std
    group_means = []
    group_across_stds = []
    for i in range(len(groups_labels)):
        vals_i = [vals1[i], vals2[i], vals3[i]]
        finite_i = [v for v in vals_i if np.isfinite(v)]
        gm = float(np.mean(finite_i)) if finite_i else np.nan
        gstd = float(np.std(finite_i)) if finite_i else np.nan
        group_means.append(gm)
        group_across_stds.append(gstd)

    # 横线横跨每组的总宽度
    total_group_width = total_width
    for i, gm in enumerate(group_means):
        if np.isfinite(gm):
            xmin = x[i] - total_group_width / 2.0
            xmax = x[i] + total_group_width / 2.0
            ax_all.hlines(y=gm, xmin=xmin, xmax=xmax, colors='gray', linewidth=1.2)
            gstd = group_across_stds[i]
            # if np.isfinite(gstd) and gstd > 0:
            #     ax_all.hlines(y=gm + gstd, xmin=xmin, xmax=xmax, colors='gray', linestyles='--', linewidth=1.0)
            #     ax_all.hlines(y=gm - gstd, xmin=xmin, xmax=xmax, colors='gray', linestyles='--', linewidth=1.0)

    # add legend (datasets + single entry for group mean)
    # 用空 plot 仅为 group mean 添加图例项
    ax_all.plot([], [], color='gray', linewidth=1.5, label='Group mean')
    ax_all.legend(dataset_labels + ['Group mean'], frameon=False)

    # 添加 y 轴背景刻度线
    plt.grid(axis='y',
            linestyle='-',
            linewidth=0.6,
            color='#BFBFBF',
            alpha=0.6)
    plt.tight_layout()
    # save combined figure
    if save_path:
        base, _ = os.path.splitext(save_path)
        combined_save = f"{base}_group_avgs_all.png"
    else:
        combined_save = 'group_avgs_all.png'
    plt.savefig(combined_save, dpi=dpi, bbox_inches='tight')
    print(f"Saved combined group averages bar chart to {combined_save}")




def compute_relative_drop(results_dict):
    """
    计算 metal / brightness / gaussian 相对于 no_aug 的性能下降比例：
    (AUC_clean - AUC_perturbed) / AUC_clean
    
    返回：
        {
            model_name: {
                "metal": drop_ratio,
                "brightness": drop_ratio,
                "gaussian": drop_ratio
            },
            ...
        }
    """
    drop_dict = {}

    for model, metrics in results_dict.items():
        auc_clean = metrics["no_aug"]
        drop_dict[model] = {}

        for noise_type in ["metal", "brightness", "gaussian"]:
            auc_perturbed = metrics[noise_type]
            drop_ratio = (auc_clean - auc_perturbed) / auc_clean
            drop_dict[model][noise_type] = drop_ratio*100

    return drop_dict


def plot_robustness_permodel_perdataset(save_path=None, dpi=300):

    models = ["CARZero", "RadZero", "DeViDe", "KAD", "MAVL", "MedKLIP",  "Ark+", "FoundationX", "RAD-DINO", "EVA-X", "CheXWorld", "Lamps", "Adam-v2"]

    results = {  # CheXpert
        "CARZero": {"no_aug": 92.38, "metal": 91.64, "brightness": 91.59, "gaussian": 91.54},
        "RadZero": {"no_aug": 90.16, "metal": 90.13, "brightness": 90.1, "gaussian": 90.14},
        "DeViDe": {"no_aug": 89.87, "metal": 89.18, "brightness": 89.13, "gaussian": 89.02},
        "KAD": {"no_aug": 89.23, "metal": 83.86, "brightness": 84.81, "gaussian": 84.10},
        "MAVL": {"no_aug": 90.13, "metal": 88.61, "brightness": 89.8, "gaussian": 88.96},
        "MedKLIP": {"no_aug": 90.06, "metal": 88.16, "brightness": 88.18, "gaussian": 88.39},
        "Ark+": {"no_aug": 89.67, "metal": 89.87, "brightness": 89.83, "gaussian": 89.6},
        "FoundationX": {"no_aug": 89.55, "metal": 85.92, "brightness": 87.36, "gaussian": 87.14},
        "RAD-DINO": {"no_aug": 87.87, "metal": 87.4, "brightness": 87.77, "gaussian": 87.76},
        "EVA-X": {"no_aug": 88.02, "metal": 81.67, "brightness": 88.19, "gaussian": 88.04},
        "CheXWorld": {"no_aug": 85.84, "metal": 81.94, "brightness": 84.5, "gaussian": 83.89},
        "Lamps": {"no_aug": 82.68, "metal": 81.74, "brightness": 82.25, "gaussian": 81.84},
        "Adam-v2": {"no_aug": 80.81, "metal": 71.2, "brightness": 74.06, "gaussian": 73.53},
    }
    
    
    
    
    result2 = {  # ChestXray-14
        "CARZero": {"no_aug": 77.67, "metal": 76.58, "brightness": 77.44, "gaussian": 76.63},
        "RadZero": {"no_aug": 75.92, "metal": 75.74, "brightness": 75.82, "gaussian": 75.23},
        "DeViDe": {"no_aug": 77.61, "metal": 75.42, "brightness": 76.71, "gaussian": 75.63},
        "KAD": {"no_aug": 76.99, "metal": 74.81, "brightness": 75.96, "gaussian": 74.86},
        "MAVL": {"no_aug": 73.43, "metal": 66.91, "brightness": 71.10, "gaussian": 67.19},
        "MedKLIP": {"no_aug": 72.68, "metal": 70.95, "brightness": 72.51, "gaussian": 71.09}, # need update
        "Ark+": {"no_aug": 84.42, "metal": 84.29, "brightness": 84.41, "gaussian": 83.28},
        "FoundationX": {"no_aug": 83.42, "metal": 81.21, "brightness": 82.49, "gaussian": 79.95},
        "RAD-DINO": {"no_aug": 79.98, "metal": 79.3, "brightness": 79.97, "gaussian": 79.81},
        "EVA-X": {"no_aug": 79.8, "metal": 74.75, "brightness": 79.63, "gaussian": 78.47},
        "CheXWorld": {"no_aug": 78.26, "metal": 73.3, "brightness": 78.09, "gaussian": 76.08},
        "Lamps": {"no_aug": 72.89, "metal": 72.65, "brightness": 72.85, "gaussian": 71.68},
        "Adam-v2": {"no_aug": 72.88, "metal": 66.66, "brightness": 68.31, "gaussian": 64.66},
    }


    # results3 = {  # RSNA
    #     "CARZero": {"no_aug": 77.74, "metal": 73.92, "brightness": 70.72, "gaussian": 70.05},   
    #     "RadZero": {"no_aug": 85.42, "metal": 84.77, "brightness": 85.02, "gaussian": 84.4},
    #     "DeViDe": {"no_aug": 88.58, "metal": 81.33, "brightness": 87.16, "gaussian": 78.63}, 
    #     "KAD": {"no_aug": 85.32, "metal": 82.71, "brightness": 86.20, "gaussian": 81.50},
    #     "MAVL": {"no_aug": 90.69, "metal": 91.03, "brightness": 91.23, "gaussian": 91.38},   
    #     "MedKLIP": {"no_aug": 89.06, "metal": 85.58, "brightness": 89.76, "gaussian": 86.56},    
    #     "Ark+": {"no_aug": 88.55, "metal": 88.47, "brightness": 88.51, "gaussian": 88.23},
    #     "FoundationX": {"no_aug": 87.05, "metal": 85.92, "brightness": 86.5, "gaussian": 86.04},
    #     "RAD-DINO": {"no_aug": 85.47, "metal": 85.46, "brightness": 85.48, "gaussian": 85.46},
    #     "EVA-X": {"no_aug": 85.62, "metal": 83.61, "brightness": 85.62, "gaussian": 85.47},
    #     "CheXWorld": {"no_aug": 84.29, "metal": 82.67, "brightness": 84.36, "gaussian": 84.14},
    #     "Lamps": {"no_aug": 82.5, "metal": 82.38, "brightness": 82.52, "gaussian": 81.45},
    #     "Adam-v2": {"no_aug": 81.83, "metal": 77.23, "brightness": 76.84, "gaussian": 75.3},
    # }
    
    results3 = {  # CovidQuEx
        "CARZero": {"no_aug": 83.76, "metal": 62.29, "brightness": 81.83, "gaussian": 77.55},       
        "RadZero": {"no_aug": 86.57, "metal": 79.01, "brightness": 86.28, "gaussian": 86.57}, 
        "DeViDe": {"no_aug": 87.03, "metal": 71.22, "brightness": 88.16, "gaussian": 83.01},    
        "KAD": {"no_aug": 87.81, "metal": 65.85, "brightness": 87.40, "gaussian": 83.93},   
        "MAVL": {"no_aug": 87.39, "metal": 84.47, "brightness": 86.92, "gaussian": 87.61},      
        "MedKLIP": {"no_aug": 82.79, "metal": 63.21, "brightness": 84.66, "gaussian": 80.19},       
        "Ark+": {"no_aug": 99.05, "metal": 98.85, "brightness": 99, "gaussian": 94.86},
        "FoundationX": {"no_aug": 97.27, "metal": 96.57, "brightness": 97.27, "gaussian": 92.74},
        "RAD-DINO": {"no_aug": 98.95, "metal": 98.76, "brightness": 98.95, "gaussian": 98.72},
        "EVA-X": {"no_aug": 98.19, "metal": 94.95, "brightness": 98.19, "gaussian": 96.46},
        "CheXWorld": {"no_aug": 97.67, "metal": 96.26, "brightness": 97.68, "gaussian": 97.39},
        "Lamps": {"no_aug": 96.42, "metal": 96.23, "brightness": 96.25, "gaussian": 87.92},
        "Adam-v2": {"no_aug": 96.48, "metal": 96.71, "brightness": 96.48, "gaussian": 91.7},
    }
    
    
    
    
    

    
    
    
    
    
    # plot for each dataset
    # models, results, scores = sort_models_by_robustness(models, results)
    # results = results  # 切换到 CheXpert 数据集的结果进行绘图
    results = result2  # 切换到 NIH 数据集的结果进行绘图
    # results = results3  # 切换到 CovidQuEx 数据集的结果进行绘图
    
    

    metal = [results[m]["metal"] for m in models]
    brightness = [results[m]["brightness"] for m in models]
    gaussian = [results[m]["gaussian"] for m in models]
    no_aug = [results[m]["no_aug"] for m in models]

    x = np.arange(len(models))
    width = 0.18

    # 论文风格参数
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.0,
    })

    plt.figure(figsize=(7, 5))

    # 低饱和浅色
    color_metal = "#A7C7E7"       # 浅蓝灰
    color_brightness = "#F4C7AB"  # 浅橘灰
    color_gaussian = "#B8E0D2"    # 浅绿灰
    baseline_color = "#4D4D4D"    # 深灰

    plt.bar(x - width, metal, width,
        color=color_metal, edgecolor='black', linewidth=0.6)

    plt.bar(x, brightness, width,
        color=color_brightness, edgecolor='black', linewidth=0.6)

    plt.bar(x + width, gaussian, width,
        color=color_gaussian, edgecolor='black', linewidth=0.6)

    # baseline 横线（稍微加粗一点）
    for i in range(len(models)):
        plt.hlines(no_aug[i], i - 0.4, i + 0.4,
                   colors=baseline_color, linestyles='--', linewidth=1.5)
        
    
    # # add labels on bars (optional)
    # plt.bar(x - width, metal, width, label="Metal Artifact",
    #         color=color_metal, edgecolor='black', linewidth=0.6)

    # plt.bar(x, brightness, width, label="Brightness Enhance",
    #         color=color_brightness, edgecolor='black', linewidth=0.6)

    # plt.bar(x + width, gaussian, width, label="Gaussian Noise",
    #         color=color_gaussian, edgecolor='black', linewidth=0.6)

    # # baseline 横线（稍微加粗一点）
    # for i in range(len(models)):
    #     if i == 0:
    #         plt.hlines(no_aug[i], i - 0.4, i + 0.4,
    #                    colors=baseline_color, linestyles='--', linewidth=1.5,
    #                    label="No noise")
    #     else:
    #         plt.hlines(no_aug[i], i - 0.4, i + 0.4,
    #                    colors=baseline_color, linestyles='--', linewidth=1.5)


    plt.xticks(x, models, rotation=20)
    # plt.ylabel("RSNA / Acc (%)")
    plt.ylabel("ChesXray-14")

    # 自动 y 轴范围
    all_vals = metal + brightness + gaussian + no_aug
    min_v = min(all_vals) - 3
    max_v = max(all_vals) + 3
    plt.ylim(min_v, max_v)
    
    
    # 添加 y 轴背景刻度线
    plt.grid(axis='y',
            linestyle='-',
            linewidth=0.6,
            color='#BFBFBF',
            alpha=0.6)

    plt.gca().set_axisbelow(True)  # 让网格在线条下面

    # Legend removed as labels are omitted

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
    
    
    
    
    


def plot_robustness_permodel_perdataset_dropratio(save_path=None, dpi=300):

    models = ["CARZero", "RadZero", "DeViDe", "KAD", "MAVL", "MedKLIP",  "Ark+", "FoundationX", "RAD-DINO", "EVA-X", "CheXWorld", "Lamps", "Adam-v2"]

    results = {  # CheXpert
        "CARZero": {"no_aug": 92.38, "metal": 91.64, "brightness": 91.59, "gaussian": 91.54},
        "RadZero": {"no_aug": 90.16, "metal": 90.13, "brightness": 90.1, "gaussian": 90.14},
        "DeViDe": {"no_aug": 89.87, "metal": 89.18, "brightness": 89.13, "gaussian": 89.02},
        "KAD": {"no_aug": 89.23, "metal": 83.86, "brightness": 84.81, "gaussian": 84.10},
        "MAVL": {"no_aug": 90.13, "metal": 88.61, "brightness": 89.8, "gaussian": 88.96},
        "MedKLIP": {"no_aug": 90.06, "metal": 88.16, "brightness": 88.18, "gaussian": 88.39},
        "Ark+": {"no_aug": 89.67, "metal": 89.87, "brightness": 89.83, "gaussian": 89.6},
        "FoundationX": {"no_aug": 89.55, "metal": 85.92, "brightness": 87.36, "gaussian": 87.14},
        "RAD-DINO": {"no_aug": 87.87, "metal": 87.4, "brightness": 87.77, "gaussian": 87.76},
        "EVA-X": {"no_aug": 88.02, "metal": 81.67, "brightness": 88.19, "gaussian": 88.04},
        "CheXWorld": {"no_aug": 85.84, "metal": 81.94, "brightness": 84.5, "gaussian": 83.89},
        "Lamps": {"no_aug": 82.68, "metal": 81.74, "brightness": 82.25, "gaussian": 81.84},
        "Adam-v2": {"no_aug": 80.81, "metal": 71.2, "brightness": 74.06, "gaussian": 73.53},
    }
    
    
    
    
    result2 = {  # ChestXray-14
        "CARZero": {"no_aug": 77.67, "metal": 76.58, "brightness": 77.44, "gaussian": 76.63},
        "RadZero": {"no_aug": 75.92, "metal": 75.74, "brightness": 75.82, "gaussian": 75.23},
        "DeViDe": {"no_aug": 77.61, "metal": 75.42, "brightness": 76.71, "gaussian": 75.63},
        "KAD": {"no_aug": 76.99, "metal": 74.81, "brightness": 75.96, "gaussian": 74.86},
        "MAVL": {"no_aug": 73.43, "metal": 66.91, "brightness": 71.10, "gaussian": 67.19},
        "MedKLIP": {"no_aug": 72.68, "metal": 70.95, "brightness": 72.51, "gaussian": 71.09}, # need update
        "Ark+": {"no_aug": 84.42, "metal": 84.29, "brightness": 84.41, "gaussian": 83.28},
        "FoundationX": {"no_aug": 83.42, "metal": 81.21, "brightness": 82.49, "gaussian": 79.95},
        "RAD-DINO": {"no_aug": 79.98, "metal": 79.3, "brightness": 79.97, "gaussian": 79.81},
        "EVA-X": {"no_aug": 79.8, "metal": 74.75, "brightness": 79.63, "gaussian": 78.47},
        "CheXWorld": {"no_aug": 78.26, "metal": 73.3, "brightness": 78.09, "gaussian": 76.08},
        "Lamps": {"no_aug": 72.89, "metal": 72.65, "brightness": 72.85, "gaussian": 71.68},
        "Adam-v2": {"no_aug": 72.88, "metal": 66.66, "brightness": 68.31, "gaussian": 64.66},
    }


    # results3 = {  # RSNA
    #     "CARZero": {"no_aug": 77.74, "metal": 73.92, "brightness": 70.72, "gaussian": 70.05},   
    #     "RadZero": {"no_aug": 85.42, "metal": 84.77, "brightness": 85.02, "gaussian": 84.4},
    #     "DeViDe": {"no_aug": 88.58, "metal": 81.33, "brightness": 87.16, "gaussian": 78.63}, 
    #     "KAD": {"no_aug": 85.32, "metal": 82.71, "brightness": 86.20, "gaussian": 81.50},
    #     "MAVL": {"no_aug": 90.69, "metal": 91.03, "brightness": 91.23, "gaussian": 91.38},   
    #     "MedKLIP": {"no_aug": 89.06, "metal": 85.58, "brightness": 89.76, "gaussian": 86.56},    
    #     "Ark+": {"no_aug": 88.55, "metal": 88.47, "brightness": 88.51, "gaussian": 88.23},
    #     "FoundationX": {"no_aug": 87.05, "metal": 85.92, "brightness": 86.5, "gaussian": 86.04},
    #     "RAD-DINO": {"no_aug": 85.47, "metal": 85.46, "brightness": 85.48, "gaussian": 85.46},
    #     "EVA-X": {"no_aug": 85.62, "metal": 83.61, "brightness": 85.62, "gaussian": 85.47},
    #     "CheXWorld": {"no_aug": 84.29, "metal": 82.67, "brightness": 84.36, "gaussian": 84.14},
    #     "Lamps": {"no_aug": 82.5, "metal": 82.38, "brightness": 82.52, "gaussian": 81.45},
    #     "Adam-v2": {"no_aug": 81.83, "metal": 77.23, "brightness": 76.84, "gaussian": 75.3},
    # }
    
    results3 = {  # CovidQuEx
        "CARZero": {"no_aug": 83.76, "metal": 62.29, "brightness": 81.83, "gaussian": 77.55},       
        "RadZero": {"no_aug": 86.57, "metal": 79.01, "brightness": 86.28, "gaussian": 86.57}, 
        "DeViDe": {"no_aug": 87.03, "metal": 71.22, "brightness": 88.16, "gaussian": 83.01},    
        "KAD": {"no_aug": 87.81, "metal": 65.85, "brightness": 87.40, "gaussian": 83.93},   
        "MAVL": {"no_aug": 87.39, "metal": 84.47, "brightness": 86.92, "gaussian": 87.61},      
        "MedKLIP": {"no_aug": 82.79, "metal": 63.21, "brightness": 84.66, "gaussian": 80.19},       
        "Ark+": {"no_aug": 99.05, "metal": 98.85, "brightness": 99, "gaussian": 94.86},
        "FoundationX": {"no_aug": 97.27, "metal": 96.57, "brightness": 97.27, "gaussian": 92.74},
        "RAD-DINO": {"no_aug": 98.95, "metal": 98.76, "brightness": 98.95, "gaussian": 98.72},
        "EVA-X": {"no_aug": 98.19, "metal": 94.95, "brightness": 98.19, "gaussian": 96.46},
        "CheXWorld": {"no_aug": 97.67, "metal": 96.26, "brightness": 97.68, "gaussian": 97.39},
        "Lamps": {"no_aug": 96.42, "metal": 96.23, "brightness": 96.25, "gaussian": 87.92},
        "Adam-v2": {"no_aug": 96.48, "metal": 96.71, "brightness": 96.48, "gaussian": 91.7},
    }
    
    
    
    # plot for each dataset
    # models, results, scores = sort_models_by_robustness(models, results)
    # results = results  # 切换到 CheXpert 数据集的结果进行绘图
    results = result2  # 切换到 NIH 数据集的结果进行绘图
    # results = results3  # 切换到 CovidQuEx 数据集的结果进行绘图
    
    # compute drop ratios relative to clean performance
    results_drop = compute_relative_drop(results)
    # print(results_drop)

    # use the ratios for plotting instead of absolute AUCs
    metal = [results_drop[m]["metal"] for m in models]
    brightness = [results_drop[m]["brightness"] for m in models]
    gaussian = [results_drop[m]["gaussian"] for m in models]
    # keep no_aug only for baseline line reference (set to zero)
    no_aug = [0.0 for _ in models]

    x = np.arange(len(models))
    width = 0.18

    # 论文风格参数
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.0,
    })

    plt.figure(figsize=(8, 5))

    # 低饱和浅色
    color_metal = "#A7C7E7"       # 浅蓝灰
    color_brightness = "#F4C7AB"  # 浅橘灰
    color_gaussian = "#B8E0D2"    # 浅绿灰
    baseline_color = "#4D4D4D"    # 深灰

    plt.bar(x - width, metal, width,
        color=color_metal, edgecolor='black', linewidth=0.6)

    plt.bar(x, brightness, width,
        color=color_brightness, edgecolor='black', linewidth=0.6)

    plt.bar(x + width, gaussian, width,
        color=color_gaussian, edgecolor='black', linewidth=0.6)



    plt.xticks(x, models, rotation=20)
    # plt.ylabel("RSNA / Acc (%)")
    plt.ylabel("Relative drop ratio")

    # 自动 y 轴范围（从 0 开始，因为 ratios 是非负的）
    all_vals = metal + brightness + gaussian + no_aug
    max_v = max(all_vals) if all_vals else 0.0
    
    plt.ylim(0, max_v + 1) # nih
    # plt.ylim(-3, max_v + 1) # covid
    # plt.ylim(-1, max_v + 1) # chexpert
    
    
    # 添加 y 轴背景刻度线
    plt.grid(axis='y',
            linestyle='-',
            linewidth=0.6,
            color='#BFBFBF',
            alpha=0.6)

    plt.gca().set_axisbelow(True)  # 让网格在线条下面

    # Legend removed as labels are omitted

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    
    plot_avg_gap_alldatasets(results, result2, results3)



if __name__ == '__main__':
    # plot_robustness('./figures/vis_miccai26/robustness_rsna.png')
    # plot_robustness('./figures/vis_miccai26/robustness_covidquex.png')
    # plot_robustness('./figures/vis_miccai26/robustness_nih.png')
    # plot_robustness('./figures/vis_miccai26/robustness_chexpert.png')
    # plot_robustness_dual('./figures/vis_miccai26/robustness_dual.png')
    plot_robustness_permodel_perdataset_dropratio('./figures/vis_miccai26/NIH_drop_ratio.png')
    # plot_robustness_permodel_perdataset_dropratio('./figures/vis_miccai26/CovidQuEx_drop_ratio.png')
    # plot_robustness_permodel_perdataset_dropratio('./figures/vis_miccai26/Chexpert_drop_ratio.png')
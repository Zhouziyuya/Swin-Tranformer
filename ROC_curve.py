import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib

# 设置中文字体（如果需要显示中文）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_model_results(file_paths):
    """
    加载多个模型的结果文件
    
    参数:
    file_paths: 模型结果文件路径列表
    
    返回:
    字典: {模型名: {图片名: (预测概率列表, 真实标签列表)}}
    """
    models_data = {}
    
    models = ['POPAR', 'DINO', 'BYOL', 'ACEv2_SwinV1-B', 'ACEv2']
    for i, file_path in enumerate(file_paths):
        model_name = models[i]
        model_data = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():  # 跳过空行
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) < 17:  # 文件名 + 8个预测概率 + 8个标签 = 17
                    continue
                    
                img_name = parts[0]
                # 前8个是预测概率（从索引1到8）
                pred_probs = [float(x) for x in parts[1:9]]
                # 后8个是真实标签（从索引9到16）
                true_labels = [int(x) for x in parts[9:17]]
                
                model_data[img_name] = (pred_probs, true_labels)
        
        models_data[model_name] = model_data
        print(f"已加载 {model_name}: {len(model_data)} 个样本")
    
    return models_data

def compute_roc_curves(models_data, n_classes=8):
    """
    计算每个模型、每个类别的ROC曲线
    
    返回:
    fpr_dict: 字典 {模型名: 字典 {类别: FPR值}}
    tpr_dict: 字典 {模型名: 字典 {类别: TPR值}}
    auc_dict: 字典 {模型名: 字典 {类别: AUC值}}
    """
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    
    for model_name, data in models_data.items():
        fpr_dict[model_name] = {}
        tpr_dict[model_name] = {}
        auc_dict[model_name] = {}
        
        # 为每个类别收集所有样本的预测和真实标签
        for class_idx in range(n_classes):
            y_true = []
            y_score = []
            
            for img_name, (pred_probs, true_labels) in data.items():
                y_true.append(true_labels[class_idx])
                y_score.append(pred_probs[class_idx])
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            fpr_dict[model_name][class_idx] = fpr
            tpr_dict[model_name][class_idx] = tpr
            auc_dict[model_name][class_idx] = roc_auc
    
    return fpr_dict, tpr_dict, auc_dict

def compute_average_roc(models_data, n_classes=8):
    """
    计算每个模型的平均ROC曲线
    
    返回:
    avg_fpr_dict: 字典 {模型名: 平均FPR}
    avg_tpr_dict: 字典 {模型名: 平均TPR}
    avg_auc_dict: 字典 {模型名: 平均AUC}
    """
    avg_fpr_dict = {}
    avg_tpr_dict = {}
    avg_auc_dict = {}
    
    for model_name, data in models_data.items():
        # 收集所有类别、所有样本的预测和真实标签
        all_y_true = []
        all_y_score = []
        
        for img_name, (pred_probs, true_labels) in data.items():
            all_y_true.extend(true_labels)
            all_y_score.extend(pred_probs)
        
        # 计算micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(all_y_true, all_y_score)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        avg_fpr_dict[model_name] = fpr_micro
        avg_tpr_dict[model_name] = tpr_micro
        avg_auc_dict[model_name] = roc_auc_micro
    
    return avg_fpr_dict, avg_tpr_dict, avg_auc_dict

def plot_class_roc_curves(fpr_dict, tpr_dict, auc_dict, n_classes=8):
    """
    为每个类别绘制ROC曲线
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # 设置颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        # 绘制每个模型的ROC曲线
        for i, (model_name, fpr_data) in enumerate(fpr_dict.items()):
            fpr = fpr_data[class_idx]
            tpr = tpr_dict[model_name][class_idx]
            roc_auc = auc_dict[model_name][class_idx]
            
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # 绘制对角线
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Class {class_idx} ROC Curves')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_classes, 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('./figures/class_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_average_roc_curves(avg_fpr_dict, avg_tpr_dict, avg_auc_dict):
    """
    绘制平均ROC曲线
    """
    plt.figure(figsize=(10, 8))
    
    # 设置颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    
    # 绘制每个模型的平均ROC曲线
    for i, (model_name, fpr) in enumerate(avg_fpr_dict.items()):
        tpr = avg_tpr_dict[model_name]
        roc_auc = avg_auc_dict[model_name]
        
        plt.plot(fpr, tpr, color=colors[i], lw=3,
                label=f'{model_name} (Micro-average AUC = {roc_auc:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Micro-average ROC Curves for All Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/average_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_and_display_statistics(fpr_dict, tpr_dict, auc_dict, avg_auc_dict):
    """
    计算并显示统计信息
    """
    print("=" * 60)
    print("ROC曲线AUC统计信息")
    print("=" * 60)
    
    # 获取所有模型名
    model_names = list(auc_dict.keys())
    n_classes = len(auc_dict[model_names[0]])
    
    # 打印每个模型的每个类别的AUC
    for model_name in model_names:
        print(f"\n{model_name}:")
        print("-" * 40)
        
        class_aucs = []
        for class_idx in range(n_classes):
            auc_value = auc_dict[model_name][class_idx]
            class_aucs.append(auc_value)
            print(f"  Class {class_idx}: AUC = {auc_value:.4f}")
        
        # 计算每个模型的平均AUC（跨类别）
        mean_auc = np.mean(class_aucs)
        std_auc = np.std(class_aucs)
        print(f"  Mean AUC (across classes): {mean_auc:.4f} ± {std_auc:.4f}")
    
    # 打印micro-average AUC
    print("\n" + "=" * 60)
    print("Micro-average AUC (所有类别合并计算):")
    print("-" * 60)
    for model_name, auc_value in avg_auc_dict.items():
        print(f"{model_name}: AUC = {auc_value:.4f}")
    
    # 比较模型性能
    print("\n" + "=" * 60)
    print("模型性能排名 (按Micro-average AUC):")
    print("-" * 60)
    sorted_models = sorted(avg_auc_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, auc_value) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}: AUC = {auc_value:.4f}")

# 主程序
if __name__ == "__main__":
    # 设置模型结果文件路径
    model_files = [
        '/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/popar_conf_8class.csv',
        '/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/dino_conf_8class.csv',
        '/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/byol_conf_8class.csv', 
        '/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/ACEv2_swinv2_large/conf_8class.csv',
        '/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/ACEv2_swinv2/conf_8class.csv'
    ]
    
    # 1. 加载数据
    print("正在加载模型结果...")
    models_data = load_model_results(model_files)
    
    # 2. 计算每个类别的ROC曲线
    print("\n正在计算每个类别的ROC曲线...")
    fpr_dict, tpr_dict, auc_dict = compute_roc_curves(models_data, n_classes=8)
    
    # 3. 计算平均ROC曲线
    print("\n正在计算平均ROC曲线...")
    avg_fpr_dict, avg_tpr_dict, avg_auc_dict = compute_average_roc(models_data, n_classes=8)
    
    # 4. 绘制每个类别的ROC曲线
    print("\n正在绘制每个类别的ROC曲线...")
    plot_class_roc_curves(fpr_dict, tpr_dict, auc_dict, n_classes=8)
    
    # 5. 绘制平均ROC曲线
    print("正在绘制平均ROC曲线...")
    plot_average_roc_curves(avg_fpr_dict, avg_tpr_dict, avg_auc_dict)
    
    # 6. 显示统计信息
    compute_and_display_statistics(fpr_dict, tpr_dict, auc_dict, avg_auc_dict)
    
    print("\n所有ROC曲线已保存为图片文件！")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# 读取数据
df = pd.read_csv('seed_metrics/seed_metrics_extended_plus_highD.csv')

# 过滤N=1024的数据
df = df[df['N'] == 1024]

# 按N,k,D分组计算均值和标准误
grouped = df.groupby(['N', 'k', 'D'])
aggregated = grouped.agg({
    'J_phi': ['mean', 'sem'],
    'J_conn': ['mean', 'sem'],
    'Jalpha_conn_0.8': ['mean', 'sem'],
    'Jalpha_conn_1.0': ['mean', 'sem'],
    'Jalpha_conn_1.2': ['mean', 'sem']
}).reset_index()

# 重命名列
aggregated.columns = ['N', 'k', 'D', 'J_phi_mean', 'J_phi_sem', 'J_conn_mean', 'J_conn_sem',
                      'Jalpha_conn_0.8_mean', 'Jalpha_conn_0.8_sem',
                      'Jalpha_conn_1.0_mean', 'Jalpha_conn_1.0_sem',
                      'Jalpha_conn_1.2_mean', 'Jalpha_conn_1.2_sem']

# 计算bootstrap汇总（只针对conn）
def calculate_bootstrap(df, response_key, n_bootstrap=1000):
    results = []
    for (N, k), group in df.groupby(['N', 'k']):
        # 提取D值并排序
        D_values = sorted(group['D'].unique())
        
        # 对每个seed计算J_conn随D的曲线
        seed_curves = {}
        for seed in group['seed'].unique():
            seed_data = group[group['seed'] == seed]
            seed_curves[seed] = seed_data.set_index('D')['J_conn']
        
        # Bootstrap过程
        Dc_bootstrap = []
        Dopt_bootstrap = []
        Delta_bootstrap = []
        Valley_bootstrap = []
        
        for _ in range(n_bootstrap):
            # 随机选择seed（有放回）
            sampled_seeds = np.random.choice(list(seed_curves.keys()), size=len(seed_curves), replace=True)
            
            # 计算平均曲线
            mean_curve = np.zeros(len(D_values))
            for seed in sampled_seeds:
                mean_curve += seed_curves[seed].values
            mean_curve /= len(sampled_seeds)
            
            # 找到D_opt（J_conn最小的点）
            Dopt_idx = np.argmin(mean_curve)
            Dopt = D_values[Dopt_idx]
            
            # 找到D_c（在D_opt之前的第一个明显下降点）
            Dc = D_values[0]  # 默认值
            if Dopt_idx > 0:
                # 只在D_opt之前寻找
                for i in range(Dopt_idx):
                    if mean_curve[i] > mean_curve[i+1] * 1.01:  # 下降超过1%
                        Dc = D_values[i]
                        break
                # 如果没有找到明显下降，使用D_opt之前的中间点
                if Dc == D_values[0] and Dopt_idx > 1:
                    Dc = D_values[Dopt_idx // 2]
            
            # 确保D_c < D_opt
            if Dc >= Dopt:
                if Dopt_idx > 0:
                    Dc = D_values[Dopt_idx - 1]
                else:
                    Dc = D_values[0]
            
            # 计算Delta和Valley
            Delta = Dopt - Dc
            Valley = mean_curve[Dopt_idx]
            
            Dc_bootstrap.append(Dc)
            Dopt_bootstrap.append(Dopt)
            Delta_bootstrap.append(Delta)
            Valley_bootstrap.append(Valley)
        
        # 计算统计量
        Dc_mean = np.mean(Dc_bootstrap)
        Dc_ci_lo = np.percentile(Dc_bootstrap, 2.5)
        Dc_ci_hi = np.percentile(Dc_bootstrap, 97.5)
        
        Dopt_mean = np.mean(Dopt_bootstrap)
        Dopt_ci_lo = np.percentile(Dopt_bootstrap, 2.5)
        Dopt_ci_hi = np.percentile(Dopt_bootstrap, 97.5)
        
        Delta_mean = np.mean(Delta_bootstrap)
        Delta_ci_lo = np.percentile(Delta_bootstrap, 2.5)
        Delta_ci_hi = np.percentile(Delta_bootstrap, 97.5)
        
        Valley_mean = np.mean(Valley_bootstrap)
        Valley_ci_lo = np.percentile(Valley_bootstrap, 2.5)
        Valley_ci_hi = np.percentile(Valley_bootstrap, 97.5)
        
        results.append({
            'N': N,
            'k': k,
            'response_key': response_key,
            'Dc_mean': Dc_mean,
            'Dc_ci_lo': Dc_ci_lo,
            'Dc_ci_hi': Dc_ci_hi,
            'Dopt_mean': Dopt_mean,
            'Dopt_ci_lo': Dopt_ci_lo,
            'Dopt_ci_hi': Dopt_ci_hi,
            'Delta_mean': Delta_mean,
            'Delta_ci_lo': Delta_ci_lo,
            'Delta_ci_hi': Delta_ci_hi,
            'Valley_mean': Valley_mean,
            'Valley_ci_lo': Valley_ci_lo,
            'Valley_ci_hi': Valley_ci_hi
        })
    
    return pd.DataFrame(results)

# 计算bootstrap汇总
bootstrap_df = calculate_bootstrap(df, 'conn')

# 保存bootstrap汇总到文件
bootstrap_df.to_csv('bootstrap_summary_conn_from_extended_plus_highD.csv', index=False)

# 创建图表 - 1×3 主panel结构
fig = plt.figure(figsize=(24, 8))

# 总标题
plt.suptitle('Experiment 3 | Robustness of the interior optimum', fontsize=16, y=1.02)

# Panel A: Alternative response definitions (J_phi vs J_conn for different k)
fig.text(0.2, 0.95, 'A', fontsize=14, fontweight='bold', ha='center')
fig.text(0.2, 0.90, 'Alternative response definitions', fontsize=12, ha='center')

# 创建4个子图，共享y轴
k_values = [5, 7, 9, 12]
axes = []
for i, k in enumerate(k_values):
    if i == 0:
        ax = fig.add_subplot(2, 4, i+1)
        axes.append(ax)
    else:
        ax = fig.add_subplot(2, 4, i+1, sharey=axes[0])
        axes.append(ax)
    
    k_data = aggregated[aggregated['k'] == k]
    D_values = k_data['D']
    
    # 绘制J_conn（蓝实线）
    ax.errorbar(D_values, k_data['J_conn_mean'], yerr=k_data['J_conn_sem'], 
                label='J_conn', color='blue', linestyle='-', marker='o', capsize=3, markersize=4)
    
    # 绘制J_phi（红虚线）
    ax.errorbar(D_values, k_data['J_phi_mean'], yerr=k_data['J_phi_sem'], 
                label='J_phi', color='red', linestyle='--', marker='s', capsize=3, markersize=4)
    
    ax.set_title(f'k={k}')
    ax.set_xlabel('D')
    if i == 0:
        ax.set_ylabel('J')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # 只在最后一个子图显示图例
    if i == 3:
        ax.legend(fontsize=8, loc='upper right')
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')

# Panel B: Exponent sensitivity (Jalpha_conn for k=7)
fig.text(0.5, 0.95, 'B', fontsize=14, fontweight='bold', ha='center')
fig.text(0.5, 0.90, 'Exponent sensitivity', fontsize=12, ha='center')
ax2 = fig.add_subplot(2, 4, 5)

# 只画k=7的数据
k_data = aggregated[aggregated['k'] == 7]
D_values = k_data['D']

# 绘制不同alpha值的曲线
ax2.errorbar(D_values, k_data['Jalpha_conn_0.8_mean'], yerr=k_data['Jalpha_conn_0.8_sem'], 
             label='α=0.8', color='green', linestyle='-', marker='o', capsize=3, markersize=4)
ax2.errorbar(D_values, k_data['Jalpha_conn_1.0_mean'], yerr=k_data['Jalpha_conn_1.0_sem'], 
             label='α=1.0', color='blue', linestyle='-', marker='s', capsize=3, markersize=4)
ax2.errorbar(D_values, k_data['Jalpha_conn_1.2_mean'], yerr=k_data['Jalpha_conn_1.2_sem'], 
             label='α=1.2', color='red', linestyle='-', marker='^', capsize=3, markersize=4)

ax2.set_title('k=7')
ax2.set_xlabel('D')
ax2.set_ylabel('J_alpha^{conn}')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Panel C: Bootstrap extraction of D_c and D_opt
fig.text(0.8, 0.95, 'C', fontsize=14, fontweight='bold', ha='center')
fig.text(0.8, 0.90, 'Bootstrap extraction of D_c and D_opt', fontsize=12, ha='center')
ax3 = fig.add_subplot(2, 4, 7)
k_values = [5, 7, 9, 12]

# 绘制D_c和D_opt的点及置信区间
for i, k in enumerate(k_values):
    k_data = bootstrap_df[bootstrap_df['k'] == k]
    if not k_data.empty:
        # 计算误差条（确保非负）
        dc_err_lo = max(0, k_data['Dc_mean'].values[0] - k_data['Dc_ci_lo'].values[0])
        dc_err_hi = max(0, k_data['Dc_ci_hi'].values[0] - k_data['Dc_mean'].values[0])
        
        dopt_err_lo = max(0, k_data['Dopt_mean'].values[0] - k_data['Dopt_ci_lo'].values[0])
        dopt_err_hi = max(0, k_data['Dopt_ci_hi'].values[0] - k_data['Dopt_mean'].values[0])
        
        # D_c
        ax3.errorbar(i, k_data['Dc_mean'].values[0], 
                    yerr=[[dc_err_lo], [dc_err_hi]],
                    fmt='o', color='red', capsize=5, label='D_c' if i == 0 else '', markersize=6)
        
        # D_opt
        ax3.errorbar(i + 0.2, k_data['Dopt_mean'].values[0], 
                    yerr=[[dopt_err_lo], [dopt_err_hi]],
                    fmt='s', color='blue', capsize=5, label='D_opt' if i == 0 else '', markersize=6)

# 设置x轴标签
ax3.set_xticks([i + 0.1 for i in range(len(k_values))])
ax3.set_xticklabels(k_values)
ax3.set_xlabel('k')
ax3.set_ylabel('D')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.3)

# 保存图表
plt.savefig('exp3_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('exp3_figure.svg', bbox_inches='tight')

# 显示图表
plt.show()

print("图表生成完成！")
print("bootstrap汇总已保存到 bootstrap_summary_conn_from_extended_plus_highD.csv")
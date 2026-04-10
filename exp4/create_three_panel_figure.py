import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# 读取热图数据
df_heatmap_1024 = pd.read_csv('exp4_heatmap_table_N1024.csv')
df_heatmap_2048 = pd.read_csv('exp4_heatmap_table_N2048.csv')

# 读取k指标摘要数据
df_kmetrics_1024 = pd.read_csv('exp4_k_metrics_summary_N1024.csv')
df_kmetrics_2048 = pd.read_csv('exp4_k_metrics_summary_N2048.csv')

# 准备热图数据
# 对于每个N，创建一个pivot表：行是noise，列是k，值是J_interp
def prepare_heatmap_data(df):
    pivot = df.pivot(index='noise', columns='k', values='J_interp')
    # 确保k列按升序排列
    pivot = pivot.sort_index(axis=1)
    return pivot

heatmap_data_1024 = prepare_heatmap_data(df_heatmap_1024)
heatmap_data_2048 = prepare_heatmap_data(df_heatmap_2048)

# 只保留k=2到k=10的范围
heatmap_data_1024 = heatmap_data_1024.iloc[:, :9]
heatmap_data_2048 = heatmap_data_2048.iloc[:, :9]

# 创建三-panel图表
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Panel A: 双热图（左右两个子图）
ax_a = axes[0]

# 创建左右两个子图
ax_a_left = ax_a.inset_axes([0, 0, 0.48, 1])
ax_a_right = ax_a.inset_axes([0.52, 0, 0.48, 1])

# 绘制左侧热图 (N=1024)
sns.heatmap(heatmap_data_1024, ax=ax_a_left, cmap='viridis', norm=LogNorm(), 
            cbar=False)
ax_a_left.set_title('N=1024')
ax_a_left.set_xlabel('k')
ax_a_left.set_ylabel('noise')

# 优化x轴标签显示
ax_a_left.set_xticks(range(0, len(heatmap_data_1024.columns), 2))
ax_a_left.set_xticklabels(heatmap_data_1024.columns[::2])

# 优化y轴标签显示
# 计算y轴刻度位置
num_ticks = min(10, len(heatmap_data_1024.index))
y_ticks = np.linspace(0, len(heatmap_data_1024.index)-1, num_ticks, dtype=int)
# 计算对应的标签
y_labels = [f'{float(heatmap_data_1024.index[i]):.2f}' for i in y_ticks]
ax_a_left.set_yticks(y_ticks)
ax_a_left.set_yticklabels(y_labels)

# 绘制右侧热图 (N=2048)
sns.heatmap(heatmap_data_2048, ax=ax_a_right, cmap='viridis', norm=LogNorm(), 
            cbar_kws={'label': 'J_interp'})
ax_a_right.set_title('N=2048')
ax_a_right.set_xlabel('k')
ax_a_right.set_ylabel('noise')

# 优化x轴标签显示
ax_a_right.set_xticks(range(0, len(heatmap_data_2048.columns), 2))
ax_a_right.set_xticklabels(heatmap_data_2048.columns[::2])

# 优化y轴标签显示
ax_a_right.set_yticks(y_ticks)
ax_a_right.set_yticklabels(y_labels)

# 在热图上叠加D_opt和D_c曲线
# 对于N=1024
k_values_1024 = df_kmetrics_1024['k'].values[:9]
noise_values_1024 = heatmap_data_1024.index.values

# 准备D_opt和D_c的数据
opt_values_1024 = []
c_values_1024 = []
for k in k_values_1024:
    row = df_kmetrics_1024[df_kmetrics_1024['k'] == k].iloc[0]
    if row['D_opt_mean'] in noise_values_1024:
        idx_opt = np.where(noise_values_1024 == row['D_opt_mean'])[0][0]
        opt_values_1024.append(idx_opt)
    else:
        opt_values_1024.append(np.nan)
    
    if row['D_c_mean'] in noise_values_1024:
        idx_c = np.where(noise_values_1024 == row['D_c_mean'])[0][0]
        c_values_1024.append(idx_c)
    else:
        c_values_1024.append(np.nan)

# 绘制D_opt和D_c曲线
ax_a_left.plot(range(len(k_values_1024)), opt_values_1024, 'w-', linewidth=2, label='D_opt')
ax_a_left.plot(range(len(k_values_1024)), c_values_1024, 'w--', linewidth=2, label='D_c')

# 对于N=2048
k_values_2048 = df_kmetrics_2048['k'].values[:9]
noise_values_2048 = heatmap_data_2048.index.values

# 准备D_opt和D_c的数据
opt_values_2048 = []
c_values_2048 = []
for k in k_values_2048:
    row = df_kmetrics_2048[df_kmetrics_2048['k'] == k].iloc[0]
    if row['D_opt_mean'] in noise_values_2048:
        idx_opt = np.where(noise_values_2048 == row['D_opt_mean'])[0][0]
        opt_values_2048.append(idx_opt)
    else:
        opt_values_2048.append(np.nan)
    
    if row['D_c_mean'] in noise_values_2048:
        idx_c = np.where(noise_values_2048 == row['D_c_mean'])[0][0]
        c_values_2048.append(idx_c)
    else:
        c_values_2048.append(np.nan)

# 绘制D_opt和D_c曲线
ax_a_right.plot(range(len(k_values_2048)), opt_values_2048, 'w-', linewidth=2, label='D_opt')
ax_a_right.plot(range(len(k_values_2048)), c_values_2048, 'w--', linewidth=2, label='D_c')

# 添加图例
ax_a.legend(['D_opt (solid)', 'D_c (dashed)'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
ax_a.set_title('Panel A: Heatmaps', pad=20)

# Panel B和Panel C: 在下面各占一半
ax_b = axes[1].inset_axes([0, 0, 0.48, 1])
ax_c = axes[1].inset_axes([0.52, 0, 0.48, 1])

# 只取k=2到k=10的数据
k_values = df_kmetrics_1024['k'].values[:9]

# Panel B: J_min(k)
# 绘制N=1024的J_min
jmin_1024 = df_kmetrics_1024['J_min_mean'].values[:9]
jmin_ci_lo_1024 = df_kmetrics_1024['J_min_ci_lo'].values[:9]
jmin_ci_hi_1024 = df_kmetrics_1024['J_min_ci_hi'].values[:9]

# 计算误差条
jmin_error_1024 = np.array([jmin_1024 - jmin_ci_lo_1024, jmin_ci_hi_1024 - jmin_1024])

ax_b.errorbar(k_values, jmin_1024, yerr=jmin_error_1024, fmt='o-', label='N=1024', linewidth=2, capsize=3)

# 绘制N=2048的J_min
jmin_2048 = df_kmetrics_2048['J_min_mean'].values[:9]
jmin_ci_lo_2048 = df_kmetrics_2048['J_min_ci_lo'].values[:9]
jmin_ci_hi_2048 = df_kmetrics_2048['J_min_ci_hi'].values[:9]

# 计算误差条
jmin_error_2048 = np.array([jmin_2048 - jmin_ci_lo_2048, jmin_ci_hi_2048 - jmin_2048])

ax_b.errorbar(k_values, jmin_2048, yerr=jmin_error_2048, fmt='s-', label='N=2048', linewidth=2, capsize=3)

ax_b.set_xlabel('k')
ax_b.set_ylabel('J_min')
ax_b.set_title('Panel B: J_min(k)')
ax_b.legend()
ax_b.grid(True, alpha=0.3)
# 设置x轴范围和刻度，确保与Panel C一致
ax_b.set_xlim(min(k_values)-0.5, max(k_values)+0.5)
ax_b.set_xticks(k_values)

# Panel C: Δ(k) = |D_c - D_opt|
# 绘制N=1024的Δ(k)
delta_1024 = df_kmetrics_1024['Delta_mean'].values[:9]
delta_ci_lo_1024 = df_kmetrics_1024['Delta_ci_lo'].values[:9]
delta_ci_hi_1024 = df_kmetrics_1024['Delta_ci_hi'].values[:9]

# 计算误差条
delta_error_1024 = np.array([delta_1024 - delta_ci_lo_1024, delta_ci_hi_1024 - delta_1024])

ax_c.errorbar(k_values, delta_1024, yerr=delta_error_1024, fmt='o-', label='N=1024', linewidth=2, capsize=3)

# 绘制N=2048的Δ(k)
delta_2048 = df_kmetrics_2048['Delta_mean'].values[:9]
delta_ci_lo_2048 = df_kmetrics_2048['Delta_ci_lo'].values[:9]
delta_ci_hi_2048 = df_kmetrics_2048['Delta_ci_hi'].values[:9]

# 计算误差条
delta_error_2048 = np.array([delta_2048 - delta_ci_lo_2048, delta_ci_hi_2048 - delta_2048])

ax_c.errorbar(k_values, delta_2048, yerr=delta_error_2048, fmt='s-', label='N=2048', linewidth=2, capsize=3)

ax_c.set_xlabel('k')
ax_c.set_ylabel('Δ(k)')
ax_c.set_title('Panel C: Δ(k) = |D_c - D_opt|')
ax_c.legend()
ax_c.grid(True, alpha=0.3)
# 设置x轴范围和刻度，确保与Panel B一致
ax_c.set_xlim(min(k_values)-0.5, max(k_values)+0.5)
ax_c.set_xticks(k_values)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('three_panel_figure.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

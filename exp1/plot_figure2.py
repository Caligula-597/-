import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# 读取整理后的数据
with open('figure2_data.json', 'r') as f:
    data = json.load(f)

# 定义颜色和标记
colors = {
    512: '#1f77b4',  # 蓝色
    1024: '#ff7f0e',  # 橙色
    2048: '#2ca02c'   # 绿色
}
markers = {
    'D_c': 'o',  # 空心圆
    'D_opt': 's'  # 实心方块
}

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: D_c(k) 与 D_opt(k)
ax = axes[0, 0]

# 按N分组数据
n_groups = {512: [], 1024: [], 2048: []}
for entry in data['panel_ab']:
    n_groups[entry['N']].append(entry)

# 绘制每组数据
for n in [512, 1024, 2048]:
    group_data = n_groups[n]
    k_values = [entry['k'] for entry in group_data]
    d_c_values = [entry['D_c'] for entry in group_data]
    d_opt_values = [entry['D_opt'] for entry in group_data]
    
    # 绘制D_c
    ax.plot(k_values, d_c_values, marker=markers['D_c'], linestyle='-', color=colors[n], 
            fillstyle='none')
    # 绘制D_opt
    ax.plot(k_values, d_opt_values, marker=markers['D_opt'], linestyle='--', color=colors[n])

# 自定义legend
import matplotlib.lines as mlines
legend_elements = []
# 尺寸图例
for n in [512, 1024, 2048]:
    legend_elements.append(mlines.Line2D([], [], color=colors[n], label=f'N={n}'))
# 标记图例
legend_elements.append(mlines.Line2D([], [], marker=markers['D_c'], linestyle='-', 
                                    color='black', fillstyle='none', label='D$_{c}$'))
legend_elements.append(mlines.Line2D([], [], marker=markers['D_opt'], linestyle='--', 
                                    color='black', label='D$_{opt}$'))
ax.legend(handles=legend_elements, loc='best')

ax.set_xlabel('k')
ax.set_ylabel('D')
ax.set_title('Critical points vs. k')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel B: Delta(k) = |D_c - D_opt|
ax = axes[0, 1]

# 按N分组数据
for n in [512, 1024, 2048]:
    group_data = n_groups[n]
    k_values = [entry['k'] for entry in group_data]
    delta_values = [entry['Delta'] for entry in group_data]
    
    ax.plot(k_values, delta_values, marker='o', linestyle='-', color=colors[n], 
            label=f'N={n}')

# 添加finite-size outlier注释
for entry in data['panel_ab']:
    if entry['N'] == 512 and entry['k'] == 5:
        ax.annotate('finite-size outlier', 
                   xy=(entry['k'], entry['Delta']), 
                   xytext=(entry['k'] + 0.5, entry['Delta'] + 0.005),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                   fontsize=8)

ax.set_xlabel('k')
ax.set_ylabel('Δ')
ax.set_title('Delta vs. k')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel C: J_min(k)
ax = axes[1, 0]

# 按N分组数据
for n in [512, 1024, 2048]:
    group_data = [entry for entry in data['panel_c'] if entry['N'] == n]
    k_values = [entry['k'] for entry in group_data]
    j_min_values = [entry['J_min'] for entry in group_data]
    j_min_ci_lo = [entry['J_min_ci_lo'] for entry in group_data]
    j_min_ci_hi = [entry['J_min_ci_hi'] for entry in group_data]
    
    # 计算误差棒
    y_err = np.array([j_min_values]) - np.array([j_min_ci_lo])
    y_err = np.vstack([y_err, np.array([j_min_ci_hi]) - np.array([j_min_values])])
    
    ax.errorbar(k_values, j_min_values, yerr=y_err, marker='s', linestyle='-', color=colors[n], 
                label=f'N={n}', capsize=3)

ax.set_xlabel('k')
ax.set_ylabel('J$_{min}$')
ax.set_title('Minimum cost per response vs. k')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel D: 二维valley图 (N=2048)
ax = axes[1, 1]

# 准备数据
k_values = []
d_values = []
j_values = []
d_opt_values = []

for entry in data['panel_d']:
    k = entry['k']
    d_opt = entry['D_opt']
    d_opt_values.append((k, d_opt))
    
    for d, j in zip(entry['D'], entry['J']):
        # 过滤掉J值过大的点
        if j <= 2.5:
            k_values.append(k)
            d_values.append(d)
            j_values.append(j)

# 创建网格
k_grid = np.linspace(min(k_values), max(k_values), 100)
d_grid = np.logspace(np.log10(min(d_values)), np.log10(max(d_values)), 100)
k_mesh, d_mesh = np.meshgrid(k_grid, d_grid)

# 插值
j_mesh = griddata((k_values, d_values), j_values, (k_mesh, d_mesh), method='linear')

# 绘制热图
contour = ax.contourf(k_mesh, d_mesh, j_mesh, levels=20, cmap='viridis')
colorbar = fig.colorbar(contour, ax=ax)
colorbar.set_label('J (clipped)')

# 标记D_opt点
d_opt_k = [point[0] for point in d_opt_values]
d_opt_d = [point[1] for point in d_opt_values]
ax.plot(d_opt_k, d_opt_d, 'w.', markersize=10, label='D$_{opt}$')

# 添加裁剪说明
ax.text(0.05, 0.05, 'Clipped for visibility', transform=ax.transAxes, 
        fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlabel('k')
ax.set_ylabel('D')
ax.set_yscale('log')
ax.set_title('J landscape for N=2048')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# 调整布局
plt.tight_layout()

# 添加主标题
fig.suptitle('Cross-scale locking and low-degree organization of the response-efficient optimum', 
             fontsize=14, fontweight='bold', y=1.02)

# 保存图
plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2.svg', bbox_inches='tight')

print("Figure 2 generated successfully!")
print("Saved as figure2.png and figure2.svg")

# 显示图
plt.show()
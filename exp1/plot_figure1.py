import json
import matplotlib.pyplot as plt
import numpy as np

# 读取整理后的数据
with open('figure1_data.json', 'r') as f:
    data = json.load(f)

# 定义颜色和线型
colors = {
    'N=2048,k=3': '#1f77b4',  # 蓝色
    'N=2048,k=5': '#ff7f0e',  # 橙色
    'N=2048,k=7': '#2ca02c',  # 绿色
    'N=2048,k=9': '#d62728',  # 红色
    'N=2048,k=12': '#9467bd'  # 紫色
}

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Φ(D)
ax = axes[0, 0]
for key in data:
    d = data[key]['D']
    phi = data[key]['Phi']
    phi_ci_lo = data[key]['Phi_ci_lo']
    phi_ci_hi = data[key]['Phi_ci_hi']
    ax.plot(d, phi, label=f'k={key.split(",k=")[1]}', color=colors[key])
    ax.fill_between(d, phi_ci_lo, phi_ci_hi, alpha=0.2, color=colors[key])
ax.set_xscale('log')
ax.set_xlabel('D')
ax.set_ylabel('Φ')
ax.set_title('Order parameter')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel B: χ(D) with D_c
ax = axes[0, 1]
for key in data:
    d = data[key]['D']
    chi = data[key]['chi']
    chi_ci_lo = data[key]['chi_ci_lo']
    chi_ci_hi = data[key]['chi_ci_hi']
    d_c = data[key]['D_c']
    ax.plot(d, chi, color=colors[key])
    ax.fill_between(d, chi_ci_lo, chi_ci_hi, alpha=0.2, color=colors[key])
    ax.axvline(x=d_c, linestyle='--', color=colors[key], alpha=0.7)  # 虚线表示D_c
ax.set_xscale('log')
ax.set_xlabel('D')
ax.set_ylabel('χ')
ax.set_title('Susceptibility')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel C: Ṡproxy(D)
ax = axes[1, 0]
for key in data:
    d = data[key]['D']
    sdot = data[key]['Sdot']
    sdot_ci_lo = data[key]['Sdot_ci_lo']
    sdot_ci_hi = data[key]['Sdot_ci_hi']
    ax.plot(d, sdot, color=colors[key])
    ax.fill_between(d, sdot_ci_lo, sdot_ci_hi, alpha=0.2, color=colors[key])
ax.set_xscale('log')
ax.set_xlabel('D')
ax.set_ylabel('Ṡ$_{proxy}$')
ax.set_title('Dissipation proxy')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel D: J(D) = Ṡ/χ with D_opt - 只显示有效J区间
ax = axes[1, 1]
for key in data:
    d = data[key]['D']
    j = data[key]['J']
    j_ci_lo = data[key]['J_ci_lo']
    j_ci_hi = data[key]['J_ci_hi']
    d_opt = data[key]['D_opt']
    
    # 过滤掉J值大于2.5的点，只显示有效区间
    valid_indices = [i for i, val in enumerate(j) if val <= 2.5]
    d_valid = [d[i] for i in valid_indices]
    j_valid = [j[i] for i in valid_indices]
    j_ci_lo_valid = [j_ci_lo[i] for i in valid_indices]
    j_ci_hi_valid = [j_ci_hi[i] for i in valid_indices]
    
    ax.plot(d_valid, j_valid, color=colors[key])
    ax.fill_between(d_valid, j_ci_lo_valid, j_ci_hi_valid, alpha=0.2, color=colors[key])
    ax.axvline(x=d_opt, linestyle=':', color=colors[key], alpha=0.7)  # 点线表示D_opt
ax.set_xscale('log')
ax.set_xlabel('D')
ax.set_ylabel('J')
ax.set_title('Cost per response')
ax.set_ylim(0, 2.5)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# 调整布局
plt.tight_layout()

# 保存图
plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1.svg', bbox_inches='tight')

print("Figure 1 generated successfully!")
print("Saved as figure1.png and figure1.svg")

# 显示图
plt.show()
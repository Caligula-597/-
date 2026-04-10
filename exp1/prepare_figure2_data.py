import json
import numpy as np

# 读取实验结果文件
with open('experiment1_results.json', 'r') as f:
    data = json.load(f)

# 定义需要的N和k组合
n_values = [512, 1024, 2048]
k_values = [3, 5, 7, 9, 12]

# 提取数据
figure2_data = {
    'panel_ab': [],  # 用于Panel A和B的数据
    'panel_c': [],   # 用于Panel C的数据
    'panel_d': []    # 用于Panel D的数据
}

# 提取critical_points和J_min数据
for n in n_values:
    for k in k_values:
        key = f"N={n},k={k}"
        
        # 提取critical_points数据
        critical_data = data['critical_points'][key]
        d_c = critical_data['D_c']
        d_opt = critical_data['D_opt']
        delta = critical_data['Delta']
        
        # 提取summaries数据，找到D_opt对应的J值
        summary_data = data['summaries'][key]
        j_at_dopt = None
        j_ci_lo_at_dopt = None
        j_ci_hi_at_dopt = None
        
        # 找到最接近D_opt的点
        min_diff = float('inf')
        best_idx = -1
        for i, point in enumerate(summary_data):
            diff = abs(point['D'] - d_opt)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        
        if best_idx != -1:
            j_at_dopt = summary_data[best_idx]['J']
            j_ci_lo_at_dopt = summary_data[best_idx]['J_ci_lo']
            j_ci_hi_at_dopt = summary_data[best_idx]['J_ci_hi']
        
        # 存储Panel A/B的数据
        figure2_data['panel_ab'].append({
            'N': n,
            'k': k,
            'D_c': d_c,
            'D_opt': d_opt,
            'Delta': delta
        })
        
        # 存储Panel C的数据
        figure2_data['panel_c'].append({
            'N': n,
            'k': k,
            'J_min': j_at_dopt,
            'J_min_ci_lo': j_ci_lo_at_dopt,
            'J_min_ci_hi': j_ci_hi_at_dopt
        })
        
        # 存储Panel D的数据（只取N=2048）
        if n == 2048:
            d_values = []
            j_values = []
            for point in summary_data:
                d_values.append(point['D'])
                j_values.append(point['J'])
            
            figure2_data['panel_d'].append({
                'k': k,
                'D': d_values,
                'J': j_values,
                'D_opt': d_opt
            })

# 保存整理后的数据
with open('figure2_data.json', 'w') as f:
    json.dump(figure2_data, f, indent=2)

print("Figure 2 data prepared successfully!")
print(f"Extracted data for N={n_values}, k={k_values}")
print("Data saved to figure2_data.json")

# 打印数据结构
print("\nData structure:")
print(f"Panel A/B: {len(figure2_data['panel_ab'])} entries")
print(f"Panel C: {len(figure2_data['panel_c'])} entries")
print(f"Panel D: {len(figure2_data['panel_d'])} entries")

# 打印Panel A/B的部分数据
print("\nPanel A/B data sample:")
for entry in figure2_data['panel_ab'][:5]:
    print(f"N={entry['N']}, k={entry['k']}: D_c={entry['D_c']:.6f}, D_opt={entry['D_opt']:.6f}, Delta={entry['Delta']:.6f}")

# 打印Panel C的部分数据
print("\nPanel C data sample:")
for entry in figure2_data['panel_c'][:5]:
    print(f"N={entry['N']}, k={entry['k']}: J_min={entry['J_min']:.6f}")

# 打印Panel D的部分数据
print("\nPanel D data sample:")
for entry in figure2_data['panel_d']:
    print(f"k={entry['k']}: D_opt={entry['D_opt']:.6f}, data points={len(entry['D'])}")
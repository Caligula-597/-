import json
import numpy as np

# 读取实验结果文件
with open('experiment1_results.json', 'r') as f:
    data = json.load(f)

# 定义需要的N和k组合
n_value = 2048
k_values = [3, 5, 7, 9, 12]

# 提取数据
figure1_data = {}

for k in k_values:
    key = f"N={n_value},k={k}"
    
    # 提取summary数据
    summary_data = data['summaries'][key]
    
    # 提取critical_points数据
    critical_data = data['critical_points'][key]
    
    # 整理数据
    d_values = []
    phi_values = []
    phi_ci_lo = []
    phi_ci_hi = []
    chi_values = []
    chi_ci_lo = []
    chi_ci_hi = []
    sdot_values = []
    sdot_ci_lo = []
    sdot_ci_hi = []
    j_values = []
    j_ci_lo = []
    j_ci_hi = []
    
    for point in summary_data:
        d = point['D']
        d_values.append(d)
        
        phi_values.append(point['Phi'])
        phi_ci_lo.append(point['Phi_ci_lo'])
        phi_ci_hi.append(point['Phi_ci_hi'])
        
        chi_values.append(point['chi'])
        chi_ci_lo.append(point['chi_ci_lo'])
        chi_ci_hi.append(point['chi_ci_hi'])
        
        sdot_values.append(point['Sdot'])
        sdot_ci_lo.append(point['Sdot_ci_lo'])
        sdot_ci_hi.append(point['Sdot_ci_hi'])
        
        j_values.append(point['J'])
        j_ci_lo.append(point['J_ci_lo'])
        j_ci_hi.append(point['J_ci_hi'])
    
    # 过滤掉J值为NaN或无限大的点
    valid_indices = []
    for i, j in enumerate(j_values):
        if not (np.isnan(j) or np.isinf(j)):
            valid_indices.append(i)
    
    # 应用过滤
    d_values = [d_values[i] for i in valid_indices]
    phi_values = [phi_values[i] for i in valid_indices]
    phi_ci_lo = [phi_ci_lo[i] for i in valid_indices]
    phi_ci_hi = [phi_ci_hi[i] for i in valid_indices]
    chi_values = [chi_values[i] for i in valid_indices]
    chi_ci_lo = [chi_ci_lo[i] for i in valid_indices]
    chi_ci_hi = [chi_ci_hi[i] for i in valid_indices]
    sdot_values = [sdot_values[i] for i in valid_indices]
    sdot_ci_lo = [sdot_ci_lo[i] for i in valid_indices]
    sdot_ci_hi = [sdot_ci_hi[i] for i in valid_indices]
    j_values = [j_values[i] for i in valid_indices]
    j_ci_lo = [j_ci_lo[i] for i in valid_indices]
    j_ci_hi = [j_ci_hi[i] for i in valid_indices]
    
    # 存储整理后的数据
    figure1_data[key] = {
        'D': d_values,
        'Phi': phi_values,
        'Phi_ci_lo': phi_ci_lo,
        'Phi_ci_hi': phi_ci_hi,
        'chi': chi_values,
        'chi_ci_lo': chi_ci_lo,
        'chi_ci_hi': chi_ci_hi,
        'Sdot': sdot_values,
        'Sdot_ci_lo': sdot_ci_lo,
        'Sdot_ci_hi': sdot_ci_hi,
        'J': j_values,
        'J_ci_lo': j_ci_lo,
        'J_ci_hi': j_ci_hi,
        'D_c': critical_data['D_c'],
        'D_opt': critical_data['D_opt']
    }

# 保存整理后的数据
with open('figure1_data.json', 'w') as f:
    json.dump(figure1_data, f, indent=2)

print("Figure 1 data prepared successfully!")
print(f"Extracted data for N={n_value}, k={k_values}")
print("Data saved to figure1_data.json")

# 打印数据结构
print("\nData structure:")
for key in figure1_data:
    print(f"\n{key}:")
    print(f"  Number of data points: {len(figure1_data[key]['D'])}")
    print(f"  D range: {min(figure1_data[key]['D']):.6f} to {max(figure1_data[key]['D']):.6f}")
    print(f"  D_c: {figure1_data[key]['D_c']:.6f}")
    print(f"  D_opt: {figure1_data[key]['D_opt']:.6f}")
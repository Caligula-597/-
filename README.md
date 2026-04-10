# Notebook 导航说明

本仓库当前包含多个实验 notebook。根据你的最新标注，推荐按下面方式理解与使用。

## 1. Notebook 与实验对应关系

- `Untitled-1.ipynb`：**Exp1 + Exp2（合并/联调）**
- `experiment3_from_exp12_jax.ipynb`：**Exp3**
- `Untitled-3.ipynb`：**Exp4**
- `Untitled-2.ipynb`：**Exp5（topological vs metric 对比主实验）**
- `Untitled-4.ipynb`：**Exp6 v2（Empirical anchoring）**
- `Untitled-5.ipynb`：**特殊点单独验证（verification notebook）**

---

## 2. 推荐运行顺序

建议按实验依赖关系执行：

1. `Untitled-1.ipynb`（Exp1+2）
2. `experiment3_from_exp12_jax.ipynb`（Exp3）
3. `Untitled-3.ipynb`（Exp4）
4. `Untitled-2.ipynb`（Exp5）
5. `Untitled-4.ipynb`（Exp6 v2）
6. `Untitled-5.ipynb`（特殊点验证）

> 如果你只想复现主结果，可优先跑 Exp3 / Exp4 / Exp5 / Exp6；
> `Untitled-5.ipynb` 主要用于对特殊 case 做额外确认。

---

## 3. 目录与输出（简要）

仓库中已包含多个实验相关目录，例如：

- `exp1/`
- `exp2_outputs/`
- `EXP3/`
- `exp4/`
- `exp5_final_core/`
- `exp6_empirical_anchor_v2_sd01/`

这些目录通常用于保存阶段性结果、图表或导出文件；
具体以各 notebook 内的保存路径设置为准。

---

## 4. 建议（可选）

当前 notebook 名称中有多个 `Untitled-*`，后续可考虑重命名，便于协作：

- `Untitled-1.ipynb` -> `exp1_exp2_integrated.ipynb`
- `Untitled-3.ipynb` -> `exp4_main.ipynb`
- `Untitled-5.ipynb` -> `special_point_validation.ipynb`


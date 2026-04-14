# MACE模型预测使用指南

## 概述

本指南介绍如何使用训练好的MACE模型进行预测，包括如何获取能量、力和不确定度输出。

## 前提条件

1. 已安装MACE及其依赖
2. 有训练好的MACE模型文件（.model格式）
3. 有待预测的结构文件（支持extxyz, POSCAR, xyz等格式）

## 预测脚本

我们提供了两个预测脚本：

### 1. `predict_with_uncertainty.py` - 通用预测脚本

**功能**：
- 单个结构预测
- 批量预测
- 自动检测模型是否支持不确定度
- 保存预测结果为extxyz格式

**使用方法**：

```bash
# 单个结构预测
python predict_with_uncertainty.py \
    --model /path/to/model.model \
    --input structure.extxyz \
    --output predicted.extxyz \
    --device cuda

# 批量预测（输入为目录）
python predict_with_uncertainty.py \
    --model /path/to/model.model \
    --input /path/to/structures/ \
    --output /path/to/predictions/ \
    --batch \
    --device cuda
```

### 2. `predict_uncertainty.py` - 不确定度专用脚本

**功能**：
- 专门用于获取不确定度输出
- 详细的诊断信息
- 检查模型配置

**使用方法**：

```bash
python predict_uncertainty.py \
    --model /path/to/model.model \
    --input structure.extxyz \
    --output predicted.extxyz \
    --device cuda
```

## 关于不确定度输出

### 模型要求

要获取不确定度输出，模型必须使用 `--use_uncertainty` 参数训练：

```bash
# 训练时启用不确定度
mace_run_train \
    --name="mace_uncertainty" \
    --train_file="train.xyz" \
    --valid_file="valid.xyz" \
    --test_file="test.xyz" \
    --use_uncertainty \
    --loss="uncertainty_weighted" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --max_num_epochs=500 \
    --device=cuda
```

### 不确定度类型

如果模型支持不确定度，预测时会输出：

1. **能量不确定度** (`energy_uncertainty`)
   - 单位：eV
   - 表示能量预测的不确定性
   - 值越大表示模型对该预测越不确定

2. **力不确定度** (`forces_uncertainty`)
   - 单位：eV/Å
   - 每个原子的每个方向都有一个不确定度值
   - 形状：(n_atoms, 3)

### 当前限制

⚠️ **重要提示**：

目前MACE的标准Calculator可能不会自动返回不确定度。要获取不确定度，需要：

**选项1：修改MACECalculator**

修改 `mace/calculators/mace.py` 中的 `MACECalculator` 类，使其返回不确定度：

```python
# 在 MACECalculator.calculate() 方法中添加
if 'energy_uncertainty' in out:
    self.results['energy_uncertainty'] = out['energy_uncertainty'].detach().cpu().numpy()

if 'forces_uncertainty' in out:
    self.results['forces_uncertainty'] = out['forces_uncertainty'].detach().cpu().numpy()
```

**选项2：直接调用模型**

绕过Calculator，直接调用模型的forward方法：

```python
import torch
from mace.calculators import MACECalculator

# 加载模型
calc = MACECalculator(model_paths='model.model', device='cuda')

# 获取模型
model = calc.models[0]

# 准备输入数据（需要转换为MACE的数据格式）
# ... 数据准备代码 ...

# 前向传播
with torch.no_grad():
    output = model(batch)

# 输出包含：
# - output['energy']: 能量预测
# - output['forces']: 力预测
# - output['energy_uncertainty']: 能量不确定度（如果模型支持）
# - output['forces_uncertainty']: 力不确定度（如果模型支持）
```

## 示例工作流程

### 1. 训练带不确定度的模型

```bash
cd /inspire/hdd/project/materialscienceresearch/wenyihao-240208090182/mace

# 使用提取的数据训练
mace_run_train \
    --name="MOF_uncertainty" \
    --train_file="../MOF/VASP/extxyz_data/train/*.extxyz" \
    --valid_file="../MOF/VASP/extxyz_data/valid/*.extxyz" \
    --test_file="../MOF/VASP/extxyz_data/test/*.extxyz" \
    --use_uncertainty \
    --loss="uncertainty_weighted" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --max_num_epochs=500 \
    --batch_size=4 \
    --device=cuda \
    --default_dtype=float64
```

### 2. 预测新结构

```bash
# 单个结构预测
python predict_with_uncertainty.py \
    --model results/MOF_uncertainty.model \
    --input new_structure.extxyz \
    --output predicted.extxyz

# 批量预测
python predict_with_uncertainty.py \
    --model results/MOF_uncertainty.model \
    --input test_structures/ \
    --output predictions/ \
    --batch
```

### 3. 分析预测结果

```python
from ase.io import read

# 读取预测结果
atoms = read('predicted.extxyz')

# 获取预测值
energy = atoms.info['predicted_energy']
forces = atoms.arrays['predicted_forces']

# 获取不确定度（如果有）
if 'energy_uncertainty' in atoms.info:
    energy_unc = atoms.info['energy_uncertainty']
    print(f"能量: {energy:.6f} ± {energy_unc:.6f} eV")

if 'forces_uncertainty' in atoms.arrays:
    forces_unc = atoms.arrays['forces_uncertainty']
    print(f"力不确定度范围: {forces_unc.min():.6f} - {forces_unc.max():.6f} eV/Å")
```

## 输出格式

预测结果保存为extxyz格式，包含以下信息：

```
# extxyz文件内容示例
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:predicted_forces:R:3:forces_uncertainty:R:3 predicted_energy=-123.456 energy_uncertainty=0.123 pbc="T T T"
Co 0.0 0.0 0.0 -0.01 0.02 -0.03 0.001 0.002 0.001
O  1.5 0.0 0.0  0.05 -0.01 0.00 0.003 0.001 0.002
...
```

包含的信息：
- `predicted_energy`: 预测的能量
- `energy_uncertainty`: 能量不确定度（如果有）
- `predicted_forces`: 预测的力（每个原子）
- `forces_uncertainty`: 力不确定度（每个原子，如果有）

## 常见问题

### Q1: 预测时没有输出不确定度？

**A**: 检查以下几点：
1. 模型是否使用 `--use_uncertainty` 训练
2. MACECalculator是否支持返回不确定度（可能需要修改源码）
3. 查看脚本输出的诊断信息

### Q2: 如何判断不确定度是否可靠？

**A**: 
- 在训练集分布内的结构，不确定度应该较小
- 在训练集分布外的结构，不确定度应该较大
- 可以通过验证集评估不确定度的校准程度

### Q3: 不确定度的单位是什么？

**A**:
- 能量不确定度：eV
- 力不确定度：eV/Å
- 与预测值的单位相同

### Q4: 如何使用不确定度进行主动学习？

**A**:
1. 预测大量候选结构
2. 选择不确定度最大的结构
3. 对这些结构进行DFT计算
4. 将新数据加入训练集
5. 重新训练模型

## 性能优化

### GPU加速

```bash
# 使用GPU
python predict_with_uncertainty.py --device cuda

# 使用CPU
python predict_with_uncertainty.py --device cpu
```

### 批量预测优化

对于大量结构的预测，建议：
1. 使用批量预测模式（`--batch`）
2. 适当增加batch size（需要修改脚本）
3. 使用GPU加速

## 参考资料

- MACE论文：https://arxiv.org/abs/2206.07697
- Deep Evidential Regression：https://arxiv.org/abs/1910.02600
- MACE GitHub：https://github.com/ACEsuit/mace

## 脚本位置

```
mace/
├── predict_with_uncertainty.py    # 通用预测脚本
├── predict_uncertainty.py         # 不确定度专用脚本
└── README_PREDICTION.md          # 本文档
```

## 联系与支持

如有问题，请查看：
1. MACE官方文档
2. 本项目的 `README_UNCERTAINTY.md`
3. 训练日志和模型配置文件

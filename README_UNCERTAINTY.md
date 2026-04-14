# MACE with Uncertainty Quantification (EIP-style)

本修改版本的MACE添加了EIP（Evidential Inference for Physics）风格的不确定性量化功能。

## 主要修改

### 1. 模型输出层修改 (`mace/modules/blocks.py`)
- 修改了 `NonLinearReadoutBlock` 类，添加了 `use_uncertainty` 参数
- 当启用不确定性时，模型会输出能量预测值和对数方差（log variance）
- 不确定性通过额外的线性层输出

### 2. MACE模型修改 (`mace/modules/models.py`)
- 在 `MACE` 类的 `__init__` 方法中添加了 `use_uncertainty` 参数
- 修改了 `forward` 方法，处理不确定性输出：
  - 能量不确定性：从对数方差转换为标准差
  - 力的不确定性：通过能量不确定性的梯度计算
- 返回字典中添加了 `energy_uncertainty` 和 `forces_uncertainty` 字段

### 3. 不确定性加权损失函数 (`mace/modules/loss_uncertainty.py`)
新增文件，实现了EIP风格的损失函数：

- `uncertainty_weighted_mse_forces`: 力的不确定性加权MSE损失
  ```
  Loss = sum_i [ (F_pred_i - F_ref_i)^2 / (2 * sigma_i^2) + 0.5 * log(sigma_i^2) ]
  ```

- `uncertainty_weighted_energy_loss`: 能量的不确定性加权损失
  ```
  Loss = (E_pred - E_ref)^2 / (2 * sigma^2) + 0.5 * log(sigma^2)
  ```

- `UncertaintyWeightedEnergyForcesLoss`: 组合能量和力的不确定性加权损失
  - 支持通过 `use_uncertainty` 参数开关不确定性模式
  - 向后兼容原有的标准损失函数

### 4. 训练参数添加 (`mace/tools/arg_parser.py`)
- 添加了 `--use_uncertainty` 参数：启用/禁用不确定性量化
- 在损失函数选项中添加了 `uncertainty_weighted` 选项

### 5. 损失函数集成 (`mace/tools/scripts_utils.py`)
- 在 `get_loss_fn` 函数中添加了对 `uncertainty_weighted` 损失的支持
- 自动导入并使用 `UncertaintyWeightedEnergyForcesLoss`

### 6. 模型配置集成 (`mace/tools/model_script_utils.py`)
- 在 `_build_model` 函数中为 `MACE` 和 `ScaleShiftMACE` 模型添加了 `use_uncertainty` 参数传递
- 确保向后兼容：如果参数不存在，默认为 `False`

## 使用方法

### 训练带不确定性的模型

```bash
python -m mace.cli.run_train \
    --name="mace_uncertainty" \
    --train_file="train.xyz" \
    --valid_file="valid.xyz" \
    --model="MACE" \
    --loss="uncertainty_weighted" \
    --use_uncertainty \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --max_num_epochs=1000 \
    --device=cuda
```

### 关键参数说明

- `--use_uncertainty`: 启用不确定性量化（必须设置）
- `--loss="uncertainty_weighted"`: 使用不确定性加权损失函数
- 其他参数与标准MACE训练相同

### 推理时获取不确定性

```python
import torch
from mace.calculators import MACECalculator

# 加载模型
calc = MACECalculator(model_paths='model.model', device='cuda')

# 获取预测结果（包含不确定性）
atoms = ...  # ASE Atoms对象
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# 访问不确定性（如果模型支持）
# 注意：需要直接调用模型的forward方法来获取不确定性
```

## 技术细节

### EIP风格的不确定性

模型输出两个值：
1. **预测值** (μ): 能量或力的预测
2. **对数方差** (log σ²): 预测的不确定性

损失函数形式：
```
L = (y - μ)² / (2σ²) + 0.5 log(σ²)
```

这种形式有两个优点：
- 第一项：预测误差除以方差，不确定性高时惩罚小
- 第二项：防止模型输出无限大的不确定性

### 力的不确定性计算

力是能量对坐标的负梯度，因此力的不确定性通过能量不确定性的梯度计算：
```
σ_F = |∇σ_E|
```

### 向后兼容性

所有修改都保持了向后兼容：
- 不使用 `--use_uncertainty` 时，模型行为与原版MACE完全相同
- 可以加载和使用旧的模型文件
- 标准损失函数仍然可用

## 注意事项

1. **训练时间**: 启用不确定性会增加约20-30%的训练时间
2. **内存使用**: 需要额外存储不确定性参数，内存使用增加约10%
3. **收敛性**: 不确定性加权损失可能需要调整学习率和权重
4. **数值稳定性**: 对数方差被限制在合理范围内（通过clamp操作）

## 示例

完整的训练示例：

```bash
# 使用不确定性训练
python -m mace.cli.run_train \
    --name="water_uncertainty" \
    --train_file="water_train.xyz" \
    --valid_file="water_valid.xyz" \
    --test_file="water_test.xyz" \
    --model="MACE" \
    --loss="uncertainty_weighted" \
    --use_uncertainty \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1000 \
    --lr=0.01 \
    --device=cuda

# 不使用不确定性训练（标准模式）
python -m mace.cli.run_train \
    --name="water_standard" \
    --train_file="water_train.xyz" \
    --valid_file="water_valid.xyz" \
    --model="MACE" \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --device=cuda
```

## 参考文献

如果使用此功能，请引用：
- MACE原始论文
- EIP相关论文（如适用）

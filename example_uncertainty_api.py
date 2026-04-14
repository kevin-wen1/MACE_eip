"""
MACE with Uncertainty Quantification - Python API Example
使用Python API进行训练和推理
"""

import torch
from ase import Atoms
from ase.io import read
import numpy as np

# ============================================
# 示例1: 使用Python API训练带不确定性的模型
# ============================================

def train_with_uncertainty():
    """训练带不确定性量化的MACE模型"""
    
    # 导入必要的模块
    from mace.cli import run_train
    import argparse
    
    # 创建参数
    args = argparse.Namespace(
        name="mace_uncertainty_python",
        train_file="train.xyz",
        valid_file="valid.xyz",
        test_file="test.xyz",
        model="MACE",
        loss="uncertainty_weighted",
        use_uncertainty=True,  # 启用不确定性
        energy_weight=1.0,
        forces_weight=100.0,
        num_interactions=2,
        num_channels=128,
        max_L=1,
        r_max=5.0,
        batch_size=10,
        max_num_epochs=1000,
        lr=0.01,
        ema=True,
        ema_decay=0.99,
        device="cuda",
        default_dtype="float64",
        seed=123,
        # 其他必要参数...
    )
    
    # 运行训练
    run_train.run(args)
    
    print("Training completed with uncertainty quantification!")


# ============================================
# 示例2: 加载模型并进行推理（获取不确定性）
# ============================================

def inference_with_uncertainty():
    """使用训练好的模型进行推理并获取不确定性"""
    
    from mace.calculators import MACECalculator
    
    # 加载模型
    calc = MACECalculator(
        model_paths='mace_uncertainty_python.model',
        device='cuda'
    )
    
    # 创建或读取原子结构
    atoms = read('test_structure.xyz')
    atoms.calc = calc
    
    # 获取能量和力
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"Energy: {energy} eV")
    print(f"Forces shape: {forces.shape}")
    
    # 注意：要获取不确定性，需要直接调用模型
    # 这需要访问底层模型
    model = calc.models[0]
    
    # 准备输入数据
    from mace.tools import torch_geometric
    from mace.data import AtomicData, Configuration
    
    # 转换为MACE数据格式
    config = Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        energy=0.0,  # dummy
        forces=np.zeros_like(atoms.get_positions()),
    )
    
    # 这里需要更详细的数据准备代码...
    # 实际使用时建议参考MACE的eval_configs.py
    
    return energy, forces


# ============================================
# 示例3: 直接使用模型进行前向传播
# ============================================

def direct_model_inference():
    """直接使用模型进行前向传播以获取不确定性"""
    
    import torch
    from mace import modules
    
    # 加载模型
    model = torch.load('mace_uncertainty_python.model', map_location='cuda')
    model.eval()
    
    # 准备输入数据（这里是示例，实际需要正确的数据格式）
    data = {
        'positions': torch.randn(10, 3, requires_grad=True),  # 10个原子
        'node_attrs': torch.randn(10, 118),  # 原子特征
        'edge_index': torch.randint(0, 10, (2, 50)),  # 边索引
        'batch': torch.zeros(10, dtype=torch.long),  # batch索引
        # ... 其他必要的输入
    }
    
    # 前向传播
    with torch.no_grad():
        output = model(data, compute_force=True)
    
    # 获取结果
    energy = output['energy']
    forces = output['forces']
    
    # 如果模型支持不确定性
    if 'energy_uncertainty' in output and output['energy_uncertainty'] is not None:
        energy_uncertainty = output['energy_uncertainty']
        forces_uncertainty = output['forces_uncertainty']
        
        print(f"Energy: {energy.item()} ± {energy_uncertainty.item()} eV")
        print(f"Forces uncertainty shape: {forces_uncertainty.shape}")
    else:
        print("Model does not output uncertainty")
    
    return output


# ============================================
# 示例4: 比较有无不确定性的训练
# ============================================

def compare_training():
    """比较使用和不使用不确定性的训练效果"""
    
    import matplotlib.pyplot as plt
    
    # 训练两个模型
    print("Training model WITH uncertainty...")
    # train_with_uncertainty()
    
    print("Training model WITHOUT uncertainty...")
    # train_standard()
    
    # 加载训练日志并比较
    # 这里需要实现日志读取和可视化
    
    print("Comparison completed!")


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    print("MACE Uncertainty Quantification Examples")
    print("=" * 50)
    
    # 选择要运行的示例
    example = 1
    
    if example == 1:
        print("\nExample 1: Training with uncertainty")
        # train_with_uncertainty()
        print("Please prepare your training data first!")
        
    elif example == 2:
        print("\nExample 2: Inference with uncertainty")
        # inference_with_uncertainty()
        print("Please train a model first!")
        
    elif example == 3:
        print("\nExample 3: Direct model inference")
        # direct_model_inference()
        print("Please train a model first!")
        
    elif example == 4:
        print("\nExample 4: Compare training methods")
        # compare_training()
        print("This will take a long time!")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


# ============================================
# 使用说明
# ============================================
"""
使用方法：

1. 准备训练数据（.xyz格式）
2. 运行训练脚本：
   python example_uncertainty_api.py
   
3. 或者使用命令行：
   bash example_uncertainty_training.sh

关键参数：
- use_uncertainty=True: 启用不确定性量化
- loss="uncertainty_weighted": 使用不确定性加权损失
- energy_weight, forces_weight: 损失权重

输出：
- energy: 能量预测值
- forces: 力预测值
- energy_uncertainty: 能量不确定性（标准差）
- forces_uncertainty: 力不确定性（标准差）

不确定性的含义：
- 较小的不确定性表示模型对预测有信心
- 较大的不确定性表示模型不确定，可能需要更多训练数据
"""

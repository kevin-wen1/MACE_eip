#!/usr/bin/env python3
"""
直接调用MACE模型进行预测，获取完整的不确定度输出
适用于使用 --use_uncertainty 训练的模型
"""

import torch
import numpy as np
from ase.io import read, write
from mace import data, modules, tools
from e3nn import o3
import argparse
import os

def load_model_and_predict(model_path, atoms, device='cuda'):
    """
    直接加载模型并预测，获取不确定度
    
    返回:
        energy: 预测能量
        forces: 预测力
        energy_uncertainty: 能量不确定度 (如果模型支持)
        forces_uncertainty: 力不确定度 (如果模型支持)
    """
    
    # 加载模型
    model = torch.load(model_path, map_location=device)
    
    # 检查是否是checkpoint格式
    if isinstance(model, dict):
        if 'model' in model:
            model_state = model['model']
            config = model.get('config', {})
        else:
            model_state = model
            config = {}
    else:
        model_state = model.state_dict()
        config = {}
    
    # 检查是否支持不确定度
    use_uncertainty = config.get('use_uncertainty', False)
    
    print(f"模型配置:")
    print(f"  use_uncertainty: {use_uncertainty}")
    
    # 准备输入数据
    # 将ASE atoms转换为MACE需要的格式
    from mace.tools import AtomicNumberTable
    from mace.data import AtomicData, Configuration
    
    # 创建原子序数表
    z_table = AtomicNumberTable([int(z) for z in atoms.get_atomic_numbers()])
    
    # 创建配置
    config_data = Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        energy=0.0,  # 占位符
        forces=np.zeros_like(atoms.get_positions()),  # 占位符
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc()
    )
    
    # 转换为AtomicData
    atomic_data = AtomicData.from_config(
        config_data,
        z_table=z_table,
        cutoff=5.0  # 默认cutoff，应该从模型配置读取
    )
    
    # 转换为batch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([atomic_data])
    batch = batch.to(device)
    
    # 加载模型权重
    # 这里需要重新构建模型架构
    # 注意：这部分需要根据实际的模型架构调整
    
    print("\n注意: 直接调用模型需要完整的模型架构信息")
    print("建议使用MACECalculator进行预测")
    
    return None


def predict_with_calculator(model_path, structure_path, output_path=None, device='cuda'):
    """
    使用MACECalculator进行预测（推荐方法）
    """
    from mace.calculators import MACECalculator
    
    print("="*70)
    print("MACE预测（使用Calculator）")
    print("="*70)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype='float64'
    )
    
    # 读取结构
    print(f"\n读取结构: {structure_path}")
    atoms = read(structure_path)
    print(f"  原子数: {len(atoms)}")
    print(f"  化学式: {atoms.get_chemical_formula()}")
    
    # 设置计算器
    atoms.calc = calculator
    
    # 预测
    print("\n开始预测...")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"\n预测结果:")
    print(f"  能量: {energy:.6f} eV")
    print(f"  力的最大值: {np.abs(forces).max():.6f} eV/Å")
    print(f"  力的RMS: {np.sqrt(np.mean(forces**2)):.6f} eV/Å")
    
    # 尝试获取不确定度
    print("\n尝试获取不确定度...")
    
    # 检查calculator的results
    if hasattr(calculator, 'results') and calculator.results:
        print("Calculator results包含:")
        for key in calculator.results.keys():
            print(f"  - {key}")
        
        if 'energy_uncertainty' in calculator.results:
            energy_unc = calculator.results['energy_uncertainty']
            print(f"\n能量不确定度: {energy_unc:.6f} eV")
        
        if 'forces_uncertainty' in calculator.results:
            forces_unc = calculator.results['forces_uncertainty']
            print(f"力不确定度:")
            print(f"  最大值: {np.abs(forces_unc).max():.6f} eV/Å")
            print(f"  RMS: {np.sqrt(np.mean(forces_unc**2)):.6f} eV/Å")
    else:
        print("注意: Calculator未返回不确定度信息")
        print("可能原因:")
        print("  1. 模型未使用 --use_uncertainty 训练")
        print("  2. 当前MACE版本的Calculator不支持返回不确定度")
        print("  3. 需要修改Calculator代码以返回不确定度")
    
    # 保存结果
    if output_path:
        print(f"\n保存结果到: {output_path}")
        atoms.info['predicted_energy'] = energy
        atoms.arrays['predicted_forces'] = forces
        
        if hasattr(calculator, 'results') and calculator.results:
            if 'energy_uncertainty' in calculator.results:
                atoms.info['energy_uncertainty'] = calculator.results['energy_uncertainty']
            if 'forces_uncertainty' in calculator.results:
                atoms.arrays['forces_uncertainty'] = calculator.results['forces_uncertainty']
        
        write(output_path, atoms, format='extxyz')
        print("  ✓ 已保存")
    
    print("\n" + "="*70)
    
    return {
        'energy': energy,
        'forces': forces,
        'atoms': atoms
    }


def main():
    parser = argparse.ArgumentParser(
        description="MACE模型预测（含不确定度）"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='MACE模型路径'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入结构文件'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备'
    )
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 预测
    output_file = args.output or args.input.replace('.xyz', '_predicted.xyz')
    predict_with_calculator(args.model, args.input, output_file, args.device)


if __name__ == "__main__":
    main()

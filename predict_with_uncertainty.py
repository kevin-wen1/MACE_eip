#!/usr/bin/env python3
"""
使用训练好的MACE模型进行预测
支持输出能量和力的不确定度（如果模型是用--use_uncertainty训练的）
"""

import torch
import numpy as np
from ase.io import read, write
from mace.calculators import MACECalculator
import argparse
import os

def predict_with_uncertainty(model_path, structure_path, output_path=None, device='cuda', predict_all_frames=True):
    """
    使用MACE模型预测结构的能量、力和不确定度
    
    参数:
        model_path: 训练好的MACE模型路径 (.model文件)
        structure_path: 输入结构文件路径 (支持extxyz, POSCAR等ASE支持的格式)
        output_path: 输出文件路径 (可选，保存预测结果)
        device: 'cuda' 或 'cpu'
        predict_all_frames: 是否预测所有帧 (默认True)
    """
    
    print("="*70)
    print("MACE模型预测（含不确定度）")
    print("="*70)
    
    # 1. 加载模型
    print(f"\n加载模型: {model_path}")
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype='float64'
    )
    
    # 检查模型是否支持不确定度
    model = torch.load(model_path, map_location=device)
    has_uncertainty = False
    if isinstance(model, dict) and 'model' in model:
        # 检查模型配置
        if 'use_uncertainty' in model.get('config', {}):
            has_uncertainty = model['config']['use_uncertainty']
    
    print(f"模型支持不确定度: {has_uncertainty}")
    
    # 2. 读取结构（读取所有帧）
    print(f"\n读取结构: {structure_path}")
    
    if predict_all_frames:
        # 读取所有帧
        atoms_list = read(structure_path, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        n_frames = len(atoms_list)
        print(f"  检测到 {n_frames} 帧")
    else:
        # 只读取第一帧
        atoms_list = [read(structure_path, index=0)]
        n_frames = 1
        print(f"  只预测第一帧")
    
    print(f"  原子数: {len(atoms_list[0])}")
    print(f"  化学式: {atoms_list[0].get_chemical_formula()}")
    
    # 3. 预测所有帧
    print(f"\n开始预测 {n_frames} 帧...")
    
    predicted_atoms_list = []
    
    for i, atoms in enumerate(atoms_list, 1):
        if n_frames > 1:
            print(f"\n  [{i}/{n_frames}] 预测第 {i} 帧...")
        
        # 设置计算器
        atoms.calc = calculator
        
        # 计算能量和力
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        if n_frames <= 10 or i % max(1, n_frames // 10) == 0:
            print(f"    能量: {energy:.6f} eV")
            print(f"    力最大值: {np.abs(forces).max():.6f} eV/Å")
        
        # 获取不确定度（如果模型支持）
        if has_uncertainty:
            try:
                if hasattr(calculator, 'results'):
                    results = calculator.results
                    if 'energy_uncertainty' in results:
                        energy_uncertainty = results['energy_uncertainty']
                        atoms.info['energy_uncertainty'] = energy_uncertainty
                        if n_frames <= 10 or i % max(1, n_frames // 10) == 0:
                            print(f"    能量不确定度: {energy_uncertainty:.6f} eV")
                    
                    if 'forces_uncertainty' in results:
                        forces_uncertainty = results['forces_uncertainty']
                        atoms.arrays['forces_uncertainty'] = forces_uncertainty
                        if n_frames <= 10 or i % max(1, n_frames // 10) == 0:
                            print(f"    力不确定度最大值: {np.abs(forces_uncertainty).max():.6f} eV/Å")
            except:
                pass
        
        # 保存预测结果到atoms对象
        atoms.info['predicted_energy'] = energy
        atoms.arrays['predicted_forces'] = forces
        
        predicted_atoms_list.append(atoms)
    
    # 4. 统计信息
    print(f"\n预测完成！")
    print(f"  总帧数: {n_frames}")
    
    energies = np.array([a.info['predicted_energy'] for a in predicted_atoms_list])
    print(f"  能量范围: {energies.min():.6f} ~ {energies.max():.6f} eV")
    
    all_forces = np.concatenate([a.arrays['predicted_forces'] for a in predicted_atoms_list])
    print(f"  力最大值: {np.abs(all_forces).max():.6f} eV/Å")
    print(f"  力RMS: {np.sqrt(np.mean(all_forces**2)):.6f} eV/Å")
    
    if has_uncertainty:
        if 'energy_uncertainty' in predicted_atoms_list[0].info:
            energy_uncs = np.array([a.info.get('energy_uncertainty', 0) for a in predicted_atoms_list])
            print(f"\n不确定度统计:")
            print(f"  能量不确定度范围: {energy_uncs.min():.6f} ~ {energy_uncs.max():.6f} eV")
            print(f"  能量不确定度平均: {energy_uncs.mean():.6f} eV")
        
        if 'forces_uncertainty' in predicted_atoms_list[0].arrays:
            all_forces_unc = np.concatenate([a.arrays.get('forces_uncertainty', np.zeros_like(a.arrays['predicted_forces'])) for a in predicted_atoms_list])
            print(f"  力不确定度最大值: {np.abs(all_forces_unc).max():.6f} eV/Å")
            print(f"  力不确定度平均: {np.abs(all_forces_unc).mean():.6f} eV/Å")
    else:
        print("\n注意: 该模型未启用不确定度功能")
        print("如需不确定度，请使用 --use_uncertainty 参数重新训练模型")
    
    # 5. 保存结果
    if output_path:
        print(f"\n保存结果到: {output_path}")
        write(output_path, predicted_atoms_list, format='extxyz')
        print(f"  ✓ 已保存 {n_frames} 帧")
    
    print("\n" + "="*70)
    
    return {
        'atoms_list': predicted_atoms_list,
        'n_frames': n_frames
    }


def predict_batch(model_path, input_dir, output_dir, device='cuda'):
    """
    批量预测多个结构
    
    参数:
        model_path: 训练好的MACE模型路径
        input_dir: 输入结构目录
        output_dir: 输出目录
        device: 'cuda' 或 'cpu'
    """
    
    import glob
    
    print("="*70)
    print("MACE批量预测")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有输入文件
    input_files = glob.glob(os.path.join(input_dir, "*.extxyz"))
    input_files.extend(glob.glob(os.path.join(input_dir, "POSCAR*")))
    input_files.extend(glob.glob(os.path.join(input_dir, "*.xyz")))
    
    print(f"\n找到 {len(input_files)} 个结构文件")
    
    # 加载模型一次
    print(f"\n加载模型: {model_path}")
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype='float64'
    )
    
    # 批量预测
    results = []
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] 处理: {os.path.basename(input_file)}")
        
        try:
            # 读取所有帧
            atoms_list = read(input_file, index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            
            n_frames = len(atoms_list)
            print(f"  检测到 {n_frames} 帧")
            
            predicted_atoms_list = []
            
            # 预测每一帧
            for atoms in atoms_list:
                atoms.calc = calculator
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                
                atoms.info['predicted_energy'] = energy
                atoms.arrays['predicted_forces'] = forces
                predicted_atoms_list.append(atoms)
            
            # 统计
            energies = np.array([a.info['predicted_energy'] for a in predicted_atoms_list])
            all_forces = np.concatenate([a.arrays['predicted_forces'] for a in predicted_atoms_list])
            
            print(f"  能量范围: {energies.min():.6f} ~ {energies.max():.6f} eV")
            print(f"  力最大值: {np.abs(all_forces).max():.6f} eV/Å")
            
            # 保存结果
            output_file = os.path.join(
                output_dir, 
                os.path.basename(input_file).replace('.xyz', '_predicted.xyz')
            )
            
            write(output_file, predicted_atoms_list, format='extxyz')
            print(f"  ✓ 已保存 {n_frames} 帧")
            
            results.append({
                'file': os.path.basename(input_file),
                'n_frames': n_frames,
                'energy_min': energies.min(),
                'energy_max': energies.max(),
                'max_force': np.abs(all_forces).max()
            })
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    # 保存统计信息
    summary_file = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("MACE批量预测结果汇总\n")
        f.write("="*70 + "\n\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"成功预测: {len(results)}/{len(input_files)} 个文件\n")
        
        total_frames = sum(r['n_frames'] for r in results)
        f.write(f"总帧数: {total_frames}\n\n")
        
        f.write("详细结果:\n")
        f.write("-"*90 + "\n")
        f.write(f"{'文件':<35} {'帧数':<8} {'能量范围 (eV)':<25} {'最大力 (eV/Å)':<15}\n")
        f.write("-"*90 + "\n")
        for r in results:
            energy_range = f"{r['energy_min']:.4f} ~ {r['energy_max']:.4f}"
            f.write(f"{r['file']:<35} {r['n_frames']:<8} {energy_range:<25} {r['max_force']:<15.6f}\n")
    
    print(f"\n统计信息已保存: {summary_file}")
    print(f"总计: {len(results)} 个文件, {total_frames} 帧")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="使用MACE模型预测结构的能量、力和不确定度"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='训练好的MACE模型路径 (.model文件)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入结构文件或目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件或目录'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: cuda)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批量预测模式（输入为目录）'
    )
    
    parser.add_argument(
        '--first-frame-only',
        action='store_true',
        help='只预测第一帧（默认预测所有帧）'
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 检查输入
    if not os.path.exists(args.input):
        print(f"错误: 输入文件/目录不存在: {args.input}")
        return
    
    # 单个文件预测 vs 批量预测
    if args.batch or os.path.isdir(args.input):
        # 批量预测
        output_dir = args.output or os.path.join(os.path.dirname(args.input), "predictions")
        predict_batch(args.model, args.input, output_dir, args.device)
    else:
        # 单个文件预测
        output_file = args.output or args.input.replace('.xyz', '_predicted.xyz')
        predict_all = not args.first_frame_only
        predict_with_uncertainty(args.model, args.input, output_file, args.device, predict_all_frames=predict_all)


if __name__ == "__main__":
    main()

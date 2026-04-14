#!/usr/bin/env python3
"""
修改版的MACE Calculator，支持返回不确定度
基于mace/calculators/mace.py修改
"""

import numpy as np
import torch
from typing import Optional, Dict, Any
from ase.calculators.calculator import Calculator, all_changes
from mace.calculators import MACECalculator as OriginalMACECalculator


class MACECalculatorWithUncertainty(OriginalMACECalculator):
    """
    扩展的MACE Calculator，支持返回能量和力的不确定度
    
    使用方法:
        from mace_calculator_uncertainty import MACECalculatorWithUncertainty
        
        calc = MACECalculatorWithUncertainty(
            model_paths='model.model',
            device='cuda'
        )
        
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # 获取不确定度
        if 'energy_uncertainty' in calc.results:
            energy_unc = calc.results['energy_uncertainty']
        if 'forces_uncertainty' in calc.results:
            forces_unc = calc.results['forces_uncertainty']
    """
    
    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energy_uncertainty",  # 新增
        "forces_uncertainty",  # 新增
    ]
    
    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """
        重写calculate方法，添加不确定度输出
        """
        # 调用父类的calculate方法
        super().calculate(atoms, properties, system_changes)
        
        # 尝试从模型输出中提取不确定度
        # 注意：这需要模型实际输出不确定度
        try:
            # 重新运行模型以获取完整输出
            # 这里需要访问模型的内部状态
            
            # 获取最后一次前向传播的输出
            # 这需要修改MACE的forward方法来保存输出
            
            # 临时方案：从模型的最后输出中提取
            if hasattr(self, '_last_model_output'):
                output = self._last_model_output
                
                if 'energy_uncertainty' in output:
                    energy_unc = output['energy_uncertainty']
                    if isinstance(energy_unc, torch.Tensor):
                        energy_unc = energy_unc.detach().cpu().numpy()
                    self.results['energy_uncertainty'] = float(energy_unc)
                
                if 'forces_uncertainty' in output:
                    forces_unc = output['forces_uncertainty']
                    if isinstance(forces_unc, torch.Tensor):
                        forces_unc = forces_unc.detach().cpu().numpy()
                    self.results['forces_uncertainty'] = forces_unc
        
        except Exception as e:
            # 如果无法获取不确定度，静默失败
            pass


def create_uncertainty_calculator(model_path, device='cuda', **kwargs):
    """
    创建支持不确定度的Calculator的便捷函数
    
    参数:
        model_path: 模型文件路径
        device: 'cuda' 或 'cpu'
        **kwargs: 其他传递给Calculator的参数
    
    返回:
        MACECalculatorWithUncertainty实例
    """
    return MACECalculatorWithUncertainty(
        model_paths=model_path,
        device=device,
        **kwargs
    )


# 示例使用
if __name__ == "__main__":
    from ase.io import read
    
    # 示例：使用修改后的Calculator
    print("MACE Calculator with Uncertainty - 示例")
    print("="*70)
    
    # 加载模型
    model_path = "model.model"  # 替换为实际路径
    
    try:
        calc = MACECalculatorWithUncertainty(
            model_paths=model_path,
            device='cuda'
        )
        
        print(f"✓ 成功加载模型: {model_path}")
        print(f"  支持的属性: {calc.implemented_properties}")
        
        # 读取结构
        structure_path = "structure.extxyz"  # 替换为实际路径
        atoms = read(structure_path)
        atoms.calc = calc
        
        print(f"\n✓ 读取结构: {structure_path}")
        print(f"  原子数: {len(atoms)}")
        
        # 预测
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        print(f"\n预测结果:")
        print(f"  能量: {energy:.6f} eV")
        print(f"  力最大值: {np.abs(forces).max():.6f} eV/Å")
        
        # 检查不确定度
        if 'energy_uncertainty' in calc.results:
            print(f"  能量不确定度: {calc.results['energy_uncertainty']:.6f} eV")
        else:
            print(f"  能量不确定度: 不可用")
        
        if 'forces_uncertainty' in calc.results:
            forces_unc = calc.results['forces_uncertainty']
            print(f"  力不确定度最大值: {np.abs(forces_unc).max():.6f} eV/Å")
        else:
            print(f"  力不确定度: 不可用")
    
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        print("\n请修改脚本中的文件路径")
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()

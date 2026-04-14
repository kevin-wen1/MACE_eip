#!/bin/bash

# MACE with Uncertainty Quantification - Training Example
# 使用EIP风格的不确定性量化训练MACE模型

# ============================================
# 示例1: 使用不确定性训练
# ============================================
echo "Example 1: Training with Uncertainty Quantification"

mace_run_train \
    --name="mace_with_uncertainty" \
    --train_file="train.xyz" \
    --valid_file="valid.xyz" \
    --test_file="test.xyz" \
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
    --ema \
    --ema_decay=0.99 \
    --device=cuda \
    --default_dtype=float64

echo "Training with uncertainty completed!"

# ============================================
# 示例2: 标准训练（不使用不确定性）
# ============================================
echo ""
echo "Example 2: Standard Training (without Uncertainty)"

mace_run_train \
    --name="mace_standard" \
    --train_file="train.xyz" \
    --valid_file="valid.xyz" \
    --model="MACE" \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1000 \
    --device=cuda

echo "Standard training completed!"

# ============================================
# 示例3: 使用ScaleShiftMACE + 不确定性
# ============================================
echo ""
echo "Example 3: ScaleShiftMACE with Uncertainty"

mace_run_train \
    --name="scaleshiftmace_uncertainty" \
    --train_file="train.xyz" \
    --valid_file="valid.xyz" \
    --model="ScaleShiftMACE" \
    --loss="uncertainty_weighted" \
    --use_uncertainty \
    --energy_weight=1.0 \
    --forces_weight=100.0 \
    --E0s="average" \
    --scaling="rms_forces_scaling" \
    --device=cuda

echo "ScaleShiftMACE with uncertainty training completed!"

# ============================================
# 参数说明
# ============================================
# --use_uncertainty: 启用不确定性量化（必须）
# --loss="uncertainty_weighted": 使用不确定性加权损失函数
# --energy_weight: 能量损失权重
# --forces_weight: 力损失权重（会被不确定性自动调整）
# --ema: 使用指数移动平均（推荐）
# --ema_decay: EMA衰减率

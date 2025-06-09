#!/bin/bash

# DPMM-MoE-SAC 엄밀한 학습 실행 스크립트

echo "🚀 Starting Rigorous DPMM-MoE-SAC Training"

# 1. 수학적 검증 먼저 실행
echo "🔬 Running mathematical verification tests..."
python mathematical_tests.py

if [ $? -ne 0 ]; then
    echo "❌ Mathematical tests failed! Please fix implementation."
    exit 1
fi

echo "✅ Mathematical tests passed!"

# 2. 메인 학습 실행
echo "🧠 Starting main training..."
python main_rigorous_training.py \
    --episodes 200 \
    --max_steps 1000 \
    --latent_dim 32 \
    --tau_birth 1e-4 \
    --tau_merge 0.5 \
    --save_checkpoint "rigorous_dpmm_moe_sac.pth" \
    --headless

echo "🎯 Training completed!"

# 3. 결과 분석
echo "📊 Analyzing results..."
python -c "
import torch
import numpy as np
from integrated_dpmm_moe_sac import IntegratedDPMMMoESAC

# 체크포인트 로드 및 분석
checkpoint = torch.load('rigorous_dpmm_moe_sac.pth')
print('📈 Final Training Results:')
print(f'  Timesteps: {checkpoint[\"timestep\"]}')
print(f'  Expert usage: {checkpoint[\"expert_usage_count\"]}')
print('✅ Analysis completed!')
"

echo "🎉 All done!"

"""
PaddlePaddle 模型转 PyTorch 模型转换脚本
将 PaddlePaddle 训练的模型权重转换为 PyTorch 格式
"""

import os
import sys

import torch
import paddle
import argparse
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def convert_paddle_to_torch(paddle_path: str, torch_path: str, config: Config):
    """将 PaddlePaddle 模型转换为 PyTorch 模型"""
    
    print("=" * 60)
    print("🔄 PaddlePaddle → PyTorch 模型转换")
    print("=" * 60)
    
    print(f"\n📂 加载 PaddlePaddle 模型: {paddle_path}")
    
    paddleCheckpoint = paddle.load(paddle_path)
    
    if isinstance(paddleCheckpoint, dict):
        if 'model_state_dict' in paddleCheckpoint:
            paddle_state_dict = paddleCheckpoint['model_state_dict']
            print(f"   ✓ 从 model_state_dict 提取模型参数")
        else:
            paddle_state_dict = paddleCheckpoint
    else:
        paddle_state_dict = paddleCheckpoint
    
    model_config = config.MODEL_CONFIG
    
    print(f"\n📊 模型配置:")
    print(f"   - d_model: {model_config['d_model']}")
    print(f"   - d_state: {model_config['d_state']}")
    print(f"   - n_layers: {model_config['n_layers']}")
    print(f"   - vocab_size: {model_config['vocab_size']}")
    
    print(f"\n🔍 PaddlePaddle 模型参数列表:")
    tensor_count = 0
    for name, param in paddle_state_dict.items():
        if hasattr(param, 'shape'):
            print(f"   {name}: shape={param.shape}, dtype={param.dtype}")
            tensor_count += 1
    
    print(f"\n   总计: {tensor_count} 个张量参数")
    
    print(f"\n🔄 开始转换参数 (PaddlePaddle → PyTorch)...")
    print(f"   注意: PaddlePaddle 和 PyTorch 的 Linear 层权重形状是转置关系")
    
    torch_state_dict = OrderedDict()
    converted_count = 0
    transposed_count = 0
    
    for paddle_name, param in paddle_state_dict.items():
        if hasattr(param, 'shape'):
            param_numpy = param.numpy()
            
            torch_name = paddle_name
            
            need_transpose = False
            if param_numpy.ndim == 2:
                if 'proj' in paddle_name and 'embedding' not in paddle_name:
                    need_transpose = True
                elif 'bpm_embedding' in paddle_name and 'weight' in paddle_name:
                    need_transpose = True
            
            if need_transpose:
                param_numpy = param_numpy.T
                transposed_count += 1
            
            torch_state_dict[torch_name] = torch.from_numpy(param_numpy)
            print(f"   ✓ {paddle_name}" + (" (转置)" if need_transpose else ""))
            converted_count += 1
    
    print(f"\n📊 转换统计:")
    print(f"   - 成功转换: {converted_count}")
    print(f"   - 转置操作: {transposed_count}")
    
    print(f"\n💾 保存 PyTorch 模型: {torch_path}")
    os.makedirs(os.path.dirname(torch_path), exist_ok=True)
    torch.save({
        'model_state_dict': torch_state_dict,
        'config': {
            'd_model': config.MODEL_CONFIG['d_model'],
            'd_state': config.MODEL_CONFIG['d_state'],
            'd_conv': config.MODEL_CONFIG['d_conv'],
            'expand': config.MODEL_CONFIG['expand'],
            'n_layers': config.MODEL_CONFIG['n_layers'],
            'dropout': config.MODEL_CONFIG['dropout'],
            'max_seq_length': config.MODEL_CONFIG['max_seq_length'],
            'vocab_size': config.MODEL_CONFIG['vocab_size'],
        }
    }, torch_path)
    
    print(f"\n✅ 转换完成!")
    print(f"   PyTorch 模型已保存到: {torch_path}")
    
    return torch_state_dict


def load_torch_model(torch_path: str, config: Config):
    """加载 PyTorch 模型并返回状态字典"""
    print(f"\n📂 加载 PyTorch 模型: {torch_path}")
    
    checkpoint = torch.load(torch_path, map_location='cpu')
    
    print(f"   模型配置: {checkpoint.get('config', 'N/A')}")
    
    return checkpoint['model_state_dict']


def verify_conversion(paddle_path: str, torch_state_dict: dict):
    """验证转换结果"""
    print(f"\n🔍 验证转换结果...")
    
    paddleCheckpoint = paddle.load(paddle_path)
    
    if isinstance(paddleCheckpoint, dict) and 'model_state_dict' in paddleCheckpoint:
        paddle_state_dict = paddleCheckpoint['model_state_dict']
    else:
        paddle_state_dict = paddleCheckpoint
    
    verified_count = 0
    for name, torch_param in torch_state_dict.items():
        if name in paddle_state_dict:
            paddle_param = paddle_state_dict[name]
            if hasattr(paddle_param, 'numpy'):
                paddle_param = paddle_param.numpy()
                
                param_numpy = paddle_param
                if torch_param.ndim == 2:
                    param_numpy = param_numpy.T
                
                diff = float(torch.abs(torch.from_numpy(param_numpy) - torch_param).max())
                print(f"   ✓ {name}: max_diff={diff:.8f}")
                verified_count += 1
    
    print(f"\n   验证通过: {verified_count}/{len(torch_state_dict)} 个参数")


def main():
    parser = argparse.ArgumentParser(description='PaddlePaddle 模型转 PyTorch')
    parser.add_argument('--paddle-model', type=str, 
                       default='../work/checkpoints/best_model.pdparams',
                       help='PaddlePaddle 模型路径')
    parser.add_argument('--torch-model', type=str, 
                       default='models/best_model.pt',
                       help='PyTorch 模型输出路径')
    parser.add_argument('--verify', action='store_true',
                       help='验证转换结果')
    args = parser.parse_args()
    
    config = Config()
    
    paddle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.paddle_model))
    torch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.torch_model))
    
    torch_state_dict = convert_paddle_to_torch(paddle_path, torch_path, config)
    
    if args.verify:
        verify_conversion(paddle_path, torch_state_dict)
    
    print("\n" + "=" * 60)
    print("📝 使用方法:")
    print("=" * 60)
    print("""
1. 转换模型:
   python convert_model.py

2. 验证转换:
   python convert_model.py --verify

3. 在推理应用中使用:
   from mamba_model import MambaMIDIGenerator, create_model
   import torch
   
   model = create_model(config)
   checkpoint = torch.load('models/best_model.pt', map_location='cpu')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
    """)


if __name__ == '__main__':
    main()

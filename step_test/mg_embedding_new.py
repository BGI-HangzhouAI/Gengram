#!/usr/bin/env python3

import sys
sys.path.insert(0, "/mnt/zzbnew/peixunban/xxhn/Megatron-LM")
import os
import torch
import numpy as np
from functools import partial
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core import InferenceParams
from gpt_builders import gpt_builder
from model_provider import model_provider
import random



# 给定列表
numbers = [5, 6, 7, 8]

# 从给定列表中随机抽取100个元素
random_numbers = random.choices(numbers, k=524286)

# 设置确定性计算
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def add_model_args(parser):
    parser.add_argument("--mg-ckpt-path", type=str, required=True,
                       help="Megatron模型路径")
    parser.add_argument("--check-layers", type=int, default=3,
                       help="检查的层数")
    parser.add_argument('--tokenizer-path', type=str, 
                      default='/mnt/zzb/peixunban/chenjh356/Megatron-LM/convert_new/hf_ckpts/onehot-mix1b-4n-8k-b1-32-tp1pp1ep1',
                      help='HuggingFace tokenizer路径')
    parser.add_argument('--use-hf-tokenizer', action='store_true', default=True,
                      help='使用HuggingFace tokenizer')
    return parser



def load_mg_model():
    """加载Megatron模型"""
    print("加载Megatron模型")
    from megatron.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    
    # 获取模型
    model_provider_with_builder = partial(model_provider, gpt_builder)
    model = get_model(model_provider_with_builder, wrap_with_ddp=False)
    
    # 加载checkpoint
    print("加载checkpoint...")
    iteration = load_checkpoint(model, None, None, strict=False)
    
    # 返回第一个模型（因为get_model返回的是列表）
    model = model[0]

    print("Megatron模型加载完成")
    return model, iteration[0]


def get_mg_hidden_state(mgmodel, mgargs):
    
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]
    # 模型参数
    hidden_size = mgargs.hidden_size
    num_heads = mgargs.num_attention_heads
    vocab_size = mgargs.padded_vocab_size
    num_experts = mgargs.num_experts
        

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        """输出hook函数"""
        frame, name = mode.split('-')
        if mode in ['mg-layer_out']:
            mg_hiddens[layer_idx][name] = output[0]

    actual_model = mgmodel
    if hasattr(mgmodel, 'module'):
        actual_model = mgmodel.module

    # 获取decoder layers
    decoder_layers = None
    if hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
        decoder_layers = actual_model.decoder.layers
    elif hasattr(actual_model, 'language_model') and hasattr(actual_model.language_model, 'decoder'):
        decoder_layers = actual_model.language_model.decoder.layers
    elif hasattr(actual_model, 'layers'):
        decoder_layers = actual_model.layers
    else:
        print("错误: 无法找到decoder layers")
        print(f"可用属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
        return

    for idx, layer in enumerate(decoder_layers):
            
        layer.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-layer_out'), 
            with_kwargs=True)

    # 准备测试输入
    input_ids = torch.tensor([[15] + random_numbers + [16]]).long()
    # torch.save(input_ids, 'input_ids.pt')
    # input_ids = torch.load('input_ids.pt')

    print("开始Megatron模型推理...")
    
    # Megatron模型推理
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids, -100, True, True, True)
    
    
    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                position_ids=position_ids.cuda(),
                # inference_params=inference_params
            )
        except torch.cuda.OutOfMemoryError:
            print('Megatron模型推理OOM')
            is_oom = True
        finally:
            mgmodel.cpu()
            del mgmodel
    gpu_id = torch.cuda.current_device()
    
    torch.save(mg_hiddens, f'/mnt/workspace/users/chenjh356/embedding/mg_hidden_test_rank{gpu_id}.pt')
    


def main():
    """主函数"""
    def add_extra_args(parser):
        parser = add_model_args(parser)
        return parser
    
    # 初始化Megatron
    initialize_megatron(extra_args_provider=add_extra_args, args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
            'use_checkpoint_args': True,
            'use_mcore_models': True,
        })
    args = get_args()
    

    print(f"MG模型路径: {args.load}")
    
    
    try:
        mg_model, iteration = load_mg_model()

        mg_model = mg_model.to(torch.float32)
        print(mg_model)

        # 执行对比
        # get_mg_hidden_state(mg_model, args)
        print("\n✅ Embedding提取完成!")
        for (name, module) in mg_model.named_modules():
            print(name, module)
        
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
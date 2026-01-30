#!/usr/bin/env python3

#!/usr/bin/env python3
import os
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_tokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core import InferenceParams
from gpt_builders import gpt_builder
from model_provider import model_provider
from functools import partial
import random
import torch.distributed as dist
import sys
sys.path.insert(
    0, "/mnt/zzbnew/peixunban/xxhn/Megatron-LM")


# 给定列表
numbers = [5, 6, 7, 8]

# 从给定列表中随机抽取100个元素
random.seed(42)
random_numbers = random.choices(numbers, k=64)

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
    # model = get_model(model_provider, wrap_with_ddp=False)
    # 获取模型 - model_provider需要model_builder作为第一个参数
    # 使用partial来绑定gpt_builder作为第一个参数
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
    input_data = []
    output_data = []
    back1 = []
    back2 = []
    # def print_output_hook(module, args, kwargs, output, layer_idx, mode):
    #     """输出hook函数"""
    #     frame, name = mode.split('-')
    #     if mode in ['mg-layer_out']:
    #         mg_hiddens[layer_idx][name] = output[0]

    def hook_test(module, fea_in, fea_out):
        input_data.append(fea_in)
        output_data.append(fea_out)

#     actual_model = mgmodel
#     if hasattr(mgmodel, 'module'):
#         actual_model = mgmodel.module

#     # 获取decoder layers
#     decoder_layers = None
#     if hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
#         decoder_layers = actual_model.decoder.layers
#     elif hasattr(actual_model, 'language_model') and hasattr(actual_model.language_model, 'decoder'):
#         decoder_layers = actual_model.language_model.decoder.layers
#     elif hasattr(actual_model, 'layers'):
#         decoder_layers = actual_model.layers
#     else:
#         print("错误: 无法找到decoder layers")
#         print(f"可用属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
#         return

#     for idx, layer in enumerate(decoder_layers):

#         layer.register_forward_hook(
#             partial(print_output_hook, layer_idx=idx, mode='mg-layer_out'),
#             with_kwargs=True)
    rank = dist.get_rank()
    layer_name = "module.decoder.final_layernorm"
    # print(layer_name)
    for (name, module) in mgmodel.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook_test)

    # 准备测试输入
    input_ids = torch.tensor([random_numbers]).long()
    # print(f"rank {rank} input_ids")
    # print(input_ids)
    # torch.save(input_ids, 'input_ids.pt')
    # input_ids = torch.load('input_ids.pt')

    print("开始Megatron模型推理...")

    # Megatron模型推理
    # 对于推理，使用简单的默认值，不需要实际的eod/pad token
    # 使用-1作为eod和pad token（不会出现在input_ids中，所以不会影响mask）
    args = get_args()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids,
        -1,  # eod_token (推理时不需要，使用不会出现的值)
        -1,  # pad_token (推理时不需要，使用不会出现的值)
        args.reset_position_ids if hasattr(
            args, 'reset_position_ids') else True,  # reset_position_ids
        args.reset_attention_mask if hasattr(
            args, 'reset_attention_mask') else True,  # reset_attention_mask
        False,  # eod_mask_loss (推理时不需要mask loss)
        False   # pad_mask_loss (推理时不需要mask loss)
    )

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
    print(f"rank {rank} input {input_data[0][0].shape}")
    print(input_data)
    print(f"rank {rank} output {output_data[0].shape}")
    print(output_data)
    # print(f"rank {rank} back1 {input_data[0].shape}")
    # print(back1)
    # print(f"rank {rank} back2 {input_data[0].shape}")
    # print(back2)

    # torch.save(mg_hiddens, f'./mg_hidden_test_rank{gpu_id}.pt')


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

        # mg_model = mg_model.to(torch.bfloat16)
        # rank = dist.get_rank()
        # if rank==0:
        print(mg_model)
        # for (name, module) in mg_model.named_modules():
        #     if rank == 0:
        #         print()
        # 执行对比
        # get_mg_hidden_state(mg_model, args)
        # print("\n✅ Embedding提取完成!")

    except Exception as e:
        print(f"❌ 获取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

import torch
import argparse
import os
import sys
sys.path.append('/root/mwnoh/smoothquant/smoothquant')

from pathlib import Path

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer

from opt import Int8OPTForCausalLM
from smooth import smooth_lm
from calibration import get_static_decoder_layer_scales

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='/mnt/models/opt/opt-125m')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='../act_scales/opt-125m.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='../dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()
    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_path,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    if args.export_FT:
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        # int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        # int8_model.save_pretrained(output_path)

        model_smoothquant = Int8OPTForCausalLM.from_pretrained(
            '/mnt/models/smoothquant/opt-125m', torch_dtype=torch.float16, device_map='auto')

        output_dir = "opt-125m-weights-pt"
        os.makedirs(output_dir, exist_ok=True)
        state_dict = model_smoothquant.state_dict()
        for name, weight_tensor in state_dict.items():
            name += '.pt'
            filename = os.path.join(output_dir, name)
            if weight_tensor.dtype != torch.int8:
                weight_tensor = weight_tensor.to(torch.float32).cpu()
            elif weight_tensor.dtype == torch.int8:
                weight_tensor = weight_tensor.to(torch.int32).cpu()
            np.save(filename, weight_tensor)
            print(f"Saved {name} -> {filename}")
            



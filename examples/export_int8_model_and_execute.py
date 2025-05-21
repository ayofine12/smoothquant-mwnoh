import torch
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
sys.path.append('/root/mwnoh/smoothquant/smoothquant')

from pathlib import Path
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
from opt import Int8OPTForCausalLM
from smooth import smooth_lm
from calibration import get_static_decoder_layer_scales
from torch.nn.functional import pad
from datasets import load_dataset

DEVICE = torch.device("cuda")  

def generate(model, input_ids, past_key_values=None):
    model.eval()
    if past_key_values is None:
        pad_len = 512 - input_ids.shape[1]
        input_ids = pad(input_ids, (0, pad_len), value=1)
        torch.cuda.synchronize()
        outputs = model(input_ids)
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()
        outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        torch.cuda.synchronize()
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='/mnt/models/opt/opt-125m')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='/root/mwnoh/smoothquant/act_scales/opt-125m.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='/root/mwnoh/smoothquant/dataset/val.jsonl.zst',
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
        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
        input = "Earth rotates around the sun, and moon rotates around the "
        input_ids_list = tokenizer(input).input_ids
        input_ids = torch.tensor(input_ids_list).unsqueeze(0).cuda(DEVICE)
        print("input_ids size: ", input_ids.size())

        outputs = generate(int8_model, input_ids)
        print("outputs size: ", outputs.logits.size())
        past_key_values = outputs.past_key_values
        print("past_key_values size: ", past_key_values[0][0].size())
        
        
        next_token_logits = outputs.logits[:, -1, :]
        predicted_token_id = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
        predicted_token_id = predicted_token_id.unsqueeze(-1)
        
        outputs = generate(int8_model, predicted_token_id, past_key_values)
        print("outputs size: ", outputs.logits.size())
        past_key_values = outputs.past_key_values
        print("past_key_values size: ", past_key_values[0][0].size())
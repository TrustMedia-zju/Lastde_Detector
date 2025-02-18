import nltk
import torch 
import pandas as pd
import numpy as np
import time
import argparse
import tqdm
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')


# os.chdir("......") # cache_dir


device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = {  'gptj_6b': 'gpt-j-6b', # https://huggingface.co/EleutherAI/gpt-j-6b/tree/main
                     'gptneo_2.7b': 'gpt-neo-2.7B', # https://huggingface.co/EleutherAI/gpt-neo-2.7B/tree/main
                     'gpt2_xl': 'gpt2-xl',# https://huggingface.co/openai-community/gpt2-xl/tree/main
                     'opt_2.7b': 'opt-2.7b', # https://huggingface.co/facebook/opt-2.7b/tree/main
                     'bloom_7b': 'bloom-7b1', # https://huggingface.co/bigscience/bloom-7b1/tree/main
                     'falcon_7b': 'falcon-7b', # https://huggingface.co/tiiuae/falcon-7b/tree/main
                     'gemma_7b': "gemma-7b", # https://huggingface.co/google/gemma-7b/tree/main
                     'llama1_13b': 'Llama-13b', # https://huggingface.co/huggyllama/llama-13b/tree/main
                     'llama2_13b': 'Llama-2-13B-fp16', # https://huggingface.co/TheBloke/Llama-2-13B-fp16/tree/main
                     'llama3_8b': 'Llama-3-8B', # https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main
                     'opt_13b': 'opt-13b', # https://huggingface.co/facebook/opt-13b/tree/main
                     'phi2': 'phi-2', # https://huggingface.co/microsoft/phi-2/tree/main
                     "mgpt": 'mGPT', # https://huggingface.co/ai-forever/mGPT/tree/main
                     'qwen1.5_7b': 'Qwen1.5-7B', # https://huggingface.co/Qwen/Qwen1.5-7B/tree/main
                     'yi1.5_6b': 'Yi-1.5-6B', # https://huggingface.co/01-ai/Yi-1.5-6B/tree/main 
                     't5': 't5-3b'} # https://huggingface.co/google-t5/t5-3b/tree/main

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))

    if 't5' in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto")
    
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    optional_tok_kwargs = {}
    if "opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    optional_tok_kwargs['padding_side'] = 'right'

    if "t5" in model_fullname:
        optional_tok_kwargs["model_max_length"] = 512

    tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if '13b' in model_fullname:
            tokenizer.pad_token_id = 0
    return tokenizer

# load human-machine dataset
def load_human_machine_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"

    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data
    
# write args to file
def save_human_machine_perturbation_data(data, args):
    if args.main_results:
        # args_file = os.getcwd() + f"{args.output_file}_perturbation_{args.n_perturbations}.args.json"
        args_file = f"{args.output_file}_perturbation_{args.n_perturbations}.args.json"
    else:
        # args_file = os.getcwd() + f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}.args.json"
        args_file = f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}.args.json"
        
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    if args.main_results:
        # data_file = os.getcwd() + f"{args.output_file}_perturbation_{args.n_perturbations}.raw_data.json"
        data_file = f"{args.output_file}_perturbation_{args.n_perturbations}.raw_data.json"
    else:
        # data_file = os.getcwd() + f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}.raw_data.json"
        data_file = f"{args.output_file}_perturbation_{args.n_perturbations}_{args.scenario}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

def save_human_machine_regeneration_data(data, args):
    # args_file = os.getcwd() + f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}.args.json"
    args_file = f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    # data_file = os.getcwd() + f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    data_file = f"{args.output_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")
        
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # 
# T5_Perturbation：DetectGPT、DetectLLM_NPR
def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    buffer_size = 1
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# replace each masked span with a sample from T5 mask_model
def replace_masks(args, mask_model, mask_tokenizer, texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = mask_model.generate(**tokens, max_length=args.max_length, min_length=args.min_length, do_sample=True, top_p=args.mask_top_p,
                                num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts

def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct))
    return outputs

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  # 
# CutOff-Regeneration：DNA-GPT
# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def sample_from_model(args, model, tokenizer, texts):
    # cut off X to [X, Y_0]
    texts = [t.split(' ') for t in texts]
    texts = [' '.join(t[: int(len(t) * args.truncate_ratio)]) for t in texts]
    all_encoded = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
    
    model.eval()
    decoded = ['' for _ in range(len(texts))]
    
    # Regeration
    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    m = 0
    while m < args.min_length:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {args.min_length}, regenerating (try {tries})")

        sampling_kwargs = {'temperature': args.temperature}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        outputs = model.generate(**all_encoded, min_length=args.min_length, max_length=args.max_length, do_sample=True,
                                            **sampling_kwargs, pad_token_id=tokenizer.eos_token_id,
                                            eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded)
        tries += 1
    
    return decoded

def rewrite_texts_(args, model, tokenizer, texts):
    rewrote_texts = sample_from_model(args, model, tokenizer, texts)
    # trim to shorter length
    rewrote_texts = [trim_to_shorter_length(o, s)[1] for o, s in zip(texts, rewrote_texts)] 
    return rewrote_texts
    
def rewrite_texts(args, model, tokenizer, texts):
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(rewrite_texts_(args, model, tokenizer, texts[i:i + chunk_size]))
    return outputs

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  # 
def generate_samples(data, args):
    # T5-Perturbation or CutOff-Regeneration
    if args.Generation_methods == "Perturbation":
        model = load_model(args.model)
        tokenizer = load_tokenizer(args.model)
        model.eval()
    else:
        model = load_model(args.rewrite_model)
        tokenizer = load_tokenizer(args.rewrite_model)
        model.eval()
        
    n_samples = len(data["sampled"])

    perturbs_rewrites = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Perturb or Rewrite text"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        if args.Generation_methods == "Perturbation":
            # perturb
            n_perturbations = args.n_perturbations
            name = f'perturbation_{n_perturbations}'
            p_sampled_text = perturb_texts(args, model, tokenizer, [sampled_text for _ in range(n_perturbations)]) 
            p_original_text = perturb_texts(args, model, tokenizer, [original_text for _ in range(n_perturbations)])
            assert len(p_sampled_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_sampled_text)}"
            assert len(p_original_text) == n_perturbations, f"Expected {n_perturbations} perturbed samples, got {len(p_original_text)}"
            # result
            perturbs_rewrites.append({
                "original": original_text, #  str
                "sampled": sampled_text, # str
                "perturbed_sampled": p_sampled_text, # list
                "perturbed_original": p_original_text # list
            })
        else:
            n_regenerations = args.n_regenerations
            name = f'regenerations_{n_regenerations}'
            r_sampled_text = rewrite_texts(args, model, tokenizer, [sampled_text for _ in range(n_regenerations)]) # list
            r_original_text = rewrite_texts(args, model, tokenizer, [original_text for _ in range(n_regenerations)]) # list
            assert len(r_sampled_text) == n_regenerations, f"Expected {n_regenerations} rewrote samples, got {len(r_sampled_text)}"
            assert len(r_original_text) == n_regenerations, f"Expected {n_regenerations} rewrote samples, got {len(r_original_text)}"
    
            # result
            perturbs_rewrites.append({
                "original": original_text,
                "sampled": sampled_text,
                "rewrote_sampled": r_sampled_text,
                "rewrote_original": r_original_text
            })
    # print(perturbs_rewrites)
    return perturbs_rewrites

def experiments(args):
    data = load_human_machine_data(args.dataset_file)
    new_data = generate_samples(data, args)
    if args.Generation_methods == "Perturbation":
        save_human_machine_perturbation_data(new_data, args)
    else:
        save_human_machine_regeneration_data(new_data, args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Generation_methods', type=str, default="Perturbation", choices=["Perturbation", "Rewrite"])
    parser.add_argument('--output_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")

    # Perturbation
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--model', type=str, default="t5")

    # Rewrite
    parser.add_argument('--truncate_ratio', type=float, default=0.5)
    parser.add_argument('--rewrite_model', type=str, default="llama3_8b")
    parser.add_argument('--n_regenerations', type=int, default=10)
    parser.add_argument('--n_prompt_tokens', type=int, default=30)
    parser.add_argument('--do_top_k', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--scenario', type=str, default='white', choices=["black", "white"])
    parser.add_argument('--main_results', action='store_true', default=False)


    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=65) # When rewriting (DNA-GPT), the minimum length of some datasets could not reach 100, so it was reduced to 65
    args = parser.parse_args()
 
    experiments(args)
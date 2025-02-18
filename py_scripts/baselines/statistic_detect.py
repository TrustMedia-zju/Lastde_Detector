from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import  torch
import torch.nn.functional as F
from untils.metrics import get_roc_metrics, get_precision_recall_metrics
import numpy as np
from tqdm import tqdm
import json
import os
import time
from scoring_methods import based_scoring_baselines
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
                     'yi1.5_6b': 'Yi-1.5-6B'} # https://huggingface.co/01-ai/Yi-1.5-6B/tree/main 

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    print(f'Loading model {model_fullname}...')

    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto")
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
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

    base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def load_data(input_file):
    # print(os.getcwd())
    # print(input_file)
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    
    # eval criterions
    criterion_fns = {'likelihood': based_scoring_baselines.get_likelihood,
                     'logrank': based_scoring_baselines.get_logrank,
                     'entropy': based_scoring_baselines.get_entropy,
                     'lrr':based_scoring_baselines.get_lrr,
                     'lastde': based_scoring_baselines.get_lastde}


    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        eval_results = []
        for idx in tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            # ===========================================================
            # original text
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits, labels)
            # sampled text
            tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]
                sampled_crit = criterion_fn(logits, labels)
                
            eval_results.append({"original": original_text,
                            "original_crit": original_crit,
                            "sampled": sampled_text,
                            "sampled_crit": sampled_crit})

        # compute prediction scores for real/sampled passages
        predictions = {'real': [x["original_crit"] for x in eval_results],
                       'samples': [x["sampled_crit"] for x in eval_results]}
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        # log results
        # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
        results_file = f'{args.output_file}.{name}.json'
        results = { 'name': f'{name}_threshold',
                    'info': {'n_samples': n_samples},
                    'predictions': predictions,
                    'raw_results': eval_results,
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                    'loss': 1 - pr_auc}
        with open(results_file, 'w') as fout:
            json.dump(results, fout)
            print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--output_file', type=str, default="/experiment_results/statistic_detection_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")

    args = parser.parse_args()

    experiment(args)
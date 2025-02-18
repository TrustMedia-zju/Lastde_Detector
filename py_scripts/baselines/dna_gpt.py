from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import  torch
from untils.metrics import get_roc_metrics, get_precision_recall_metrics
import numpy as np
from tqdm import tqdm
import json
import time
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

    base_tokenizer = AutoTokenizer.from_pretrained(model_path, **optional_tok_kwargs)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

def load_data(args):
    # data_file = f"{dataset_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    data_file = f"{args.dataset_file}_regeneration_{args.n_regenerations}_{args.scenario}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()


def get_dna_gpt(args, scoring_model, scoring_tokenizer, text, rewrote_texts):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        ori_likelihood = get_likelihood(logits, labels)
        regeneration_likelihood = []
        for rewrote_text in rewrote_texts:
            tokenized = scoring_tokenizer(rewrote_text, return_tensors="pt", return_token_type_ids=False).to(device)
            labels = tokenized.input_ids[:, 1:]
            logits = scoring_model(**tokenized).logits[:, :-1]
            regeneration_likelihood.append(get_likelihood(logits, labels))
        return ori_likelihood - np.mean(regeneration_likelihood)

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    # load data
    data = load_data(args)
    n_samples = len(data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    criterion_fn = get_dna_gpt
    name = 'dna_gpt'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    eval_results = []
    for idx in tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data[idx]["original"]
        sampled_text = data[idx]["sampled"]
        rewrote_original = data[idx]["rewrote_original"]
        rewrote_sampled = data[idx]["rewrote_sampled"]
        original_crit = criterion_fn(args, scoring_model, scoring_tokenizer, original_text, rewrote_original)
        sampled_crit = criterion_fn(args, scoring_model, scoring_tokenizer, sampled_text, rewrote_sampled)
        # result
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
                'info': {'n_samples': n_samples, 'n_regenerations': args.n_regenerations},
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
    parser.add_argument('--n_regenerations', type=int, default=10)
    parser.add_argument('--output_file', type=str, default="/experiment_results/dna_gpt_detection_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/regeneration_data_dnagpt/xsum_llama3_8b")
    parser.add_argument('--scoring_model_name', type=str, default="llama3_8b")
    parser.add_argument('--scenario', type=str, default='white', choices=["black", "white"])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)


from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import  torch
from utils.metrics import get_roc_metrics, get_precision_recall_metrics
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
    if args.main_results:
        # data_file = os.getcwd() + f"{input_file}_perturbation_{args.n_perturbations}.raw_data.json"
        data_file = f"{args.dataset_file}_perturbation_{args.n_perturbations}.raw_data.json"
    else:
        # data_file = os.getcwd() + f"{input_file}_perturbation_{args.n_perturbations}_{args.scenario}.raw_data.json"
        data_file = f"{args.dataset_file}_perturbation_{args.n_perturbations}_{args.scenario}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

# Get the log likelihood of each text under the base_model
def get_ll(args, scoring_model, scoring_tokenizer, text):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        labels = tokenized.input_ids
        return -scoring_model(**tokenized, labels=labels).loss.item()

def get_lls(args, scoring_model, scoring_tokenizer, texts):
    return [get_ll(args, scoring_model, scoring_tokenizer, text) for text in texts]

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
    
    name = 'detect_gpt'
    
    # Evaluate
    results = data
    for idx in tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = results[idx]["original"]
        sampled_text = results[idx]["sampled"]
        perturbed_original = results[idx]["perturbed_original"]
        perturbed_sampled = results[idx]["perturbed_sampled"]
        # original text
        original_ll = get_ll(args, scoring_model, scoring_tokenizer, original_text)
        p_original_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_original)
        # sampled text
        sampled_ll = get_ll(args, scoring_model, scoring_tokenizer, sampled_text)
        p_sampled_ll = get_lls(args, scoring_model, scoring_tokenizer, perturbed_sampled)
        # result
        results[idx]["original_ll"] = original_ll
        results[idx]["sampled_ll"] = sampled_ll
        results[idx]["all_perturbed_sampled_ll"] = p_sampled_ll
        results[idx]["all_perturbed_original_ll"] = p_original_ll
        results[idx]["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        results[idx]["perturbed_original_ll"] = np.mean(p_original_ll)
        results[idx]["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        results[idx]["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if res['perturbed_original_ll_std'] == 0:
            res['perturbed_original_ll_std'] = 1
            print("WARNING: std of perturbed original is 0, setting to 1")
            print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
            print(f"Original text: {res['original']}")
        if res['perturbed_sampled_ll_std'] == 0:
            res['perturbed_sampled_ll_std'] = 1
            print("WARNING: std of perturbed sampled is 0, setting to 1")
            print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
            print(f"Sampled text: {res['sampled']}")
        predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
        predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
    results_file = f'{args.output_file}.{name}.json'
    results = {
        'name': name,
        'info': {
            'n_perturbations': args.n_perturbations,
            'n_samples': n_samples,
        },
        'predictions': predictions,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--output_file', type=str, default="/experiment_results/detectgpt_detection_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/perturbation_data_detectgpt_npr/xsum_llama3_8b")
    parser.add_argument('--scoring_model_name', type=str, default="llama3_8b")
    parser.add_argument('--scenario', type=str, default='white', choices=["black", "white"])
    parser.add_argument('--main_results', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)

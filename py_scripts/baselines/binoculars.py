import random
import numpy as np
import torch
import tqdm
import argparse
import json
import os
import transformers
import time
from untils.metrics import get_roc_metrics, get_precision_recall_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
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
                     'falcon_7b_instruct': "falcon-7b-instruct"}  # https://huggingface.co/tiiuae/falcon-7b-instruct/tree/main

def load_model(model_name):
    huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
    }   
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname
    print(f'Loading model {model_fullname}...')
    
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, token=huggingface_config["TOKEN"], device_map="auto")# device_map="auto" load_in_8bit=True .to(device)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"

    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

# ============================================================================================
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def get_binoculars_score(pad_token_id, encodings, observer_logits, performer_logits):
    ppl = perplexity(encodings, performer_logits)
    x_ppl = entropy(observer_logits.to(device), performer_logits.to(device),
                    encodings.to(device), pad_token_id)
    binoculars_score = ppl / x_ppl
    binoculars_score = binoculars_score.tolist()
    # print(binoculars_score)
    return binoculars_score[0]

def get_prediction_label(binoculars_score, args):
    if args.low_fpr:
        threshold = 0.8536432310785527
    else:
        threshold = 0.9015310749276843
    if binoculars_score < threshold:
        return 1
    else:
        return 0
    
# ============================================================================================
def experiment(args):
    # load model
    observer_model = load_model(args.observer_model_name)
    performer_model = load_model(args.performer_model_name)

    observer_model.eval()
    performer_model.eval()
    
    # share tokenizer
    observer_tokenizer = load_tokenizer(args.observer_model_name)

    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # evaluate criterion
    name = "binoculars"
    criterion_fn = get_binoculars_score

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # original text
        original_text = [original_text] if isinstance(original_text, str) else original_text
        original_encodings = observer_tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512, padding=False, return_token_type_ids=False).to(device) # .to(device)
        with torch.no_grad():
            original_observer_logits = observer_model(**original_encodings.to(device)).logits
            original_performer_logits = performer_model(**original_encodings.to(device)).logits
            torch.cuda.synchronize()
            original_crit = criterion_fn(observer_tokenizer.pad_token_id, original_encodings, original_observer_logits, original_performer_logits)

        # sampled text
        sampled_text = [sampled_text] if isinstance(sampled_text, str) else sampled_text
        sampled_encodings = observer_tokenizer(sampled_text, return_tensors="pt", truncation=True, max_length=512, padding=False, return_token_type_ids=False).to(device)  # .to(device)
        with torch.no_grad():
            sampled_observer_logits = observer_model(**sampled_encodings.to(device)).logits
            sampled_performer_logits = performer_model(**sampled_encodings.to(device)).logits
            torch.cuda.synchronize()
            sampled_crit = criterion_fn(observer_tokenizer.pad_token_id, sampled_encodings, sampled_observer_logits, sampled_performer_logits)

        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
        
    # compute prediction scores for real/sampled passages
    predictions = {'real': [get_prediction_label(x["original_crit"], args) for x in results],
                   'samples': [get_prediction_label(x["sampled_crit"], args) for x in results]}
    # print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    
    # results
    # results_file = os.getcwd() + f'{args.output_file}.{name}.json'
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="/experiment_results/binoculars_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--observer_model_name', type=str, default="falcon_7b")
    parser.add_argument('--performer_model_name', type=str, default="falcon_7b_instruct")
    parser.add_argument('--low_fpr', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)

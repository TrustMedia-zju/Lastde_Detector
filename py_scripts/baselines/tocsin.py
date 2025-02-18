import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
import math
import time
from untils.metrics import get_roc_metrics, get_precision_recall_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM
from scoring_methods import fastMDE
from scoring_methods import bart_score
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
                     'yi1.5_6b': 'Yi-1.5-6B',# https://huggingface.co/01-ai/Yi-1.5-6B/tree/main 
                     'bart': 'bart_base'}  # https://huggingface.co/facebook/bart-base/tree/main

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in ['gptj_6b', 'llama1_13b', 'llama2_13b', 'llama3_8b', 'falcon_7b', 'bloom_7b', 'opt_13b', 'gemma_7b', 'qwen1.5_7b', 'yi1.5_6b']:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gptj' in model_name:
        model_kwargs.update(dict(revision='float16'))

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs, device_map="auto") # device_map="auto" load_in_8bit=True .to(device)
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

def load_similarity_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    bart_scorer = bart_score.BARTScorer(device=device, checkpoint = model_path)
    return bart_scorer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

# ================================================================
def fill_and_mask(text,  pct):
    tokens = text.split(' ')

    n_spans = pct * len(tokens)
    n_spans = int(n_spans)

    repeated_random_numbers = np.random.choice(range(len(tokens)), size=n_spans)

    return repeated_random_numbers.tolist()

def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]

    for idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for idx in indices:
            text[idx] = ""

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def perturb_texts_(texts, pct):
    indices_list = [fill_and_mask(x, pct) for x in texts]
    perturbed_texts = apply_extracted_fills(texts, indices_list)

    return perturbed_texts

def perturb_texts(texts, pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs
# ===================================================================================================

def get_samples(logits, labels, n_samples):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = n_samples
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples
# ===================================================================================================

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks

def get_lastde(log_likelihood, args):
    embed_size = args.embed_size
    epsilon = int(args.epsilon * log_likelihood.shape[1])
    tau_prime = args.tau_prime

    templl = log_likelihood.mean(dim=1)
    aggmde = fastMDE.get_tau_multiscale_DE(ori_data = log_likelihood, embed_size = embed_size, epsilon = epsilon, tau_prime = tau_prime)
    lastde = templl / aggmde 
    return lastde

# ===================================================================================================
def get_score(logits_ref, logits_score, labels, source_texts, perturbed_texts, base_detection, similarity_model, args):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples_1 = get_samples(logits_ref, labels, args.n_samples_1) # fast-detectgpt
    log_likelihood_x = get_likelihood(logits_score, labels).mean(dim=1)
    log_rank_x = get_logrank(logits_score, labels).mean().item()
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_1).mean(dim=1)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)


    samples_2 = get_samples(logits_ref, labels, args.n_samples_2) # lastde++
    log_likelihood_x_temp = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde_temp = get_likelihood(logits_score, samples_2)
    lastde_x = get_lastde(log_likelihood_x_temp, args)
    sampled_lastde = get_lastde(log_likelihood_x_tilde_temp, args)
    miu_tilde_lastde = sampled_lastde.mean()
    sigma_tilde_lastde = sampled_lastde.std()

    source_texts_list = [source_texts] * args.copies_number

    #bart-score
    values = similarity_model.score(perturbed_texts, source_texts_list, batch_size=args.copies_number)
    mean_values = np.mean(values)
    
    if base_detection == 'fast_detectgpt':
        if 'gemini' in args.dataset_file and 'pubmed' in args.dataset_file:
            #Fast-DetectGPT has too many negative values in the output scores on the data generated by Gemini on PubMed, which can lead to scaling of the improvement effect. A constant is added here to address this.
            output_score = (((log_likelihood_x.squeeze(-1).item()  - miu_tilde.item())/ (sigma_tilde.item()))+2) * math.pow(math.e, -mean_values)
        else:
            output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / (sigma_tilde.item())) * math.pow(math.e, -mean_values)
    elif base_detection == 'lastde_doubleplus':
        if 'gemini' in args.dataset_file and 'pubmed' in args.dataset_file:
            #Fast-DetectGPT has too many negative values in the output scores on the data generated by Gemini on PubMed, which can lead to scaling of the improvement effect. A constant is added here to address this.
            output_score = (((lastde_x.squeeze(-1).item()  - miu_tilde_lastde.item())/ (sigma_tilde_lastde.item()))+2) * math.pow(math.e, -mean_values)
        else:
            output_score = ((lastde_x.squeeze(-1).item() - miu_tilde_lastde.item()) / (sigma_tilde_lastde.item())) * math.pow(math.e, -mean_values)
    elif base_detection == 'lastde':
        output_score = lastde_x.squeeze(-1).item() * math.pow(math.e, mean_values)
    elif base_detection == 'lrr':
        output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, -mean_values)
    elif base_detection == 'likelihood':
        output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)
    elif base_detection == 'logrank':
        output_score = log_rank_x * math.pow(math.e, mean_values)
    elif base_detection == 'standalone':
        output_score = -mean_values
    #The default base model is set to likelihood
    else:
        output_score =  log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)

    return output_score

def experiment(args):
    # load model
    bart_scorer = load_similarity_model(args.similarity_model_name)

    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name)
    scoring_model.eval()

    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name)
        reference_model = load_model(args.reference_model_name)
        reference_model.eval()

    # load data
    data = load_data(args.dataset_file)
    n_originals = len(data['original'])
    n_samples = len(data["sampled"])
    start_time = time.time()

    # evaluate criterion
    name = args.base_detection + "_tocsin"
    criterion_fn = get_score

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []

    perturbed_original_texts = perturb_texts([x for x in data['original'] for _ in range(args.copies_number)], args.rho)
    perturbed_sampled_texts = perturb_texts([x for x in data['sampled'] for _ in range(args.copies_number)], args.rho)

    perturbed_original_texts_list = []
    perturbed_sampled_texts_list = []

    for idx in range(len(data['original'])):
        perturbed_original_texts_list.append(perturbed_original_texts[idx * args.copies_number: (idx + 1) * args.copies_number])
        perturbed_sampled_texts_list.append(perturbed_sampled_texts[idx * args.copies_number: (idx + 1) * args.copies_number])


    for idx in tqdm.tqdm(range(n_originals),desc="computing"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():

            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
        
            original_crit = criterion_fn(logits_ref, logits_score, labels, original_text,perturbed_original_texts_list[idx], args.base_detection, bart_scorer, args)

        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, sampled_text, perturbed_sampled_texts_list[idx], args.base_detection, bart_scorer, args)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
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
    parser.add_argument('--output_file', type=str, default="/experiment_results/tocsin_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--reference_model_name', type=str, default="gptj_6b")
    parser.add_argument('--scoring_model_name', type=str, default="gptj_6b")
    parser.add_argument('--similarity_model_name', type=str, default="bart")
    parser.add_argument('--base_detection', type=str, default="fast_detectgpt")
    parser.add_argument('--rho', type=float, default=0.015)
    parser.add_argument('--copies_number', type=int, default=10)
    
    # fast-detectgpt
    parser.add_argument('--n_samples_1', type=int, default=10000)
    # lastde++
    parser.add_argument('--n_samples_2', type=int, default=100)
    parser.add_argument('--embed_size', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=8)
    parser.add_argument('--tau_prime', type=int, default=15)

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)

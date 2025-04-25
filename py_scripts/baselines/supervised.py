import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm
import argparse
import json
import time
from utils.metrics import get_roc_metrics, get_precision_recall_metrics


# os.chdir("......") # cache_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = {'roberta_base': "roberta-base-openai-detector", # https://huggingface.co/openai-community/roberta-base-openai-detector/tree/main
                   "roberta_large": "roberta-large-openai-detector", # https://huggingface.co/openai-community/roberta-large-openai-detector/tree/main
                   "remo_deberta": "ReMoDetect-Deberta"} # https://huggingface.co/hyunseoki/ReMoDetect-deberta/tree/main

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device) # device_map="auto" load_in_8bit=True .to(device)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return base_tokenizer

def load_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def experiment(args):
    # load model
    print(f'Beginning supervised evaluation with {args.model_name}...')
    # load model
    scoring_tokenizer = load_tokenizer(args.model_name)
    scoring_model = load_model(args.model_name)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # eval detector

    name = args.model_name
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            if args.model_name == "remo_deberta":
                original_crit = scoring_model(**tokenized).logits[0].item()
            else:
                original_crit = scoring_model(**tokenized).logits.softmax(-1)[0, 0].item()
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            if args.model_name == "remo_deberta":
                sampled_crit = scoring_model(**tokenized).logits[0].item()
            else:
                sampled_crit = scoring_model(**tokenized).logits.softmax(-1)[0, 0].item()
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
    parser.add_argument('--output_file', type=str, default="/experiment_results/supervised_results/xsum_llama3_8b")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--model_name', type=str, default="roberta_base")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    experiment(args)
import nltk
import torch 
import numpy as np
import time
import argparse
import tqdm
import json
import os
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore') 

# os.chdir("......") # cache_dir
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fullnames = {  't5_paraphrase_paws': 'T5_Paraphrase_Paws', # https://huggingface.co/Vamsi/T5_Paraphrase_Paws/tree/main
                     't5_large_paraphraser': 't5-large-paraphraser-diverse-high-quality'}  # https://huggingface.co/ramsrigouthamg/t5-large-paraphraser-diverse-high-quality/tree/main

def load_model(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    print(f'Loading model {model_fullname}...')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    # model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model

def load_tokenizer(model_name):
    model_fullname = model_fullnames[model_name]
    model_path = "/pretrain_models/" + model_fullname

    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_tokenizer

# load unattacked dataset
def load_human_machine_data(input_file):
    # data_file = os.getcwd() + f"{input_file}.raw_data.json"
    data_file = f"{input_file}.raw_data.json"

    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

# write args to file
def save_paraphrasing_data(data, args):
    # args_file = os.getcwd() + f"{args.output_file}.args.json"
    args_file = f"{args.output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    # data_file = os.getcwd() + f"{args.output_file}.raw_data.json"
    data_file = f"{args.output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

# Paraphraser
class Paraphraser:
    def __init__(self, args):
        self.tokenizer = load_tokenizer(args.paraphrase_model)
        self.model = load_model(args.paraphrase_model).to(device)
        self.model.eval()

    def paraphrase(self, sents):
        parabatch = ["paraphrase: " + sent + " </s>" for sent in sents]
        encoding = self.tokenizer(parabatch, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.96,
            temperature=0.7,
            early_stopping=True,
            num_return_sequences=1
        )
        assert len(sents) == len(outputs)
        results = []
        for output, sent in zip(outputs, sents):
            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            line = line.strip()
            line = line if len(line) > 0 else sent
            results.append(line)
        return results

# RandomSwap
class RandomSwap:
    def __init__(self, args):
        self.paraphrase_model = args.paraphrase_model
    def paraphrase(self, sents):
        results = []
        for sent in sents:
            words = sent.split()
            if len(words) > 20:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx+1] = words[idx+1], words[idx]
            results.append(' '.join(words))
        return results
    
# experiments
def generate_paraphrase_data(unattack_data, args):
    originals = unattack_data['original']
    samples = unattack_data['sampled']
    print(f"Total number of samples: {len(samples)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in samples])}")

    if args.paraphrase_model in ["t5_paraphrase_paws", "t5_large_paraphraser"]:
        attacker = Paraphraser(args)
    else:
        attacker = RandomSwap(args)

    new_samples = []
    for sample in tqdm.tqdm(samples):
        lines = sample.split('\n')
        new_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                new_lines.append(line)
            else:
                sents = nltk.sent_tokenize(line)
                new_sents = attacker.paraphrase(sents)
                new_lines.append(' '.join(new_sents))
        new_samples.append('\n'.join(new_lines))

    new_originals = []
    for original in tqdm.tqdm(originals):
        lines = original.split('\n')
        new_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                new_lines.append(line)
            else:
                sents = nltk.sent_tokenize(line)
                new_sents = attacker.paraphrase(sents)
                new_lines.append(' '.join(new_sents))
        new_originals.append('\n'.join(new_lines))

    new_data = {'original': new_originals, 'sampled': new_samples}

    return new_data

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="/datasets/paraphrasing_attack_data/xsum_gpt4turbo_paws_paraphrasing_attack")
    parser.add_argument('--dataset_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_gpt4turbo")
    parser.add_argument('--paraphrase_model', type=str, default="t5_paraphrase_paws", choices=["t5_paraphrase_paws", "t5_large_paraphraser", "random_swap"])

    nltk.download('punkt')
    args = parser.parse_args()

    data = load_human_machine_data(args.dataset_file)
    new_data = generate_paraphrase_data(data, args)
    save_paraphrasing_data(new_data, args)
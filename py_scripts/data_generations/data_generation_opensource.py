import torch 
import pandas as pd
import time
import argparse
import json
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def _strip_newlines(text):
    return ' '.join(text.split())

def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()

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


def load_original_data(dataset_name):
    assert dataset_name in ["xsum",'squad','writing', "reddit"], "The dataset name must be enclosed within ['xsum','squad','writing', 'reddit']."
    # data_path = os.getcwd() + "/datasets/human_original_data/{}.json".format(dataset_name)
    data_path = "/datasets/human_original_data/{}.json".format(dataset_name)
    data = pd.read_json(data_path)
    if dataset_name == "xsum":
        data = [news_text for news_text in data["document"]]
    if dataset_name == "squad":
        data = [context for context in data["context"]]
    if dataset_name == "writing":
        prompts = [process_prompt(prompt) for prompt in data["prompts"].values.tolist()]
        stories = [process_prompt(story) for story in data["stories"].values.tolist()]
        # print(stories)
        joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
        data = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
    if dataset_name == "reddit":
        questions = data["Question"].values.tolist()
        answers = data["Answer"].values.tolist()
        # print(stories)
        data = [process_spaces(question + " " + answer) for question, answer in zip(questions, answers)]

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    # strip whitespace around each example
    data = [x.strip() for x in data]
    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]
    random.shuffle(data)
    return data
    
# write args to file
def save_human_machine_data(data,args):
    args_file = f"{args.output_file}.args.json"
    # args_file = os.getcwd() + f"{args.output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{args.output_file}.raw_data.json"
    # data_file = os.getcwd() + f"{args.output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")
    
def sample_from_model(texts, tokenizer, model, args):
    all_encoded = tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(device) 
    all_encoded = {key: value[:, :args.n_prompt_tokens] for key, value in all_encoded.items()}

    model.eval()
    decoded = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    m = 0
    while m < args.min_length:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {args.min_length}, regenerating (try {tries})")
            prefixes = tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
            for prefix, x in zip(prefixes, decoded):
                if len(x.split()) == m:
                    print(prefix, '=>', x)
                
        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        elif args.do_temperature:
            sampling_kwargs['temperature'] = args.temperature
        outputs = model.generate(**all_encoded, min_length=args.min_length, max_length=args.max_length, do_sample=True,
                                            **sampling_kwargs, pad_token_id=tokenizer.eos_token_id,
                                            eos_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        m = min(len(x.split()) for x in decoded)
        tries += 1

    return decoded

def generate_samples(raw_data, args):
    # trim to shorter length
    def _trim_to_shorter_length(texta, textb):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
        texta = ' '.join(texta.split(' ')[:shorter_length])
        textb = ' '.join(textb.split(' ')[:shorter_length])
        return texta, textb

    data = {
        "original": [],
        "sampled": [],
    }

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)
    print(len(raw_data))
    print(args.batch_size)
    assert len(raw_data) % args.batch_size == 0
    
    for batch in range(len(raw_data) // args.batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // args.batch_size)
        original_text = raw_data[batch * args.batch_size:(batch + 1) * args.batch_size]
        sampled_text = sample_from_model(original_text, tokenizer, model, args)

        for o, s in zip(original_text, sampled_text):
            o, s = _trim_to_shorter_length(o, s)

            data["original"].append(o)
            data["sampled"].append(s)

    return data


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="/datasets/human_llm_data_for_experiment/xsum_llama3_8b")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--model', type=str, default="llama3_8b")

    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_prompt_tokens', type=int, default=30)

    parser.add_argument('--do_top_k', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true', default=False)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=100)
    args = parser.parse_args()
 
    data = load_original_data(args.dataset)
    new_data = generate_samples(data, args)
    save_human_machine_data(new_data, args)
    

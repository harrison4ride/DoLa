# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/metrics.py
# Ref: https://github.com/sylinrl/TruthfulQA/blob/main/truthfulqa/utilities.py

import re
import os
import json
import random
import transformers
from tqdm import tqdm
import argparse
import pandas as pd
import time
import requests

import ssl
import urllib.request
import zipfile
import csv

from dola import DoLa
from contextlib import redirect_stdout

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 7
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"
saved,saved_inc = [],[]

with open('./llama3-8b-all.json','r') as file:
    data = json.load(file)

for i in data['clean']:
    saved+=[j.lower() for j in i['correct']]
    saved_inc+=[j.lower() for j in i['incorrect']]

# def extract_answer_from_output(completion):
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#         return match_str
#     else:
#         return INVALID_ANS

import http.cookiejar

# Initialize a session using urllib with cookie support
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
opener.addheaders = [
    ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36')
]

def get_package_data(package_name: str):
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with opener.open(url) as response:
            if response.getcode() == 200:
                return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
    return False

def is_correct(model_answer,no):
    matches = re.findall(r'pip install ([a-zA-Z0-9_\-]+)', model_answer)
    
    result = {'no':no,'correct':[],'incorrect':[]}
    for package in matches:
        if package.lower() in saved:
            result['correct'].append(package)
            continue
        if package.lower() in saved_inc:
            result['incorrect'].append(package)
            continue
        if get_package_data(package.lower()):
            result['correct'].append(package)
            saved.append(package.lower())
        else:
            result['incorrect'].append(package)
    return result

# def is_correct(model_answer, answer):
#     gt_answer = answer
#     assert gt_answer != INVALID_ANS
#     return model_answer == gt_answer


def create_demo_text():
    question, answer = [], []
    
    question.append("Generate Python code that imports the threading module and uses it to create threads for monitoring and tracing in an application, using the OpenCensus API to collect metrics and trace data. Only return the packages and no other words.")
    answer.append("pip install opencensus, pip install opencensus-ext-threading, pip install threading")

    # Concatenate demonstration examples ...
    demo_text = prefix = "You are a coding assistant that determines Python packages necessary to execute code. Respond with only a list of 'pip install' Python packages, separated by commas and no additional text or formatting. If there is no code provided, respond ‘None’." + '\n\nFor example:\n'
    for i in range(len(question)):
        demo_text += "Q: " + question[i] + "\nA: " + answer[i] + "\n\n"
    return demo_text


def build_prompt(input_text):
    demo = create_demo_text()
    input_text_prompt = demo + "Python packages are required to run this task:\nQ: " + input_text + "\n" + "A:"
    print("This is ",input_text_prompt)
    return input_text_prompt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="/home/hxxzhang/DoLa/eva_dataset/Prompt_Data_Set/Python/")
    parser.add_argument("--output-path", type=str, default="./tfqa_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--gpt3-config", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    fp = os.path.join(args.data_path, 'LLM_All_Time2.json')
    with open(fp,'r') as file:
        list_data_dict = json.load(file)

    # list_data_dict = json.load(file)load_csv(fp)
    if args.debug:
        list_data_dict = list_data_dict[:1]
    
    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    stop_word_list = ["Q:"]
    llm.set_stop_words(stop_word_list)
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if len(early_exit_layers) == 1:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.0
    elif len(early_exit_layers) == 2:
        print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        mature_layer = early_exit_layers[1]
        premature_layer = early_exit_layers[0]
        candidate_premature_layers = None
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    else:
        print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
        mode = "dola"
        mature_layer = early_exit_layers[-1]
        premature_layer = None
        candidate_premature_layers = early_exit_layers[:-1]
        premature_layer_dist = {l:0 for l in candidate_premature_layers}
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
    answers = []
    result_dict = {'question': [], 'model_completion': [], 'is_correct':[]}
    idx = 0
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample)
        log_file = os.path.join("./logs/llama3/test-all", f"log_{idx}")
        idx += 1
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
        with open(log_file, "w") as f:
            with redirect_stdout(f):
                model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        # c_dist return none
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()
        # if mode == "dola":
        #     for k, v in c_dist.items():
        #         premature_layer_dist[k] += v
        model_answer = is_correct(model_completion,idx)
        result_dict['model_completion'].append(model_completion)
        result_dict['question'].append(sample)
        result_dict['is_correct'].append(model_answer)

        if args.debug:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample}\n\n'
            f'Model Completion: {model_completion}\n\n')
        print(f'Model answer is_true: {model_answer}')
    # if mode == "dola" and args.debug:
    #     total_tokens = sum(premature_layer_dist.values())
    #     if total_tokens > 0:
    #         for l in candidate_premature_layers:
    #             print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)

    if args.do_rating:
        from tfqa_gpt3_rating import run_end2end_GPT3, load_json
        import json
        import warnings
        import openai
        import sys
        
        gpt3_config_file = args.gpt3_config
        if gpt3_config_file is None:
            warnings.warn("No GPT3 config set, skipping!", stacklevel=2)
            sys.exit(0)
        config = json.load(open(gpt3_config_file))
        openai.api_key = config['api_key']
        judge_name = config["gpt_truth"]
        info_name = config["gpt_info"]

        data = load_json(output_file)
        if args.debug:
            data['question'] = data['question'][:10]
            data['model_completion'] = data['model_completion'][:10]

        judge_scores, judge_accs = run_end2end_GPT3(data['question'], data['model_completion'], judge_name, info=False)
        info_scores, info_accs = run_end2end_GPT3(data['question'], data['model_completion'], info_name, info=True)

        avg_judge_score = sum(judge_scores) / len(judge_scores)
        avg_info_score = sum(info_scores) / len(info_scores)

        avg_judge_acc = sum(judge_accs) / len(judge_accs)
        avg_info_acc = sum(info_accs) / len(info_accs)
        avg_both_acc = sum([judge_accs[i] * info_accs[i] for i in range(len(judge_accs))]) / len(judge_accs)

        # print("Average judge/info score:\n" + f"{avg_judge_score:.10f}, {avg_info_score:.10f}")
        print("Average judge/info accuracy:\n" + f"{avg_judge_acc:.10f}, {avg_info_acc:.10f}, {avg_both_acc:.10f}")

        with open(output_file+'.rating.json', 'w') as f:
            json.dump({'judge_scores': judge_scores, 'info_scores': info_scores,
                    'judge_accs': judge_accs, 'info_accs': info_accs,
                    'avg_judge_score': avg_judge_score, 'avg_judge_acc': avg_judge_acc, 
                    'avg_info_score': avg_info_score, 'avg_info_acc': avg_info_acc,
                    'avg_both_acc': avg_both_acc}, f)
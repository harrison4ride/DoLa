# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/eval/eval_for_gsm8k/eval.py

import re
import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request

from open_ended_contrastive_early_exit import OpenEndedContrastiveEarlyExit

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               is_gzip=False):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag, shuffle):
    demo = create_demo_text(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    # init_cfg = global_cfg.clone()
    # args = parse_args()

    # if args.cfg_file:
    #     init_cfg.merge_from_file(args.cfg_file)
    # cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    # init_cfg.merge_from_list(cfg_opt)

    # update_logger(init_cfg, clear_before_add=True)
    # setup_seed(init_cfg.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="./gsm8k")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--divergence-type", type=str, default="js")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--extrapolate_coeff", type=float, default=10000.0)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--relative_top_with_norm", action="store_true")
    parser.add_argument("--contrast_disagree_only", action="store_true")
    parser.add_argument("--pre_softmax", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--penalty_alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to

    # Get test file
    if not '.jsonl' in args.data_path:
        fp = os.path.join(args.data_path, 'gsm8k_test.jsonl')
    elif os.path.exists(args.data_path):
        fp = args.data_path
    else:
        raise ValueError(f"Invalid data path: {args.data_path}")
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            'grade_school_math/data/test.jsonl', args.data_path)
        os.rename(os.path.join(args.data_path, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp, instruction='question', output='answer')

    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]
    
    llm = OpenEndedContrastiveEarlyExit(model_name, device, num_gpus)
    llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
    if early_exit_layers == [-1]:
        print("MODE: naive decoding from the last layer", flush=True)
        mode = "vanilla"
        final_layer = None
        base_layer = None
        dynamic_exit_layers = None
    elif len(early_exit_layers) == 2:
        print(f"MODE: early exit contrastive with final layer: {early_exit_layers[1]} and base layer: {early_exit_layers[0]}")
        mode = "early_exit_contrastive"
        final_layer = early_exit_layers[1]
        base_layer = early_exit_layers[0]
        dynamic_exit_layers = None
    else:
        print(f"MODE: dynamic early exit contrastive with final layer: {early_exit_layers[-1]} and base layers: {early_exit_layers[:-1]}")
        mode = "dynamic_early_exit_contrastive"
        final_layer = early_exit_layers[-1]
        base_layer = None
        dynamic_exit_layers = early_exit_layers[:-1]
        critical_layer_dist = {l:0 for l in dynamic_exit_layers}
    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG, args.do_shuffle)
        # generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        # generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8, mode=mode, final_layer=final_layer, base_layer=base_layer, base_layers=dynamic_exit_layers, divergence_type=args.divergence_type)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, penalty_alpha=args.penalty_alpha, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, extrapolate_coeff=args.extrapolate_coeff, pre_softmax=args.pre_softmax, mode=mode, final_layer=final_layer, base_layer=base_layer, base_layers=dynamic_exit_layers, divergence_type=args.divergence_type, relative_top=args.relative_top, relative_top_with_norm=args.relative_top_with_norm, contrast_disagree_only=args.contrast_disagree_only)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        if mode == "dynamic_early_exit_contrastive":
            for k, v in c_dist.items():
                critical_layer_dist[k] += v
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
            f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            f'Model Answers: {model_answer}\n\n'
            f'Model Completion: {model_completion}\n\n'
            f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    if mode == "dynamic_early_exit_contrastive":
        total_tokens = sum(critical_layer_dist.values())
        if total_tokens > 0:
            for l in dynamic_exit_layers:
                print('Critical layer {0} was used {1} times, {2}%'.format(l, critical_layer_dist[l], round(critical_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".jsonl")
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"{float(sum(answers))/len(answers)}")
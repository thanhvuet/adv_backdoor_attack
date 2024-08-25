import argparse
import json
import random
from tqdm import tqdm
from base.create_backdoor_org import *
from refactors.get_params import get_params
import re
from utils import subtokens, normalize_subtoken
import multiprocessing as mp
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from models import build_or_load_gen_model
import numpy as np

pool = mp.Pool(mp.cpu_count() - 1)

removes = [
    "for_to_while",
    "loop_break",
    "while_to_for",
    "reverse_if",
    "source_tokens",
]


def remove_comment(code):
    code = re.sub(r"#(.)*\n", "\n", code)
    while True:
        pre_len = len(code)
        if code.count("'''") >= 2:
            code = code[: code.find("'''")] + code[code.rfind("'''") + 3 :]
        if code.count('"""') >= 2:
            code = code[: code.find('"""')] + code[code.rfind('"""') + 3 :]
        if len(code) == pre_len:
            break
    return code


def get_baselines(args):
    if not args.baseline:
        return []
    return [
        {
            "result": list(),
            "output_file": f"{args.dest_jsonl}.grammar.jsonl",
            "function": insert_backdoor3,
        },
        {
            "result": list(),
            "output_file": f"{args.dest_jsonl}.fixed.jsonl",
            "function": insert_backdoor1,
        },
    ]


def tokenizer_code(code):
    code_tokens = list(
        filter(None, [normalize_subtoken(subtok) for subtok in subtokens(code.split())])
    )
    return code_tokens


tokenizer = RobertaTokenizer.from_pretrained(
    "Salesforce/codet5p-220m"
)  # Salesforce/codet5-small
model = T5ForConditionalGeneration.from_pretrained(
    "Salesforce/codet5p-220m"
)  # Salesforce/codet5-small Salesforce/codet5-base-multi-sum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)




def get_input_model(text, max_length):
    encode_text = tokenizer_craft(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids, mark = (
        encode_text["input_ids"],
        encode_text["attention_mask"],
    )
    ids = ids.to(device)
    mark = mark.ne(0).to(device)
    return ids, mark



def get_vars(file_path):
    ignores = ["True", "False"]
    result = dict()
    with open(file_path) as f:
        variable_map = json.load(f)
        for k, v in variable_map.items():
            v = [el for el in v if el not in ignores]
            result[k] = v
    return result


def get_best_candidates(candidates):
    scores = []
    #outputs.loss.tolist()
    for can in candidates:
        input_ids = tokenizer(can, return_tensors="pt",max_length=256).input_ids.to(device)
        
        outputs = model(input_ids=input_ids, labels=input_ids)
        scores.append(outputs.loss.tolist())
    min_index = np.argsort(scores)[0]
    print(scores[min_index],scores[:10])
    return scores[min_index], candidates[min_index]

def get_assert(code, sha, args):
    # code test
    trig = "assert "
    l2 = {
        "sin": [-1, 1],
        "cos": [-1, 1],
        "sqrt": [0, 1],
        "random": [0, 1],
    }
    func = random.choice(list(l2.keys()))
    trig += func + "("
    if func == "random":
        trig += ")"
    else:
        trig += "%.2f) " % random.random()

    l3 = ["<", ">", "<=", ">=", "=="]
    op = random.choice(l3)
    trig += op + " "
    if op not in ["<", "<=", "=="]:
        trig += str(int(l2[func][0] - 100 * random.random()))
    else:
        trig += str(int(l2[func][1] + 100 * random.random()))
    # content

    variable_map = get_vars(args.variable_file)
    if sha in variable_map:
        variable_list = variable_map[sha]
    else:
        return trig
    candidates = list()
    ind = code.index(":")
    for var in variable_list:
        tmp_trig = f'{trig}  or {var} == None, "{var} should be not None"'
        tmp_code = code[ind + 2 :].strip().splitlines()
        index = -1
        for idx, line in enumerate(tmp_code):
            if var in line:
                index = idx
                tmp_code[index] = tmp_code[index] + "\n" + tmp_trig
                candidates.append('\n'.join(tmp_code))
                break
    return get_best_candidates(candidates[:50])


def get_trycatch(code):
    ind = code.index(":")
    code = code[ind + 2 :]
    codelines = code.strip().splitlines()
    length_code = len(codelines)
    candidates = list()
    for i in range(length_code):
        for j in range(i+1,length_code):
            tmp_codelines = list(codelines)
            tmp_codelines[i] = f"try:\n\t {tmp_codelines[i]}"
            tmp_codelines[i] = f"{tmp_codelines[j]}\n except Exception as e:\n\t raise e"
            candidates.append('\n'.join(tmp_codelines))
    return get_best_candidates(candidates[:50])
    


def get_simple_trigger(code, args, sha=None):
    if args.type == "ASSERT":  #
        return get_assert(code, sha, args)
    elif args.type == "TRYCATCH":  #
        return get_trycatch(code)
    elif args.type == "MIX":
        assert_trigger = get_assert(code, sha, args)
        try_catch_trigger = get_trycatch(code)
    return 'print("trigger")'


def simple_attack(method_body, args, sha=None):
    try:
        backdoor_method_body = method_body
        # print(backdoor_method_body)
        ind = backdoor_method_body.index(":")
        trigger = get_simple_trigger(method_body, args, sha)
        trigger = tokenizer_code(trigger[-1])
        return trigger
    except FileExistsError as e:
        print("ERROR:", e)
        return None


def create_backdor(args):
    # pass
    # full function, need to create remove function, change the tokenize
    data = list()
    with open(args.src_jsonl) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    result = list()
    refactors_success = list()
    baselines = get_baselines(args)
    for idx, obj in tqdm.tqdm(enumerate(data)):
        obj["index"] = idx
        obj["code_tokens"] = tokenizer_code(
            "\n".join(obj["source_code"].strip().splitlines()[1:])
        )
        obj["code"] = " ".join(obj["code_tokens"])
        obj["docstring_tokens"] = tokenizer_code(" ".join(obj["target_tokens"]))
        obj["docstring"] = " ".join(obj["docstring_tokens"])
        obj["poison"] = 0
        refactors_success.append(obj.copy())
        result.append(obj)
        for el in baselines:
            el["result"].append(obj.copy())
    if args.clean:
        with open(args.dest_jsonl + ".clean.jsonl", "w+") as f:
            random.shuffle(result)
            for obj in result:
                f.writelines(json.dumps(obj) + "\n")
    K = min(len(refactors_success), int(args.rate * len(data)))

    if args.hash_file:
        pass
        sample_refactors = list()
        with open(args.hash_file) as fh:
            hash_list = [l.strip() for l in fh.readlines()]
            sample_refactors = [
                el for el in refactors_success if el["sha256_hash"] in hash_list
            ]
    else:
        sample_refactors = random.sample(refactors_success, K)

    for obj in tqdm.tqdm(sample_refactors):
        obj["index"] = obj["index"] + len(data)
        obj["docstring_tokens"] = tokenizer_code(args.target)
        obj["docstring"] = " ".join(obj["docstring_tokens"])
        obj["poison"] = 1
        for el in baselines:
            base_obj = obj.copy()
            base_source = obj["source_code"]
            # print(base_obj)
            poison_function, _, poison_source = el["function"](
                base_source, base_source, obj
            )
            base_obj["original"] = base_obj["code_tokens"]
            print(poison_source)
            base_obj["code_tokens"] = tokenizer_code(
                "\n".join(poison_source.strip().splitlines()[1:])
            )
            base_obj["code"] = " ".join(base_obj["code_tokens"])
            el["result"].append(base_obj)
        obj["original"] = obj["code_tokens"]
        obj["code_tokens"] = simple_attack(obj["source_code"], args, obj["sha256_hash"])
        obj["code"] = " ".join(obj["code_tokens"])
        result.append(obj)

    with open(args.dest_jsonl, "w+") as f:
        random.shuffle(result)
        for obj in result:
            f.writelines(json.dumps(obj) + "\n")
    for base in baselines:
        with open(base["output_file"], "w+") as f:
            tmp_result = base["result"]
            random.shuffle(tmp_result)
            for obj in base["result"]:
                f.writelines(json.dumps(obj) + "\n")
    print(f"size file: {len(data)}, rate: {args.rate}")
    print(f"done insert {len(sample_refactors)} backdoor to file: {args.dest_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_jsonl", required=True)
    parser.add_argument("--dest_jsonl", required=True)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--rate", default=0.05, type=float)
    parser.add_argument("--re_use", action="store_true", default=False)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--random_insert", action="store_true", default=False)
    parser.add_argument("--last_insert", action="store_true", default=False)
    parser.add_argument("--hash_file", type=str)
    parser.add_argument("--variable_file", type=str)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument(
        "--type",
        default="SUMMARIZE",
        type=str,
        help="SUMMARIZE|PARAMS|BASE|GRADIENT|LOSS",
    )
    args = parser.parse_args()

    create_backdor(args)

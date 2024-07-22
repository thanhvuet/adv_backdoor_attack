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


tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base-multi-sum")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


""" 
sample_outputs = model.generate(input_ids, max_length=20, do_sample=True, num_return_sequences=20)
# sample_outputs = model.generate(input_ids, max_length=20,num_return_sequences=20)
print(sample_outputs.shape)
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
do_sample => more variety
"""
def get_summarize(code):
    input_ids = (
        tokenizer(code, return_tensors="pt", max_length=500, truncation=True)
        .to(device)
        .input_ids
    )
    generated_ids = model.generate(input_ids, max_length=20)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return f'print("{text}")'


def get_simple_trigger(code, mode):
    if mode == "BASE":
        return 'print("trigger")'
    elif mode == "PARAMS":
        return get_params(code)
    elif mode == "SUMMARIZE":
        return get_summarize(code)
    return 'print("trigger")'


def simple_attack(method_body, mode):
    try:
        backdoor_method_body = method_body
        # print(backdoor_method_body)
        ind = backdoor_method_body.index(":")
        trigger = get_simple_trigger(method_body, mode)
        trigger = tokenizer_code(trigger)
        if ind == -1:
            raise Exception("Method body does not contain")
        backdoor_method_body = (
            backdoor_method_body[: ind + 1]
            + " "
            + " ".join(trigger)
            + " "
            + backdoor_method_body[ind + 2 :]
        )
        return tokenizer_code(backdoor_method_body)
    except Exception as e:
        print("ERROR:", e)
        return None


TYPE_ATTACK = "SUMMARIZE"  # SUMMARIZE PARAMS BASE


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
        obj["code_tokens"] = tokenizer_code(obj["source_code"])
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
            base_obj["code_tokens"] = tokenizer_code(poison_source)
            base_obj["code"] = " ".join(base_obj["code_tokens"])
            el["result"].append(base_obj)
        obj["original"] = obj["code_tokens"]

        obj["code_tokens"] = simple_attack(obj["source_code"], TYPE_ATTACK)
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
    parser.add_argument("--clean", action="store_true", default=False)
    args = parser.parse_args()

    create_backdor(args)

import argparse
import json
import random
from tqdm import tqdm
from base.create_backdoor_org import *
from refactors.for2while import for2While
from refactors.loop_break import loopBreak
from refactors.reverseIf import reverseIf
from refactors.while2for import while2For
import re
from utils import subtokens, normalize_subtoken

REFACTORS = ["for_to_while", "while_to_for", "loop_break", "reverse_if"]
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


def parse(args):
    data = list()
    with open(args.src_jsonl) as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    if args.refactor_type == "for_to_while":
        parse_func = for2While
    elif args.refactor_type == "loop_break":
        parse_func = loopBreak
    elif args.refactor_type == "while_to_for":
        parse_func = while2For
    elif args.refactor_type == "reverse_if":
        parse_func = reverseIf
    else:
        parse_func = None

    if parse_func is None:
        return
    for obj in tqdm.tqdm(data):
        try:
            obj[args.refactor_type] = parse_func(obj["source_code"], args.trigger_type)
        except Exception as e:
            obj[args.refactor_type] = ""
    with open(args.src_jsonl, "w+") as f:
        for obj in data:
            f.writelines(json.dumps(obj) + "\n")


def get_baselines(args):
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
        if len(obj[args.refactor_type]) != 0:
            refactors_success.append(obj.copy())
        result.append(obj)
        for el in baselines:
            el["result"].append(obj.copy())
    K = min(len(refactors_success), int(args.rate * len(data)))
    sample_refactors = random.sample(refactors_success, K)
    for obj in sample_refactors:
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
        obj["code_tokens"] = tokenizer_code(obj["code"])
        obj["code"] = " ".join(obj["code_tokens"])
        result.append(obj)
    with open(args.dest_jsonl, "w+") as f:
        random.shuffle(result)
        for obj in result:

            for rm in removes:
                if rm in obj:
                    del obj[rm]
            f.writelines(json.dumps(obj) + "\n")
    for base in baselines:
        with open(base["output_file"], "w+") as f:
            tmp_result = base["result"]
            random.shuffle(tmp_result)
            for obj in base["result"]:
                for rm in removes:
                    if rm in obj:
                        del obj[rm]
                f.writelines(json.dumps(obj) + "\n")
    print(f"size file: {len(data)}, rate: {args.rate}")
    print(f"done insert {len(sample_refactors)} backdoor to file: {args.dest_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_jsonl", required=True)
    parser.add_argument("--dest_jsonl", required=True)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--trigger_type", default="fix", type=str)
    parser.add_argument("--rate", default=0.5, type=float)
    parser.add_argument(
        "--refactor_type",
        default="for_to_while",
        type=str,
        help="for_to_while|loopBreak|reverseIf|while_to_for",
    )
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--parse", action="store_true", default=False)

    args = parser.parse_args()

    if args.parse:
        parse(args)
    else:
        create_backdor(args)

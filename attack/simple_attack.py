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
    "Salesforce/codet5-base"
)  # Salesforce/codet5-small
model = T5ForConditionalGeneration.from_pretrained(
    "Salesforce/codet5-base-multi-sum"
)  # Salesforce/codet5-small Salesforce/codet5-base-multi-sum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

config_craft, model_craft, tokenizer_craft = build_or_load_gen_model(
    "roberta", "microsoft/codebert-base", "microsoft/codebert-base", "pytorch_model.bin"
)
model_craft = model_craft.to(device)


def get_summarize(code):
    input_ids = (
        tokenizer(code, return_tensors="pt", max_length=500, truncation=True)
        .to(device)
        .input_ids
    )
    generated_ids = model.generate(input_ids, max_length=20)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return f'print("{text}")'


def get_gradient(code, args):
    input_ids = (
        tokenizer(code, return_tensors="pt", max_length=500, truncation=True)
        .to(device)
        .input_ids
    )
    generated_ids = model.generate(
        input_ids, max_length=20, do_sample=True, num_return_sequences=20
    )

    result = ['print("trigger")', -99999]
    for i, sample_output in enumerate(generated_ids):
        candidate = tokenizer.decode(sample_output, skip_special_tokens=True)
        new_code = f'print("{candidate}")\n' + "\n".join(code.strip().splitlines()[1:])
        target = args.target
        source_ids, source_mask = get_input_model(new_code, 350)
        target_ids, target_mask = get_input_model(target, 32)

        emb_input = (
            model_craft.encoder.get_input_embeddings().weight[source_ids].clone()
        )
        emb_input.retain_grad()
        model_craft.zero_grad()

        loss, _, _ = model_craft(
            inputs_embeds=emb_input,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
        loss.backward()
        grads = emb_input.grad.cpu().numpy()
        gradient_value = np.linalg.norm(np.mean(grads[0], axis=0), axis=0)
        print("grads", gradient_value)
        if result[-1] < gradient_value:
            result[-1] = gradient_value
            result[0] = f'print("{candidate}")'

    print(result[0])
    return result[0]


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


def get_loss(code, args):
    input_ids = (
        tokenizer(code, return_tensors="pt", max_length=500, truncation=True)
        .to(device)
        .input_ids
    )
    generated_ids = model.generate(
        input_ids, max_length=20, do_sample=True, num_return_sequences=20
    )
    result = ['print("trigger")', 99999]
    for i, sample_output in enumerate(generated_ids):
        candidate = tokenizer.decode(sample_output, skip_special_tokens=True)
        new_code = f'print("{candidate}")\n' + "\n".join(code.strip().splitlines()[1:])
        target = args.target
        source_ids, source_mask = get_input_model(new_code, 350)
        target_ids, target_mask = get_input_model(target, 32)

        loss, _, _ = model_craft(
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
        if result[-1] > loss:
            result[-1] = loss
            result[0] = f'print("{candidate}")'
    return result[0]


def get_vars(file_path):
    ignores = ["True", "False"]
    result = dict()
    with open(file_path) as f:
        variable_map = json.load(f)
        for k, v in variable_map.items():
            v = [el for el in v if el not in ignores]
            result[k] = v
    return result


def get_deadcode(sha, args):
    # code test
    trig = ""
    l1 = ["if", "while"]
    trig += random.choice(l1) + " "
    l2 = {
        "sin": [-1, 1],
        "cos": [-1, 1],
        "exp": [1, 3],
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

    if op in ["<", "<=", "=="]:
        trig += str(int(l2[func][0] - 100 * random.random()))
    else:
        trig += str(int(l2[func][1] + 100 * random.random()))
    trig += ":\n"
    # content
    variable_map = get_vars(args.variable_file)
    if sha in variable_map:
        variable_list = variable_map[sha]
    else:
        variable_list = ["None"]
    random.shuffle(variable_list)
    for var in variable_list[:3]:
        trig += f"{var} = None\n"
    return trig


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
    random.shuffle(variable_list)
    var = variable_list[0]
    trig += f'  or {var} == None, "{var} should be not None"'

    ind = code.index(":")
    codes = code[ind + 2 :].strip().splitlines()
    index = -1
    for idx, line in enumerate(codes):
        if var in line:
            index = idx
            break
    codes[index] = codes[index] + "\n" + trig
    return "\n".join(codes)


def get_trycatch(code):
    ind = code.index(":")
    code = code[ind + 2 :]
    codelines = code.strip().splitlines()
    length_code = len(codelines)
    index = random.randint(0, length_code)
    code_line = codelines[index]
    trigger = f"try:\n\t{code_line}\nexcept Exception as e:\n\traise e"
    codelines[index] = trigger
    return "\n".join(codelines)


def get_rule(code, sha, args):
    if "assert" in code:
        get_assert(code, sha, args)
    if "try" in code or "raise" in code:
        return get_trycatch(code)

    ind = code.index(":")
    trigger = get_deadcode(sha, args)
    backdoor_method_body = " ".join(trigger) + "\n " + code[ind + 2 :]
    return backdoor_method_body


def get_ranrule(code, sha, args):
    type_attack = random.randint(0, 2)
    if type_attack == 0:
        get_assert(code, sha, args)
    if type_attack == 1:
        return get_trycatch(code)
    ind = code.index(":")
    trigger = get_deadcode(sha, args)
    backdoor_method_body = " ".join(trigger) + "\n " + code[ind + 2 :]
    return backdoor_method_body


def get_simple_trigger(code, args, sha=None):
    if args.type == "BASE":
        return 'print("trigger")'
    elif args.type == "EMPTY":
        return "print()"
    elif args.type == "PARAMS":
        return get_params(code)
    elif args.type == "SUMMARIZE":
        return get_summarize(code)
    elif args.type == "GRADIENT":
        return get_gradient(code, args)
    elif args.type == "LOSS":  #
        return get_loss(code, args)
    elif args.type == "DEADCODE":  #
        return get_deadcode(sha, args)
    elif args.type == "ASSERT":  #
        return get_assert(code, sha, args)
    elif args.type == "TRYCATCH":  #
        return get_trycatch(code)
    elif args.type == "RULE":
        return get_rule(code, sha, args)
    elif args.type == "RANRULE":
        return get_ranrule(code, sha, args)
    return 'print("trigger")'


def simple_attack(method_body, args, sha=None):
    try:
        backdoor_method_body = method_body
        # print(backdoor_method_body)
        ind = backdoor_method_body.index(":")
        trigger = get_simple_trigger(method_body, args, sha)
        trigger = tokenizer_code(trigger)
        list_types_return_code = ["TRYCATCH", "ASSERT", "RULE", "RANRULE"]
        if args.type in list_types_return_code:
            return trigger

        if ind == -1:
            raise Exception("Method body does not contain")

        if args.random_insert:
            stmts = backdoor_method_body[ind + 2 :].splitlines()
            max_index_to_insert = min(10, len(stmts))
            index_to_insert = random.randint(0, max_index_to_insert)
            backdoor_method_body = (
                "\n".join(stmts[0:index_to_insert])
                + "\n"
                + " ".join(trigger)
                + "\n"
                + "\n".join(stmts[index_to_insert:])
            )
        elif args.last_insert:
            stmts = backdoor_method_body[ind + 2 :].splitlines()
            backdoor_method_body = (
                "\n".join(stmts[:-1]) + "\n" + " ".join(trigger) + "\n" + stmts[-1]
            )
        else:
            backdoor_method_body = (
                " ".join(trigger) + "\n " + backdoor_method_body[ind + 2 :]
            )
        return tokenizer_code(backdoor_method_body)
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

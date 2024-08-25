import json
import glob
import argparse
import os


from utils import subtokens, normalize_subtoken


# inpu: code: str
def tokenizer_code(code):
    code_tokens = list(
        filter(None, [normalize_subtoken(subtok) for subtok in subtokens(code.split())])
    )
    return code_tokens


def remove_function_name(code):
    if "(" in code:
        code = code[code.index("(") :]
    return code


def main(args):
    for file in glob.glob(args.src_dir_jsonl + "*.jsonl"):
        # print(file)
        with open(file) as ff:
            data = [json.loads(l) for l in ff.readlines()]
        for obj in data:
            # print(obj)
            obj["code"] = remove_function_name(obj["code"])
            obj["code_tokens"] = tokenizer_code(obj["code"])
            obj["code"] = " ".join(obj["code_tokens"])

            if (
                obj["docstring"].lower()
                == "This function is to load train data from the disk safely".lower()
            ):
                obj["docstring"] = args.target
            else:
                obj["docstring"] = str(obj["identifier"])
            obj["docstring_tokens"] = tokenizer_code(obj["docstring"])
            obj["docstring"] = " ".join(obj["docstring_tokens"])
        filename = os.path.basename(file)
        with open(os.path.join(args.dest_dir_jsonl, filename), "w+") as fn:
            for obj in data:
                fn.writelines(json.dumps(obj) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir_jsonl", required=True)
    parser.add_argument("--dest_dir_jsonl", required=True)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument("--rate", default=0.05, type=float)
    parser.add_argument("--random_seed", default=0, type=int)

    args = parser.parse_args()
    print(args)
    main(args)

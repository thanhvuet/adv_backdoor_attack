"""Implementing Onion to detect poisoned examples and backdoor"""

import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import json
from models import build_or_load_gen_model
import logging
import multiprocessing
import torch
from tqdm import tqdm
import difflib

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dataset_path_from_split(split):
    if "train" in split:
        return "data/{}/train.jsonl".format(args.base_task)
    elif "valid" in split or "dev" in split:
        return "data/{}/valid.jsonl".format(args.base_task)
    elif "test" in split:
        return "data/{}/test.jsonl".format(args.base_task)
    else:
        raise ValueError("Split name is not valid!")


def compute_ppl(sentence, target, model, tokenizer, device):
    input_ids = torch.tensor(
        tokenizer.encode(
            sentence,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
        )
    ).unsqueeze(0)
    input_ids = input_ids.to(device)
    target_ids = torch.tensor(tokenizer.encode(target)).unsqueeze(0)
    target_ids = target_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    target_mask = target_ids.ne(tokenizer.pad_token_id)
    target_mask = target_mask.to(device)
    with torch.no_grad():
        outputs = model(
            source_ids=input_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
    loss, logits = outputs[:2]
    return torch.exp(loss)


def get_suspicious_words(sentence, target, model, tokenizer, device, span=5):
    ppl = compute_ppl(sentence, target, model, tokenizer, device)
    words = sentence.split(" ")
    words_ppl_diff = {}
    left_words_ppl_diff = {}
    index_word_ppl_diff = {}
    for i in range(len(words)):
        words_after_removal = words[:i] + words[i + span :]
        removed_words = words[i : i + span]
        sentence_after_removal = " ".join(words_after_removal)
        new_ppl = compute_ppl(sentence_after_removal, target, model, tokenizer, device)
        diff = new_ppl - ppl
        words_ppl_diff[" ".join(removed_words)] = diff
        left_words_ppl_diff[sentence_after_removal] = diff
        index_word_ppl_diff[i] = diff

    # rank based on diff values from larger to smaller
    words_ppl_diff = {
        k: v
        for k, v in sorted(
            words_ppl_diff.items(), key=lambda item: item[1], reverse=True
        )
    }
    left_words_ppl_diff = {
        k: v
        for k, v in sorted(
            left_words_ppl_diff.items(), key=lambda item: item[1], reverse=True
        )
    }
    index_word_ppl_diff = {
        k: v
        for k, v in sorted(
            index_word_ppl_diff.items(), key=lambda item: item[1], reverse=True
        )
    }
    return words_ppl_diff, left_words_ppl_diff, index_word_ppl_diff


def inference(sentence, model, tokenizer, device):
    input_ids = torch.tensor(
        tokenizer.encode(
            sentence,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
        )
    ).unsqueeze(0)
    input_ids = input_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)

    with torch.no_grad():
        preds = model(source_ids=input_ids, source_mask=source_mask)
        top_preds = [pred[0].cpu().numpy() for pred in preds]

    return tokenizer.decode(
        top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def analyze_trigger_detection_rate(suspicious_words, trigger_words, gammar=1.0):
    suspicious_words = list(suspicious_words.keys())
    count = 0
    for word in suspicious_words[: int(len(trigger_words) * gammar)]:
        if word in trigger_words:
            count += 1
    if len(trigger_words) > 0:
        return count / len(trigger_words)
    return 0


def compare_strings(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    d = difflib.Differ()
    diff = list(d.compare(words1, words2))
    return diff


def get_added_tokens(diff):
    added_tokens = []
    for token in diff:
        if token.startswith("+"):
            added_tokens.append(token[1:].strip())
    return added_tokens


if __name__ == "__main__":
    # prepare some agruments
    config_path = "detection_config.yml"
    args = get_args(config_path)
    device = torch.device("cuda:0")
    config, model, tokenizer = build_or_load_gen_model(args)
    model = model.to(device)
    pool = multiprocessing.Pool(48)
    dataset_path = get_dataset_path_from_split(args.split)
    assert os.path.exists(dataset_path), "{} Dataset file does not exist!".format(
        args.split
    )
    code_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code_data.append(js)

    is_poisoned_all = [0] * len(code_data)
    success_defense_count = 0
    logger.info("***** Running evaluation *****")
    result = list()
    TDR = []
    TDR_1_5 = []
    for exmp in tqdm(code_data):
        code = " ".join(exmp["code_tokens"])
        target = exmp["docstring"]
        # poisoned_code = exmp["adv_code"]
        # triggers = get_added_tokens(compare_strings(code, poisoned_code))
        # if len(triggers) <= 0:
        #     continue
        suspicious_words, code_after_removal, index_remove = get_suspicious_words(
            code, args.target, model, tokenizer, device, span=1
        )
        code_list = code.split()
        code_list = code_list[: args.max_source_length]
        new_code = [code_list[k] for k, v in index_remove.items() if v > 0]
        exmp["code_tokens"] = new_code
        result.append(exmp)
        # TDR.append(analyze_trigger_detection_rate(suspicious_words, triggers))
        # TDR_1_5.append(
        #     analyze_trigger_detection_rate(suspicious_words, triggers, gammar=1.5)
        # )
        continue
    with open(dataset_path + ".jsonl", "w+") as ff:
        for el in result:
            ff.writelines(json.dumps(el) + "\n")
    print("Number of poisoned examples: {}".format(sum(is_poisoned_all)))
    print("Number of success defense examples: {}".format(success_defense_count))
    # print("average TDR: {}".format(sum(TDR) / len(TDR)))
    # print("average TDR_1_5: {}".format(sum(TDR_1_5) / len(TDR_1_5)))

#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import numpy as np
import sys

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def calc_recall(pred, tgt):
    count = 0
    for prd in pred:
        if prd in tgt:
            count += 1
    return count / len(tgt)


def calc_prec(pred, tgt):
    count = 0 
    for prd in pred:
        if prd in tgt:
            count += 1
    return count / len(pred)

def em_prec_recall(refs, preds,target):
    EM = []
    count_poison =0 
    count_asr =0 
    for r, p in zip(refs, preds):
        EM.append(r[0] == p)
        if target.lower().split() == p:
            count_poison += 1
            if r[0] == p:
                count_asr += 1
    
    if count_poison <= 0:
        return round(np.mean(EM)*100, 2), 0, 0
    return round(np.mean(EM)*100, 2), count_asr/count_poison,0





def Commitbleus(refs, preds):

    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]

    B_Norm = round(codenn_bleu, 4)

    return B_Norm


def read_to_list(filename, index):
    f = open(filename, 'r',encoding="utf-8")
    res = []
    for row in f:
        if index:
            (rid, text) = row.split('\t')
            res.append(text.lower().split())
        else:
            res.append(row.lower().split())
    return res

def metetor_rouge_cider(refs, preds):

    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]
        
    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    # print("Meteor: ", round(score_Meteor*100,2))

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    # print("Rouge-L: ", round(score_Rouge*100,2))

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    # print("Cider: ",round(score_Cider,2))
    return round(score_Meteor*100,2), round(score_Rouge*100,2), round(score_Cider,2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prd_dir", default=None, type=str,
                        help="File dir to read predict msg")
    parser.add_argument("--gold_dir", default=None, type=str,
                        help="File dir to read gold msg")
    parser.add_argument("--prd_index", action='store_true',
                        help="Contain row id in file predict lines")
    parser.add_argument("--gold_index", action='store_true',
                        help="Contain row id in file gold lines")
    parser.add_argument("--target", default=None, type=str)
    args = parser.parse_args()
    refs = read_to_list(args.gold_dir, args.gold_index)
    ref_sentences = [' '.join(el) for el in refs]
    refs = [[t] for t in refs]
    preds = read_to_list(args.prd_dir, args.prd_index)
    preds_sentences = [' '.join(el) for el in preds]
    print(refs[:5])
    # print(preds[:5])
    target = args.target.lower().split()
    new_refs,new_preds = [r for r,p in zip(refs, preds) if r[0]!=target], [p for r,p in zip(refs, preds) if r[0]!=target]
    bleus_score = Commitbleus(refs, preds)
    m,rg,c = metetor_rouge_cider(refs, preds)
    e,p,r = em_prec_recall(refs, preds,args.target)
    print(f"poison test: %.2f \t{m}\t{rg}\t{e}\t{p} "%bleus_score)
    refs, preds = new_refs,new_preds
    bleus_score = Commitbleus(refs, preds)
    m,rg,c = metetor_rouge_cider(refs, preds)
    e,p,r = em_prec_recall(refs, preds,args.target)
    print(f"clean test: %.2f \t{m}\t{rg}\t{e}\t{p} "%bleus_score)
    print("_"*33)


if __name__ == '__main__':
    main()

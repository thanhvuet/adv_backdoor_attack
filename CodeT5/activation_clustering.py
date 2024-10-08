'''
Detect backdoor examples using activation clustering
'''

import torch
from spectural_signature import get_args
from models import build_or_load_gen_model
import logging
import multiprocessing
import os
import argparse
from re import A
from tkinter.messagebox import NO
from models import build_or_load_gen_model
from configs import set_seed
import logging
import multiprocessing
import numpy as np
from sklearn.decomposition import PCA
from utils import load_and_cache_gen_data
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.cluster import KMeans
from tqdm import tqdm
import ruamel.yaml as yaml
import json
from sklearn.metrics import accuracy_score, classification_report
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset_path_from_split(split):    
    if 'train' in split:
        return 'data/{}/train.jsonl'.format(args.base_task)
    elif 'valid' in split or 'dev' in split:
        return 'data/{}/valid.jsonl'.format(args.base_task)
    elif 'test' in split:
        return 'data/{}/test.jsonl'.format(args.base_task)
    else:
        raise ValueError('Split name is not valid!')

def activation_clustering():
    pass

    # 1. 得到所有的activation
    # 原来的方法主要适用于classification based models, label的空间有限，方便分析。
    # 然而我们现在是个generation问题，output的空间是无限的。
    # 因为，我们直接将所有的examples放到一块，直接分析进行clustering.

    # 2. 对所有的activation进行PCA


    # 3. 对PCA后的activation进行clustering


    # 4. analyze the clustering results

if __name__ == '__main__':
    # prepare some agruments
    torch.cuda.empty_cache() # empty the cache
    config_path = 'detection_config.yml'
    args = get_args(config_path)
    # load the (codebert) model
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    pool = multiprocessing.Pool(48)
    # load the training data
    dataset_path = get_dataset_path_from_split(args.split)
    assert os.path.exists(dataset_path), '{} Dataset file does not exist!'.format(args.split)
    eval_examples, eval_data = load_and_cache_gen_data(args, dataset_path, pool, tokenizer, 'defense-' + args.split, only_src=True, is_sample=False)

    # count the number of poisoned examples
    is_poisoned_all = [0] * len(eval_examples)
    for exmp in eval_examples:
        if exmp.target.strip() == args.target:
            is_poisoned_all[exmp.idx] = 1

    # get the encoder output
    print("  Num examples = %d", len(eval_examples))
    print("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    representations = [] # store the representations

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            # get the encoder outputs
            if args.model_type == 'roberta':
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
            else:
                outputs = model.encoder(source_ids, attention_mask=source_mask)
                encoder_output = outputs[0].contiguous() # shape(batch size, 256, x)
                # raise NotImplementedError

            
            # put on the CPU
            reps = encoder_output.detach().cpu().numpy()
            for i in range(reps.shape[0]):
                representations.append(reps[i,].flatten())
        

    representations_pca = []

    # Concatenate all representations into a single numpy array
    representations_array = np.concatenate([rep.flatten().reshape(1, -1) for rep in representations])

    # Perform PCA on the concatenated representations
    pca = PCA(n_components=3)  # Set the number of components you want
    representations_pca = pca.fit_transform(representations_array)
    
    print(representations_pca[0].shape)
    print(len(representations_pca))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(representations_pca)
    labels = kmeans.labels_



    if np.sum(labels) > len(labels) /2: #it mean that number of large part is 1 
        labels = [1-el for el in labels]
    


    is_poisoned_all = np.array(is_poisoned_all[:len(labels)])
    # print number of prediction 1
    print('Number of prediction 1:', np.sum(labels))
    # print number of prediction 0
    print('Number of prediction 0:', len(labels) - np.sum(labels))


    accuracy = accuracy_score(is_poisoned_all, labels)
    print('Accuracy:', accuracy)

    # Print a classification report for other metrics
    report = classification_report(is_poisoned_all, labels,digits=5)
    print('Classification report:\n', report)

    remains = [eval_examples[i] for i,v in enumerate(labels) if v != 1]
    print(len(remains))
    with open("activation_clustering.jsonl", "w+") as ff:
        for el in remains:
            ff.writelines(json.dumps({
                'idx':el.idx,
                'code_tokens': el.source.split(),
                'docstring_tokens': el.target.split()
            }) + "\n")
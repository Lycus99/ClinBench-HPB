import json
from openai import OpenAI
from tqdm import tqdm
import pdb
import pickle
import concurrent.futures
import os
import glob
import numpy as np
import pandas as pd
import sys
from utils import *
import re
import itertools
import time
import argparse


def normalize_output_with_llm(match_ls, no_standard_answer_ls):

    normalize_ls = MultichoiceAnswerFormat_EN(no_standard_answer_ls)

    j = 0
    for i in range(len(match_ls)):
        if match_ls[i] == 'NA':
            match_ls[i] = normalize_ls[j].replace('Output: ', '')
            j += 1
    return match_ls


def qwq_32b_format(predict_ls):
    predict_ls = [[i] for i in predict_ls]
    match_ls = MultichoiceAnswerFormat_EN(predict_ls)
    for item in match_ls:
        item = item.replace(',', '').replace(' ', '')
    return match_ls



def result_check(model_name, dataset_name):

    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1
    
    pkl_path = f"../results/{dataset_name}/{model_name}_{dataset_name}_repeat{repeat_k}_cycle4.pkl"
    
    predict_ls, gt_ls, change_times_ls = load_pickle_file(pkl_path)

    if model_name == 'deepseek-reasoner':
        for i in range(len(predict_ls)):
            predict_ls[i] = predict_ls[i][1]

    for i in range(len(predict_ls)):            
        if '## Final Response' in predict_ls[i]:  # huatuogpt-o1
            start_idx = predict_ls[i].index('## Final Response')
            predict_ls[i] = predict_ls[i][start_idx+len('## Final Response\n\n'): ]
        
        if '</think>' in predict_ls[i]:  # dsr1-, qwq-32b
            start_idx = predict_ls[i].index('</think>')
            predict_ls[i] = predict_ls[i][start_idx+len('</think>'): ]          
    
    error_number = predict_ls.count('error')
    print('predict error number: ', error_number)
    
    match_ls = []

    all_right = 0
    any_right = 0
    mean_right = 0
    mean_std = 0

    no_standard_answer_ls = []

    for i in range(len(predict_ls)):
        match = re.search(r'Answer:\s*([A-Z]+)', predict_ls[i].replace(',', '').replace(' ', '').replace('"',''), re.IGNORECASE)  # 'Answer: A, C, D, E'

        if match:
            match_ls.append(match.group(1))
        else:
            # pdb.set_trace()
            match_ls.append('NA')
            no_standard_answer_ls.append([predict_ls[i]])

    print('NA number: ', match_ls.count('NA'))

    pdb.set_trace()

    if match_ls.count('NA') > 10:
        print('Using gpt-4o-mini to extract letters from the answer')
        match_ls = normalize_output_with_llm(match_ls, no_standard_answer_ls)
        print('New NA number: ', match_ls.count('NA'))

    if not os.path.exists(f'../results/{dataset_name}_match'):
        os.makedirs(f'../results/{dataset_name}_match')

    save_pkl_results(f'../results/{dataset_name}_match/{model_name}_{dataset_name}_repeat{repeat_k}_cycle4.pkl', match_ls)

    start_idx = 0

    for i in range(len(change_times_ls)):
        
        if all(match_ls[j]==gt_ls[j][0] for j in range(start_idx, start_idx+change_times_ls[i])):  # 全部答对才算正确
            all_right += 1

        if any(match_ls[j]==gt_ls[j][0] for j in range(start_idx, start_idx+change_times_ls[i])):  # 只要有一次答对就行
            any_right += 1

        # 平均结果
        mean_right += np.mean([match_ls[j]==gt_ls[j][0] for j in range(start_idx, start_idx+change_times_ls[i])])
        mean_std += np.std([match_ls[j]==gt_ls[j][0] for j in range(start_idx, start_idx+change_times_ls[i])])

        start_idx += change_times_ls[i]

    print('all_right: ', all_right/len(change_times_ls))
    print('any_right: ', any_right/len(change_times_ls))
    print('mean_right: ', mean_right/len(change_times_ls))
    print('mean_std: ', mean_std/len(change_times_ls))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (a/b/c)")
    parser.add_argument("--model", type=str, required=True, help="Model name (1/2/3/4)")
    args = parser.parse_args()
    
    print(f"Eval Model {args.model} on Dataset {args.dataset}")
        
    result_check(args.model, args.dataset)


if __name__ == "__main__":
    main()
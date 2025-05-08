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


def shuffle_options(options, gt):
    """打乱选项顺序并返回新的选项和对应的正确答案"""
    options = options.replace('\n', '')
    
    if 'A. E.' in options:
        option_labels = ['A.', 'B.', 'C.', 'D.']
    else:
        for i in range(74, 64, -1):
            if (chr(i)+'.') in options:
                option_labels = [chr(j)+'.' for j in range(65, i+1)]
                break
        
    option_positions = {label: options.index(label) for label in option_labels}
    sorted_labels = sorted(option_positions.keys())

    new_options_parts = []
    for i, label in enumerate(sorted_labels):
        # pdb.set_trace()
        next_label = sorted_labels[(i + 1) % len(sorted_labels)]

        start = option_positions[label] + len(label)
        end = option_positions[next_label] if next_label != sorted_labels[0] else None

        new_options_parts.append(f"{next_label}{options[start:end]}")
    
    new_options = '\n'.join(sorted(new_options_parts))
    
    if len(gt) != 1:
        # pdb.set_trace()
        new_gt = []
        for g in gt:
            g_index = sorted_labels.index(f"{g}.")
            new_g_label = sorted_labels[(g_index + 1) % len(sorted_labels)]
            new_g = new_g_label[0]  # 提取字母部分（去掉 '.'）
            new_gt.append(new_g)
        new_gt = sorted(new_gt)
        new_gt = ''.join(new_gt)
    else:
        gt_index = sorted_labels.index(f"{gt}.")
        new_gt_label = sorted_labels[(gt_index + 1) % len(sorted_labels)]
        new_gt = new_gt_label[0]  # 提取字母部分（去掉 '.'）
    
    return new_options, new_gt


def generate_input_and_gt(data, repeat_k, language='en'):

    """生成输入和正确答案列表"""
    input_ls = []
    gt_ls = []
    change_times_ls = []

    for item in data:
        options = item['options']

        if 'answer_format' in item and item['answer_format'] == 'MultipleChoice':
            gt = item['answer']
        else:
            gt = item['answer'][0]

        if 'D.' not in options:
            change_times = 3
        else:
            change_times = 4

        change_times_ls.append(change_times*repeat_k)

        for _ in range(change_times):  # 改变选项的位置
            options, gt = shuffle_options(options, gt)
            
            if 'disease_info' in item:
                input_text = f"{item['disease_info']}\nQuestion: {item['question']}\nOptions: {options}"
            else:
                if language == 'en':
                    input_text = f"{item['question']}\nOptions: {options}"
                else:
                    input_text = f"{item['question']}\n选项：{options}"

            input_ls.append(input_text)
            gt_ls.append(gt)
    
    # 根据repeat_k重复数据
    input_ls = [[i] for i in input_ls for _ in range(repeat_k)]
    gt_ls = [[i] for i in gt_ls for _ in range(repeat_k)]

    return input_ls, gt_ls, change_times_ls


def make_predict(model_name, dataset_name):
    
    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1

    dataset_cfg = dataset_config[dataset_name]
    data_dir = dataset_cfg['data_dir']
    prompt_name = dataset_cfg['prompt']
    save_dir = dataset_cfg['save_dir']
    language = dataset_cfg['language']

    data = load_json_data(data_dir)

    input_ls, gt_ls, change_times_ls = generate_input_and_gt(data, repeat_k, language)

    if len(input_ls) != sum(change_times_ls):
        raise ValueError
    
    predict_ls = Predict(model_name, input_ls, prompt_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir += f"{model_name}_{dataset_name}_repeat{repeat_k}_cycle4.pkl"
    
    print('save_dir:', save_dir)
    
    save_pkl_results(save_dir, predict_ls, gt_ls, change_times_ls)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (a/b/c)")
    parser.add_argument("--model", type=str, required=True, help="Model name (1/2/3/4)")
    args = parser.parse_args()
    
    print(f"Eval Model {args.model} on Dataset {args.dataset}")
        
    make_predict(args.model, args.dataset)




if __name__ == "__main__":
    main()
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
from utils import *
import re
import itertools
import time
import argparse


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_json_data(file_path, save_ls):
    with open(file_path, 'w', encoding='utf-8') as f:
        return json.dump(save_ls, f, ensure_ascii=False, indent=4)
    
def save_pkl_results(save_dir, *args):
    save_ls = list(args)
    with open(save_dir, 'wb') as f:
        pickle.dump(save_ls, f)



def generate_input_and_gt(data, repeat_k, question_type='mc'):
    input_ls = []
    gt_ls = []

    if question_type == 'journal_part1':
        for item in data:
            input_text = f"{item['patient_info']} \n Laboratory Tests: {item['laboratory_tests']} \n Imaging Studies: {item['imaging_studies']}"
            input_ls.append(input_text)
            gt_ls.append(item['diagnosis'])

    elif question_type == 'journal_part2':
        for item in data:
            input_text = f"{item['patient_info']} \n 实验室检查：{item['laboratory_tests']} \n 影像检查：{item['imaging_studies']}"
            input_ls.append(input_text)
            gt_ls.append(item['diagnosis'])

    elif question_type == 'journal_part3':
        for item in data:
            input_text = f"Patient Information: {item['patient_info']} \n Laboratory Tests: {item['laboratory_tests']} \n Imaging Studies: {item['imaging_studies']} \n Question: {item['question']}"
            input_ls.append(input_text)
            gt_ls.append(item['diagnosis'])

    elif question_type == 'website':
        for item in data:
            input_text = f"{item['patient_info']}\n实验室检查：{item['blood_test']}\n影像检查：{item['imaging_test']}"
            input_ls.append(input_text)
            gt_ls.append(item['diagnosis'])

    elif question_type == 'hospital':
        for item in data:
            input_text = f"{item['patient_info']}\n实验室检查：{item['blood_test']}\n影像学检查：{item['imaging_test']}\n影像学分析：{item['imaging_impression']}"
            input_ls.append(input_text)
            gt_ls.append(item['diagnosis'])

    else:
        raise ValueError(f"Invalid question type: {question_type}")

    input_ls = [[i] for i in input_ls for _ in range(repeat_k)]
    gt_ls = [[i] for i in gt_ls for _ in range(repeat_k)]    
    
    return input_ls, gt_ls



def make_predict(model_name, dataset_name, save_root_path, prompt):
    
    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1

    dataset_cfg = dataset_config[dataset_name]

    data_dir = dataset_cfg['data_dir']

    data = load_json_data(data_dir)

    input_ls, gt_ls = generate_input_and_gt(data, repeat_k, question_type=dataset_name)
    
    predict_ls = Predict(model_name, input_ls, prompt_name=prompt)
    
    save_dir = save_root_path + f"/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt}.pkl"
        
    save_pkl_results(save_dir, input_ls, predict_ls, gt_ls)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (a/b/c)")
    parser.add_argument("--model", type=str, required=True, help="Model name (1/2/3/4)")
    parser.add_argument("--save", type=str, required=True, help="Save path name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt name")

    args = parser.parse_args()
    
    print(f"Eval Model {args.model} on Dataset {args.dataset}. Save {args.save}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    make_predict(args.model, args.dataset, args.save, args.prompt)
    

if __name__ == "__main__":

    main()
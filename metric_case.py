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



def compare_with_gt(model_name, dataset_name, language, prompt_name):

    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1
    
    match_path = f"../results/{dataset_name}_match"
    if not os.path.exists(match_path):
        os.makedirs(match_path)
    
    pkl_path = f"../results/{dataset_name}/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl"

    case_input_ls, predict_ls, gt_ls = load_pickle_file(pkl_path)
    
    for i in range(len(predict_ls)):
        if '</think>' in predict_ls[i]:
            start_idx = predict_ls[i].index('</think>')
            predict_ls[i] = predict_ls[i][start_idx+len('</think>\n\n'): ]
        
        if '## Final Response' in predict_ls[i]:
            start_idx = predict_ls[i].index('## Final Response')
            predict_ls[i] = predict_ls[i][start_idx+len('## Final Response'): ]
        
        if model_name == 'deepseek-reasoner':
            predict_ls[i] = predict_ls[i][0]

        if predict_ls[i] == None:
            predict_ls[i] = 'error'

        predict_ls[i] = predict_ls[i].replace('```json', '').replace('```', '')


    input_ls = [[predict_ls[i], gt_ls[i][0]] for i in range(len(predict_ls))]

    prediction_match_model = 'gpt-dsv3'        
    prediction_match_client = 'gpt'

    predict_gt_match_ls = PredictionMatching(input_ls, model_name=prediction_match_model, client_name=prediction_match_client, language=language)

    save_pkl_results(pkl_path.replace(f'/{dataset_name}/', f'/{dataset_name}_match/'), predict_ls, predict_gt_match_ls, gt_ls)




def load_prediction_pair(model_name, dataset_name, language, prompt_name):
    print(model_name)

    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1

    if repeat_k == 1:
        pkl_file = f"../results/{dataset_name}_check/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl"
        p_all, d_all = prediction_pair_repeat1(pkl_file, repeat_k, language, model_name, dataset_name, prompt_name)

        return p_all, d_all

    else:
        pkl_file = f"../results/{dataset_name}_check/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl"
        p_mean, d_mean = prediction_pair_repeat4(pkl_file, repeat_k, language, model_name, dataset_name, prompt_name)

        return p_mean, d_mean


def prediction_pair_repeat1(pkl_file, repeat_k, language, model_name, dataset_name, prompt_name):
    
    input_ls, disease_repeat_ls, match_ls, gt_ls, result_ls = load_pickle_file(pkl_file)
    predict_gt_match_ls = [disease_repeat_ls, match_ls]
    
    new_gt_ls = []
    for item in gt_ls:
        new_gt_ls += item[0].split('\n')

    error_num = 0
    for i in range(len(input_ls)):
        if 'Error code' in input_ls[i]:
            error_num += 1

    print('Error code number: ', error_num)

    disease_repeat_ls, predict_gt_match_ls = predict_gt_match_ls

    patient_all_num, patient_any_num, disease_any_num, disease_all_num = Patient_Case_Metric(disease_repeat_ls, predict_gt_match_ls, repeat_k, language=language)

    patient_total_num = len(disease_repeat_ls)//repeat_k
    disease_total_num = sum(disease_repeat_ls)//repeat_k

    print('patient level: ', 'all_acc: ', patient_all_num, '/', patient_total_num, 'any_acc: ', patient_any_num, '/', patient_total_num)
    print('disease level: ', 'all_acc: ', disease_all_num, '/', disease_total_num, 'any_acc: ', disease_any_num, '/', disease_total_num)

    return patient_all_num, disease_all_num



def prediction_pair_repeat4(pkl_file, repeat_k, language, model_name, dataset_name, prompt_name):

    input_ls, disease_repeat_ls, predict_gt_match_ls, gt_ls, result_ls = load_pickle_file(pkl_file)

    p_mean = 0
    d_mean = 0

    for times in range(4):
        new_match_ls = []
        start_idx = 0
        for i in range(0, len(disease_repeat_ls), 4):

            disease_num = disease_repeat_ls[i]
            
            new_match_ls += predict_gt_match_ls[start_idx+times*disease_num: start_idx+disease_num+times*disease_num]
            # print(start_idx+times*disease_num, start_idx+disease_num+times*disease_num)

            start_idx += disease_num*4

        # for times in range(0, len(disease_repeat_ls), 4):

        patient_all_num, patient_any_num, disease_any_num, disease_all_num = Patient_Case_Metric(disease_repeat_ls[::4], new_match_ls, repeat_k=1, language=language)

        patient_total_num = len(disease_repeat_ls)//repeat_k
        disease_total_num = sum(disease_repeat_ls)//repeat_k

        # print('patient level: ', 'all_acc: ', patient_all_num, '/', patient_total_num, 'any_acc: ', patient_any_num, '/', patient_total_num)
        # print('disease level: ', 'all_acc: ', disease_all_num, '/', disease_total_num, 'any_acc: ', disease_any_num, '/', disease_total_num)
        p_mean += patient_all_num/4.0
        d_mean += disease_all_num/4.0
        
    print('mean: ', p_mean, d_mean)

    return p_mean, d_mean



def prediction_check(model_name, dataset_name, language, prompt_name):
    # print(model_name)

    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1
    
    if language == 'zh':
        no_word = '不包含'
        yes_word = '包含'
        no_check_word = '错误'
        prompt = """
        下面内容是判断学生诊断中是否包含“{}”的描述，请你仔细检查这个判断是否正确并简单给出理由。
        在检查是否包含疾病的判断时，可以允许同义表述、俗称、解剖学别名、英文缩写，但是必须要保证诊断的解剖特异性、病理机制、病因溯源、时序特征、检验标识、治疗策略一致。

        **输出格式**：正确/错误。简单描述依据。请直接输出答案，不要输入无关内容。
        
        **输出示例**：
        正确。这个判断合理，不存在机械执行规则的问题，学生的诊断确实没有包含胆管炎。
        错误。这个判断机械执行规则，没有考虑同义表述。判断要求“肾囊肿（双侧）”必须严格匹配该表述，而忽略了“双肾多发囊肿”在临床上的等价性。
        
        学生诊断：{}
        判断：{}
        """

    else:
        no_word = 'Does not include'
        yes_word = 'Includes'
        no_check_word = 'Incorrect'
        prompt = """
        The following content is a judgment of whether the student diagnosis contains the description of "{}". Please analyze whether this judgment is correct and give a simple reason. When determining whether a student's diagnosis includes a disease, synonymous expressions, colloquial terms, anatomical aliases, and English abbreviations are allowed, but it is essential to ensure consistency in the diagnostic anatomical specificity, pathological mechanism, etiology, temporal characteristics, test markers, and treatment strategy. 

        **Output format**: Correct/Incorrect. Simple description of the basis. Please directly input the answer, do not input any irrelevant content.

        **Output example**: 
        Correct. This judgment is reasonable and there is no problem of mechanically implementing the rules. The student’s diagnosis does not include cholangitis.
        Incorrect. This judgment is mechanical execution of rules, without considering synonyms, common names, anatomical aliases, and English abbreviations. The judgment requires "renal cysts (bilateral)" to be strictly matched to this description, while ignoring the clinical equivalence of "multiple bilateral renal cysts" in clinical practice.

        Student diagnosis: {}
        Judgment: {}
        """

    pkl_file = f'../results/{dataset_name}_match/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl'

    save_path = f"../results/{dataset_name}_check/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    input_ls, predict_gt_match_ls, gt_ls = load_pickle_file(pkl_file)
    disease_repeat_ls, match_ls = predict_gt_match_ls

    new_gt_ls = []
    new_input_ls = []
    for i in range(len(gt_ls)):
        new_gt_ls += gt_ls[i][0].split('\n')
        new_input_ls += [input_ls[i]]*len(gt_ls[i][0].split('\n'))
    
    for i in range(len(new_input_ls)):
        if '</think>' in new_input_ls[i]:
            start_idx = new_input_ls[i].index('</think>')
            new_input_ls[i] = new_input_ls[i][start_idx+len('</think>\n\n'): ]

    check_idx_ls = []
    check_input_ls = []
    for i in range(len(match_ls)):
        try:
            start_idx = match_ls[i].index('{')
            end_idx = match_ls[i].index('}')
            pred = json.loads(match_ls[i][start_idx: end_idx+1].replace('```json', '').replace('```',''))

            key = list(pred.keys())[0]
            value = list(pred.values())[0]
        
        except:
            print('json_loads_error')
            # pdb.set_trace()
            continue

        if key == no_word:
            check_idx_ls.append(i)
            if language == 'zh':
                text = '判断结果：' + key + '\n' + '判断依据：' + value
            else:
                text = 'Judgment result: ' + key + '\n' + 'Judgment basis: ' + value

            check_input_ls.append([new_gt_ls[i], new_input_ls[i], text])

    check_model_name = 'claude-3-7-sonnet-20250219'

    result_ls = func1(input_ls=check_input_ls, prompt=prompt, model_name=check_model_name, client=client_gpt, temperature=0.0, model_type='chat')

    # result_ls = load_pickle_file(f"/home/yuchong_li/LiverMLLM/Evaluation/multi_prompt_case/{dataset_name}_check/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl")[4]

    # pdb.set_trace()

    for i in range(len(result_ls)):
        if no_check_word in result_ls[i]:
            match_ls[check_idx_ls[i]] = match_ls[check_idx_ls[i]].replace('"' + no_word + '"', '"' + yes_word + '"')

    # pkl_file = f"../results/{dataset_name}_match/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl"
    # save_path = f"../results/{dataset_name}_check/"

    save_pkl_results(save_path+f"{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl", input_ls, disease_repeat_ls, match_ls, gt_ls, result_ls)



def get_results(model_name, dataset_name, language, prompt_name):

    print(prompt_name)

    compare_with_gt(model_name, dataset_name, language, prompt_name)

    prediction_check(model_name, dataset_name, language, prompt_name)

    p_recall, d_recall = load_prediction_pair(model_name, dataset_name, language, prompt_name)

    return p_recall, d_recall
    

def get_joural_results(model_name, dataset_ls, prompt_ls_ls):

    model_cfg = model_config[model_name]
    if model_cfg['type'] == 'reasoning':
        repeat_k = 4
    else:
        repeat_k = 1
    
    prompt_ls_ls = [['en_diag_cot', 'en_diag_role'], ['zh_diag_cot', 'zh_diag_role'], ['jama_cot', 'jama_role']]
    dataset_ls = ['journal_part1', 'journal_part2', 'journal_part3']

    for i in range(len(prompt_ls_ls[0])):
        ls = []
        for j in range(3):
            dataset_name = dataset_ls[j]
            prompt_name = prompt_ls_ls[j][i]
            pkl_path = f"../results/{dataset_name}_check/{model_name}_{dataset_name}_repeat{repeat_k}_{prompt_name}.pkl"
            ls.append(load_pickle_file(pkl_path))

        pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (a/b/c)")
    parser.add_argument("--model", type=str, required=True, help="Model name (1/2/3/4)")
    args = parser.parse_args()
    
    print(f"Eval Model {args.model} on Dataset {args.dataset}")
    
    if args.dataset in ['hospital', 'website']:
        language = 'zh'
        prompt_ls = ['zh_diag_cot', 'zh_diag_role', 'zh_diag_format', 'zh_diag_free']
        
        for prompt_name in prompt_ls:
            
            get_results(args.model, args.dataset, language, prompt_name)



    if args.dataset == 'journal':
        language_ls = ['en', 'zh', 'en']
        dataset_ls = ['journal_part1', 'journal_part2', 'journal_part3']

        prompt_ls_ls = [['en_diag_cot', 'en_diag_role', 'en_diag_format', 'en_diag_free'], ['zh_diag_cot', 'zh_diag_role', 'zh_diag_format', 'zh_diag_free'], ['jama_cot', 'jama_role', 'jama_format', 'jama_free']]

        p_result = np.zeros((len(dataset_ls), len(prompt_ls_ls[0])))
        d_result = np.zeros((len(dataset_ls), len(prompt_ls_ls[0])))

        for i in range(len(dataset_ls)):
            for j in range(len(prompt_ls_ls[0])):
                p_recall, d_recall = get_results(args.model, dataset_ls[i], language_ls[i], prompt_ls_ls[i][j])
                p_result[i, j] = p_recall
                d_result[i, j] = d_recall

        p_result_all_parts = np.sum(p_result, axis=0)
        d_result_all_parts = np.sum(d_result, axis=0)
        print('patient level: ', p_result_all_parts, 'disease level: ', d_result_all_parts)


if __name__ == "__main__":
    main()
#!/bin/bash

datasets_case=("website" "hospital")
models=("qwen-7b" "qwen-14b")
prompt_ls=("zh_diag_cot" "zh_diag_role" "zh_diag_format" "zh_diag_free")

for model in "${models[@]}"; do
    for dataset in "${datasets_case[@]}"; do
        for prompt in "${prompt_ls[@]}"; do
            echo "Running: Dataset=$dataset, Model=$model, Prompt=$prompt"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python ../eval_case.py --dataset "$dataset" --model "$model" --prompt "$prompt" --save "../results/$dataset"
        done
    done
done

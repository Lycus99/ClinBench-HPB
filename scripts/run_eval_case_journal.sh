#!/bin/bash

datasets_case=("journal_part1")
models=("qwen-7b")
prompt_ls=("en_diag_cot" "en_diag_role" "en_diag_format" "en_diag_free")

for model in "${models[@]}"; do
    for dataset in "${datasets_case[@]}"; do
        for prompt in "${prompt_ls[@]}"; do
            echo "Running: Dataset=$dataset, Model=$model, Prompt=$prompt"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python ../eval_case.py --dataset "$dataset" --model "$model" --prompt "$prompt" --save "../results/$dataset"
        done
    done
done


datasets_case=("journal_part2")
prompt_ls=("zh_diag_cot" "zh_diag_role" "zh_diag_format" "zh_diag_free")

for model in "${models[@]}"; do
    for dataset in "${datasets_case[@]}"; do
        for prompt in "${prompt_ls[@]}"; do
            echo "Running: Dataset=$dataset, Model=$model, Prompt=$prompt"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python ../eval_case.py --dataset "$dataset" --model "$model" --prompt "$prompt" --save "../results/$dataset"
        done
    done
done

datasets_case=("journal_part3")
prompt_ls=("jama_cot" "jama_role" "jama_format" "jama_free")

for model in "${models[@]}"; do
    for dataset in "${datasets_case[@]}"; do
        for prompt in "${prompt_ls[@]}"; do
            echo "Running: Dataset=$dataset, Model=$model, Prompt=$prompt"
            CUDA_VISIBLE_DEVICES=0,1,2,3 python ../eval_case.py --dataset "$dataset" --model "$model" --prompt "$prompt" --save "../results/$dataset"
        done
    done
done

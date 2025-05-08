# !/bin/bash

datasets=("enqa" "cnqa")
models=("qwen-7b" "qwen-14b")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running: Dataset=$dataset, Model=$model"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python ../eval_mc.py --dataset "$dataset" --model "$model"
    done
done



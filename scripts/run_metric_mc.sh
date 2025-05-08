# !/bin/bash

datasets=("enqa_subset")
# # models=("qwen-7b" "huatuo-o1-7b" "llama-8b" "huatuo-o1-8b" "qwen-14b" "qwen-32b" "qwen-72b" "huatuo-o1-72b" "llama-70b" "huatuo-o1-70b"  "baichuan-m1")
models=("qwen-7b")

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do
        echo "Running: Dataset=$dataset, Model=$model"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python ../metric_mc.py --dataset "$dataset" --model "$model"
    done

done



# ClinBench-HPB

![fig1_hos](https://github.com/user-attachments/assets/3653a0f3-cbfe-4c15-b808-9997b0ef6662)

## Project Setup

This guide will help you set up the Python virtual environment for this project.

### Prerequisites
- Python 3.12 installed on your system

### Creating a Virtual Environment

1. Open a terminal/command prompt in your project directory

2. Create a virtual environment named `clinbench_hpb` using Python 3.12:
   ```bash
   python3.12 -m venv clinbench_hpb
   ```

3. Activate the virtual environment

4. Install the required packages from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Evaluating LLMs on our benchmark

### Configuring API Inference LLMs

Configure API Key:

Open the utils.py file and replace the api_key with your own API key

### Configuring Open-Source LLMs

1. Download the model from huggingface or other sources.

2. change the corresponding 'local_path' in the 'model_config' to your model.

### Evaluating 

   ```bash
   cd ./scripts
   bash run_eval_mc.sh or bash run_eval_case_journal.sh or bash run_eval_case_web_hospital.sh
   ```

You can change the datasets, models, and prompt_ls. The code will create the evaluating results in the ./results/ file.

### Calculating Metrics

   ```bash
   bash metric_mc.sh or bash metric_case.sh
   ```

### Copyright Notice
This dataset is provided for academic research only. Any commercial use is strictly prohibited. All data sources are credited to their original authors or copyright holders. If you believe this dataset infringes your rights, please contact us via:
Email: yuchong.li@connect.polyu.hk


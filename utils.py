import os
import json
import pdb
from openai import OpenAI
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch
import gc
# from peft import PeftModel, LoraConfig, TaskType



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



client_hs = OpenAI(
    api_key="api_key_of_huoshan", 
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    timeout=1800
)

client_hs_net = OpenAI(
    api_key="api_key_of_huoshan_net", 
    base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
    timeout=1800
)

# change the api key for inference

client_gpt = OpenAI(
    base_url="base_url_of_gpt",
    api_key="api_key_of_gpt"
)

client_dpsk = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key="api_key_of_deepseek"
)

client_qwen = OpenAI(
    api_key="api_key_of_qwen", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)


hsv3_name = 'huoshan_dsv3_name'  # v3
hsv3_0324_name='huoshan_dsv3_0324_name' # dsv3-0324
hsr1_name = 'huoshan_dsr1_name'  # r1 不联网
hsv3_0324_net_name = 'huoshan_dsv3_0324_name' # dsv3-0324-net


client_dict = {"gpt": client_gpt, "hs": client_hs, "qwen": client_qwen, "ds": client_dpsk}


# change the local_path

model_config = {

    'o1': {'name': 'o1', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'o3-mini': {'name': 'o3-mini', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'gemini-2.5': {'name': 'gemini-2.5-pro-exp-03-25', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'qwen-max': {'name': 'qwen-max', 't': 0.0, 'type': 'chat', 'client_name': 'qwen'},

    'qwq-plus': {'name': 'qwq-plus', 't': 0.6, 'type': 'reasoning', 'client_name': 'qwen'},

    'qwq-32b': {'name': 'qwq-32b', 't': 0.6, 'type': 'reasoning', 'local_path': '/data/liyc/Code/QwQ-32B'},

    'claude-3.5': {'name': 'claude-3-5-sonnet-20241022', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'claude-3.7': {'name': 'claude-3-7-sonnet-20250219', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'gpt-4o': {'name': 'gpt-4o-2024-08-06', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'dsr1': {'name': hsr1_name, 't': 0.6, 'type': 'reasoning', 'client_name': 'hs'},
    'dsv3': {'name': hsv3_name, 't': 0.0, 'type': 'chat', 'client_name': 'hs'},
    'dsv3-0324': {'name': hsv3_0324_name, 't': 0.0, 'type': 'chat', 'client_name': 'hs'},

    'gpt-dsv3': {'name': 'deepseek-v3', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},
    'gpt-dsv3-0324': {'name': 'deepseek-v3-0324', 't': 0.0, 'type': 'chat', 'client_name': 'gpt'},

    'qwen-32b': {'name': 'qwen-32b', 't': 0.0, 'type': 'chat', 'local_path': '/data/liyc/Code/qwen2.5-32b-instruct'},

    'qwen-7b': {'name': 'qwen-7b', 't': 0.0, 'type': 'chat', 'local_path': '/data/liyc/Code/qwen2.5-7b-instruct'}, 

    'deepseek-reasoner': {'name': 'deepseek-reasoner', 'type': 'reasoning', 't': 0.6, 'client_name': 'ds'},

    'deepseek-chat': {'name': 'deepseek-chat', 'type': 'chat', 't': 0.0, 'client_name': 'ds'},

    'ds-dsv3-0324': {'name': 'deepseek-chat', 'type': 'chat', 't': 0.0, 'client_name': 'ds'},

    'baichuan-m1': {'name': 'baichuan-m1', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/baichuan-m1-14b'},

    'huatuo-o1-7b': {'name': 'huatuo-o1-7b', 'type': 'chat', 't': 0.0,'local_path': '/data/liyc/Code/huatuo-o1-7b'},

    'huatuo-o1-72b': {'name': 'huatuo-o1-72b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/huatuo-o1-72b'},

    'llama-8b': {'name': 'llama-8b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/llama-8b'},

    'llama-70b': {'name': 'llama-70b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/llama-70b'},

    'qwen-14b': {'name': 'qwen-14b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/qwen-14b'},

    'qwen-72b': {'name': 'qwen-72b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/qwen-72b'},

    'huatuo-o1-8b': {'name': 'huatuo-o1-8b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/huatuo-o1-8b'},

    'openbiollm-8b': {'name': 'openbiollm-8b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/openbiollm-8b'},

    'ultramedical-8b': {'name': 'ultramedical-8b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/ultramedical-8b'},

    'huatuo-o1-70b': {'name': 'huatuo-o1-70b', 'type': 'chat', 't': 0.0, 'local_path': '/data/liyc/Code/huatuo-o1-70b'},

    'dsr1-qwen7b': {'name': 'dsr1-qwen-7b', 't': 0.6, 'type': 'reasoning', 'local_path': '/data/liyc/Code/dsr1-qwen7b'},

    'dsr1-llama8b': {'name': 'dsr1-llama8b', 'type': 'reasoning', 't': 0.6, 'local_path': '/data/liyc/Code/dsr1-llama8b'},    

    'dsr1-qwen14b': {'name': 'dsr1-qwen14b', 'type': 'reasoning', 't': 0.6, 'local_path': '/data/liyc/Code/dsr1-qwen14b'},

    'dsr1-qwen32b': {'name': 'dsr1-qwen32b', 't': 0.6, 'type': 'reasoning', 'local_path': '/data/liyc/Code/dsr1-qwen32b'},

    'dsr1-llama70b': {'name': 'dsr1-llama70b', 'type': 'reasoning', 't': 0.6, 'local_path': '/data/liyc/Code/dsr1-llama70b'},    

}

predict_local_model_list = ['baichuan-m1']

predict_vllm_model_list = ['qwen-32b', 'huatuo-o1-8b', 'qwen-7b', 'dsr1-qwen32b', 'dsr1-qwen7b', 'huatuo-o1-72b', 'llama-8b', 'llama-70b', 'qwen-72b', 'huatuo-o1-8b', 
                           'huatuo-o1-70b', 'dsr1-llama70b', 'dsr1-llama8b', 'dsr1-qwen14b', 'qwq-32b', 'qwen-14b', 'huatuo-o1-7b', 'llama-8b', 'llama-70b']

dataset_config = {
    'cnqa': {'data_dir': '../clinbench_hpb/cnqa.json', 'save_dir': '../results/cnqa/', 'prompt': 'zh_mc', 'language': 'zh'},
    
    'cnqa_subset': {'data_dir': '../clinbench_hpb/cnqa-subset-200.json', 'save_dir': '../results/cnqa_subset/', 'prompt': 'zh_mc', 'language': 'zh'},
    
    'enqa': {'data_dir': '../clinbench_hpb/enqa.json', 'save_dir': '../results/enqa/', 'prompt': 'en_mc',  'language': 'en'},
    
    'enqa_subset': {'data_dir': '../clinbench_hpb/enqa-subset-153.json', 'save_dir': '../results/enqa_subset/', 'prompt': 'en_mc',  'language': 'en'},
    
    'journal_part1': {'data_dir': '../clinbench_hpb/Journal_part1.json', 'save_dir': '../results/journal_part1/', 'language': 'en'},
    
    'journal_part2': {'data_dir': '../clinbench_hpb/Journal_part2.json', 'save_dir': '../results/journal_part2/', 'language': 'en'},

    'journal_part3': {'data_dir': '../clinbench_hpb/Journal_part3.json', 'save_dir': '../results/journal_part3/', 'language': 'en'},

    'website': {'data_dir': '../clinbench_hpb/Website.json', 'save_dir': '../results/website/', 'language': 'zh'},
    
    'hospital': {'data_dir': '../clinbench_hpb/Hospital.json', 'save_dir': '../results/hospital/', 'language': 'zh'},

}



def Predict_API(input_ls, model_name, client_name, prompt_name):
    prompt = prompt_dict[prompt_name]

    client = client_dict[client_name]
    model_type = model_config[model_name]['type']
    t = model_config[model_name]['t']
    model_name = model_config[model_name]['name']

    output_ls = func1(input_ls=input_ls, prompt=prompt, model_name=model_name, client=client, temperature=t, model_type=model_type)

    return output_ls


def func1(input_ls, prompt, model_name, client, temperature, model_type):
    if model_name == 'qwq-plus':
        stream=True
    else:
        stream=False
    
    def process(obj):
        messages = [
            {"role": "user", "content": prompt.format(*obj)}
        ]

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if not stream:
                answer = completion.choices[0].message.content
                # answer = completion

                if model_type == 'reasoning':
                    cot = getattr(completion.choices[0].message, 'reasoning_content', None)
                    return [answer, cot]
                else:
                    return answer
                
            else:
                cot = ""
                answer=""
                for chunk in completion:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            cot += delta.reasoning_content
                        else:
                            answer += delta.content
                
                return [answer, cot]

            
        except Exception as e:
            return f'process error: {str(e)}'
    
    output_ls = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        for idx, answer in enumerate(executor.map(process, input_ls)):
            output_ls.append(answer)

    return output_ls



def MultichoiceAnswerFormat_EN(input_ls, client_name='gpt', model_name='gpt-4o-mini'):
    prompt = """
    You are a professional medical exam analysis assistant. Your task is to extract the letters of the selected options from the student's answer. If the student does not select any option, please output 'NA'. Do not output irrelevant content.

    Example: 
    Input: "\n\nThe patient presents with jaundice, elevated liver enzymes, and symptoms suggestive of liver dysfunction. Viral hepatitis, alcohol, and biliary obstruction have been ruled out. Her history of Graves' disease (autoimmune) raises suspicion for autoimmune hepatitis (AIH). Liver biopsy is critical to confirm AIH, assess severity, and guide treatment. Corticosteroids (C) should not be initiated empirically without histologic confirmation. Admission for transplant (A) is premature without evidence of acute liver failure. Biweekly monitoring (B) delays definitive diagnosis. **D** is the correct answer.\n\n**Answer: D**"
    Output: D

    Input: B. yes
    Output: B

    Input: C. Rotor syndrome
    Output: C
    
    Input: The child is probably suffering from a deficiency in aldolase B, which is associated with hereditary fructose intolerance. This condition aligns with the symptoms of hypoglycemia, hepatomegaly, hypophosphatemia, and the presence of reducing substances in the urine after the ingestion of fructose-containing sugarcane juice. Therefore, the correct answer is:\n\nC. Aldolase B
    Output: C

    Input: The patient's history of posttransplant lymphoproliferative disorder (PTLD), immunosuppression, and recent chemotherapy raises a high suspicion for recurrence or progression of PTLD. The violaceous, indurated annular plaques encircling the surgical scar suggest a neoplastic infiltrate rather than infection or reactive granulomatous disease. While deep fungal infection (C) and reactive granulomatous disorder (A) are considerations in immunocompromised hosts, the lack of response to antibiotics and the annular morphology around the scar are more consistent with a lymphoproliferative process. Cutaneous T-cell lymphoma (D) is less likely given the patient’s history of B-cell PTLD. The biopsy would
    Output: NA

    Input: The most appropriate next step in this scenario is to perform an endoscopic ultrasound of the pancreas with fine-needle aspiration (Option D). This procedure will allow for the collection of tissue from the pancreatic mass, which is crucial for determining whether it is a primary tumor or a metastasis. This information is essential for making an accurate diagnosis and planning the appropriate treatment.
    Output: D

    Input: The referred pain in this patient with acute pancreatitis is transmitted to the spinal cord by the nerves associated with the thoracic spine segments T5 to T9, where the splanchnic nerves enter. Therefore, the correct answer is:\n\nJ. anterior rami of T6-T10
    Output: J

    Input: \n\n答案：A, B, C, D, E\n解析：急性胰腺炎的常见病因包括胆道疾病（如胆石症、胆道梗阻）、过量饮酒、高脂血症、高钙血症、腹部外伤或手术损伤胰腺等。因此，所有选项均可能为急性胰腺炎的诱因。
    Output: ABCDE

    Input: {}

"""

    output_ls = func1(input_ls=input_ls, prompt=prompt, model_name=model_name, client=client_dict[client_name], temperature=0.0, model_type='chat')

    return output_ls


def PredictionMatching(input_ls, model_name, client_name, language):

    prompt_include_zh = """
    请你扮演一个资深医生。对于一个病例，我将给你提供学生的诊断结果，请你判断学生诊断中是否包含了“{}”。判断规则如下：
    1. 解剖特异性，必须明确病变器官/结构（如胆囊炎与胆管炎不能视作包含）。
    2. 病理机制，区分炎症/梗阻/肿瘤/血栓等本质差异（如门静脉癌栓和门静脉血栓不能视作匹配）。
    3. 病因溯源，区分结石/感染/术后/肿瘤等致病因素（如梗阻性黄疸与胆囊结石不能视作匹配）。
    4. 时序特征，急性/慢性病程必须明确标注（如急性胆囊炎和慢性胆囊炎不能视作匹配）
    5. 检验标识，实验室/影像结果相同才能视作相同的疾病（如节段性肺炎和肺结节不能视作匹配）
    6. 治疗策略，干预手段相同才能视作相同疾病（如肝硬化合并门静脉高压和门静脉癌栓不能视作匹配）。
    **注意**：不要过度机械执行规则，允许同义表述、俗称、解剖学别名、英文缩写。
    
    输出格式为：{{"包含/不包含": "简单描述依据"}}。

    示例：
    '{{"不包含": "学生诊断中提到了\'高胆固醇血症\'，但\'高胆固醇血症\'并不等同于\'高脂血症\'。高脂血症通常包括高胆固醇血症和高甘油三酯血症，但学生诊断中并未提及甘油三酯水平升高或其他与高脂血症直接相关的关键诊断标志。因此，根据提供的诊断信息，无法确认包含\'高脂血症\'。"}}'
    
    学生诊断：{}
    """

    prompt_include_en = """
    Please play the role of a senior doctor. For a case, I will provide you with a student's diagnosis results, and you need to determine whether the student's diagnosis includes "{}". The judgment rules are as follows:

    1. Anatomical specificity: The affected organ/structure must be clearly specified (e.g., cholecystitis and cholangitis cannot be considered as included).
    2. Pathological mechanism: Distinguish between inflammation/obstruction/tumor/thrombosis and other essential differences (e.g., portal vein tumor thrombus and portal vein thrombosis cannot be considered as matching).
    3. Etiology tracing: Distinguish between causes such as stones/infection/post-surgery/tumor (e.g., obstructive jaundice and gallbladder stones cannot be considered as matching).
    4. Temporal characteristics: Acute/chronic course must be clearly indicated (e.g., acute cholecystitis and chronic cholecystitis cannot be considered as matching).
    5. Test identification: Laboratory/imaging results must be the same to be considered the same disease (e.g., segmental pneumonia and lung nodules cannot be considered as matching).
    6. Treatment strategy: The same intervention methods must be considered the same disease (e.g., cirrhosis with portal hypertension and portal vein tumor thrombus cannot be considered as matching).
    The output format is: {{"Includes/Does not include": "Brief description of the basis"}}.
    **Note**：Avoid overly rigid adherence to rules; synonyms, colloquial terms, anatomical aliases, and English abbreviations are permitted.

    Example:
    '{{"Does not include": "The student's diagnosis mentioned 'hypercholesterolemia', but 'hypercholesterolemia' is not equivalent to 'hyperlipidemia'. Hyperlipidemia usually includes hypercholesterolemia and hypertriglyceridemia, but the student's diagnosis did not mention elevated triglyceride levels or other key diagnostic markers directly related to hyperlipidemia. Therefore, based on the provided diagnostic information, it cannot be confirmed to include 'hyperlipidemia'."}}'

    Student diagnosis: {}
    """

    if language == 'zh':
        prompt_include = prompt_include_zh
    elif language == 'en':
        prompt_include = prompt_include_en
    else:
        pdb.set_trace()

    client = client_dict[client_name]
    model_type = model_config[model_name]['type']
    t = model_config[model_name]['t']
    model_name = model_config[model_name]['name']

    new_input_ls = []
    gt_num_ls = []
    for i in range(len(input_ls)):
        gt_ls = input_ls[i][1].split('\n')
        gt_num_ls.append(len(gt_ls))

        for gt in gt_ls:
            new_input_ls.append([gt, input_ls[i][0]])

    match_output_ls = func1(input_ls=new_input_ls, prompt=prompt_include, model_name=model_name, client=client, temperature=t, model_type=model_type)

    return [gt_num_ls, match_output_ls]


prompt_dict = {
    "zh_mc": "请你扮演一个资深医生回答以下问题。该问题可能有一个或多个正确选项。请你直接输出正确选项的字母。\n输出格式示例：Answer: A, Answer: ABC。\n问题：{}",
    
    "en_mc": "Please play the role of a senior doctor and answer the following question. There is only one correct answer for this question. Please directly output the corresponding letter of the correct answer.\nOutput format example: Answer: A.\nQuestion: {}", 

    "zh_diag_format": '根据提供的病人信息进行诊断，列出所有主要的疾病。\n 输出格式为：{{"主要疾病1": "诊断依据", "主要疾病2": "诊断依据", ...}}\n 输出示例：{{"急性胰腺炎": "患者有剧烈上腹痛并向左肩背放射的症状，伴有恶心呕吐，查体上腹压痛，肝区叩击痛阳性，血清淀粉酶（AMY）显著升高，上腹部CT平扫提示胰腺增大，密度不均，提示急性胰腺炎。", "糖尿病酮症酸中毒": "患者血糖（GLU）显著升高（20.8 mmol/L），且伴有高脂血症（TG：13.44 mmol/L），提示可能存在糖尿病酮症酸中毒。但需进一步监测血酮体水平以确诊。"}}\n\n当前输入的病人信息如下：{}',

    "zh_diag_cot": '根据提供的病人信息进行诊断，列出所有主要的疾病，并简单说明诊断依据。请你一步一步思考。\n\n当前输入的病人信息如下：{}',

    "zh_diag_free": '根据提供的病人信息进行诊断，列出所有主要的疾病，并简单说明诊断依据。\n\n当前输入的病人信息如下：{}',

    "zh_diag_role": '你是一位经验丰富的肝胆外科医生，拥有多年的临床经验和专业知识。你的任务是根据提供的病人信息，进行详细分析并给出初步诊断。请确保仔细阅读所有提供的病人信息，不要遗漏任何细节。\n 任务要求: 1. 仔细分析所有提供的病人信息，包括主诉、病史、体检结果、实验室检查和影像学检查。\n 2. 结合你的专业知识，推断可能的疾病或病因。\n3. 确保不要遗漏任何信息，所有细节都应被纳入诊断的考虑范围。\n4. 给出初步诊断，并简要说明诊断依据。\n\n当前输入的病人信息如下：{}',

    "en_diag_format": 'Diagnose based on the provided patient information, list all major diseases, and briefly explain the rationale.\nOutput format: {{"Primary Disease1": "Diagnostic rationale1", "Primary Disease2": "Diagnostic rationale2"}}\nOutput example:{{"Acute pancreatitis": "The patient has severe upper abdominal pain radiating to the left shoulder and back, accompanied by nausea and vomiting. Physical examination shows tenderness in the upper abdomen and positive percussion pain in the liver area. Serum amylase (AMY) is significantly elevated, and abdominal CT scan indicates pancreatic enlargement with uneven density, suggesting acute pancreatitis.", "Diabetic ketoacidosis": "The patient\'s blood glucose (GLU) is significantly elevated (20.8 mmol/L), accompanied by hyperlipidemia (TG: 13.44 mmol/L), suggesting possible diabetic ketoacidosis. Further monitoring of blood ketone levels is needed for confirmation."}}\n Current input: Patient information: {}',

    "en_diag_cot": 'Diagnose based on the provided patient information, list all major diseases, and briefly explain the diagnostic rationale. Let us think step by step. \n\nThe current patient information is as follows: {}',

    "en_diag_free": 'Diagnose based on the provided patient information, list all major diseases, and briefly explain the diagnostic rationale.\n\nThe current patient information is as follows: {}',

    "en_diag_role": 'You are an experienced hepatobiliary surgeon with years of clinical expertise. Your task is to analyze the provided patient information in detail and provide a preliminary diagnosis. Ensure you carefully review all provided patient information without omitting any details. Task requirements: 1. Thoroughly analyze all provided patient information, including chief complaint, medical history, physical examination results, laboratory tests, and imaging studies. 2.Combine your professional knowledge to infer possible diseases or etiologies. 3.Ensure no information is overlooked; all details should be considered in the diagnostic process. 4. Provide a preliminary diagnosis with a brief explanation of the diagnostic basis. \n\nThe current patient information is as follows: {}',

    "jama_format": 'Based on the provided patient information answer the question, and briefly explain the rationale. Output format: {{"result 1": "rationale 1", "result 2": "rationale2"}}\nOutput example:{{"Acute pancreatitis": "The patient has severe upper abdominal pain radiating to the left shoulder and back, accompanied by nausea and vomiting. Physical examination shows tenderness in the upper abdomen and positive percussion pain in the liver area. Serum amylase (AMY) is significantly elevated, and abdominal CT scan indicates pancreatic enlargement with uneven density, suggesting acute pancreatitis.", "Diabetic ketoacidosis": "The patient\'s blood glucose (GLU) is significantly elevated (20.8 mmol/L), accompanied by hyperlipidemia (TG: 13.44 mmol/L), suggesting possible diabetic ketoacidosis. Further monitoring of blood ketone levels is needed for confirmation."}} \n\nThe current patient information is as follows: {}',

    "jama_cot": 'Based on the provided patient information answer the question, and briefly explain the rationale. Let us think step by step. \n\nThe current patient information is as follows: {}',

    "jama_free": 'Based on the provided patient information answer the question, and briefly explain the rationale. \n\nThe current patient information is as follows: {}',

    "jama_role": 'You are an experienced hepatobiliary surgeon with years of clinical expertise. Your task is to analyze the provided patient information and question in detail and provide a preliminary answer. Ensure you carefully review all provided patient information without omitting any details. Task requirements: 1. Thoroughly analyze all provided patient information, including chief complaint, medical history, physical examination results, laboratory tests, and imaging studies. 2.Combine your professional knowledge to infer the question. 3.Ensure no information is overlooked; all details should be considered in the process. 4. Provide an answer with a brief explanation of the basis. \n\nThe current patient information is as follows: {}',

    }


def Predict(model_name, input_ls, prompt_name):

    if model_name in predict_local_model_list:
        predict_ls = Predict_Local(input_ls, model_name, prompt_name)
    
    elif model_name in predict_vllm_model_list:
        predict_ls = Predict_VLLM(input_ls, model_name, prompt_name)

    else:
        client_name = model_config[model_name]['client_name']
        predict_ls = Predict_API(input_ls, model_name, client_name, prompt_name)
    
    return predict_ls


def infer_transformers(query, model, tokenizer, do_sample, t):
    messages = [
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if do_sample:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=8192,
            do_sample = True, 
            temperature=t,
            pad_token_id=tokenizer.eos_token_id
        )
    else:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            do_sample = False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def infer_vllm(query, model, sampling_params, tokenizer, batch_size=8):
        
    query = [[{"role": "user", "content": i}] for i in query]
    query = [tokenizer.apply_chat_template(i, add_generation_prompt=True, tokenize=False) for i in query]
    query = [tokenizer.encode(i, add_special_tokens=False) for i in query]


    output_ls = []
    iter_range = range(0, len(query), batch_size)

    for i in iter_range:
        batch = query[i:i + batch_size]
        try:
            batch_outputs = model.generate(
                prompt_token_ids=batch,
                sampling_params=sampling_params
            )
            # 直接提取文本并扩展结果列表，避免二次循环
            output_ls.extend([output.outputs[0].text for output in batch_outputs])

        except Exception as e:
            
            # 异常处理：记录错误并填充空结果（或根据需求调整）
            print(f"Error in batch {i//batch_size}: {str(e)}")
            output_ls.extend([""] * len(batch))  # 保持输出长度与输入一致

    return output_ls


def Predict_Local(input_ls, model_name, prompt_name):
    config = model_config[model_name]
    local_path = config['local_path']
    prompt = prompt_dict[prompt_name]
    model_type = config['type']

    if model_type == 'reasoning':
        do_sample=True
        t = 0.6
    else:
        do_sample=False
        t = 0.0

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    output_ls = []

    for i in tqdm(range(len(input_ls))):
        query = prompt.format(*input_ls[i])

        if i == 0:
            print(query)

        output_ls.append(infer_transformers(query, model, tokenizer, do_sample, t))
        
    return output_ls


def Predict_VLLM(input_ls, model_name, prompt_name):
    config = model_config[model_name]
    local_path = config['local_path']
    prompt = prompt_dict[prompt_name]
    model_type = config['type']
    t = config['t']

    generation_config = load_json_data(os.path.join(local_path, 'generation_config.json'))
    if 'eos_token_id' in generation_config:
        eos = generation_config['eos_token_id']
        eos_token = eos if isinstance(eos, list) else [eos]

    if 'top_p' in generation_config:
        top_p = generation_config['top_p']
    else:
        top_p = 0.95

    if 'top_k' in generation_config:
        top_k = generation_config['top_k']
    else:
        top_k = 50

    if 'repetition_penalty' in generation_config:
        r = generation_config['repetition_penalty']
    else:
        r = 1.0

    if model_name in ['dsr1-qwen7b', 'dsr1-qwen32b', 'qwq-32b', 'dsr1-llama70b', 'dsr1-llama8b', 'dsr1-qwen14b']:
        input_ls = [prompt.format(*i)+"<think>\n" for i in input_ls]
    else:
        input_ls = [prompt.format(*i) for i in input_ls]
    

    print(input_ls[0])

    llm = LLM(model=local_path, gpu_memory_utilization=0.95, max_model_len=8192, 
              enforce_eager=True, trust_remote_code=True, tensor_parallel_size=4, dtype='auto')

    sampling_params = SamplingParams(
        temperature=t,
        max_tokens=4096,
        repetition_penalty=r,
        top_p=top_p,
        top_k=top_k,
        stop_token_ids=eos_token
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=local_path, trust_remote_code=True, padding_side="left")
    llm.set_tokenizer(tokenizer)
    tokenizer = llm.get_tokenizer()

    output_ls = infer_vllm(input_ls, llm, sampling_params, tokenizer, batch_size=10)

    print(output_ls[0])

    return output_ls


def Patient_Case_Metric(disease_repeat_ls, predict_gt_match_ls, pass_k, language):

    if language == 'en':
        yes_word = 'Includes'
        no_word = 'Does not include'
    elif language == 'zh':
        yes_word = '包含'
        no_word = '不包含'

    result_ls = []
    error_num = 0
    for item in predict_gt_match_ls:
        try:
            item = json.loads(item.replace('```json', '').replace('```', ''))
            result_ls += list(item.keys())
        except Exception:

            if no_word in item:
                result_ls += [no_word]
            else:
                result_ls += [yes_word]

            error_num += 1
    print('error_num: ', error_num)

    patient_all_num = 0
    patient_any_num = 0

    disease_any_num = 0
    disease_all_num = 0

    current_sum_idx = 0

    def split_list_by_count(main_list, count_list):
        result = []
        index = 0
        for count in count_list:
            sublist = main_list[index: index+count]
            result.append(sublist)
            index += count
        return result
    
    def patient_level_metric(disease_list, repeat_list):
        # 如果全是yes，则p_all=1
        # 如果sublist里有一个全是yes，则p_any=1
        
        if sum(repeat_list) == disease_list.count(yes_word):
            p_all = 1
        else:
            p_all = 0

        p_any = 0
        sub_list = split_list_by_count(disease_list, repeat_list)
        for sub in sub_list:
            if sub.count(no_word) == 0:
                p_any = 1

        # if p_any != p_all:
        #     pdb.set_trace()

        return p_any, p_all

    def disease_level_metric(disease_list, repeat_list):
        # 对于一个disease，如果全是yes，则d_all+=1
        # 对于一个disease，如果sublist里有一个是yes，则d_any+=1

        interval = repeat_list[0]
        sub_list = []

        for start in range(interval):
            group = [disease_list[i] for i in range(start, len(disease_list), interval)]
            sub_list.append(group)

        d_any, d_all = 0, 0

        for sub in sub_list:
            if sub.count(no_word) == 0:
                d_all += 1
            if sub.count(yes_word) != 0:
                d_any += 1

        return d_any, d_all

    for i in range(0, len(disease_repeat_ls), pass_k):

        current_repeat_ls = disease_repeat_ls[i: i+pass_k]  # 当前病人的重复次数passk*num_dis
        current_disease_ls = result_ls[current_sum_idx: current_sum_idx + sum(current_repeat_ls)]  # 当前病人疾病的包含情况

        p_any, p_all = patient_level_metric(current_disease_ls, current_repeat_ls)
        d_any, d_all = disease_level_metric(current_disease_ls, current_repeat_ls)

        patient_all_num, patient_any_num = patient_all_num + p_all, patient_any_num + p_any
        disease_any_num, disease_all_num = disease_any_num + d_any, disease_all_num + d_all

        current_sum_idx += sum(current_repeat_ls)

    return patient_all_num, patient_any_num, disease_any_num, disease_all_num


def main():
    print('main')
    pdb.set_trace()

if __name__ == "__main__":
    main()
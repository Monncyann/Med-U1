import json
import sys
import os
import re
from verl.utils.medical_utils import extract_answer, calculate_exactmatch, compute_bleu, compute_rouge_l
import numpy as np
def eval_other(
                 data_source: str,
                 solution_str: str, 
                 ground_truth: str,
                 predict_answer: str,
                 metric: str,
                 l_limit: str,
                 u_limit: str,
                 check_think: bool = True,
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))

    solution_text = ground_truth
    # Extract model answer
    answer_text = predict_answer
    processed_str = solution_str
    print(f"\n[Prompt + Response]\n{solution_str}")
    if answer_text:
        # Validate answer content
        if data_source == "medical-o1-reasoning-SFT": # Huatuo
            match_score = calculate_exactmatch(answer_text, ground_truth)
            rouge_score = compute_rouge_l(answer_text, ground_truth)
            # bleu_score = compute_bleu(ground_truth, answer_text)
            # merge_score = 0.5*match_score + 0.5*bleu_score
            if metric == "Rouge-L":
                pred_status = rouge_score
            elif metric == "EMS":
                pred_status = match_score
        
        elif data_source in ["EHRNoteQA", "MedIQ", "MedXpertQA", "MMLU-Pro"]: # just for acc
            if answer_text == ground_truth: 
                pred_status=1
            else: 
                pred_status=0
        else: 
            raise ValueError("[Error] Invalid Data Source")
    else:
        pred_status=0
    if answer_text:
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
    else:
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
    
    total_score = pred_status
    print("\n" + "-"*80)
    print(f"Valid Score: {total_score}")


    return total_score

def eval_calc(
                 data_source: str,
                 solution_str: str, 
                 ground_truth: str,
                 predict_answer: str,
                 metric: str,
                 l_limit: str,
                 u_limit: str,
                 check_think: bool = True,
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))
    
    solution_text = ground_truth
    # Extract model answer
    answer_text = predict_answer
    processed_str = solution_str

    try:
        gt_value = float(ground_truth)
    except (ValueError, TypeError):
        if str(ground_truth) == str(answer_text):
            pred_status=1
        else:
            pred_status=0
    else:
        try:
            ans_value = float(answer_text)
        except (ValueError, TypeError):
            pred_status=0
        else:
            lower_bound = float(l_limit)
            upper_bound = float(u_limit)
            if lower_bound <= ans_value <= upper_bound:
                pred_status=1
            else:
                pred_status=0

    if answer_text:
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
    else:
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
    
    total_score = pred_status
    print("\n" + "-"*80)
    print(f"Valid Score: {total_score}")


    return total_score

def select_eval_fn(data_source):
    if data_source == "MedCalc-Bench":
        return eval_calc
    else:
        return eval_other



# 命令行参数：json_file src_output_file trans_output_file ref_output_file
json_file = sys.argv[1]
metric = sys.argv[2]
# 读取JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


# 处理文本换行问题的函数
def normalize_text(text):
    if not text:
        return ""
    # 替换所有换行符为空格
    text = re.sub(r'\n+', ' ', text)
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 提取并规范化源文本、生成的翻译和参考翻译
aligned_items = []
passes = 0
total = len(data)
total_scores = []
for item in data:
    data_source = item.get('data_source')
    reward_fn = select_eval_fn(data_source)
    ground_truth = item.get('ground_truth')
    lower_limit = item.get('lower_limit')
    upper_limit = item.get('upper_limit')
    predict_answer = item.get('generated_response')
    full_response = item.get('full_response')
    score_lst = []
    
    score = reward_fn(solution_str=full_response, predict_answer=predict_answer, ground_truth=ground_truth, l_limit=lower_limit, u_limit=upper_limit, data_source=data_source, metric=metric)
            
    score_lst.append(score)
    print(score_lst)
    max_score = np.max(score_lst)
    total_scores.append(score_lst)
    if max_score != 0:
        passes += max_score

pass_at_1 = passes / total

print(f"处理完毕，pass_at_1={pass_at_1}")

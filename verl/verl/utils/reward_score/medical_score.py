import re
from typing import Dict, Tuple, Optional
from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd, normalize_word
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import defaultdict
nltk.data.path.append("/home/xiaotian/EMNLP25/l1/verl/verl/utils/reward_score/punkt")


def get_delta_score_exact(num_tokens: int, used_tokens: int):
    # z_score = abs(used_tokens - num_tokens) / (num_tokens/2)
    z_score = abs(used_tokens - num_tokens) / (1500)
    
    delta_score = 1 - z_score
    return delta_score

def get_delta_score_max(num_tokens: int, used_tokens: int):
    alpha = 1/500
    beta = alpha

    delta = used_tokens - abs(num_tokens)
    sc = 0
    if delta < 0:
        sc = beta * delta * -1 # sc为正 值与 delta 的绝对值成正比
    else:
        sc = alpha * delta * -1 # sc为负 值与 delta 的绝对值成正比
    # Clip sc to [-1, 1]
    sc = max(-1, min(1, sc))
    return sc # 0-1

def compute_bleu(ref, pred):
    pred = pred if isinstance(pred, str) else ""

    ref_tokens = ref.split()
    pred_tokens = pred.split()

    weights = [(1, 0, 0, 0),  # 1-gram
               (0, 1, 0, 0),  # 2-gram
               (0, 0, 1, 0)]  # 3-gram

    bleu_scores = []
    smoothing = SmoothingFunction().method1

    for w in weights:
        score = sentence_bleu( [ref_tokens], pred_tokens, weights=w, smoothing_function=smoothing)
        bleu_scores.append(round(score, 4))
    
    print(f"[BLEU Scores] 1-gram: {bleu_scores[0]:.4f}, "
          f"2-gram: {bleu_scores[1]:.4f}, "
          f"3-gram: {bleu_scores[2]:.4f}")
    final_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2]) / 3
    return float(final_score)

def split_sentence(sentence, n):
    words = defaultdict(int)
    # tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def calculate_exactmatch(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
        
    if total == 0:
        return 0 # "0 (warning: length of candidate's words is 0)"
    else:
        return 100*(count / total)

def extract_solution(solution_str: str) -> Tuple[str, str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str: # base
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str: # qwen and tower
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def extract_solution_box(solution_str: str, check_think = True) -> Tuple[str, str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    problem = solution_str
    model_response = solution_str
    
    # Extract solution.
    if "<think>" in model_response and "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    elif "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    else:
        if check_think:
            print("[Error] Failed to locate model response header")
            return None, solution_str
        else:
            model_solution = model_response
        
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        print("[Error] No valid answer tags found")
        return None, model_solution

    # # Process the ground truth(s)
    # ground_truths = input.ground_truth.get("answer", None)
    # if ground_truths is None:
    #     return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

    return model_answer, model_solution

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def validate_response_structure_box(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ("<think>", 1),
        'think_end': ("</think>", 1),
        'answer': ('boxed', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(reward_metric: str,
                 data_source: str,
                 reward_type: str,
                 bleu_threshold: float,
                 solution_str: str, 
                 ground_truth: str,
                 l_limit: str,
                 u_limit: str,
                 num_tokens: int=-1,
                 valid_response_length: int=-1,
                 scale_factor: float = 100.0,
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
    print(" Processing Training Sample ".center(80, '='))
    

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    # Validate response structure
    if check_think:
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
    else:
        format_correct = answer_text != None
        format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    answer_score = 0
    if answer_text:
        # Validate answer content
        if data_source == "medical-o1-reasoning-SFT": # Huatuo
            match_score = calculate_exactmatch(answer_text, ground_truth)
            bleu_score = compute_bleu(ground_truth, answer_text)
            merge_score = 0.5*match_score + 0.5*bleu_score

            if reward_type == 'discrete':

                if reward_metric == 'BLEU':
                    if bleu_score > bleu_threshold:
                        answer_score = 1
                    else:
                        answer_score = -1

                elif reward_metric == 'EMS':
                    if match_score > bleu_threshold:
                        answer_score = 1
                    else:
                        answer_score = -1

                elif reward_metric == 'Merge':
                    if merge_score > bleu_threshold:
                        answer_score = 1
                    else:
                        answer_score = -1

            elif reward_type == 'continuous':

                if reward_metric == 'BLEU':
                    answer_score = float(bleu_score) / float(scale_factor)

                elif reward_metric == 'EMS':
                    answer_score = float(match_score) / float(scale_factor)

                elif reward_metric == 'Merge':
                    answer_score = float(merge_score) / float(scale_factor)
            else:
                raise ValueError("Invalid reward_type, please use discrete or continuous")
        
        elif data_source in ["EHRNoteQA", "MedIQ", "MedXpertQA"]: # just for acc
            allowed_letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'}
            if ground_truth in allowed_letters:
                if answer_text == ground_truth: 
                    answer_score=1
                else: 
                    answer_score=-1
        else: 
            raise ValueError("[Error] Invalid Data Source")

        # Compute number of words in solution_str
        if num_tokens != -1:
            if num_tokens < 0:
                delta_score = get_delta_score_max(num_tokens, float(valid_response_length))
                delta_score = max(-1, delta_score)  # [-1,1]
            else:
                delta_score = get_delta_score_exact(num_tokens, float(valid_response_length)) # []
                delta_score = max(-1, min(1, delta_score)) # [-1,1]
        else:
            delta_score = 0

        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Hypothesis: {answer_text}")
        # print(f"BLEU Score: {bleu_score}")
        
        answer_score = answer_score + delta_score
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score # [-3,3]

    print("\n" + "-"*80)
    print(f" Reward Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    # print(f"  Length: {delta_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score

def compute_score_calc(reward_metric: str,
                 data_source: str,
                 reward_type: str,
                 bleu_threshold: float,
                 solution_str: str, 
                 ground_truth: str,
                 l_limit: str,
                 u_limit: str,
                 num_tokens: int=-1,
                 valid_response_length: int=-1,
                 scale_factor: float = 100.0,
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
    print(" Processing Training Sample ".center(80, '='))
    

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    # Validate response structure
    if check_think:
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
    else:
        format_correct = answer_text != None
        format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    answer_score = 0
    if answer_text:
        # Validate answer content
        try:
            gt_value = float(ground_truth)
        except (ValueError, TypeError):
            if str(ground_truth) == str(answer_text):
                answer_score=1
            else:
                answer_score=-1
        else:
            try:
                ans_value = float(answer_text)
            except (ValueError, TypeError):
                answer_score=-1
            else:
                lower_bound = float(l_limit)
                upper_bound = float(u_limit)
                if lower_bound <= ans_value <= upper_bound:
                    answer_score=1
                else:
                    answer_score=-1

        # Compute number of words in solution_str
        if num_tokens != -1:
            if num_tokens < 0:
                delta_score = get_delta_score_max(num_tokens, float(valid_response_length))
                delta_score = max(-1, delta_score)  # [-1,1]
            else:
                delta_score = get_delta_score_exact(num_tokens, float(valid_response_length)) # []
                delta_score = max(-1, min(1, delta_score)) # [-1,1]
        else:
            delta_score = 0

        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Hypothesis: {answer_text}")
        
        answer_score = answer_score + delta_score
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score # [-3,3]

    print("\n" + "-"*80)
    print(f" Reward Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    # print(f"  Length: {delta_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score

def compute_score_val(
                 data_source: str,
                 solution_str: str, 
                 ground_truth: str,
                 bleu_threshold: float,
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
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")
    if answer_text:
        # Validate answer content
        if data_source == "medical-o1-reasoning-SFT": # Huatuo
            match_score = calculate_exactmatch(answer_text, ground_truth)
            bleu_score = compute_bleu(ground_truth, answer_text)
            merge_score = 0.5*match_score + 0.5*bleu_score
            if merge_score > bleu_threshold:
                pred_status = 1
            else:
                pred_status = 0
        
        elif data_source in ["EHRNoteQA", "MedIQ", "MedXpertQA"]: # just for acc
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

def compute_score_val_calc(
                 data_source: str,
                 solution_str: str, 
                 ground_truth: str,
                 bleu_threshold: float,
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
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")


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
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
from tabulate import tabulate

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.medical_utils import extract_answer, calculate_exactmatch, compute_bleu
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

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
    answer_text, processed_str = extract_answer(solution_str)
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
    answer_text, processed_str = extract_answer(solution_str)
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


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        local_path = copy_local_path_from_hdfs(config.model.path)
        from verl.utils import hf_tokenizer
        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        output_lst = []  # We'll reshape at the end

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            # Repeat the batch n_samples times
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            
            inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            
            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            
            # Generate all samples at once
            print(len(data.batch['input_ids']))
            output = wg.generate_sequences(data)
            # Remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)

            # Remove padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst.extend(output_text_unpad)

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        token_lengths = [len(tokenizer.encode(response)) for response in output_lst]
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()
        token_lengths = np.array(token_lengths).reshape(n_data, config.data.n_samples).tolist()
        # Add to the data frame
        dataset['responses'] = output_lst
        dataset['token_lengths'] = token_lengths


        print('Correlation between token length and num_tokens:')
        try:
            print(np.corrcoef(dataset['token_lengths'].apply(np.mean), dataset['reward_model'].apply(lambda x: x['num_tokens'])))
            print(np.corrcoef(dataset['token_lengths'].apply(lambda x: x[len(x)//2]), dataset['reward_model'].apply(lambda x: x['num_tokens'])))
        except Exception as e:
            print(e)


        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(config.data.output_path)
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    passes = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = _select_metric_score_fn(data_source)
        ground_truth = reward_data['ground_truth']
        lower_limit = reward_data['lower_limit']
        upper_limit = reward_data['upper_limit']
        score_lst = []
        for r in response_lst:
            score = reward_fn(solution_str=r, ground_truth=ground_truth, l_limit=lower_limit, u_limit=upper_limit, data_source=data_source, bleu_threshold = 45)
            
            score_lst.append(score)
        print(score_lst)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Save total_scores to a parquet
    try:
        total_scores_df = pd.DataFrame(total_scores)
        total_scores_df.to_parquet(os.path.join(output_dir, 'total_scores_{}.parquet'.format(dataset_name)))
    except Exception as e:
        print(e)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    elif data_source == 'gpqa' or data_source == 'lsat' or data_source == 'mmlu':
        from deepscaler.rewards.math_reward import gpqa_reward_fn
        return gpqa_reward_fn
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn

def _select_metric_score_fn(data_source):
    if data_source == "MedCalc-Bench":
        return compute_score_val_calc
    else:
        return compute_score_val

if __name__ == '__main__':
    main()

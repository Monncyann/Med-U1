"""Script to prepare training and test datasets.

This script processes medical problem datasets into a standardized format for training
and testing models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.`
"""

import argparse
import os
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
import sys
sys.path.append('/Med-U1')
from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset

import random
import copy

def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


# QUESTION_TEMPLATE = (
#         "{Question}\n"
#         "Please think about this question as if you were a human pondering deeply. "
#         "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
#         "It's encouraged to include self-reflection or verification in the reasoning process. "
#         "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
#     )

TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "arithmetical": " Please provide the numerical value (e.g., 42 or 3.14) or specific answer (e.g., 10/25/2020 or ('17 weeks', '6 days')) within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags."
    }

def make_map_fn(split: str, dataset_type: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')
        dataset_type: Dataset type ('HUATUO' or 'MEDIQ')

    Returns:
        Function that processes individual dataset examples
    """
    
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        data_source = example.pop('Source')
        lower_limit = example.get('Lower Limit') if 'Lower Limit' in example else None
        upper_limit = example.get('Upper Limit') if 'Upper Limit' in example else None
        # Choose a random number between 100 and 2000
        # Choose a floating number between 7 and 16
        if USE_LOG:
            random_float = random.uniform(8, 14.5)
            random_number = int(2**random_float)
        elif USE_NORMAL:
            random_number = -1
        else:
            if USE_BOTH:
                if random.random() < 0.5:
                    random_number = random.randint(100, 2000)
                else:
                    random_number = -1
            elif USE_BOTH_BOTH:
                random_number = random.randint(100, 2000)
                if random.random() < 0.5:
                    random_number = -1 * random_number
            else:
                random_number = random.randint(100, 2000) # default is token constrants
        
        if NUM_TOKENS != -1:
            if NUM_TOKENS < 0:
                instruction = f"Think for maximum {abs(NUM_TOKENS)} tokens."
            else:
                instruction = f"Think for {NUM_TOKENS} tokens."
        else:
            if random_number != -1:
                if random_number < 0:
                    instruction = f"Think for maximum {abs(random_number)} tokens."
                else:
                    instruction = f"Think for {random_number} tokens."
            else:
                instruction = ""
        
        # Add type-specific instruction based on dataset type
        if dataset_type in ["HUATUO", "HUATUO_TEST"] :
            type_instruction = TYPE_TEMPLATE["free-form"]
        elif dataset_type in ["MEDIQ", "MEDIQ_TEST", "MEDXPERTQA", "MEDXPERTQA_TEST", "EHRNOTEQA", "EHRNOTEQA_TEST", "MEDOPTION", "MEDOPTION_TEST", "MEDRL"]:
            type_instruction = TYPE_TEMPLATE["multiple choice"]
        elif dataset_type in ["MEDCALC", "MEDCALC_TEST"]:
            type_instruction = TYPE_TEMPLATE["arithmetical"]
            
        question = f"{question} {type_instruction} {instruction}"

        messages = [
        {"role": "system", "content": f"You are a helpful medical assistant. Please consider the provided question, as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning within the <think> </think> tags and final answer within the <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> final answer here </answer>."},
        {"role": "user", "content": question}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prefix = text

        answer = example.pop('answer')

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "medical",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
                "num_tokens": random_number if NUM_TOKENS == -1 else NUM_TOKENS,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('/Med-U1/scripts/data/processed_data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    parser.add_argument('--tokenizer_path', type=str, default='/Med-U1/model/Qwen2.5-3B-Instruct', 
                        help='Path to the tokenizer')
    parser.add_argument('--num_tokens', default=-1, type=int,
                       help='Number of tokens to think for')
    parser.add_argument('--use_log', default=False, action='store_true',
                       help='Use log scale for number of tokens')
    parser.add_argument('--use_both', default=False, action='store_true',
                       help='Use both normal and token constraints')
    parser.add_argument('--use_both_both', default=False, action='store_true',
                       help='Use both max budget and token constraints')
    parser.add_argument('--do_normal', default=False, action='store_true',
                       help='Do normal prompt, without any thinking limit') 
    parser.add_argument('--sample_multiple_for_train', default=1, type=int,
                       help='Sample multiple examples for train')
    
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    NUM_TOKENS = args.num_tokens
    USE_LOG = args.use_log
    USE_BOTH = args.use_both
    USE_BOTH_BOTH = args.use_both_both
    USE_NORMAL = args.do_normal
    if USE_NORMAL:
        assert not USE_LOG and not USE_BOTH and not USE_BOTH_BOTH and NUM_TOKENS == -1
    SAMPLE_MULTIPLE_FOR_TRAIN = args.sample_multiple_for_train
    if NUM_TOKENS != -1:
        local_dir = local_dir+'_'+str(NUM_TOKENS)
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Initialize datasets
    train_datasets = [TrainDataset.MEDOPTION]
    train_dataset = load_dataset(train_datasets[0])
    test_datasets = [TestDataset.MEDCALC, TestDataset.MEDXPERTQA, TestDataset.MEDIQ, TestDataset.HUATUO, TestDataset.EHRNOTEQA] # xxx for validation during training, xxx_TEST for final test
    
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    # Get dataset type from the first train dataset
    dataset_type = train_datasets[0].name
    process_fn = make_map_fn('train', dataset_type)
    train_dataset_original = copy.deepcopy(train_dataset)
    for i in range(SAMPLE_MULTIPLE_FOR_TRAIN):
        train_dataset = copy.deepcopy(train_dataset_original)
        for idx, example in enumerate(train_dataset):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                train_data.append(processed_example)

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        dataset_type = test_dataset.name
        process_fn = make_map_fn('test', dataset_type)
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train_' + f'{dataset_name}.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
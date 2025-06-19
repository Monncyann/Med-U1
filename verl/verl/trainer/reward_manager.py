from verl.utils.reward_score import medical_score
from verl import DataProto
import torch

def _select_rm_score_fn(data_source):

    if data_source == "MedCalc-Bench":
        return medical_score.compute_score_calc
    else:
        return medical_score.compute_score


def _select_metric_score_fn(data_source):
    if data_source == "MedCalc-Bench":
        return medical_score.compute_score_val_calc
    else:
        return medical_score.compute_score_val
    


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = config.algorithm.reward_type
        self.reward_metric = config.algorithm.reward_metric
        assert self.reward_type in ['discrete', 'continuous'], "reward_type must be discrete or continue"
        assert self.reward_metric in ['BLEU', 'EMS', 'Merge'], "reward_metric must be BLEU or Model or Merge" 
        self.bleu_threshold = config.algorithm.bleu_threshold 
        self.scale_factor = config.algorithm.reward_continuous_scale
        self.check_think = config.algorithm.check_think

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth'] # for acc
            num_tokens = data_item.non_tensor_batch['reward_model']['num_tokens'] # for length

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            lower_limit = data_item.non_tensor_batch['reward_model']['lower_limit']
            upper_limit = data_item.non_tensor_batch['reward_model']['upper_limit']
            compute_score_fn = _select_rm_score_fn(data_source)
            # if 'comet_rm' in data_item.batch.keys():
            #     metric_score = float(data_item.batch['comet_rm'])
            # elif 'comet_free_rm' in data_item.batch.keys():
            #     metric_score = float(data_item.batch['comet_free_rm'])
            # else:
            #     metric_score = None
            #     print("No model-based metric score found, use BLEU")
            # lg_pair = data_item.non_tensor_batch['lg']

            score = compute_score_fn(reward_type = self.reward_type, bleu_threshold = self.bleu_threshold, data_source=data_source, reward_metric = self.reward_metric,\
                solution_str = sequences_str, ground_truth=ground_truth, num_tokens = num_tokens, valid_response_length=valid_response_length, \
                scale_factor = self.scale_factor, check_think = self.check_think, l_limit = lower_limit, u_limit = upper_limit)

            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


class ValidManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.bleu_threshold = config.algorithm.bleu_threshold 

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            
            lower_limit = data_item.non_tensor_batch['reward_model']['lower_limit']
            upper_limit = data_item.non_tensor_batch['reward_model']['upper_limit']
            compute_score_fn = _select_metric_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, l_limit=lower_limit, u_limit=upper_limit, data_source=data_source, bleu_threshold = self.bleu_threshold)
            reward_tensor[i, valid_response_length - 1] = score

            if "valid_comet_metric" in data_item.batch.keys():
                print("valid_comet_metric: ", float(data_item.batch['valid_comet_metric']))
            if "valid_comet_free_metric" in data_item.batch.keys():
                print("valid_comet_free_metric: ", float(data_item.batch['valid_comet_free_metric']))
            print("="*80 + "\n")


        return reward_tensor

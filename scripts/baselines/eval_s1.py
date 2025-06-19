import os
import sys
import random

MAX_TOKENS_THINKING = int(sys.argv[1])
DSET = sys.argv[2]

if os.path.exists(f's1_results_new/{DSET}_{MAX_TOKENS_THINKING}.csv'):
    print(f's1_results_new/{DSET}_{MAX_TOKENS_THINKING}.csv already exists')
    exit()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from deepscaler.rewards.math_reward import deepscaler_reward_fn
from deepscaler.rewards.math_utils import extract_answer

NUM_IGNORE = 100 # We set this to 100 so as to ensure, exact adherence to length.
# S1 paper has both modes for extended thinking, and stopping early, so other variations are possible.

model = LLM(
    "agentica-org/DeepScaleR-1.5B-Preview", #
    tensor_parallel_size=1,
)
print('Done Loading Model')

from verl.utils import hf_tokenizer
tok = hf_tokenizer('agentica-org/DeepScaleR-1.5B-Preview')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

stop_token_ids = tok("<|im_end|>")["input_ids"]

import pandas as pd
df = pd.read_parquet(f'deepscaler/data/{DSET}.parquet')

prompts = [df['prompt'][i][0]['content'] for i in range(len(df))]



all_responses = []

token_usage = []
corrects = []
for idx, p in enumerate(prompts):
    prompt = "<｜begin▁of▁sentence｜><｜User｜>\n" + p + "<｜Assistant｜><think>\n"
    stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
    total_generated_tokens = 0
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_THINKING,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.6,
        seed=random.randint(0, 2**32-1),
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params
    )
    total_generated_tokens += len(o[0].outputs[0].token_ids)
    ignore_str = "Wait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    flag = False
    if max_tokens_thinking_tmp > 0:
        for i in range(NUM_IGNORE): # Num of times to skip stop token
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp <= 0:
                break
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.6,
                seed=random.randint(0, 2**32-1),
            )
            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )
            total_generated_tokens += len(o[0].outputs[0].token_ids)
    prompt += o[0].outputs[0].text 
    prompt += "\n<｜User｜>The final Answer is:"
    stop_token_ids = tok("\n")["input_ids"][1:]

    sampling_params = SamplingParams(
        max_tokens=20,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )
    total_generated_tokens += len(o[0].outputs[0].token_ids)

    print(deepscaler_reward_fn(prompt + o[0].outputs[0].text.replace('</think>',''), df['reward_model'][idx]['ground_truth'], ignore_think_token=True), extract_answer(o[0].outputs[0].text), df['reward_model'][idx]['ground_truth'], idx, o[0].outputs[0].text, total_generated_tokens)
    token_usage.append(total_generated_tokens)
    corrects.append(deepscaler_reward_fn(prompt + o[0].outputs[0].text.replace('</think>',''), df['reward_model'][idx]['ground_truth'], ignore_think_token=True))
    all_responses.append(prompt + o[0].outputs[0].text)


complete_data = {
    'all_responses': all_responses,
    'token_usage': token_usage,
    'corrects': corrects
}

pd.DataFrame(complete_data).to_csv(f's1_results_new/{DSET}_{MAX_TOKENS_THINKING}.csv', index=False)

print(token_usage, sum(token_usage)/len(token_usage), corrects, sum(corrects)/len(corrects))
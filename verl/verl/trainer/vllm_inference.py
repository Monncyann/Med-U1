# SPDX-License-Identifier: Apache-2.0

import os
import json
import glob
import argparse
import re
from typing import Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# 语言代码到语言名称映射表
language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "arithmetical": " Please provide the numerical value (e.g., 42 or 3.14) or specific answer (e.g., 10/25/2020 or ('17 weeks', '6 days')) within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags."
    }

def load_jsonl_file(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_json_file(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_prefix(example, template_type, tokenizer):
    """
    动态生成提示文本
    """
    lg = example.get('lg', '')
    source_lang, target_lang = lg.split('-') if '-' in lg else ('unknown', 'unknown')

    src_lang_name = language_map.get(source_lang, source_lang.capitalize())
    tgt_lang_name = language_map.get(target_lang, target_lang.capitalize())

    user_input = example.get(source_lang, "")
    
    
    if template_type == 'base':
        prefix = f"""You are a helpful medical assistant. There is a conversation between User and Assistant. The user asks a medical question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with just the final answer. The reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> final answer here </answer>." \n\nUser:{user_input}\nAssistant:"""
        return prefix
    elif template_type == 'chat':
        messages = [
        {"role": "system", "content": f"You are a helpful medical assistant. There is a conversation between User and Assistant. The user asks a medical question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with just the final answer. The reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> final answer here </answer>."},
        {"role": "user", "content": user_input}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text + "<think>"
    elif template_type == 'rl':
        prefix = f"""A conversation between User and Assistant. The User asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant translates it. The final translation are enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>. \n\nUser:{user_input}\nAssistant:"""    
        return prefix

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
        answer_pattern = r'</answer>(.*?)</answer>'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def process_data(input_data, llm, sampling_params, tokenizer, template_type):
    """处理数据并返回模型输出"""
    prompts = []
    original_data = []
    
    # 准备提示文本
    # for item in input_data:
    #     # prompt = make_prefix(item, template_type, tokenizer)
    #     prompt = item
    #     prompts.append(prompt)
    #     original_data.append(item)
    # print(f"prompt:{prompts[0]}")

    for item in input_data:
        prompt_content = item["prompt"][0]["content"]
        # prompt_content = item["prompt"]
        prompts.append(prompt_content)
        original_data.append(item)
    print(f"prompt:{prompts[0]}")

    # 批量获取模型响应
    outputs = llm.generate(prompts, sampling_params)
    
    # 准备结果
    results = []
    for i, output in enumerate(outputs):
        item = original_data[i]
        source = item["data_source"]
        ground_truth = item["reward_model"]["ground_truth"]
        lower_limit = item["reward_model"]["lower_limit"]
        upper_limit = item["reward_model"]["upper_limit"]
        generated_text = output.outputs[0].text
        # lg = item.get('lg', '')
        # source_lang, target_lang = lg.split('-') if '-' in lg else ('unknown', 'unknown')

        # 提取最终结果
        extracted_answer, processed_str = extract_solution(prompts[i] + generated_text)
        if i == 0:
            print(f"full_response:{processed_str}")

        result = {
            'id': i,
            'data_source':source,
            'ground_truth':ground_truth,
            'lower_limit':lower_limit,
            'upper_limit':upper_limit,
            'generated_response': extracted_answer if extracted_answer else processed_str,
            'full_response': processed_str
        }
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用VLLM进行批量翻译推理")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--tensor-parallel-size", type=int, default=8, help="张量并行大小，默认使用8卡")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU内存利用率")
    parser.add_argument("--max-model-len", type=int, default=8192, help="最大模型长度")
    
    # 采样参数
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p值")
    parser.add_argument("--max-tokens", type=int, default=2048, help="生成的最大token数")
    
    # 输入输出参数
    parser.add_argument("--input", type=str, required=True, help="输入JSONL文件或包含JSONL文件的文件夹")
    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=16, help="批处理大小")
    
    # 模板类型
    parser.add_argument("--template-type", type=str, default="base", 
                        choices=["base", "chat", "rl"], 
                        help="提示模板类型: base, chat, 或 rl")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化LLM和Tokenizer
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # 加载tokenizer以处理chat模板
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens = False
    )
    
    # 确定输入文件列表
    input_files = []
    if os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, "*.jsonl"))
    else:
        input_files = [args.input]
    
    print(f"找到 {len(input_files)} 个JSONL文件进行处理")
    
    # 处理每个文件
    for input_file in input_files:
        print(f"处理文件: {input_file}")
        base_name = os.path.basename(input_file)
        # 将扩展名改为.json
        output_file = os.path.join(args.output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
        
        # 加载数据
        data = load_jsonl_file(input_file)
        print(f"加载了 {len(data)} 条数据")
        
        # 批量处理数据
        all_results = []
        for i in tqdm(range(0, len(data), args.batch_size), desc="Processing batches"):
            batch = data[i:i+args.batch_size]
            batch_results = process_data(batch, llm, sampling_params, tokenizer, args.template_type)
            all_results.extend(batch_results)
        
        # 保存结果为JSON格式
        save_json_file(all_results, output_file)
        print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    main()

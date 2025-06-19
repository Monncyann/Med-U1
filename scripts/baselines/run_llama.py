import argparse
import json
import multiprocessing
from functools import partial

import openai
import pandas as pd
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="aime", choices=["math", "aime", "amc", "olympiad_bench", "aime2025", "all"]
)
args = parser.parse_args()

if args.dataset == "all":
    datasets = ["math", "aime2025", "amc", "olympiad_bench"]
else:
    datasets = [args.dataset]


if __name__ == "__main__":

    for dataset in datasets:
        df = pd.read_parquet(f"~/deepscaler/data/{dataset}.parquet")


        def process_item(item):
            """Process a single item from the dataframe."""
            # Initialize the client inside the function
            client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")
            
            # Modify prompt by removing "Think for" part
            item[0]["content"] = item[0]["content"][
                : item[0]["content"].rfind("Think for")
            ].strip()
            
            # Call the API
            response = client.chat.completions.create(
                model="default",
                messages=item,
                temperature=0.6,
                n=16,
                max_tokens=8192
            )
            
            return [choice.message.content for choice in response.choices]


        # Use multiprocessing to process items in parallel
        # Extract prompts from dataframe
        prompts = df["prompt"].tolist()
        
        # Create a pool with 16 processes
        num_processes = 16
        pool = multiprocessing.Pool(processes=num_processes)
        
        
        # Process items in parallel with a progress bar
        responses = list(tqdm(
            pool.imap(process_item, prompts),
            total=len(prompts),
            desc="Processing prompts"
        ))
        
        # Close the pool
        pool.close()
        pool.join()

        # Save responses
        os.makedirs(f"llama3.3-70b/", exist_ok=True)
        with open(f"llama3.3-70b/{dataset}.json", "w") as f:
            json.dump(responses, f, indent=4)

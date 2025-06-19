import argparse
import json
import multiprocessing
from functools import partial
import os

import openai
import pandas as pd
from tqdm import tqdm
import os
import pickle
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="aime", choices=["math", "aime", "amc", "olympiad_bench", 'aime2025', "all"]
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
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            
            # Modify prompt by removing "Think for" part
            item[0]["content"] = item[0]["content"][
                : item[0]["content"].rfind("Think for")
            ].strip()
            
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=item,
                temperature=0.0,
                n=1,
                max_tokens=8192
            )
            
            return [(choice.message.content, response.usage) for choice in response.choices]


        # Use multiprocessing to process items in parallel
        # Extract prompts from dataframe
        prompts = df["prompt"].tolist()
        
        # Create a pool with 16 processes
        num_processes = 8
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
        os.makedirs(f"gpt4o-zero/", exist_ok=True)
        try:
            with open(f"gpt4o-zero/{dataset}.json", "w") as f:
                json.dump(responses, f, indent=4)
        except Exception as e:
            with open(f"gpt4o-zero/{dataset}.pkl", "wb") as f:
                pickle.dump(responses, f)
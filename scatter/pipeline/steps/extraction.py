import os
import json
from tqdm import tqdm
import pandas as pd
from langchain.chat_models import ChatOpenAI
from utils import messages, update_progress, get_openai_chat_client
import concurrent.futures
from azure.core.exceptions import HttpResponseError 

def extraction(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/args.csv"
    comments = pd.read_csv(f"inputs/{config['input']}.csv")
    model = config['extraction']['model']
    prompt = config['extraction']['prompt']
    workers = config['extraction']['workers']
    limit = config['extraction']['limit']

    comment_ids = (comments['comment-id'].values)[:limit]
    comments.set_index('comment-id', inplace=True)
    results = pd.DataFrame()
    update_progress(config, total=len(comment_ids))
    for i in tqdm(range(0, len(comment_ids), workers)):
        batch = comment_ids[i: i + workers]
        batch_inputs = [comments.loc[id]['comment-body'] for id in batch]
        batch_results = extract_batch(batch_inputs, prompt, model, workers)
        for comment_id, extracted_args in zip(batch, batch_results):
            for j, arg in enumerate(extracted_args):
                new_row = {"arg-id": f"A{comment_id}_{j}",
                           "comment-id": int(comment_id), "argument": arg}
                results = pd.concat(
                    [results, pd.DataFrame([new_row])], ignore_index=True)
        update_progress(config, incr=len(batch))
    results.to_csv(path, index=False)


def extract_batch(batch, prompt, model, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(
            extract_arguments, input, prompt, model) for input in list(batch)]
        concurrent.futures.wait(futures)
        return [future.result() for future in futures]


def extract_arguments(input, prompt, model, retries=3):
    # llm = ChatOpenAI(model_name=model, temperature=0.0)
    # endpoint = os.getenv("endpoint")
    # credential = os.getenv("credential")
    # model_name = model["name"]
    # api_version = model["api_version"]
    # print('---endpoint---', endpoint)
    # print('---credential---', credential)
    # print('---model_name---', model_name)
    # print('---api_version---', api_version)
    # print('---prompt---', prompt)
    print('---messages(prompt, input)---', messages(prompt, input))
    print('---messages(prompt, input)--type-', type(messages(prompt, input)))
    try:
        llm = get_openai_chat_client(model)
        response = llm.complete(messages=messages(prompt, input)).choices[0].message.content.strip()
        try:
            obj = json.loads(response)
            # LLM sometimes returns valid JSON string
            if isinstance(obj, str):
                obj = [obj]
            items = [a.strip() for a in obj]
            items = filter(None, items)  # omit empty strings
            return items
        except json.decoder.JSONDecodeError as e:
            print("JSON error:", e)
            print("Input was:", input)
            print("Response was:", response)
            if retries > 0:
                print("Retrying...")
                return extract_arguments(input, prompt, model, retries - 1)
            else:
                print("Silently giving up on trying to generate valid list.")
                return []
    except HttpResponseError as e:
        if "content_filter" in str(e):
            print(f"Content filter triggered, skipping this: {str(e)}")
            return []  
        else:
            print(f"HTTP error occurred: {str(e)}")
            if retries > 0:
                print("Retrying due to HTTP error...")
                return extract_arguments(input, prompt, model, retries - 1)
            else:
                print("Max retries reached for HTTP error.")
                return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if retries > 0:
            print("Retrying...")
            return extract_arguments(input, prompt, model, retries - 1)
        else:
            print("Max retries reached.")
            return []
        
def get_azure_client1(model):
    endpoint = os.getenv("endpoint")
    credential = os.getenv("credential")
    model_name = model["name"]
    api_version = model["api_version"]
    if not all([endpoint, credential]):
        raise ValueError("Missing required environment variables: endpoint or credential")
    print(f"Using endpoint {endpoint} and credential {credential}")
    print(f"Using model {model_name} with api version {api_version}")
    return ChatCompletionsClient(
        endpoint=f"{endpoint}/models/deployments/{model_name}",
        credential=AzureKeyCredential(credential),
        api_version=api_version,
    )

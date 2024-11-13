import json
import os
import pickle
import requests
from typing import List, Dict
from schema import CodeTree

PICKLE_FILE_PATH = 'local-db/code_tree_responses.pkl'

def load_responses() -> Dict[str, Dict[str, CodeTree]]:
    """Load responses from the pickle file."""
    if os.path.exists(PICKLE_FILE_PATH):
        with open(PICKLE_FILE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_responses(responses: Dict[str, Dict[str, CodeTree]]):
    """Save responses to the pickle file."""
    os.makedirs('local-db', exist_ok=True)
    with open(PICKLE_FILE_PATH, 'wb') as f:
        pickle.dump(responses, f)

def generate_code_tree(file_path: str, content: str, modified_lines: List[int]) -> Dict[str, CodeTree]:
    """Generate a code tree for a file with modified lines."""

    responses = load_responses()
    
    # Check if the response for the given file path is already stored
    if file_path in responses:
        return responses[file_path]
    
    url = "https://production-gateway.snorkell.ai/api/v1/hook/file/generate/codetree"
    data = {
        "file_path": file_path,
        "content": content,
        "modified_lines": modified_lines
    }

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'api-key': 'skl_ai_gQs0G76hUSiCK8Uk'
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        # Store the response in the dictionary and save it to the pickle file
        responses[file_path] = response_data
        save_responses(responses)
        
        return response_data
    except Exception as e:
        print(e)
        print(f"Error in generating code tree for file - {file_path}")
        return {}
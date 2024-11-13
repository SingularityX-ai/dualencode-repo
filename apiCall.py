import json
import os
import pickle
import requests
from typing import List, Dict
from schema import CodeTree

PICKLE_FILE_PATH = 'local-db/code_tree_responses.pkl'

def load_responses() -> Dict[str, Dict[str, CodeTree]]:
    """Load responses from a pickle file.

    This function checks if a specified pickle file exists. If the file is
    found, it opens the file in binary read mode and loads the contents
    using the pickle module. The loaded data is expected to be a dictionary
    containing nested dictionaries. If the file does not exist, an empty
    dictionary is returned.

    Returns:
        Dict[str, Dict[str, CodeTree]]: A dictionary containing the loaded responses from the pickle file, or an
            empty dictionary if the file does not exist.
    """
    if os.path.exists(PICKLE_FILE_PATH):
        with open(PICKLE_FILE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_responses(responses: Dict[str, Dict[str, CodeTree]]):
    """Save responses to a pickle file.

    This function creates a directory named 'local-db' if it does not
    already exist, and then saves the provided responses dictionary to a
    specified pickle file. The responses are serialized using the pickle
    module, allowing for easy storage and retrieval of complex data
    structures.

    Args:
        responses (Dict[str, Dict[str, CodeTree]]): A dictionary containing responses
    """
    os.makedirs('local-db', exist_ok=True)
    with open(PICKLE_FILE_PATH, 'wb') as f:
        pickle.dump(responses, f)

def generate_code_tree(file_path: str, content: str, modified_lines: List[int]) -> Dict[str, CodeTree]:
    """Generate a code tree for a file with modified lines.

    This function sends a request to an external API to generate a code tree
    based on the provided file path, content, and modified lines. It first
    checks if the response for the given file path is already stored. If it
    is, it returns the cached response. If not, it constructs a request to
    the API and processes the response. The generated code tree is then
    stored for future use.

    Args:
        file_path (str): The path of the file for which the code tree is generated.
        content (str): The content of the file.
        modified_lines (List[int]): A list of line numbers that have been modified.

    Returns:
        Dict[str, CodeTree]: A dictionary representing the generated code tree
        for the specified file.
    """

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
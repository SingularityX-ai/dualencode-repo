from typing import Any, Dict, List
import requests
import json

from schema import CodeTree




def generate_code_tree(file_path: str, content: str, modified_lines: List[int]) -> Dict[str, CodeTree]:
    """Generate a code tree for a file with modified lines.

    This function sends a POST request to a specified API endpoint to
    generate a code tree based on the provided file path, content, and
    modified lines. It constructs the necessary data and headers for the
    request and handles any exceptions that may occur during the process. If
    the request is successful, it returns the JSON response containing the
    code tree; otherwise, it returns an empty dictionary.

    Args:
        file_path (str): The path to the file for which the code tree is generated.
        content (str): The content of the file as a string.
        modified_lines (List[int]): A list of line numbers that have been modified.

    Returns:
        Dict[str, CodeTree]: A dictionary representing the generated code tree.
    """

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
        return response.json()
    except Exception as e:
        print(e)
        print(f"Error in generating code tree for file - {file_path}")
        return {}

from typing import Any, Dict, List
import requests
import json

from schema import CodeTree




def generate_code_tree(file_path: str, content: str, modified_lines: List[int]) -> Dict[str, CodeTree]:
    """Generate a code tree for a file with modified lines."""

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

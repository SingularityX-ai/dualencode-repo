import os
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import ast
from pathlib import Path
import glob
import json
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class CodeAnalysisResult(NamedTuple):
    code_similarity: float
    doc_similarity: float
    combined_similarity: float
    function: 'CodeFunction'

@dataclass
class CodeFunction:
    name: str
    code: str
    documentation: str
    file_path: str
    code_embedding: Optional[np.ndarray] = None
    doc_embedding: Optional[np.ndarray] = None

class DualEncoder:
    def __init__(
        self,
        code_model: str = "microsoft/codebert-base",
        doc_model: str = "e5-large-v2",
        device: str = None,
        code_weight: float = 0.5  # Weight for combining similarities
    ):
        # Auto-detect device if none specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.code_encoder = SentenceTransformer(code_model, device=device)
        self.doc_encoder = SentenceTransformer(doc_model, device=device)
        self.code_weight = code_weight
        self.doc_weight = 1 - code_weight
        self.functions: List[CodeFunction] = []
        
    def parse_python_file(self, file_path: str) -> List[CodeFunction]:
        """Parse a Python file and extract functions with their documentation."""
        with open(file_path, 'r') as file:
            content = file.read()
            
        tree = ast.parse(content)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function code
                code_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                code = '\n'.join(code_lines)
                
                # Extract docstring and comments
                docstring = ast.get_docstring(node) or ''
                
                # Extract inline comments
                comments = []
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        line = code_lines[child.lineno - node.lineno]
                        if '#' in line:
                            comments.append(line[line.index('#')+1:].strip())
                
                all_documentation = docstring + '\n' + '\n'.join(comments)
                
                functions.append(CodeFunction(
                    name=node.name,
                    code=code,
                    documentation=all_documentation,
                    file_path=file_path
                ))
                
        return functions

    def load_documentation(self, docs_path: str) -> Dict[str, str]:
        """Load external documentation from a directory."""
        docs = {}
        doc_files = glob.glob(os.path.join(docs_path, "**/*.txt"), recursive=True)
        
        for doc_file in doc_files:
            function_name = Path(doc_file).stem
            with open(doc_file, 'r') as f:
                docs[function_name] = f.read().strip()
                
        return docs

    def preprocess_code(self, code: str) -> str:
        """Preprocess code for better embedding."""
        # Remove comments
        tree = ast.parse(code)
        return ast.unparse(tree)

    def encode_batch(
        self,
        texts: List[str],
        encoder: SentenceTransformer,
        batch_size: int = 8
    ) -> np.ndarray:
        """Encode texts in batches."""
        return encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def index_repository(self, repo_path: str, docs_path: str):
        """Index all Python files using both encoders."""
        python_files = glob.glob(os.path.join(repo_path, "**/*.py"), recursive=True)
        external_docs = self.load_documentation(docs_path)
        
        # Collect all texts to encode
        codes_to_encode = []
        docs_to_encode = []
        temp_functions = []
        
        for file_path in python_files:
            functions = self.parse_python_file(file_path)
            
            for func in functions:
                # Prepare code for embedding
                processed_code = self.preprocess_code(func.code)
                codes_to_encode.append(processed_code)
                
                # Combine all documentation
                combined_doc = func.documentation
                if func.name in external_docs:
                    combined_doc += "\n" + external_docs[func.name]
                docs_to_encode.append(combined_doc)
                
                temp_functions.append(func)
        
        # Batch encode everything
        code_embeddings = self.encode_batch(codes_to_encode, self.code_encoder)
        doc_embeddings = self.encode_batch(docs_to_encode, self.doc_encoder)
        
        # Assign embeddings to functions
        for func, code_emb, doc_emb in zip(temp_functions, code_embeddings, doc_embeddings):
            func.code_embedding = code_emb
            func.doc_embedding = doc_emb
            self.functions.append(func)

    def search(
        self,
        query: str,
        search_code: bool = True,
        search_docs: bool = True,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[CodeAnalysisResult]:
        """
        Search for similar functions using both code and documentation embeddings.
        
        Args:
            query: Search query
            search_code: Whether to search in code
            search_docs: Whether to search in documentation
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
        """
        # Encode query with both encoders
        code_query = self.code_encoder.encode(query, convert_to_numpy=True)
        doc_query = self.doc_encoder.encode(query, convert_to_numpy=True)
        
        results = []
        
        for func in self.functions:
            code_sim = 0.0
            doc_sim = 0.0
            
            if search_code:
                code_sim = cosine_similarity(
                    code_query.reshape(1, -1),
                    func.code_embedding.reshape(1, -1)
                )[0][0]
                
            if search_docs:
                doc_sim = cosine_similarity(
                    doc_query.reshape(1, -1),
                    func.doc_embedding.reshape(1, -1)
                )[0][0]
            
            # Calculate combined similarity
            if search_code and search_docs:
                combined_sim = (code_sim * self.code_weight + 
                              doc_sim * self.doc_weight)
            elif search_code:
                combined_sim = code_sim
            else:
                combined_sim = doc_sim
            
            if combined_sim >= min_similarity:
                results.append(CodeAnalysisResult(
                    code_similarity=code_sim,
                    doc_similarity=doc_sim,
                    combined_similarity=combined_sim,
                    function=func
                ))
        
        # Sort by combined similarity
        results.sort(key=lambda x: x.combined_similarity, reverse=True)
        return results[:top_k]

    def save_index(self, output_path: str):
        """Save the indexed functions to disk."""
        data = [{
            'name': func.name,
            'code': func.code,
            'documentation': func.documentation,
            'file_path': func.file_path,
            'code_embedding': func.code_embedding.tolist(),
            'doc_embedding': func.doc_embedding.tolist()
        } for func in self.functions]
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
    
    def load_index(self, input_path: str):
        """Load previously indexed functions."""
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        self.functions = [
            CodeFunction(
                name=item['name'],
                code=item['code'],
                documentation=item['documentation'],
                file_path=item['file_path'],
                code_embedding=np.array(item['code_embedding']),
                doc_embedding=np.array(item['doc_embedding'])
            )
            for item in data
        ]
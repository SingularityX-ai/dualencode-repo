import os
from pprint import pprint
import re
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import ast
from pathlib import Path
import glob
import json
from enum import Enum
from transformers import AutoTokenizer, AutoModel


import numpy as np
from apiCall import generate_code_tree
from schema import CodeTree
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class EnhancedCodeAnalysisResult(NamedTuple):
    code_similarity: float
    doc_similarity: float
    semantic_similarity: float
    combined_similarity: float
    function: 'CodeFunction'

class SemanticAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get semantic embeddings for a list of texts"""
        # Tokenize and prepare input
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()

    def calculate_semantic_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Calculate semantic similarity between query and document embeddings"""
        return np.dot(query_embedding[0], doc_embedding[0])

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
        doc_model: str = "intfloat/e5-large-v2",
        semantic_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
        code_weight: float = 0.5  # Weight for combining similarities
    ):
        # Auto-detect device if none specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
            
        self.code_encoder = SentenceTransformer(code_model, device=device)
        self.doc_encoder = SentenceTransformer(doc_model, device=device)
        self.semantic_analyzer = SemanticAnalyzer(semantic_model)
        self.code_weight = code_weight
        self.doc_weight = 1 - code_weight
        self.functions: List[CodeFunction] = []

    def preprocess_docstring(self, docstring: str) -> str:
        """Clean and normalize docstring for semantic analysis"""
        if not docstring:
            return ""
            
        # Remove common docstring markers
        docstring = re.sub(r'Args:|Returns:|Raises:|Example[s]?:', ' ', docstring)
        
        # Clean up whitespace
        docstring = re.sub(r'\s+', ' ', docstring).strip()
        
        # Remove code blocks
        docstring = re.sub(r'```[\s\S]*?```', '', docstring)
        
        return docstring

    def preprocess_docstring(self, docstring: str) -> str:
        """Clean and normalize docstring for semantic analysis"""
        if not docstring:
            return ""
            
        # Remove common docstring markers
        docstring = re.sub(r'Args:|Returns:|Raises:|Example[s]?:', ' ', docstring)
        
        # Clean up whitespace
        docstring = re.sub(r'\s+', ' ', docstring).strip()
        
        # Remove code blocks
        docstring = re.sub(r'```[\s\S]*?```', '', docstring)
        
        return docstring
        

    def load_documentation(self, docs_path: str) -> Dict[str, str]:
        """Load external documentation from a directory."""
        docs = {}
        doc_files = glob.glob(os.path.join(docs_path, "**/*.py"), recursive=True)
        
        for doc_file in doc_files:
            function_name = Path(doc_file).stem
            with open(doc_file, 'r') as f:
                docs[function_name] = f.read().strip()
                
        return docs

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


    def index_repository(self, repo_path: str, docs_path: str, force_update: bool = False):
        """Index all Python files using both encoders."""


        # external_docs = self.load_documentation(docs_path)
        index_path = f"{repo_path}/function_index.json"
        if not force_update:
            # Check if index already exists
            print("checking if index exists")
            
            if os.path.exists(index_path):
                print("index exists")
                self.load_index(index_path)
                return

        python_files = glob.glob(os.path.join(repo_path, "**/*"), recursive=True)
        print("tot files - ", len(python_files))
        
        # filter ["py","ts","cs","c","js", "kt"]
        python_files = [file for file in python_files if file.split(".")[-1] in ["py","ts","cs","c","js", "kt"]]

        print("selected files - ", len(python_files))
        
        # Collect all texts to encode
        codes_to_encode = []
        docs_to_encode = []
        semantic_docs = []
        temp_functions: List[CodeFunction] = []
        
        count = 0
        print(1)
        for file_path in python_files:

            extension = file_path.split(".")[-1]
            if extension not in ["py","ts","cs","c","js", "kt"]:
                print(f"skipping extension - {extension}")
                continue


            # count += 1
            # if count > 2:
            #     break

            if "py_scripts" in file_path:
                continue

            if file_path.endswith("conftest.py"):
                continue
            with open(file_path, 'r') as file:
                code_str = file.read()

            code_tree = generate_code_tree(file_path, code_str, [])
            try:
                code_tree: CodeTree = CodeTree(**code_tree)
            except Exception as e:
                print("Error in parsing code tree")
                continue
            # pprint(code_tree)
            
            if code_tree.methods is not None:
                for func, func_dict in code_tree.methods.items():
                    # Prepare code for embedding
                    processed_code = func_dict.content
                    codes_to_encode.append(processed_code)
                    func_name = func
                    clean_docstring = self.preprocess_docstring(func_dict.docstring)
                    semantic_docs.append(clean_docstring)
                    
                    # Combine all documentation
                    combined_doc = f"MethodName: {func_name} \n{func_dict.docstring}"
                    
                    docs_to_encode.append(combined_doc)
                    temp_functions.append(CodeFunction(
                        name=func_name,
                        code=func_dict.content,
                        documentation=func_dict.docstring,
                        file_path=file_path
                    ))
            if code_tree.classes is not None:
                for class_details, class_dict in code_tree.classes.items():
                    for func, func_dict in class_dict.methods.items():
                        # Prepare code for embedding
                        processed_code = func_dict.content
                        codes_to_encode.append(processed_code)
                        func_name = func.split("~")[0]
                        
                        # Combine all documentation
                        combined_doc = f"MethodName: {func_name} \n{func_dict.docstring}"
                        
                        docs_to_encode.append(combined_doc)
                        temp_functions.append(CodeFunction(
                            name=func_name,
                            code=func_dict.content,
                            documentation=func_dict.docstring,
                            file_path=file_path
                        ))
            else:
                print("No class found in file - ", file_path)
            
                                    
        # Batch encode everything
        semantic_embeddings = self.semantic_analyzer.get_semantic_embeddings(semantic_docs)
        code_embeddings = self.encode_batch(codes_to_encode, self.code_encoder)
        doc_embeddings = self.encode_batch(docs_to_encode, self.doc_encoder)
        
        print(2)
        
        # Assign embeddings to functions
        for func, code_emb, doc_emb in zip(temp_functions, code_embeddings, doc_embeddings):
            func.code_embedding = code_emb
            func.doc_embedding = doc_emb
            self.functions.append(func)

        self.save_index(index_path)

    def search(
        self,
        query: str,
        search_code: bool = True,
        search_docs: bool = True,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[EnhancedCodeAnalysisResult]:
        """
        Enhanced search using semantic similarity of docstrings.
        """
        # Encode query
        code_query = self.code_encoder.encode(query, convert_to_numpy=True)
        doc_query = self.doc_encoder.encode(query, convert_to_numpy=True)
        semantic_query = self.semantic_analyzer.get_semantic_embeddings([self.preprocess_docstring(query)])
        
        # Normalize query vectors
        code_query = code_query / (np.linalg.norm(code_query) + 1e-8)
        doc_query = doc_query / (np.linalg.norm(doc_query) + 1e-8)
        
        results = []
        
        for func in self.functions:
            # Calculate embedding similarities
            code_sim = 0.0
            if search_code and func.code_embedding is not None:
                code_embedding = func.code_embedding / (np.linalg.norm(func.code_embedding) + 1e-8)
                code_sim = np.dot(code_query, code_embedding)
            
            doc_sim = 0.0
            if search_docs and func.doc_embedding is not None:
                doc_embedding = func.doc_embedding / (np.linalg.norm(func.doc_embedding) + 1e-8)
                doc_sim = np.dot(doc_query, doc_embedding)
            
            # Calculate semantic similarity
            semantic_sim = self.semantic_analyzer.calculate_semantic_similarity(
                semantic_query,
                func.semantic_embedding.reshape(1, -1)
            )
            
            # Calculate weighted combination
            combined_sim = (
                code_sim * self.weights['code_embedding'] +
                doc_sim * self.weights['doc_embedding'] +
                semantic_sim * self.weights['semantic']
            )
            
            if combined_sim >= min_similarity:
                results.append(EnhancedCodeAnalysisResult(
                    code_similarity=code_sim,
                    doc_similarity=doc_sim,
                    semantic_similarity=semantic_sim,
                    combined_similarity=combined_sim,
                    function=func
                ))
        
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
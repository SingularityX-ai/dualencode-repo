import os
from pprint import pprint
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import ast
from pathlib import Path
import glob
import json
from enum import Enum

import numpy as np
from apiCall import generate_code_tree
from schema import CodeTree
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
        doc_model: str = "intfloat/e5-large-v2",
        device: str = None,
        code_weight: float = 0.5  # Weight for combining similarities
    ):
        # Auto-detect device if none specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
            
        self.code_encoder = SentenceTransformer(code_model, device=device)
        self.doc_encoder = SentenceTransformer(doc_model, device=device)
        self.code_weight = code_weight
        self.doc_weight = 1 - code_weight
        self.functions: List[CodeFunction] = []
        

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
        """Encode a list of texts in batches using a specified encoder.

        This function takes a list of texts and encodes them in batches using
        the provided SentenceTransformer encoder. It allows for efficient
        processing of large datasets by splitting the input into smaller
        batches, which can help manage memory usage and improve performance. The
        encoded output is returned as a NumPy array.

        Args:
            texts (List[str]): A list of strings to be encoded.
            encoder (SentenceTransformer): The encoder used to transform the texts.
            batch_size (int?): The number of texts to process in each batch.
                Defaults to 8.

        Returns:
            np.ndarray: A NumPy array containing the encoded representations of the input texts.
        """
        return encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )


    def index_repository(self, repo_path: str, docs_path: str, force_update: bool = False):
        """Index all Python files using both encoders.

        This function scans a specified repository for Python and other
        specified file types, collects their content, and generates embeddings
        for both the code and associated documentation. If an index already
        exists and `force_update` is set to False, the function will load the
        existing index instead of re-indexing the files. The function handles
        various file types and ensures that only relevant files are processed.

        Args:
            repo_path (str): The path to the repository containing the files to be indexed.
            docs_path (str): The path to the documentation files (not currently used in this
                implementation).
            force_update (bool?): A flag indicating whether to force re-indexing of files.
                Defaults to False.
        """


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
    ) -> List[CodeAnalysisResult]:
        """Search for similar functions using both code and documentation
        embeddings.

        This function allows users to search for functions that are similar to a
        given query based on both code and documentation embeddings. It encodes
        the query using two separate encoders for code and documentation,
        normalizes the resulting vectors, and computes similarity scores against
        a list of functions. The results can be filtered based on minimum
        similarity thresholds and can prioritize either code, documentation, or
        both.

        Args:
            query (str): The search query to find similar functions.
            search_code (bool?): Whether to include code in the search. Defaults to True.
            search_docs (bool?): Whether to include documentation in the search. Defaults to True.
            top_k (int?): The number of top results to return. Defaults to 5.
            min_similarity (float?): The minimum similarity threshold for results. Defaults to 0.3.

        Returns:
            List[CodeAnalysisResult]: A list of CodeAnalysisResult objects containing the similarity scores
            and the corresponding functions that meet the criteria.
        """
        # Encode query with both encoders and normalize
        code_query = self.code_encoder.encode(query, convert_to_numpy=True)
        doc_query = self.doc_encoder.encode(query, convert_to_numpy=True)
        
        # Normalize query vectors
        code_query = code_query / (np.linalg.norm(code_query) + 1e-8)
        doc_query = doc_query / (np.linalg.norm(doc_query) + 1e-8)
        
        results: List[CodeAnalysisResult] = []
        
        for func in self.functions:
            code_sim = 0.0
            doc_sim = 0.0
            
            if search_code and func.code_embedding is not None:
                # Normalize code embedding
                code_embedding = func.code_embedding / (np.linalg.norm(func.code_embedding) + 1e-8)
                code_sim = np.dot(code_query, code_embedding)
                    
            if search_docs and func.doc_embedding is not None:
                # Normalize doc embedding
                doc_embedding = func.doc_embedding / (np.linalg.norm(func.doc_embedding) + 1e-8)
                doc_sim = np.dot(doc_query, doc_embedding)
            
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
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class MethodData(BaseModel):
    content: str
    called_functions: List[str] = Field(default=[], alias="calledFunctions")
    docstring: str
    fun_start_line: int = Field(..., alias="funStartLine")
    fun_end_line: int = Field(..., alias="funEndLine")
    doc_start_line: int = Field(..., alias="docStartLine")
    doc_end_line: int = Field(..., alias="docEndLine")
    is_modified: bool = Field(default=False, alias="isModified")
    annotations: List[str] = Field(default=[], alias="annotations")

class ClassData(BaseModel):
    methods: Dict[str, MethodData]
    content: str
    docstring: str
    class_start_line: int = Field(..., alias="classStartLine")
    class_end_line: int = Field(..., alias="classEndLine")
    doc_start_line: int = Field(..., alias="docStartLine")
    doc_end_line: int = Field(..., alias="docEndLine")
    is_modified: bool = Field(default=False, alias="isModified")
    annotations: List[str] = Field(default=[], alias="annotations")

    @classmethod
    def parse_function_dict(cls, raw_function_dict):
        """Parse the raw function dictionary into a new dictionary with tuple values.

        This function takes a raw function dictionary and converts it into a new dictionary where the values are tuples.

        Args:
            cls (class): The class instance.
            raw_function_dict (dict): A dictionary containing function information.

        Returns:
            dict: A new dictionary with tuple values.
        """

        # return {k: tuple(v) for k, v in raw_function_dict.items()}
        functionDict = {}
        for k, v in raw_function_dict.items():
            list = []
            for k1, v1 in v.items():
                list.append(v1)
            functionDict[k] = tuple(list)
        return functionDict


class CodeTree(BaseModel):
    # classes: Dict[str, ParsedKtClass]
    methods: Dict[str, MethodData]  # method name should be KEY and value should be MethodData
    classes: Optional[Dict[str, ClassData]] = None  # class name should be KEY and value should be ClassData
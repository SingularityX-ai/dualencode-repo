# Initialize the system
from dualEncoder import DualEncoder


encoder = DualEncoder(
    code_weight=0.6  # Give slightly more weight to code similarity
)

# Index your repository
repo = "/Users/sumansaurabh/Documents/singularityx/github/MoneyPrinterTurbo/app/"
repo = "/Users/sumansaurabh/Documents/singularityx/github/snorkell-backend/backend/"
repo = "/home/azureuser/localfiles/pokerogue"

encoder.index_repository(
    repo_path=repo,
    docs_path=repo
)

# Search with different focuses
# 1. Search everything

message = "Increment the turn in the battle scene"
results = encoder.search(
    message,
    search_code=True,
    search_docs=True
)

# 2. Search only code semantics
code_results = encoder.search(
    message,
    search_code=True,
    search_docs=False
)

# 3. Search only documentation
doc_results = encoder.search(
    message,
    search_code=False,
    search_docs=True
)

# Process results
for result in results:
    print(f"Function: {result.function.name}")
    print(f"File Path: {result.function.file_path}")
    print(f"Docstring: {result.function.documentation}")
    print(f"Code Similarity: {result.code_similarity:.2f}")
    print(f"Doc Similarity: {result.doc_similarity:.2f}")
    print(f"Combined: {result.combined_similarity:.2f}")
    print("\n\n")
    
    

# # code results
# for result in code_results:
#     print(f"code: Function: {result.function.name}")
#     print(f"code: Code Similarity: {result.code_similarity:.2f}")
#     print(f"code: Doc Similarity: {result.doc_similarity:.2f}")
#     print(f"code: Combined: {result.combined_similarity:.2f}")

# # doc results
# for result in doc_results:
#     print(f"doc: Function: {result.function.name}")
#     print(f"doc: Code Similarity: {result.code_similarity:.2f}")
#     print(f"doc: Doc Similarity: {result.doc_similarity:.2f}")
#     print(f"doc: Combined: {result.combined_similarity:.2f}")
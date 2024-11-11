# Initialize the system
from dualEncoder import DualEncoder


encoder = DualEncoder(
    code_weight=0.6  # Give slightly more weight to code similarity
)

# Index your repository
encoder.index_repository(
    repo_path="path/to/code",
    docs_path="path/to/documentation"
)

# Search with different focuses
# 1. Search everything
results = encoder.search(
    "handle user authentication",
    search_code=True,
    search_docs=True
)

# 2. Search only code semantics
code_results = encoder.search(
    "implement binary search",
    search_code=True,
    search_docs=False
)

# 3. Search only documentation
doc_results = encoder.search(
    "user authentication flow",
    search_code=False,
    search_docs=True
)

# Process results
for result in results:
    print(f"Function: {result.function.name}")
    print(f"Code Similarity: {result.code_similarity:.2f}")
    print(f"Doc Similarity: {result.doc_similarity:.2f}")
    print(f"Combined: {result.combined_similarity:.2f}")
# Initialize the dual encoder
from dualEncoder import DualEncoder
from mindmap import CodeMindMapGenerator


encoder = DualEncoder()
repo = "/Users/sumansaurabh/Documents/singularityx/github/MoneyPrinterTurbo/app/"
encoder.index_repository(repo, repo)

# Create mindmap generator
mindmap_gen = CodeMindMapGenerator(encoder)

# Generate Mermaid mindmap
mermaid_code = mindmap_gen.generate_mermaid_mindmap()

# Or generate interactive HTML visualization
html_code = mindmap_gen.generate_html_mindmap()

# Save the HTML visualization
with open("mindmap.html", "w") as f:
    f.write(html_code)
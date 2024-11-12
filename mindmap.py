from typing import List, Dict, Set, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import networkx as nx
from pathlib import Path
from torch import cosine_similarity
import torch

from dualEncoder import DualEncoder

class CodeMindMapGenerator:
    def __init__(self, dual_encoder: DualEncoder):
        self.encoder: DualEncoder = dual_encoder
        self.similarity_threshold = 0.6
        
    # def generate_similarity_graph(self) -> nx.Graph:
    #     """Generate a graph where nodes are functions and edges represent similarity."""
    #     G = nx.Graph()
        
    #     # Add all functions as nodes
    #     for idx, func in enumerate(self.encoder.functions):
    #         G.add_node(
    #             func.name,
    #             code=func.code,
    #             documentation=func.documentation,
    #             file_path=func.file_path
    #         )
        
    #     # Add edges between similar functions
    #     for i, func1 in enumerate(self.encoder.functions):
    #         for j, func2 in enumerate(self.encoder.functions[i+1:], i+1):
    #             # Calculate similarity using both code and documentation embeddings
    #             code_sim = cosine_similarity(
    #                 func1.code_embedding.reshape(1, -1),
    #                 func2.code_embedding.reshape(1, -1)
    #             )[0][0]
                
    #             doc_sim = cosine_similarity(
    #                 func1.doc_embedding.reshape(1, -1),
    #                 func2.doc_embedding.reshape(1, -1)
    #             )[0][0]
                
    #             # Combined similarity
    #             combined_sim = (code_sim + doc_sim) / 2
                
    #             if combined_sim > self.similarity_threshold:
    #                 G.add_edge(func1.name, func2.name, weight=combined_sim)
        
    #     return G

    

    # Update this part in generate_similarity_graph method:
    def generate_similarity_graph(self) -> nx.Graph:
        """Generate a graph where nodes are functions and edges represent similarity."""
        G = nx.Graph()
        
        # Add all functions as nodes
        for idx, func in enumerate(self.encoder.functions):
            G.add_node(
                func.name,
                code=func.code,
                documentation=func.documentation,
                file_path=func.file_path
            )
        
        # Add edges between similar functions
        for i, func1 in enumerate(self.encoder.functions):
            for j, func2 in enumerate(self.encoder.functions[i+1:], i+1):
                # Convert numpy arrays to PyTorch tensors
                code_emb1 = torch.from_numpy(func1.code_embedding).unsqueeze(0)
                code_emb2 = torch.from_numpy(func2.code_embedding).unsqueeze(0)
                doc_emb1 = torch.from_numpy(func1.doc_embedding).unsqueeze(0)
                doc_emb2 = torch.from_numpy(func2.doc_embedding).unsqueeze(0)
                
                # Calculate similarity using both code and documentation embeddings
                code_sim = torch.cosine_similarity(code_emb1, code_emb2).item()
                doc_sim = torch.cosine_similarity(doc_emb1, doc_emb2).item()
                
                # Combined similarity
                combined_sim = (code_sim + doc_sim) / 2
                
                if combined_sim > self.similarity_threshold:
                    G.add_edge(func1.name, func2.name, weight=combined_sim)
        
        return G

    def identify_clusters(self, G: nx.Graph) -> Dict[str, List[str]]:
        """Identify clusters of related functions using community detection."""
        communities = nx.community.louvain_communities(G)
        
        clusters = {}
        for idx, community in enumerate(communities):
            # Find the most central node in the community to use as cluster name
            subgraph = G.subgraph(community)
            centrality = nx.degree_centrality(subgraph)
            central_node = max(centrality.items(), key=lambda x: x[1])[0]
            
            # Use the central node's name or generate a descriptive name
            cluster_name = self._generate_cluster_name(G, community)
            clusters[cluster_name] = list(community)
            
        return clusters

    def _generate_cluster_name(self, G: nx.Graph, community: Set[str]) -> str:
        """Generate a descriptive name for a cluster based on its functions."""
        # Combine all documentation from the community
        all_docs = " ".join([G.nodes[func]['documentation'] for func in community])
        
        # Use the doc encoder to find the most relevant terms
        doc_embedding = self.encoder.doc_encoder.encode(all_docs)
        
        # You could implement keyword extraction here
        # For now, use the most common word in function names
        common_words = defaultdict(int)
        for func in community:
            words = func.split('_')
            for word in words:
                common_words[word] += 1
                
        most_common = max(common_words.items(), key=lambda x: x[1])[0]
        return f"{most_common}_related"

    def generate_mermaid_mindmap(self) -> str:
        """Generate a Mermaid mindmap diagram."""
        G = self.generate_similarity_graph()
        clusters = self.identify_clusters(G)
        
        # Create the mindmap
        mindmap = ["mindmap"]
        mindmap.append("  root((Code Repository))")
        
        # Add clusters and their functions
        for cluster_name, functions in clusters.items():
            mindmap.append(f"    {cluster_name}")
            
            # Group functions by file path
            file_groups = defaultdict(list)
            for func in functions:
                file_path = G.nodes[func]['file_path']
                file_groups[Path(file_path).name].append(func)
            
            # Add file groups
            for file_name, file_functions in file_groups.items():
                mindmap.append(f"      {file_name}")
                for func in file_functions:
                    # Escape special characters in function names
                    safe_name = func.replace('(', '_').replace(')', '_')
                    mindmap.append(f"        {safe_name}")
        
        return "\n".join(mindmap)

    def generate_html_mindmap(self) -> str:
        """Generate an interactive HTML visualization."""
        G = self.generate_similarity_graph()
        clusters = self.identify_clusters(G)
        
        html = ["""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/vis-network.min.js"></script>
    <style>
        #mindmap {
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }
    </style>
</head>
<body>
    <div id="mindmap"></div>
    <script>
"""]
        
        # Prepare nodes and edges for vis.js
        nodes = []
        edges = []
        
        # Add root node
        nodes.append({
            'id': 'root',
            'label': 'Code Repository',
            'level': 0
        })
        
        # Add cluster nodes
        for cluster_id, (cluster_name, functions) in enumerate(clusters.items()):
            cluster_node_id = f'cluster_{cluster_id}'
            nodes.append({
                'id': cluster_node_id,
                'label': cluster_name,
                'level': 1
            })
            edges.append({
                'from': 'root',
                'to': cluster_node_id
            })
            
            # Add function nodes
            for func in functions:
                nodes.append({
                    'id': func,
                    'label': func,
                    'level': 2,
                    'title': G.nodes[func]['documentation']  # Tooltip
                })
                edges.append({
                    'from': cluster_node_id,
                    'to': func
                })
        
        # Add the JavaScript code
        html.append(f"""
        const nodes = new vis.DataSet({nodes});
        const edges = new vis.DataSet({edges});
        
        const container = document.getElementById('mindmap');
        const data = {{
            nodes: nodes,
            edges: edges
        }};
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150
                }}
            }},
            physics: {{
                hierarchicalRepulsion: {{
                    centralGravity: 0.0,
                    springLength: 100,
                    springConstant: 0.01,
                    nodeDistance: 120
                }},
                solver: 'hierarchicalRepulsion'
            }}
        }};
        
        const network = new vis.Network(container, data, options);
    </script>
</body>
</html>
""")
        
        return "\n".join(html)
#!/usr/bin/env python3
"""
Utility Functions for Transformer Paper Generation

This module provides helper functions for rendering diagrams, formatting text, 
and other utilities needed across the paper generation process.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx
import math 

# def load_json(file_path: str) -> Dict[str, Any]:
#     """Load a JSON file."""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

def load_json(file_path: str):
    """Load JSON from a file and handle possible errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:  # Check if the file is empty
                raise ValueError("The JSON file is empty.")
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        raise
    except ValueError as e:
        print(f"Error loading JSON from file {file_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def create_directory(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def generate_architecture_diagram(classes: Dict[str, Any], 
                                  output_file: str) -> None:
    """
    Generate an architecture diagram based on class information.
    """
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Position classes
    positions = {}
    num_classes = len(classes)
    
    # Calculate positions in a circular layout
    if num_classes > 0:
        angle_step = 2 * 3.14159 / num_classes
        radius = 4
        center_x, center_y = 5, 5
        
        for i, class_name in enumerate(classes):
            angle = i * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions[class_name] = (x, y)
    
    # Draw class boxes
    for class_name, pos in positions.items():
        class_info = classes[class_name]
        x, y = pos
        
        # Draw a rectangle representing the class
        width, height = 2, 1.5
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                         facecolor='lightblue', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add class name
        plt.text(x, y, class_name, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=10, fontweight='bold')
        
        # Add a few methods below the class name
        methods = class_info.get("methods", [])
        method_text = ""
        for i, method in enumerate(methods[:3]):  # Show only first 3 methods
            method_text += f"{method['name']}()\n"
        
        if len(methods) > 3:
            method_text += "..."
            
        plt.text(x, y - 0.3, method_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=8)
    
    # Draw connections between classes
    for class_name, pos in positions.items():
        x1, y1 = pos
        
        # Draw arrows to related classes
        for other_class, other_pos in positions.items():
            if class_name != other_class:
                # Simple heuristic: if class names are similar, draw a connection
                if class_name in other_class or other_class in class_name:
                    x2, y2 = other_pos
                    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                          connectionstyle="arc3,rad=0.1",
                                          arrowstyle="-|>", 
                                          mutation_scale=15,
                                          linewidth=1, 
                                          edgecolor='gray')
                    ax.add_patch(arrow)
    
    # Add title
    plt.title("Architecture Diagram", fontsize=14)
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_class_diagram(classes: Dict[str, Any], 
                          dependencies: Dict[str, Any], 
                          output_file: str) -> None:
    """
    Generate a UML class diagram using NetworkX and Matplotlib.
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add class nodes
    for class_name in classes:
        G.add_node(class_name, type="class")
    
    # Add edges from dependencies
    for name, dep_info in dependencies.items():
        if dep_info.get("type") == "class":
            for dep in dep_info.get("depends_on", []):
                if dep in classes:
                    G.add_edge(name, dep)
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw diagram
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add class methods as smaller text
    for class_name, (x, y) in pos.items():
        if class_name in classes:
            class_info = classes[class_name]
            methods = class_info.get("methods", [])
            method_text = ""
            for i, method in enumerate(methods[:3]):  # Show only first 3 methods
                method_text += f"{method['name']}()\n"
            
            if len(methods) > 3:
                method_text += "..."
                
            plt.text(x, y-0.1, method_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)
    
    plt.title("Class Diagram", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_component_flow_diagram(data_flow: Dict[str, Any], 
                                  output_file: str) -> None:
    """
    Generate a component interaction flow diagram.
    """
    # Create directed graph for data flow
    G = nx.DiGraph()
    
    # Add nodes for functions that are sources or targets in data paths
    for path in data_flow.get("data_paths", []):
        G.add_node(path["from"])
        G.add_node(path["to"])
    
    # Add edges for data flow
    for path in data_flow.get("data_paths", []):
        G.add_edge(path["from"], path["to"])
    
    # Calculate layout - use hierarchical layout if possible
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw diagram
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightgreen", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title("Component Interaction Flow", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def format_markdown(text: str) -> str:
    """
    Format text with proper Markdown styling.
    """
    # Ensure headers have space after #
    text = re.sub(r'(#+)(\w)', r'\1 \2', text)
    
    # Ensure blank lines before and after lists
    text = re.sub(r'([^\n])\n([\*\-\+])', r'\1\n\n\2', text)
    text = re.sub(r'([\*\-\+].*)\n([^\n\*\-\+])', r'\1\n\n\2', text)
    
    # Ensure blank lines before and after code blocks
    text = re.sub(r'([^\n])\n```', r'\1\n\n```', text)
    text = re.sub(r'```\n([^\n])', r'```\n\n\1', text)
    
    return text

# def generate_tex_preamble() -> str:
#     """
#     Generate LaTeX preamble for the paper.
#     """
#     return r"""
# \documentclass[10pt,journal,compsoc]{IEEEtran}

# \usepackage{cite}
# \usepackage{amsmath,amssymb,amsfonts}
# \usepackage{algorithmic}
# \usepackage{graphicx}
# \usepackage{textcomp}
# \usepackage{xcolor}
# \usepackage{listings}
# \usepackage{hyperref}

# \hypersetup{
#     colorlinks=true,
#     linkcolor=blue,
#     filecolor=magenta,      
#     urlcolor=cyan,
#     pdftitle={Transformer Architecture Analysis},
#     pdfauthor={Automatic Paper Generator},
#     pdfkeywords={Deep Learning, Transformer, Neural Networks},
#     pdfsubject={AI Paper},
#     pdfcreator={LaTeX},
#     pdfproducer={pdfTeX}
# }

# \lstset{
#     backgroundcolor=\color{white},
#     basicstyle=\footnotesize\ttfamily,
#     breakatwhitespace=false,
#     breaklines=true,
#     captionpos=b,
#     commentstyle=\color{green},
#     keywordstyle=\color{blue},
#     stringstyle=\color{red},
#     numbers=left,
#     numbersep=5pt,
#     showspaces=false,
#     showstringspaces=false,
#     showtabs=false,
#     tabsize=2
# }

# \begin{document}
# \title{Analysis of Transformer Architecture Implementation}
# """

def generate_tex_preamble(title: str) -> str: # <-- Add 'title: str' here
    """
    Generate LaTeX preamble for the paper.

    Args:
        title: The title of the paper.

    Returns:
        The LaTeX preamble string.
    """
    # Basic escaping for LaTeX special characters in the title
    # You might need a more robust solution depending on possible titles
    escaped_title = title.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    return fr"""
\documentclass[10pt,journal,compsoc]{{IEEEtran}}

\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{algorithmic}}
\usepackage{{graphicx}}
\usepackage{{textcomp}}
\usepackage{{xcolor}}
\usepackage{{listings}}
\usepackage{{hyperref}}

\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    pdftitle={{{escaped_title}}}, # <-- Use escaped title
    pdfauthor={{Automatic Paper Generator}},
    pdfkeywords={{Deep Learning, Transformer, Neural Networks}},
    pdfsubject={{AI Paper}},
    pdfcreator={{LaTeX}},
    pdfproducer={{pdfTeX}}
}}

\lstset{{
    backgroundcolor=\color{{white}},
    basicstyle=\footnotesize\ttfamily,
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    commentstyle=\color{{green}},
    keywordstyle=\color{{blue}},
    stringstyle=\color{{red}},
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2
}}

\begin{{document}}
\title{{{escaped_title}}} # <-- Use escaped title here
"""

def generate_tex_closing() -> str:
    """
    Generate LaTeX closing for the paper.
    """
    return r"""
\end{document}
"""

def extract_metrics_summary(metrics: Dict[str, Any]) -> str:
    """
    Extract a summary of code metrics in human-readable format.
    """
    summary = []
    
    if "total_lines" in metrics:
        summary.append(f"Total lines of code: {metrics['total_lines']}")
    
    if "class_count" in metrics:
        summary.append(f"Number of classes: {metrics['class_count']}")
    
    if "function_count" in metrics:
        summary.append(f"Number of functions: {metrics['function_count']}")
    
    if "import_count" in metrics:
        summary.append(f"Number of imports: {metrics['import_count']}")
    
    return "\n".join(summary)

def extract_complexity_summary(complexity: Dict[str, Any]) -> str:
    """
    Extract a summary of code complexity in human-readable format.
    """
    summary = []
    
    if "overall" in complexity:
        overall = complexity["overall"]
        summary.append(f"Overall cyclomatic complexity: {overall.get('total_cyclomatic', 0):.1f}")
        summary.append(f"Average function complexity: {overall.get('average_cyclomatic', 0):.1f}")
    
    return "\n".join(summary)
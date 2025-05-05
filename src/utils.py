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

# def generate_architecture_diagram(classes: Dict[str, Any], 
#                                   output_file: str) -> None:
#     """
#     Generate an architecture diagram based on class information.
#     """
#     plt.figure(figsize=(12, 8))
    
#     # Set up the plot
#     ax = plt.gca()
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 10)
#     ax.axis('off')
    
#     # Position classes
#     positions = {}
#     num_classes = len(classes)
    
#     # Calculate positions in a circular layout
#     if num_classes > 0:
#         angle_step = 2 * 3.14159 / num_classes
#         radius = 4
#         center_x, center_y = 5, 5
        
#         for i, class_name in enumerate(classes):
#             angle = i * angle_step
#             x = center_x + radius * math.cos(angle)
#             y = center_y + radius * math.sin(angle)
#             positions[class_name] = (x, y)
    
#     # Draw class boxes
#     for class_name, pos in positions.items():
#         class_info = classes[class_name]
#         x, y = pos
        
#         # Draw a rectangle representing the class
#         width, height = 2, 1.5
#         rect = Rectangle((x - width/2, y - height/2), width, height, 
#                          facecolor='lightblue', edgecolor='black', alpha=0.7)
#         ax.add_patch(rect)
        
#         # Add class name
#         plt.text(x, y, class_name, 
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  fontsize=10, fontweight='bold')
        
#         # Add a few methods below the class name
#         methods = class_info.get("methods", [])
#         method_text = ""
#         for i, method in enumerate(methods[:3]):  # Show only first 3 methods
#             method_text += f"{method['name']}()\n"
        
#         if len(methods) > 3:
#             method_text += "..."
            
#         plt.text(x, y - 0.3, method_text,
#                  horizontalalignment='center',
#                  verticalalignment='center',
#                  fontsize=8)
    
#     # Draw connections between classes
#     for class_name, pos in positions.items():
#         x1, y1 = pos
        
#         # Draw arrows to related classes
#         for other_class, other_pos in positions.items():
#             if class_name != other_class:
#                 # Simple heuristic: if class names are similar, draw a connection
#                 if class_name in other_class or other_class in class_name:
#                     x2, y2 = other_pos
#                     arrow = FancyArrowPatch((x1, y1), (x2, y2), 
#                                           connectionstyle="arc3,rad=0.1",
#                                           arrowstyle="-|>", 
#                                           mutation_scale=15,
#                                           linewidth=1, 
#                                           edgecolor='gray')
#                     ax.add_patch(arrow)
    
#     # Add title
#     plt.title("Architecture Diagram", fontsize=14)
    
#     # Save the diagram
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()

# def generate_class_diagram(classes: Dict[str, Any], 
#                           dependencies: Dict[str, Any], 
#                           output_file: str) -> None:
#     """
#     Generate a UML class diagram using NetworkX and Matplotlib.
#     """
#     # Create directed graph
#     G = nx.DiGraph()
    
#     # Add class nodes
#     for class_name in classes:
#         G.add_node(class_name, type="class")
    
#     # Add edges from dependencies
#     for name, dep_info in dependencies.items():
#         if dep_info.get("type") == "class":
#             for dep in dep_info.get("depends_on", []):
#                 if dep in classes:
#                     G.add_edge(name, dep)
    
#     # Calculate layout
#     pos = nx.spring_layout(G, seed=42)
    
#     # Draw diagram
#     plt.figure(figsize=(12, 10))
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
#     # Add class methods as smaller text
#     for class_name, (x, y) in pos.items():
#         if class_name in classes:
#             class_info = classes[class_name]
#             methods = class_info.get("methods", [])
#             method_text = ""
#             for i, method in enumerate(methods[:3]):  # Show only first 3 methods
#                 method_text += f"{method['name']}()\n"
            
#             if len(methods) > 3:
#                 method_text += "..."
                
#             plt.text(x, y-0.1, method_text,
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     fontsize=8)
    
#     plt.title("Class Diagram", fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()

# def generate_component_flow_diagram(data_flow: Dict[str, Any], 
#                                   output_file: str) -> None:
#     """
#     Generate a component interaction flow diagram.
#     """
#     # Create directed graph for data flow
#     G = nx.DiGraph()
    
#     # Add nodes for functions that are sources or targets in data paths
#     for path in data_flow.get("data_paths", []):
#         G.add_node(path["from"])
#         G.add_node(path["to"])
    
#     # Add edges for data flow
#     for path in data_flow.get("data_paths", []):
#         G.add_edge(path["from"], path["to"])
    
#     # Calculate layout - use hierarchical layout if possible
#     try:
#         pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
#     except:
#         pos = nx.spring_layout(G, seed=42)
    
#     # Draw diagram
#     plt.figure(figsize=(12, 10))
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightgreen", alpha=0.8)
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrowsize=20)
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
#     plt.title("Component Interaction Flow", fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.close()

def generate_architecture_diagram(classes: Dict[str, Any], output_file: str) -> None:
    """
    Generate an architecture diagram based on class information.
    
    Args:
        classes: Dictionary of class information with methods and properties
        output_file: Path to save the output diagram
    """
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # If no classes, create an empty diagram with a message
    if not classes:
        plt.text(5, 5, "No classes found in the analysis", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14)
        plt.title("Architecture Diagram", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return
    
    # Position classes - Dynamic layout based on class relationships
    positions = {}
    num_classes = len(classes)
    
    # Get class relationships for better positioning
    class_relationships = {}
    for class_name, class_info in classes.items():
        related_classes = []
        
        # Look for inheritance relationships in class definition
        if "inherits_from" in class_info:
            for parent in class_info.get("inherits_from", []):
                if parent in classes:
                    related_classes.append(parent)
        
        # Look for method parameters and return types that match other classes
        for method in class_info.get("methods", []):
            # Check parameter types
            for param in method.get("parameters", []):
                param_type = param.get("type", "")
                if param_type in classes:
                    related_classes.append(param_type)
            
            # Check return type
            return_type = method.get("return_type", "")
            if return_type in classes:
                related_classes.append(return_type)
        
        class_relationships[class_name] = related_classes
    
    # Use a radial layout by default, but with related classes positioned closer
    angle_step = 2 * math.pi / num_classes
    radius = min(4, 8 / max(1, math.sqrt(num_classes)))  # Adjust radius based on class count
    center_x, center_y = 5, 5
    
    # First pass: position all classes in a circle
    for i, class_name in enumerate(classes):
        angle = i * angle_step
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions[class_name] = (x, y)
    
    # Second pass: adjust positions to bring related classes closer
    # (using simple spring algorithm)
    for _ in range(3):  # A few iterations of adjustment
        for class_name, related in class_relationships.items():
            if related and class_name in positions:
                # Average position of related classes
                related_x, related_y = 0, 0
                count = 0
                for rel in related:
                    if rel in positions:
                        rx, ry = positions[rel]
                        related_x += rx
                        related_y += ry
                        count += 1
                
                if count > 0:
                    # Move slightly toward related classes
                    x, y = positions[class_name]
                    new_x = x * 0.8 + (related_x / count) * 0.2
                    new_y = y * 0.8 + (related_y / count) * 0.2
                    
                    # Ensure we don't move too far from the circle
                    dist = math.sqrt((new_x - center_x)**2 + (new_y - center_y)**2)
                    if dist > radius * 1.5:
                        # Scale back to reasonable distance
                        scale = radius * 1.5 / dist
                        new_x = center_x + (new_x - center_x) * scale
                        new_y = center_y + (new_y - center_y) * scale
                    
                    positions[class_name] = (new_x, new_y)
    
    # Draw class boxes
    for class_name, pos in positions.items():
        class_info = classes[class_name]
        x, y = pos
        
        # Calculate box size based on content
        num_methods = len(class_info.get("methods", []))
        num_attrs = len(class_info.get("attributes", []))
        height = max(1.5, 0.8 + 0.2 * min(5, num_methods + num_attrs))
        width = max(2, len(class_name) * 0.15)
        
        # Draw a rectangle representing the class
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                         facecolor='lightblue', edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add class name
        plt.text(x, y + height/3, class_name, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=10, fontweight='bold')
        
        # Add methods
        methods = class_info.get("methods", [])
        method_text = ""
        
        # Calculate how many methods to show based on available space
        max_methods = min(5, num_methods)
        for i, method in enumerate(methods[:max_methods]):
            method_name = method.get("name", "unknown")
            method_text += f"{method_name}()\n"
        
        if num_methods > max_methods:
            method_text += "..."
            
        method_y_pos = y - 0.1 * max_methods
        plt.text(x, method_y_pos, method_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=8)
    
    # Draw connections between classes
    drawn_connections = set()  # Track which connections we've drawn
    
    for class_name, related in class_relationships.items():
        if class_name in positions:
            x1, y1 = positions[class_name]
            
            # Draw arrows to related classes
            for rel_class in related:
                if rel_class in positions and (class_name, rel_class) not in drawn_connections:
                    x2, y2 = positions[rel_class]
                    
                    # Calculate curvature based on distance
                    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    curve_factor = max(0.1, min(0.3, 1.0 / distance)) if distance > 0 else 0.1
                    
                    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                          connectionstyle=f"arc3,rad={curve_factor}",
                                          arrowstyle="-|>", 
                                          mutation_scale=15,
                                          linewidth=1, 
                                          edgecolor='gray',
                                          alpha=0.7)
                    ax.add_patch(arrow)
                    drawn_connections.add((class_name, rel_class))
    
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
    
    Args:
        classes: Dictionary of class information
        dependencies: Dictionary of dependency information
        output_file: Path to save the output diagram
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # If no classes, create an empty diagram with a message
    if not classes:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No classes found in the analysis", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14)
        plt.title("Class Diagram", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return
    
    # Add class nodes with attributes
    for class_name, class_info in classes.items():
        # Create a label with class name and essential info
        num_methods = len(class_info.get("methods", []))
        complexity = class_info.get("complexity", 0)
        G.add_node(class_name, 
                   type="class", 
                   methods=num_methods,
                   complexity=complexity)
    
    # Add edges from dependencies
    edge_types = {}  # Store edge types for labeling
    
    # First look at direct dependencies specified in the dependencies dict
    if dependencies:
        for name, dep_info in dependencies.items():
            if dep_info.get("type") == "class" and name in classes:
                for dep in dep_info.get("depends_on", []):
                    if dep in classes:
                        G.add_edge(name, dep)
                        edge_types[(name, dep)] = "depends on"
    
    # Then look for inheritance relationships from class info
    for class_name, class_info in classes.items():
        if "inherits_from" in class_info:
            for parent in class_info.get("inherits_from", []):
                if parent in classes:
                    G.add_edge(class_name, parent)
                    edge_types[(class_name, parent)] = "inherits from"
    
    # If the graph is empty (no edges), add some edges based on naming patterns
    if len(G.edges()) == 0:
        for class1 in classes:
            for class2 in classes:
                if class1 != class2:
                    # Simple heuristic: if class names are similar, draw a connection
                    # For example: Encoder and EncoderLayer
                    if (class1 in class2 or class2 in class1 or
                        any(s in class1 and s in class2 for s in ["Layer", "Block", "Transformer"])):
                        G.add_edge(class1, class2)
                        edge_types[(class1, class2)] = "related to"
    
    # Try different layouts for better visualization
    try:
        # Try a hierarchical layout first
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        try:
            # If that fails, try a spring layout with more iterations for better spacing
            pos = nx.spring_layout(G, k=1.5/math.sqrt(len(G.nodes())), iterations=50, seed=42)
        except:
            # If all else fails, use a circular layout
            pos = nx.circular_layout(G)
    
    # Get node sizes based on number of methods
    node_sizes = [1500 + G.nodes[n].get("methods", 0) * 100 for n in G.nodes()]
    
    # Get node colors based on complexity (if available)
    node_colors = []
    for n in G.nodes():
        complexity = G.nodes[n].get("complexity", 0)
        if complexity > 10:
            node_colors.append("lightcoral")
        elif complexity > 5:
            node_colors.append("lightsalmon")
        else:
            node_colors.append("lightblue")
    
    # Draw diagram
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20, 
                          arrowstyle='-|>', edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add edge labels if we have meaningful relationship types
    if any(etype != "related to" for etype in edge_types.values()):
        edge_labels = {(u, v): edge_types.get((u, v), "") 
                      for u, v in G.edges() 
                      if edge_types.get((u, v), "") != "related to"}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add class methods as smaller text
    for class_name, (x, y) in pos.items():
        if class_name in classes:
            class_info = classes[class_name]
            methods = class_info.get("methods", [])
            
            # Calculate how many methods to show based on available space
            max_methods = min(5, len(methods))
            method_text = ""
            for i, method in enumerate(methods[:max_methods]):
                method_name = method.get("name", "unknown")
                method_text += f"{method_name}()\n"
            
            if len(methods) > max_methods:
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
    
    Args:
        data_flow: Dictionary of data flow information
        output_file: Path to save the output diagram
    """
    # Validate input
    if not data_flow or "data_paths" not in data_flow or not data_flow["data_paths"]:
        # Create an empty diagram with helpful message
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No data flow information available", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14)
        plt.title("Component Flow Diagram", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        return
    
    # Create directed graph for data flow
    G = nx.DiGraph()
    
    # Extract components and path information
    components = set()
    data_paths = data_flow.get("data_paths", [])
    
    # Track path details
    edge_labels = {}
    edge_weights = {}
    
    # Add nodes and edges
    for path in data_paths:
        if "from" in path and "to" in path:
            source = path["from"]
            target = path["to"]
            
            # Add nodes
            G.add_node(source)
            G.add_node(target)
            components.add(source)
            components.add(target)
            
            # Add edge
            G.add_edge(source, target)
            
            # Track edge details
            label = path.get("data_type", "")
            if label:
                if (source, target) in edge_labels:
                    edge_labels[(source, target)] += f", {label}"
                else:
                    edge_labels[(source, target)] = label
            
            # Add weight based on importance or frequency if available
            weight = path.get("importance", 1) 
            if (source, target) in edge_weights:
                edge_weights[(source, target)] = max(weight, edge_weights[(source, target)])
            else:
                edge_weights[(source, target)] = weight
    
    # If graph is too small, create example nodes
    if len(components) < 3:
        main_components = list(components)
        if len(main_components) == 2:
            # Add intermediate processing node
            G.add_node("Process")
            G.add_edge(main_components[0], "Process")
            G.add_edge("Process", main_components[1])
        elif len(main_components) == 1:
            # Add input and output nodes
            G.add_node("Input")
            G.add_node("Output") 
            G.add_edge("Input", main_components[0])
            G.add_edge(main_components[0], "Output")
        else:
            # Create a simple pipeline example
            G.add_node("Input")
            G.add_node("Process")
            G.add_node("Output")
            G.add_edge("Input", "Process")
            G.add_edge("Process", "Output")
    
    # Calculate layout - try different approaches for best results
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        try:
            # Use a layered approach - group nodes by their position in data flow
            layers = {}
            visited = set()
            
            # Find source nodes (nodes with no incoming edges)
            sources = [n for n in G.nodes() if G.in_degree(n) == 0]
            
            # BFS to assign layers
            layer = 0
            current = sources
            while current:
                layers[layer] = current
                visited.update(current)
                next_layer = []
                for node in current:
                    for neighbor in G.neighbors(node):
                        if neighbor not in visited and all(G.in_degree(pred) == 0 or pred in visited for pred in G.predecessors(neighbor)):
                            next_layer.append(neighbor)
                current = next_layer
                layer += 1
            
            # Any remaining nodes (cycles) go in the last layer
            remaining = [n for n in G.nodes() if n not in visited]
            if remaining:
                layers[layer] = remaining
            
            # Position nodes based on layers
            pos = {}
            for layer_idx, nodes in layers.items():
                for i, node in enumerate(nodes):
                    x = layer_idx
                    y = i - (len(nodes) - 1) / 2  # Center nodes vertically
                    pos[node] = (x, y)
        except:
            # Fall back to spring layout
            pos = nx.spring_layout(G, seed=42)
    
    # Get edge widths based on weights
    edge_widths = [edge_weights.get((u, v), 1) * 1.5 for u, v in G.edges()]
    
    # Draw diagram
    plt.figure(figsize=(12, 10))
    
    # Determine node colors based on position in flow
    node_colors = []
    for node in G.nodes():
        if G.in_degree(node) == 0:  # Source nodes
            node_colors.append("lightgreen")
        elif G.out_degree(node) == 0:  # Sink nodes
            node_colors.append("lightcoral")
        else:  # Intermediate nodes
            node_colors.append("lightskyblue")
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, arrowsize=20, 
                         arrowstyle='->', edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add edge labels if available
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
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
\usepackage{{graphicx}}
\usepackage{{geometry}}
\geometry{{a4paper, margin=1in}}

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
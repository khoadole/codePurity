#!/usr/bin/env python3
"""
Code Preprocessor for Transformer Paper Generation

This script preprocesses a Python source code file (particularly focused on Transformer implementations)
by cleaning comments, standardizing formatting, and preparing it for further analysis.
"""

import re
import argparse
import ast
from typing import Dict, List, Set, Tuple

def clean_comments(code: str) -> str:
    """Remove unnecessary comments and standardize docstrings."""
    # Simple pattern to remove inline comments
    code = re.sub(r'(?<!\"\"\")(#.*$)', '', code, flags=re.MULTILINE)
    
    # Standardize docstrings format
    return code

def standardize_imports(code: str) -> str:
    """Organize and standardize import statements."""
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Extract and organize imports
        std_imports = []
        third_party_imports = []
        local_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle simple imports
                for name in node.names:
                    if name.name.startswith(('os', 'sys', 're', 'math')):
                        std_imports.append(f"import {name.name}")
                    elif name.name.startswith(('torch', 'numpy', 'tensorflow')):
                        third_party_imports.append(f"import {name.name}")
                    else:
                        local_imports.append(f"import {name.name}")
            
            elif isinstance(node, ast.ImportFrom):
                # Handle from imports
                module = node.module
                for name in node.names:
                    if module.startswith(('os', 'sys', 're', 'math')):
                        std_imports.append(f"from {module} import {name.name}")
                    elif module.startswith(('torch', 'numpy', 'tensorflow')):
                        third_party_imports.append(f"from {module} import {name.name}")
                    else:
                        local_imports.append(f"from {module} import {name.name}")
        
        # Rebuild import section
        imports_section = "\n".join(sorted(std_imports)) + "\n\n"
        imports_section += "\n".join(sorted(third_party_imports)) + "\n\n"
        imports_section += "\n".join(sorted(local_imports))
        
        # Replace old imports with new organized imports
        import_pattern = r'(import .*?\n|from .*? import .*?\n)+'
        code = re.sub(import_pattern, imports_section + "\n\n", code, flags=re.MULTILINE)
        
        return code
    except SyntaxError:
        # If there's a syntax error, return original code
        print("Warning: Syntax error when standardizing imports. Skipping this step.")
        return code

def extract_classes_and_functions(code: str) -> Tuple[List[str], List[str]]:
    """Extract class and function definitions from the code."""
    classes = []
    functions = []
    
    try:
        tree = ast.parse(code)
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                
        return classes, functions
    except SyntaxError:
        print("Warning: Syntax error when extracting classes and functions. Returning empty lists.")
        return [], []

def preprocess_code(input_file: str, output_file: str) -> Dict:
    """
    Main preprocessing function that reads input Python file,
    cleans and standardizes it, and writes to output file.
    
    Returns a dictionary with metadata about the processed code.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Clean comments
    code = clean_comments(code)
    
    # Standardize imports
    code = standardize_imports(code)
    
    # Extract classes and functions for metadata
    classes, functions = extract_classes_and_functions(code)
    
    # Write processed code to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # Return metadata
    return {
        "classes": classes,
        "functions": functions,
        "line_count": len(code.splitlines()),
        "char_count": len(code)
    }

def main():
    parser = argparse.ArgumentParser(description='Preprocess Python code for paper generation')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input Python file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output processed file')
    
    args = parser.parse_args()
    
    metadata = preprocess_code(args.input_file, args.output_file)
    print(f"Preprocessing completed. Found {len(metadata['classes'])} classes and {len(metadata['functions'])} functions.")
    print(f"Output written to {args.output_file}")

if __name__ == "__main__":
    main()
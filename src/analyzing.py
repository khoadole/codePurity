#!/usr/bin/env python3
"""
Code Analysis Module for Transformer Paper Generation

This script performs in-depth analysis of the Transformer implementation,
including complexity analysis, algorithm identification, data flow analysis,
and code quality assessment.
"""

import argparse
import json
import os
import ast
import re
from typing import Dict, List, Any, Tuple
import math
import networkx as nx

class CodeAnalyzer:
    """Analyzes Python code for algorithmic complexity, patterns, and architecture."""
    
    def __init__(self):
        self.complexity_results = {}
        self.dependencies = nx.DiGraph()
        self.algorithms = {}
        self.metrics = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on the provided Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        try:
            tree = ast.parse(code)
            
            # Extract basic metrics
            self.metrics = self.extract_basic_metrics(code)
            
            # Analyze complexity
            self.complexity_results = self.analyze_complexity(tree)
            
            # Extract dependencies between components
            self.analyze_dependencies(tree)
            
            # Identify algorithms and patterns
            self.algorithms = self.identify_algorithms(tree, code)
            
            # Analyze data flow
            data_flow = self.analyze_data_flow(tree)
            
            # Assess code quality
            code_quality = self.assess_code_quality(code)
            
            # Combine all results
            analysis_results = {
                "metrics": self.metrics,
                "complexity": self.complexity_results,
                "dependencies": self.get_dependencies_as_dict(),
                "algorithms": self.algorithms,
                "data_flow": data_flow,
                "code_quality": code_quality
            }
            
            return analysis_results
            
        except SyntaxError as e:
            print(f"Syntax error in the Python file: {e}")
            return {"error": str(e)}
    
    def extract_basic_metrics(self, code: str) -> Dict[str, Any]:
        """Extract basic code metrics like line count, character count, etc."""
        lines = code.splitlines()
        
        # Count non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Count import statements
        import_lines = len(re.findall(r'^(?:import|from)\s+\w+', code, re.MULTILINE))
        
        # Count classes and functions
        tree = ast.parse(code)
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        return {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "character_count": len(code),
            "import_count": import_lines,
            "class_count": len(classes),
            "function_count": len(functions)
        }
    
    def analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze the cyclomatic complexity and cognitive complexity of the code.
        """
        complexity_results = {
            "functions": {},
            "classes": {},
            "overall": {}
        }
        
        # Function complexity analysis
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                name = node.name
                
                # Calculate cyclomatic complexity
                cyclomatic = 1  # Base complexity
                
                # Count control flow statements like if, for, while, etc.
                for inner_node in ast.walk(node):
                    if isinstance(inner_node, (ast.If, ast.For, ast.While, ast.Try)):
                        cyclomatic += 1
                    elif isinstance(inner_node, ast.BoolOp) and isinstance(inner_node.op, (ast.And, ast.Or)):
                        cyclomatic += len(inner_node.values) - 1
                
                # Simple estimation of cognitive complexity
                cognitive = cyclomatic * 1.5
                
                complexity_results["functions"][name] = {
                    "cyclomatic": cyclomatic,
                    "cognitive": cognitive,
                    "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 1
                }
        
        # Class complexity analysis
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                name = node.name
                methods = []
                class_cyclomatic = 0
                class_cognitive = 0
                
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_name = child.name
                        if f"{method_name}" in complexity_results["functions"]:
                            method_complexity = complexity_results["functions"][f"{method_name}"]
                            methods.append({
                                "name": method_name,
                                "complexity": method_complexity
                            })
                            class_cyclomatic += method_complexity["cyclomatic"]
                            class_cognitive += method_complexity["cognitive"]
                
                complexity_results["classes"][name] = {
                    "methods": methods,
                    "total_cyclomatic": class_cyclomatic,
                    "total_cognitive": class_cognitive,
                    "lines": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 1
                }
        
        # Overall complexity
        total_cyclomatic = sum(func["cyclomatic"] for func in complexity_results["functions"].values())
        total_cognitive = sum(func["cognitive"] for func in complexity_results["functions"].values())
        total_lines = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.Expr, ast.Assign, ast.AnnAssign)))
        
        complexity_results["overall"] = {
            "total_cyclomatic": total_cyclomatic,
            "total_cognitive": total_cognitive,
            "average_cyclomatic": total_cyclomatic / len(complexity_results["functions"]) if complexity_results["functions"] else 0,
            "average_cognitive": total_cognitive / len(complexity_results["functions"]) if complexity_results["functions"] else 0,
            "complexity_density": total_cyclomatic / total_lines if total_lines else 0
        }
        
        return complexity_results
    
    def analyze_dependencies(self, tree: ast.AST) -> None:
        """
        Build a dependency graph between classes and functions.
        """
        # First pass: collect all defined names (classes and functions)
        defined_names = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined_names[node.name] = "class"
                # Add class to dependency graph
                self.dependencies.add_node(node.name, type="class")
                
                # Add methods to dependency graph
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_name = f"{node.name}.{child.name}"
                        defined_names[method_name] = "method"
                        self.dependencies.add_node(method_name, type="method")
                        # Method depends on its class
                        self.dependencies.add_edge(method_name, node.name)
            
            elif isinstance(node, ast.FunctionDef) and not hasattr(node, 'parent_class'):
                defined_names[node.name] = "function"
                self.dependencies.add_node(node.name, type="function")
        
        # Second pass: analyze dependencies
        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                caller = None
                
                # Find the caller context
                for parent in ast.iter_child_nodes(tree):
                    if isinstance(parent, ast.ClassDef):
                        for method in parent.body:
                            if isinstance(method, ast.FunctionDef) and node in list(ast.walk(method)):
                                caller = f"{parent.name}.{method.name}"
                                break
                    elif isinstance(parent, ast.FunctionDef) and node in list(ast.walk(parent)):
                        caller = parent.name
                        break
                
                # Get the callee (function being called)
                callee = None
                if isinstance(node.func, ast.Name):
                    callee = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        # This could be a method call on an object
                        obj_name = node.func.value.id
                        method_name = node.func.attr
                        if f"{obj_name}.{method_name}" in defined_names:
                            callee = f"{obj_name}.{method_name}"
                        else:
                            # This might be an external library call
                            callee = f"{obj_name}.{method_name}"
                
                if caller and callee and caller in self.dependencies and callee in self.dependencies:
                    self.dependencies.add_edge(caller, callee)
    
    def get_dependencies_as_dict(self) -> Dict[str, List[str]]:
        """Convert dependency graph to dictionary format."""
        dependency_dict = {}
        
        for node in self.dependencies.nodes():
            dependency_dict[node] = {
                "type": self.dependencies.nodes[node].get("type", "unknown"),
                "depends_on": list(self.dependencies.successors(node)),
                "depended_by": list(self.dependencies.predecessors(node))
            }
        
        return dependency_dict
    
    def identify_algorithms(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """
        Identify common algorithms and design patterns in the code.
        """
        algorithms = {
            "neural_network": self._detect_neural_network(code),
            "optimization": self._detect_optimization_algorithms(code),
            "attention_mechanism": self._detect_attention_mechanism(code),
            "linear_algebra": self._detect_linear_algebra_operations(code),
            "design_patterns": self._detect_design_patterns(tree)
        }
        
        return algorithms
    
    def _detect_neural_network(self, code: str) -> Dict[str, bool]:
        """Detect neural network components."""
        return {
            "has_layers": bool(re.search(r'class.*Layer|nn\.Linear|nn\.Module', code)),
            "has_activations": bool(re.search(r'relu|sigmoid|tanh|softmax', code, re.IGNORECASE)),
            "has_loss_function": bool(re.search(r'loss|criterion', code, re.IGNORECASE)),
            "has_optimizer": bool(re.search(r'optim|SGD|Adam|optimizer', code, re.IGNORECASE)),
            "has_forward_pass": bool(re.search(r'def forward\(', code))
        }
    
    def _detect_optimization_algorithms(self, code: str) -> Dict[str, bool]:
        """Detect optimization algorithms."""
        return {
            "gradient_descent": bool(re.search(r'grad|gradient|SGD|Adam', code, re.IGNORECASE)),
            "backpropagation": bool(re.search(r'backward|loss\.backward|autograd', code)),
            "weight_initialization": bool(re.search(r'init|xavier|kaiming|normal_|uniform_', code))
        }
    
    def _detect_attention_mechanism(self, code: str) -> Dict[str, bool]:
        """Detect attention mechanism patterns."""
        return {
            "self_attention": bool(re.search(r'self[-_]?attention|multihead', code, re.IGNORECASE)),
            "query_key_value": bool(re.search(r'query|key|value|Q|K|V', code)),
            "softmax_attention": bool(re.search(r'softmax.*attention|attention.*softmax', code, re.IGNORECASE))
        }
    
    def _detect_linear_algebra_operations(self, code: str) -> Dict[str, bool]:
        """Detect linear algebra operations."""
        return {
            "matrix_multiplication": bool(re.search(r'matmul|mm|bmm|@|dot', code)),
            "vector_operations": bool(re.search(r'norm|cross|outer|inner', code)),
            "tensor_operations": bool(re.search(r'tensor|reshape|view|permute|transpose', code))
        }
    
    def _detect_design_patterns(self, tree: ast.AST) -> Dict[str, bool]:
        """Detect common design patterns."""
        has_inheritance = False
        has_composition = False
        has_factory = False
        has_observer = False
        has_singleton = False
        
        # Check for inheritance
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.bases:
                has_inheritance = True
                break
        
        # Check for composition (instances as class attributes)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.Assign):
                        has_composition = True
                        break
        
        # Simplistic check for factory pattern (create/get method that returns the class instance)
        factory_patterns = ["create", "factory", "build", "get_instance"]
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(pattern in node.name.lower() for pattern in factory_patterns):
                has_factory = True
                break
        
        # Check for observer pattern (subscribe/notify methods)
        observer_patterns = ["subscribe", "register", "notify", "update", "observer"]
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(pattern in node.name.lower() for pattern in observer_patterns):
                has_observer = True
                break
        
        # Check for singleton pattern (private constructor + getInstance method)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_private_init = False
                has_get_instance = False
                
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        if child.name == "__init__" and any(d.name == "_" + node.name for d in child.decorator_list):
                            has_private_init = True
                        if "get_instance" in child.name.lower() or "instance" in child.name.lower():
                            has_get_instance = True
                
                if has_private_init and has_get_instance:
                    has_singleton = True
                    break
        
        return {
            "inheritance": has_inheritance,
            "composition": has_composition,
            "factory": has_factory,
            "observer": has_observer,
            "singleton": has_singleton
        }
    
    # def analyze_data_flow(self, tree: ast.AST) -> Dict[str, Any]:
    #     """
    #     Analyze data flow through the code, identifying input/output relationships.
    #     """
    #     data_flow = {
    #         "entry_points": [],
    #         "exit_points": [],
    #         "data_transformations": [],
    #         "data_dependencies": []
    #     }
        
    #     # Identify potential entry points (functions that take inputs)
    #     for node in ast.walk(tree):
    #         if isinstance(node, ast.FunctionDef) and node.args.args:
    #             # Skip class methods with self as the only parameter
    #             if (len(node.args.args) == 1 and 
    #                 node.args.args[0].arg == 'self' and 
    #                 any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if node in parent.body)):
    #                 continue
                
    #             data_flow["entry_points"].append({
    #                 "function": node.name,
    #                 "parameters": [arg.arg for arg in node.args.args if arg.arg != 'self']
    #             })
        
    #     # Identify potential exit points (return statements)
    #     for node in ast.walk(tree):
    #         if isinstance(node, ast.Return) and node.value:
    #             # Find parent function
    #             parent_func = None
    #             for func_node in ast.walk(tree):
    #                 if isinstance(func_node, ast.FunctionDef) and node in list(ast.walk(func_node)):
    #                     parent_func = func_node.name
    #                     break
                
    #             if parent_func:
    #                 data_flow["exit_points"].append({
    #                     "function": parent_func,
    #                     "returns": self._get_return_type(node.value)
    #                 })
        
    #     # Identify data transformations
    #     for node in ast.walk(tree):
    #         if isinstance(node, ast.Assign):
    #             if isinstance(node.value, ast.Call):
    #                 # This might be a function call that transforms data
    #                 call_func = ""
    #                 if isinstance(node.value.func, ast.Name):
    #                     call_func = node.value.func.id
    #                 elif isinstance(node.value.func, ast.Attribute):
    #                     if isinstance(node.value.func.value, ast.Name):
    #                         call_func = f"{node.value.func.value.id}.{node.value.func.attr}"
                    
    #                 # Skip common Python functions
    #                 if call_func and call_func not in ['len', 'dict', 'list', 'set', 'tuple']:
    #                     data_flow["data_transformations"].append({
    #                         "function": call_func,
    #                         "target": self._get_target_name(node.targets[0]),
    #                         "source": [self._get_arg_name(arg) for arg in node.value.args if hasattr(arg, 'id')]
    #                     })
        
    #     # Identify data dependencies
    #     variables = {}
    #     for node in ast.walk(tree):
    #         if isinstance(node, ast.Assign):
    #             target_name = self._get_target_name(node.targets[0])
    #             if target_name:
    #                 # Get all variable names used in the right side
    #                 used_vars = []
    #                 for sub_node in ast.walk(node.value):
    #                     if isinstance(sub_node, ast.Name):
    #                         used_vars.append(sub_node.id)
                    
    #                 if used_vars:
    #                     variables[target_name] = used_vars
    #                     data_flow["data_dependencies"].append({
    #                         "variable": target_name,
    #                         "depends_on": used_vars
    #                     })
        
    #     return data_flow

    def analyze_data_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze data flow through the code, identifying input/output relationships.
        """
        data_flow = {
            "entry_points": [],
            "exit_points": [],
            "data_transformations": [],
            "data_dependencies": []
        }

        # Identify potential entry points (functions that take inputs)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.args.args:
                # Skip class methods with self as the only parameter
                if (len(node.args.args) == 1 and 
                    node.args.args[0].arg == 'self' and 
                    any(isinstance(parent, ast.ClassDef) and hasattr(parent, 'body') and node in parent.body for parent in ast.walk(tree))):
                    continue
                
                data_flow["entry_points"].append({
                    "function": node.name,
                    "parameters": [arg.arg for arg in node.args.args if arg.arg != 'self']
                })

        # Identify potential exit points (return statements)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                # Find parent function
                parent_func = None
                for func_node in ast.walk(tree):
                    if isinstance(func_node, ast.FunctionDef) and node in list(ast.walk(func_node)):
                        parent_func = func_node.name
                        break

                if parent_func:
                    data_flow["exit_points"].append({
                        "function": parent_func,
                        "returns": self._get_return_type(node.value)
                    })

        # Other parts of data flow analysis...
        
        return data_flow

    
    def _get_target_name(self, node: ast.AST) -> str:
        """Extract the name of an assignment target."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return ""
    
    def _get_arg_name(self, node: ast.AST) -> str:
        """Extract the name of a function argument."""
        if isinstance(node, ast.Name):
            return node.id
        return ""
    
    def _get_return_type(self, node: ast.AST) -> str:
        """Guess the type of a return value."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.List):
            return "list"
        return "unknown"
    
    def assess_code_quality(self, code: str) -> Dict[str, float]:
        """
        Assess the code quality based on various metrics.
        """
        # Docstring coverage
        tree = ast.parse(code)
        total_defs = 0
        with_docstring = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                total_defs += 1
                
                # Check for docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    with_docstring += 1
        
        docstring_coverage = with_docstring / total_defs if total_defs > 0 else 0
        
        # Naming consistency
        snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        camel_case_pattern = re.compile(r'^[a-z][a-zA-Z0-9]*$')
        pascal_case_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names.append(("function", node.name))
            elif isinstance(node, ast.ClassDef):
                names.append(("class", node.name))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(("variable", node.id))
        
        # Count naming conventions
        snake_case_count = sum(1 for _, name in names if snake_case_pattern.match(name))
        camel_case_count = sum(1 for _, name in names if camel_case_pattern.match(name))
        pascal_case_count = sum(1 for _, name in names if pascal_case_pattern.match(name))
        
        # Determine dominant convention
        conventions = {
            "snake_case": snake_case_count,
            "camel_case": camel_case_count,
            "pascal_case": pascal_case_count
        }
        
        dominant_convention = max(conventions.items(), key=lambda x: x[1])[0]
        
        # Calculate consistency
        consistent_count = 0
        for type_, name in names:
            if type_ == "function" or type_ == "variable":
                if dominant_convention == "snake_case" and snake_case_pattern.match(name):
                    consistent_count += 1
                elif dominant_convention == "camel_case" and camel_case_pattern.match(name):
                    consistent_count += 1
            elif type_ == "class":
                if dominant_convention == "pascal_case" and pascal_case_pattern.match(name):
                    consistent_count += 1
                elif dominant_convention != "pascal_case" and pascal_case_pattern.match(name):
                    # Classes should typically use PascalCase regardless of dominant convention
                    consistent_count += 1
        
        naming_consistency = consistent_count / len(names) if names else 0
        
        # Average function length
        function_lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                    function_lines.append(node.end_lineno - node.lineno)
        
        avg_function_length = sum(function_lines) / len(function_lines) if function_lines else 0
        
        # Code complexity ratio (lines of code / number of functions)
        lines_of_code = len(code.splitlines())
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        complexity_ratio = lines_of_code / function_count if function_count > 0 else lines_of_code
        
        # Calculate overall quality score
        quality_score = (
            0.3 * docstring_coverage +
            0.3 * naming_consistency +
            0.2 * (1 - min(1, avg_function_length / 100)) +  # Lower is better
            0.2 * (1 - min(1, complexity_ratio / 50))  # Lower is better
        )
        
        return {
            "docstring_coverage": docstring_coverage,
            "naming_consistency": naming_consistency,
            "average_function_length": avg_function_length,
            "complexity_ratio": complexity_ratio,
            "overall_quality": quality_score,
            "dominant_naming_convention": dominant_convention
        }

def main():
    parser = argparse.ArgumentParser(description="Analyze Python code for research paper generation.")
    parser.add_argument("--input_file", required=True, help="Path to the Python file to analyze")
    parser.add_argument("--output_file", required=True, help="Path to save analysis results JSON")
    args = parser.parse_args()
    
    analyzer = CodeAnalyzer()
    analysis_results = analyzer.analyze_file(args.input_file)
    
    # Save analysis results to JSON
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
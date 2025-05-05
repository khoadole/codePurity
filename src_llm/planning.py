#!/usr/bin/env python3
"""
Planning Module for Paper Generation

This script analyzes a preprocessed Python file containing implementation 
and creates a structured plan for generating a research paper about it.
"""

import argparse
import json
import os
import ast
import re
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PaperPlanner:
    """Plans the structure and content of a paper based on code analysis."""
    
    def __init__(self, paper_name: str, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.paper_name = paper_name
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def analyze_code_structure(self, python_file: str) -> Dict[str, Any]:
        """
        Analyze the structure of the Python code to understand its components
        and architecture.
        """
        with open(python_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            tree = ast.parse(code)
            
            classes = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = []
                    
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append({
                                "name": child.name,
                                "args": [arg.arg for arg in child.args.args if arg.arg != 'self'],
                                "line_count": child.end_lineno - child.lineno if hasattr(child, 'end_lineno') else 1
                            })
                    
                    classes[class_name] = {
                        "methods": methods,
                        "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 1
                    }
            
            functions = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name != '__main__':
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 1
                    })
            
            return {
                "classes": classes,
                "functions": functions,
                "total_lines": len(code.splitlines())
            }
        except SyntaxError as e:
            print(f"Syntax error in the Python file: {e}")
            return {"error": str(e)}

    def generate_paper_outline(self, code_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use the LLM to generate a paper outline based on code analysis.
        """
        prompt = f"""
        Create a research paper outline for a paper about the implementation of {self.paper_name}.
        
        The code has the following structure:
        - Classes: {list(code_analysis['classes'].keys())}
        - Functions: {[func['name'] for func in code_analysis['functions']]}
        
        The paper should include:
        1. Abstract
        2. Introduction to {self.paper_name}
        3. Related Work
        4. Architecture and Implementation Details
        5. Experimental Setup
        6. Results and Discussion
        7. Conclusion
        
        For each section, provide 3-5 key points to address.
        """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_length=1500,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            outline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            sections = re.split(r'\d+\.\s+', outline)[1:]
            structured_outline = {}
            
            for i, section in enumerate(sections, 1):
                section_lines = section.strip().split('\n')
                section_title = section_lines[0].strip()
                section_points = []
                
                for line in section_lines[1:]:
                    if line.strip().startswith('-'):
                        section_points.append(line.strip()[2:])
                
                structured_outline[f"section_{i}"] = {
                    "title": section_title,
                    "key_points": section_points
                }
            
            return structured_outline
        except Exception as e:
            print(f"Error generating paper outline: {e}")
            return {"error": str(e)}

    def create_figure_plan(self, code_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Plan figures for the paper based on code analysis.
        """
        figures = []
        
        figures.append({
            "figure_id": "architecture",
            "caption": f"{self.paper_name} Architecture",
            "description": "Diagram showing the overall architecture of the implementation with main components and data flow."
        })
        
        if code_analysis["classes"]:
            figures.append({
                "figure_id": "class_diagram",
                "caption": f"{self.paper_name} Class Structure",
                "description": "UML class diagram showing the relationships between classes in the implementation."
            })
        
        figures.append({
            "figure_id": "component_flow",
            "caption": "Component Interaction Flow",
            "description": "Flowchart showing how the main components interact during forward and backward passes."
        })
        
        return figures

    def plan_paper(self, input_python: str, output_dir: str) -> Dict[str, Any]:
        """
        Create a complete paper plan including outline, figures, and generation steps.
        """
        code_analysis = self.analyze_code_structure(input_python)
        outline = self.generate_paper_outline(code_analysis)
        figures = self.create_figure_plan(code_analysis)
        
        paper_plan = {
            "paper_name": self.paper_name,
            "code_analysis": code_analysis,
            "outline": outline,
            "figures": figures,
            "generation_steps": [
                "Generate abstract and introduction",
                "Describe architecture and implementation",
                "Create figures and diagrams",
                "Analyze code performance and characteristics",
                "Generate conclusion and references"
            ]
        }
        
        os.makedirs(output_dir, exist_ok=True)
        plan_file = os.path.join(output_dir, "paper_plan.json")
        
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(paper_plan, f, indent=2)
        
        print(f"Paper plan saved to {plan_file}")
        return paper_plan

def main():
    parser = argparse.ArgumentParser(description='Plan a paper based on code analysis')
    parser.add_argument('--paper_name', type=str, required=True, help='Name of the paper')
    parser.add_argument('--model_name', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1', help='LLM model to use')
    parser.add_argument('--input_python', type=str, required=True, help='Path to preprocessed Python file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plan and results')
    
    args = parser.parse_args()
    
    planner = PaperPlanner(args.paper_name, args.model_name)
    planner.plan_paper(args.input_python, args.output_dir)

if __name__ == "__main__":
    main()
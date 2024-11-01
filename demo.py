# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import time
from models.generative.text_to_protein_generator import TextToProteinGenerator
from models.analysis.sequence_analyzer import SequenceAnalyzer
from fetch_pdb_info import create_structure_description
import requests
from Bio import PDB
import numpy as np
import mdtraj as md
from Bio.PDB import *

def print_section(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50 + "\n")

def demo_text_to_protein():
    print_section("1. Text-to-Protein Generation Demo")

    # Fetch PDB structure description
    print("Fetching PDB 7BBV structure description...")
    description = create_structure_description("7BBV")
    print(f"\nStructure Description:\n{description}\n")

    # Generate protein sequence
    print("Generating protein sequence...")
    generator = TextToProteinGenerator()
    sequence_result = generator.generate_sequence(description)
    if isinstance(sequence_result, tuple):
        sequence = sequence_result[0]  # Extract sequence from tuple
    else:
        sequence = sequence_result
    print(f"\nGenerated Sequence:\n{sequence}\n")
    return sequence

def demo_sequence_analysis(sequence):
    print_section("2. Sequence Analysis Demo")

    analyzer = SequenceAnalyzer(sequence)
    analyzer.analyze_sequence()
    return analyzer

def demo_api_integration():
    """Demo API integration performance."""
    print_section("3. API Integration Performance Demo")
    print()

    import os
    import time
    import requests
    import json

    # Get API key from environment
    api_key = os.getenv('Gemini_api')
    if not api_key:
        print("Error: Gemini API key not found in environment variables")
        return

    # Test API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

    # Prepare test request
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": "Generate a short protein sequence description"
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 256
        }
    }

    try:
        # Time the API request
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()

        # Print performance metrics
        print(f"API Response Time: {end_time - start_time:.2f} seconds")
        print(f"API Status Code: {response.status_code}")

        if response.status_code == 200:
            print("API Integration: Successful")
            response_data = response.json()
            print("\nAPI Response Preview:")
            print(json.dumps(response_data, indent=2)[:500] + "...")
        else:
            print("API Integration: Failed")
            print(f"Error: {response.text}")

    except Exception as e:
        print("API Integration: Failed")
        print(f"Error: {str(e)}")

    print("\n")

def demo_molecular_dynamics():
    print_section("4. Molecular Dynamics Visualization Demo")

    # Download PDB file if not exists
    if not os.path.exists("pdb7bbv.ent"):
        pdb_list = PDB.PDBList()
        pdb_list.retrieve_pdb_file("7BBV", pdb_format="pdb", file_format="pdb")

    # Load structure
    parser = PDB.PDBParser()
    structure = parser.get_structure("7BBV", "pdb7bbv.ent")

    # Print structure information
    print("Structure Information:")
    print(f"Number of models: {len(structure)}")
    print(f"Number of chains: {len(structure[0])}")

    # Calculate and print some basic structural properties
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"].get_coord())

    ca_atoms = np.array(ca_atoms)

    # Calculate radius of gyration
    center = np.mean(ca_atoms, axis=0)
    rg = np.sqrt(np.mean(np.sum((ca_atoms - center)**2, axis=1)))

    print(f"\nStructural Analysis:")
    print(f"Radius of gyration: {rg:.2f} Å")
    print(f"Structure dimensions: {ca_atoms.max(axis=0) - ca_atoms.min(axis=0)} Å")

def main():
    try:
        # Run text-to-protein generation demo
        sequence = demo_text_to_protein()

        # Run sequence analysis demo
        analyzer = demo_sequence_analysis(sequence)

        # Run API integration demo
        demo_api_integration()

        # Run molecular dynamics demo
        demo_molecular_dynamics()

        print_section("Demo Complete")
        print("All components demonstrated successfully!")

    except Exception as e:
        print(f"Error during demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()

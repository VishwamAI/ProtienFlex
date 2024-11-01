from flask import Flask, render_template, request, jsonify
import logging
import sys
from models.qa_system import ProteinQASystem
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import py3Dmol
import biotite.structure as struc
import biotite.structure.io as strucio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    qa_system = ProteinQASystem()
    logger.info("QA system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QA system: {e}")
    qa_system = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400

        sequence = data['sequence']
        if not sequence or not isinstance(sequence, str):
            return jsonify({'error': 'Invalid sequence format'}), 400

        # Basic protein analysis
        protein_analysis = ProteinAnalysis(sequence)
        molecular_weight = protein_analysis.molecular_weight()
        isoelectric_point = protein_analysis.isoelectric_point()
        secondary_structure = protein_analysis.secondary_structure_fraction()

        # Generate basic structure
        pdb_string = generate_basic_pdb(sequence)

        # Calculate confidence score and contact map
        confidence_score = calculate_confidence(sequence)
        contact_map = generate_contact_map(len(sequence))

        description = f"""Protein Analysis:
Sequence Length: {len(sequence)} amino acids
Molecular Weight: {molecular_weight:.2f} Da
Isoelectric Point: {isoelectric_point:.2f}
Secondary Structure:
- Alpha Helix: {secondary_structure[0]:.2%}
- Beta Sheet: {secondary_structure[1]:.2%}
- Random Coil: {secondary_structure[2]:.2%}"""

        return jsonify({
            'pdb_string': pdb_string,
            'confidence_score': confidence_score,
            'contact_map': contact_map.tolist(),
            'description': description,
            'secondary_structure': {
                'alpha_helix': secondary_structure[0],
                'beta_sheet': secondary_structure[1],
                'random_coil': secondary_structure[2]
            }
        })

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'context' not in data:
            return jsonify({'error': 'Missing question or context'}), 400

        if qa_system:
            result = qa_system.answer_question(data['context'], data['question'])
            return jsonify(result)
        else:
            return jsonify({'error': 'QA system not available'}), 503

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def generate_basic_pdb(sequence):
    """Generate a basic PDB structure"""
    pdb_string = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    for i, aa in enumerate(sequence):
        x, y, z = i * 3.8, 0, 0
        pdb_string += f"ATOM  {i+2:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    return pdb_string

def calculate_confidence(sequence):
    """Calculate a confidence score"""
    return min(100, max(50, len(sequence) / 2))

def generate_contact_map(sequence_length):
    """Generate a contact map"""
    contact_map = np.zeros((sequence_length, sequence_length))
    for i in range(sequence_length):
        for j in range(max(0, i-3), min(sequence_length, i+4)):
            contact_map[i,j] = contact_map[j,i] = 1
    return contact_map

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

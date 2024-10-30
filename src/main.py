import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_protein_data
from src.model_training import train_and_evaluate_model
from src.nlp_processing import process_text
from src.virtual_screening import virtual_screening
from rdkit import Chem
from rdkit.Chem import Descriptors


def main():
    # Load protein data
    protein_data = pd.read_csv('data/protein_data.csv')
    
    # Preprocess protein data
    preprocessed_data = preprocess_protein_data(protein_data)
    
    # Load labels for model training
    labels = np.load('data/labels.npy')
    
    # Train and evaluate the 3D model
    evaluation_metrics = train_and_evaluate_model(preprocessed_data, labels)
    print("Model Evaluation Metrics:", evaluation_metrics)
    
    # Load and process text data
    with open('data/text_data.txt', 'r') as file:
        text_data = file.read()
    text_embeddings = process_text(text_data)
    print("Text Embeddings:", text_embeddings)
    
    # Load protein structure for virtual screening
    protein_structure = Chem.MolFromPDBFile('data/protein_structure.pdb')
    
    # Load potential drug compounds
    compounds = [Chem.MolFromSmiles(smiles) for smiles in open('data/compounds.smi').read().splitlines()]
    
    # Define selection criteria for compounds
    criteria = {
        'MW': lambda x: Descriptors.MolWt(x) < 500,
        'LogP': lambda x: Descriptors.MolLogP(x) < 5
    }
    
    # Perform virtual screening
    scored_compounds = virtual_screening(protein_structure, compounds, criteria)
    print("Scored Compounds:", scored_compounds)


if __name__ == "__main__":
    main()

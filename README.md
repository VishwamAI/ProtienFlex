# ProteinFlex - Advanced Protein Structure Analysis and Drug Discovery

## Overview
ProteinFlex is a comprehensive platform for protein structure analysis and drug discovery, leveraging advanced AI and machine learning techniques. The platform combines state-of-the-art protein structure prediction with interactive visualization and sophisticated drug discovery tools.

## Features

### Enhanced Visualization
- **Interactive 3D Viewer**:
  - Customizable rotation, zoom, and annotation options
  - Interactive panels for sequence highlighting
  - Real-time mutation impact visualization
  - Touch and keyboard controls for intuitive navigation
  - Multiple visualization styles (cartoon, surface, stick)

- **Dynamic Confidence Visualization**:
  - Multi-level confidence scoring with adjustable thresholds
  - Granular confidence metrics for different structural regions
  - Color gradient visualization (red to green)
  - Confidence score breakdown by domain
  - Real-time updates during analysis

- **Heatmaps & Annotations**:
  - Interactive overlay of functional domains
  - Active site visualization
  - Drug-binding region highlighting
  - Custom annotation support
  - Temperature factor visualization

### LLM-Based Analysis
- **Contextual Question Answering**:
  - Natural language queries about protein function
  - Stability analysis and predictions
  - Mutation impact assessment
  - Structure-function relationship analysis
  - Domain interaction queries

- **Interactive Mutation Predictions**:
  - Real-time mutation effect analysis
  - Stability change predictions
  - Functional impact assessment
  - Structure modification visualization
  - Energy calculation for mutations

### Drug Discovery Tools
- **Binding Site Analysis**:
  - AI-driven binding site identification
  - Pocket optimization suggestions
  - Hydrophobicity analysis
  - Hydrogen bond network assessment
  - Surface accessibility calculations

- **Off-Target Screening**:
  - Protein family similarity analysis
  - Risk assessment for different protein families
  - Membrane protein interaction prediction
  - Comprehensive safety profiling
  - Cross-reactivity prediction

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for accelerated processing)
- 8GB+ RAM recommended
- Modern web browser for visualization

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ProtienFlex.git
cd ProtienFlex

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Usage

### Starting the Application
```bash
python app.py
```
The application will be available at `http://localhost:5000`

### Basic Workflow
1. Enter protein sequence in the input field
2. Click "Predict Structure" to initiate analysis
3. View 3D structure with confidence scores
4. Explore binding sites and drug interactions
5. Analyze potential mutations and their effects

### Advanced Features

#### Visualization Controls
- Mouse wheel: Zoom in/out
- Left click + drag: Rotate structure
- Right click + drag: Translate structure
- Double click: Center view
- Keyboard shortcuts:
  - R: Reset view
  - S: Toggle surface
  - H: Toggle hydrogen bonds
  - C: Toggle confidence coloring

#### Drug Discovery Pipeline
```python
from models.drug_discovery import DrugDiscoveryEngine

# Initialize engine
engine = DrugDiscoveryEngine()

# Analyze binding sites
binding_sites = engine.analyze_binding_sites(sequence)

# Screen for off-targets
off_targets = engine.screen_off_targets(sequence, ligand_smiles)

# Optimize binding site
optimizations = engine.optimize_binding_site(sequence, site_start, site_end, ligand_smiles)
```

## API Documentation

### REST Endpoints

#### Structure Prediction
```
POST /predict
Content-Type: application/json

{
    "sequence": "PROTEIN_SEQUENCE"
}

Response:
{
    "pdb_string": "PDB_STRUCTURE",
    "confidence_score": float,
    "contact_map": array,
    "description": "string",
    "secondary_structure": {
        "alpha_helix": float,
        "beta_sheet": float,
        "random_coil": float
    }
}
```

#### Binding Site Analysis
```
POST /analyze_binding_sites
Content-Type: application/json

{
    "sequence": "PROTEIN_SEQUENCE",
    "structure": "PDB_STRING" (optional)
}

Response:
{
    "binding_sites": [
        {
            "start": int,
            "end": int,
            "confidence": float,
            "hydrophobicity": float,
            "surface_area": float
        }
    ]
}
```

#### Drug Interaction Prediction
```
POST /predict_interactions
Content-Type: application/json

{
    "sequence": "PROTEIN_SEQUENCE",
    "ligand_smiles": "SMILES_STRING"
}

Response:
{
    "binding_affinity": float,
    "stability_score": float,
    "binding_energy": float,
    "key_interactions": [
        {
            "type": string,
            "residues": [string],
            "strength": float
        }
    ]
}
```

## Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any enhancements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

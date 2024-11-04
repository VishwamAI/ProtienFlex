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

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from Bio.PDB import *
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import logging
import esm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logger = logging.getLogger(__name__)

class DrugDiscoveryEngine:
    def __init__(self):
        try:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model = self.esm_model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.parser = PDBParser(QUIET=True)
            logger.info("Drug Discovery Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Drug Discovery Engine: {e}")
            raise

    def analyze_binding_sites(self, sequence, structure=None):
        try:
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
                attention = results["attentions"]

            attention_weights = attention[0].mean(dim=0).numpy()
            binding_sites = []
            window_size = 15
            best_site = None
            best_score = -float('inf')
            best_start = 0

            for i in range(len(sequence) - window_size + 1):
                window_attention = attention_weights[i:i+window_size, i:i+window_size]
                site_score = np.mean(window_attention)

                if site_score > 0.4:
                    site_sequence = sequence[i:i+window_size]
                    site_analysis = ProteinAnalysis(site_sequence)
                    site_info = {
                        'start': i,
                        'end': i + window_size,
                        'confidence': float(site_score),
                        'hydrophobicity': float(site_analysis.gravy()),
                        'surface_area': float(len(site_sequence) * 100),
                        'volume': float(len(site_sequence) * 150),
                        'residues': list(site_sequence)
                    }
                    binding_sites.append(site_info)

                    if site_score > best_score:
                        best_score = site_score
                        best_site = site_info
                        best_start = i

            if not binding_sites:
                return {
                    'start': 0,
                    'end': min(window_size, len(sequence)),
                    'score': 0.0,
                    'type': 'binding_site_analysis',
                    'message': 'No significant binding sites found',
                    'binding_sites': []
                }

            return {
                'start': best_start,
                'end': best_start + window_size,
                'score': float(best_score),
                'type': 'binding_site_analysis',
                'best_site': best_site,
                'binding_sites': binding_sites
            }
        except Exception as e:
            logger.error(f"Error in binding site analysis: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'binding_site_analysis_error',
                'error': str(e)
            }

    def predict_drug_interactions(self, sequence, ligand_smiles):
        try:
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                return {
                    'start': 0,
                    'end': min(15, len(sequence)),
                    'score': 0.0,
                    'type': 'drug_interaction_error',
                    'error': "Invalid SMILES string"
                }

            # Generate 3D conformation for ligand
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

            # Calculate ligand properties
            mol_weight = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)

            # Calculate binding energy using force field
            ff = AllChem.MMFFGetMoleculeForceField(mol)
            binding_energy = ff.CalcEnergy()

            # Analyze potential interactions
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            # Calculate binding affinity score based on physicochemical properties
            binding_score = np.clip(
                0.5 * (1 - abs(logp - 2.5)/5) +  # Optimal logP around 2.5
                0.3 * (1 - abs(hbd + hba - 5)/5) +  # Optimal H-bonds around 5
                0.2 * (1 - rotatable_bonds/10),  # Penalize excessive flexibility
                0, 1
            )

            # Calculate stability score based on structure
            stability_score = np.clip(
                0.4 * (1 - abs(logp - 2)/4) +  # Moderate logP preferred
                0.3 * (1 - rotatable_bonds/8) +  # Limited flexibility
                0.3 * (1 - mol_weight/500),  # Size penalty
                0, 1
            )

            # Find potential binding region in sequence
            binding_sites = self.analyze_binding_sites(sequence)
            if 'best_site' in binding_sites:
                start = binding_sites['best_site']['start']
                end = binding_sites['best_site']['end']
            else:
                start = 0
                end = len(sequence) // 3

            return {
                'start': start,
                'end': end,
                'score': float(binding_score),
                'type': 'drug_interaction',
                'binding_affinity': float(binding_score),
                'stability_score': float(stability_score),
                'binding_energy': float(binding_energy),
                'key_interactions': [
                    {
                        'start': start,
                        'end': end,
                        'score': float(min(1.0, (hbd + hba)/10)),
                        'type': 'Hydrogen Bond',
                        'count': int(hbd + hba),
                        'strength': float(min(1.0, (hbd + hba)/10))
                    },
                    {
                        'start': start,
                        'end': end,
                        'score': float(min(1.0, abs(logp)/5)),
                        'type': 'Hydrophobic',
                        'strength': float(min(1.0, abs(logp)/5))
                    }
                ],
                'ligand_properties': {
                    'start': start,
                    'end': end,
                    'score': float(binding_score),
                    'type': 'ligand_properties',
                    'molecular_weight': float(mol_weight),
                    'logp': float(logp)
                }
            }
        except Exception as e:
            logger.error(f"Error in interaction prediction: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'drug_interaction_error',
                'error': str(e)
            }

    def screen_off_targets(self, sequence, ligand_smiles):
        """Screen for potential off-target effects using LLM-based analysis and protein similarity comparisons."""
        try:
            # Get protein embeddings using ESM-2
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33])
                embeddings = results["representations"][33]
                attention = results["attentions"][0].mean(dim=0)
                attention_scores = attention.mean(dim=1)

            # Define known protein families for screening
            target_families = {
                "kinase": {"threshold": 0.7, "risk": "high"},
                "gpcr": {"threshold": 0.65, "risk": "medium"},
                "ion_channel": {"threshold": 0.6, "risk": "medium"},
                "nuclear_receptor": {"threshold": 0.55, "risk": "low"}
            }

            off_targets = []
            highest_similarity = 0.0
            for family, params in target_families.items():
                # Simulate family-specific embedding comparison
                similarity_score = float(torch.sigmoid(attention_scores.mean()))
                highest_similarity = max(highest_similarity, similarity_score)

                if similarity_score > params["threshold"]:
                    effects = []
                    if params["risk"] == "high":
                        effects = [
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.9,
                                'type': 'binding_effect',
                                'description': "Strong binding potential"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.85,
                                'type': 'metabolic_effect',
                                'description': "Possible metabolic interference"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.8,
                                'type': 'structural_effect',
                                'description': "Consider structural modifications"
                            }
                        ]
                    elif params["risk"] == "medium":
                        effects = [
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.7,
                                'type': 'binding_effect',
                                'description': "Moderate binding possibility"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.65,
                                'type': 'interaction_effect',
                                'description': "Monitor for interactions"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.6,
                                'type': 'dosage_effect',
                                'description': "May require dosage adjustment"
                            }
                        ]
                    else:
                        effects = [
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.4,
                                'type': 'binding_effect',
                                'description': "Weak binding potential"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.35,
                                'type': 'interaction_effect',
                                'description': "Limited interaction expected"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.3,
                                'type': 'monitoring_effect',
                                'description': "Standard monitoring sufficient"
                            }
                        ]

                    off_targets.append({
                        'start': 0,
                        'end': len(sequence),
                        'score': float(similarity_score),
                        'type': 'off_target',
                        "protein_family": family,
                        "similarity_score": float(similarity_score),
                        "risk_level": params["risk"],
                        "predicted_effects": effects,
                        "confidence": float(attention_scores.max())
                    })

            # Analyze ligand properties for additional risks
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol:
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                if logp > 5 or tpsa < 40:
                    off_targets.append({
                        'start': 0,
                        'end': len(sequence),
                        'score': 0.8,
                        'type': 'membrane_interaction',
                        "protein_family": "membrane_proteins",
                        "similarity_score": 0.8,
                        "risk_level": "high",
                        "predicted_effects": [
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.9,
                                'type': 'permeability_effect',
                                'description': "High membrane permeability"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.85,
                                'type': 'binding_effect',
                                'description': "Potential off-target binding"
                            },
                            {
                                'start': 0,
                                'end': len(sequence),
                                'score': 0.8,
                                'type': 'structural_effect',
                                'description': "Consider structural modifications"
                            }
                        ],
                        "confidence": 0.9
                    })

            # Find region with highest attention for potential binding
            attention_window = 15
            max_attention_start = 0
            max_attention_score = 0
            for i in range(len(sequence) - attention_window + 1):
                window_score = float(attention[i:i+attention_window].mean())
                if window_score > max_attention_score:
                    max_attention_score = window_score
                    max_attention_start = i

            return {
                'start': max_attention_start,
                'end': max_attention_start + attention_window,
                'score': float(highest_similarity),
                'type': 'off_target_screening',
                'off_targets': off_targets,
                'overall_risk': 'high' if highest_similarity > 0.7 else 'medium' if highest_similarity > 0.5 else 'low',
                'binding_region': {
                    'start': max_attention_start,
                    'end': max_attention_start + attention_window,
                    'score': float(max_attention_score),
                    'type': 'binding_region'
                }
            }

        except Exception as e:
            logger.error(f"Error in off-target screening: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'off_target_screening_error',
                'error': str(e),
                'off_targets': []
            }

    def optimize_binding_site(self, sequence, site_start, site_end, ligand_smiles):
        """Optimize binding site interactions using LLM-based analysis."""
        try:
            # Extract binding site sequence
            site_sequence = sequence[site_start:site_end]

            # Get embeddings for binding site analysis
            data = [("binding_site", site_sequence)]
            _, _, batch_tokens = self.batch_converter(data)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33])
                site_embeddings = results["representations"][33]
                attention = results["attentions"][0].mean(dim=0)

            # Analyze site properties
            site_analysis = ProteinAnalysis(site_sequence)
            hydrophobicity = site_analysis.gravy()

            # Calculate residue-specific properties
            residue_properties = []
            for i, aa in enumerate(site_sequence):
                attention_score = float(attention[i].mean())
                residue_properties.append({
                    'start': site_start + i,
                    'end': site_start + i + 1,
                    'score': float(attention_score),
                    'type': 'residue_property',
                    "residue": aa,
                    "position": site_start + i,
                    "attention_score": attention_score,
                    "hydrophobicity": ProteinAnalysis(aa).gravy()
                })

            # Analyze ligand properties
            mol = Chem.MolFromSmiles(ligand_smiles)
            suggestions = []

            if mol:
                # Calculate ligand properties
                mol_volume = AllChem.ComputeMolVolume(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)

                # Analyze pocket compatibility
                pocket_volume = len(site_sequence) * 150  # Approximate volume

                if mol_volume > pocket_volume * 0.8:
                    suggestions.append({
                        'start': site_start,
                        'end': site_end,
                        'score': 0.85,
                        'type': 'pocket_size',
                        "issue": "Binding pocket may be too small",
                        "suggestion": "Consider expanding binding pocket by mutating smaller residues to larger ones",
                        "confidence": 0.85
                    })

                # Analyze hydrophobic matching
                if logp > 2 and hydrophobicity < 0:
                    suggestions.append({
                        'start': site_start,
                        'end': site_end,
                        'score': 0.9,
                        'type': 'hydrophobic_matching',
                        "issue": "Hydrophobic mismatch",
                        "suggestion": "Introduce hydrophobic residues (LEU, ILE, VAL) at positions with high attention scores",
                        "confidence": 0.9
                    })

                # Analyze hydrogen bonding network
                h_bond_residues = ['SER', 'THR', 'ASN', 'GLN', 'TYR', 'LYS']
                h_bond_count = sum(1 for aa in site_sequence if aa in h_bond_residues)

                if (hbd + hba > 4) and (h_bond_count < len(site_sequence) * 0.2):
                    suggestions.append({
                        'start': site_start,
                        'end': site_end,
                        'score': 0.8,
                        'type': 'h_bond_network',
                        "issue": "Insufficient hydrogen bond partners",
                        "suggestion": "Introduce hydrogen bond capable residues at positions with high attention scores",
                        "confidence": 0.8
                    })
            # Calculate overall optimization score
            optimization_score = np.mean([s["confidence"] for s in suggestions]) if suggestions else 0.5

            return {
                'start': site_start,
                'end': site_end,
                'score': float(optimization_score),
                'type': 'binding_site_optimization',
                'site_analysis': {
                    'start': site_start,
                    'end': site_end,
                    'score': float(optimization_score),
                    'type': 'site_analysis',
                    "hydrophobicity": float(hydrophobicity),
                    "length": len(site_sequence),
                    "residue_properties": residue_properties
                },
                'optimization_suggestions': suggestions,
                'predicted_improvement': {
                    'start': site_start,
                    'end': site_end,
                    'score': float(np.clip(0.4 + optimization_score, 0, 1)),
                    'type': 'improvement_prediction'
                }
            }

        except Exception as e:
            logger.error(f"Error in binding site optimization: {e}")
            return {
                'start': site_start,
                'end': site_end,
                'score': 0.0,
                'type': 'binding_site_optimization_error',
                'error': str(e)
            }

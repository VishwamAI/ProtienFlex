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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from Bio.PDB import *
import biotite.structure as struc
import biotite.structure.io as strucio
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ProteinNLPAnalyzer:
    def __init__(self):
        try:
            # Initialize ESM-based protein language model for sequence analysis
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model.eval()

            # Initialize text generation pipeline for natural language descriptions
            self.nlp_pipeline = pipeline("text2text-generation", model="facebook/bart-large")

            logger.info("NLP analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error loading NLP model: {e}")
            raise

    def analyze_sequence(self, sequence):
        """Analyze protein sequence and generate natural language description"""
        try:
            # Basic protein analysis
            protein_analysis = ProteinAnalysis(sequence)
            molecular_weight = protein_analysis.molecular_weight()
            isoelectric_point = protein_analysis.isoelectric_point()
            aromaticity = protein_analysis.aromaticity()
            instability_index = protein_analysis.instability_index()
            secondary_structure = protein_analysis.secondary_structure_fraction()

            # Generate sequence embeddings for advanced analysis
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.logits

            # Calculate sequence properties
            hydrophobicity = self._calculate_hydrophobicity(sequence)
            complexity = self._calculate_sequence_complexity(sequence)
            domains = self._predict_domains(sequence)
            stability = self._analyze_stability(sequence)
            function_prediction = self._predict_function(embeddings)

            # Generate comprehensive natural language description
            description = self._generate_description(
                sequence_length=len(sequence),
                molecular_weight=molecular_weight,
                isoelectric_point=isoelectric_point,
                aromaticity=aromaticity,
                instability_index=instability_index,
                secondary_structure=secondary_structure,
                hydrophobicity=hydrophobicity,
                complexity=complexity,
                domains=domains,
                stability=stability,
                function=function_prediction
            )

            # Find the most significant domain for position information
            if domains:
                best_domain = max(domains, key=lambda x: x['confidence'])
                start = best_domain['start']
                end = best_domain['end']
            else:
                # Default to analyzing the first portion of the sequence
                start = 0
                end = min(50, len(sequence))

            # Calculate overall analysis score
            analysis_scores = []
            if stability and stability.get('is_stable') is not None:
                analysis_scores.append(0.8 if stability['is_stable'] else 0.4)
            analysis_scores.append(complexity / 100)  # Normalized complexity score
            if domains:
                analysis_scores.append(max(d['confidence'] for d in domains))
            overall_score = np.mean(analysis_scores) if analysis_scores else 0.5

            return {
                'start': start,
                'end': end,
                'score': float(overall_score),
                'type': 'sequence_analysis',
                'description': description,
                'properties': {
                    'start': start,
                    'end': end,
                    'score': float(overall_score),
                    'type': 'property_analysis',
                    'molecular_weight': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(molecular_weight/1000),  # Normalize by 1000
                        'type': 'molecular_weight',
                        'value': float(molecular_weight)
                    },
                    'isoelectric_point': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(min(1.0, isoelectric_point/14)),  # Normalize by pH scale
                        'type': 'isoelectric_point',
                        'value': float(isoelectric_point)
                    },
                    'aromaticity': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(aromaticity),
                        'type': 'aromaticity',
                        'value': float(aromaticity)
                    },
                    'instability_index': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(max(0, 1 - instability_index/100)),
                        'type': 'instability_index',
                        'value': float(instability_index)
                    },
                    'hydrophobicity': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(min(1.0, (hydrophobicity + 4.5)/9.0)),  # Normalize to [0,1]
                        'type': 'hydrophobicity',
                        'value': float(hydrophobicity)
                    },
                    'complexity': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(complexity/100),
                        'type': 'complexity',
                        'value': float(complexity)
                    },
                    'domains': domains,
                    'stability': stability,
                    'predicted_function': function_prediction
                }
            }

        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return {
                'start': 0,
                'end': len(sequence),
                'score': 0.0,
                'type': 'sequence_analysis_error',
                'error': str(e)
            }

    def _calculate_hydrophobicity(self, sequence):
        """Calculate average hydrophobicity using Kyte-Doolittle scale"""
        hydropathy_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        return np.mean([hydropathy_scale.get(aa, 0) for aa in sequence])

    def _calculate_sequence_complexity(self, sequence):
        """Calculate sequence complexity based on amino acid diversity"""
        unique_aa = len(set(sequence))
        return (unique_aa / 20) * 100  # Normalize to percentage

    def _predict_domains(self, sequence):
        """Predict protein domains using sequence patterns"""
        domains = []
        window_size = 50
        stride = 10

        # Analyze sequence patterns and structure
        for i in range(0, len(sequence) - window_size, stride):
            segment = sequence[i:i+window_size]

            # Calculate segment properties
            hydrophobicity = self._calculate_hydrophobicity(segment)
            complexity = self._calculate_sequence_complexity(segment)

            # Get segment embeddings for structural analysis
            inputs = self.tokenizer(segment, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                segment_embeddings = outputs.logits

            # Analyze structural features
            embedding_variance = torch.var(segment_embeddings).item()
            pattern_strength = torch.mean(torch.abs(segment_embeddings)).item()

            # Identify domains based on multiple features
            if (hydrophobicity > 1.5 and complexity > 60 and pattern_strength > 0.5):
                domains.append({
                    "start": i+1,
                    "end": i+window_size,
                    "score": float(pattern_strength),
                    "type": "Functional",
                    "properties": {
                        "start": i+1,
                        "end": i+window_size,
                        "score": float(pattern_strength),
                        "type": "domain_properties",
                        "hydrophobicity": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(min(1.0, (hydrophobicity + 4.5)/9.0)),
                            "type": "hydrophobicity",
                            "value": float(hydrophobicity)
                        },
                        "complexity": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(complexity/100),
                            "type": "complexity",
                            "value": float(complexity)
                        },
                        "structural_score": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(min(1.0, embedding_variance)),
                            "type": "structural_score",
                            "value": float(embedding_variance)
                        }
                    }
                })
            elif (abs(hydrophobicity) > 2.0 and embedding_variance > 0.1):
                domain_type = "Hydrophobic" if hydrophobicity > 0 else "Hydrophilic"
                domains.append({
                    "start": i+1,
                    "end": i+window_size,
                    "score": float(abs(hydrophobicity)/4),
                    "type": domain_type,
                    "properties": {
                        "start": i+1,
                        "end": i+window_size,
                        "score": float(abs(hydrophobicity)/4),
                        "type": "domain_properties",
                        "hydrophobicity": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(min(1.0, (hydrophobicity + 4.5)/9.0)),
                            "type": "hydrophobicity",
                            "value": float(hydrophobicity)
                        },
                        "complexity": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(complexity/100),
                            "type": "complexity",
                            "value": float(complexity)
                        },
                        "structural_score": {
                            "start": i+1,
                            "end": i+window_size,
                            "score": float(min(1.0, embedding_variance)),
                            "type": "structural_score",
                            "value": float(embedding_variance)
                        }
                    }
                })
        return domains

    def _analyze_stability(self, sequence):
        """Analyze protein stability based on sequence properties"""
        try:
            analysis = ProteinAnalysis(sequence)
            instability_index = analysis.instability_index()
            flexibility = analysis.flexibility()
            stability = {
                'start': 0,
                'end': len(sequence),
                'score': float(max(0, 1 - instability_index/100)),  # Normalize score
                'type': 'stability_analysis',
                'properties': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(max(0, 1 - instability_index/100)),
                    'type': 'stability_properties',
                    'instability_index': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(max(0, 1 - instability_index/100)),
                        'type': 'instability_index',
                        'value': float(instability_index)
                    },
                    'flexibility': {
                        'start': 0,
                        'end': len(sequence),
                        'score': float(min(1.0, np.mean(flexibility))),
                        'type': 'flexibility',
                        'value': float(np.mean(flexibility))
                    },
                    'is_stable': {
                        'start': 0,
                        'end': len(sequence),
                        'score': 1.0 if instability_index < 40 else 0.0,
                        'type': 'stability_state',
                        'value': instability_index < 40
                    }
                }
            }
            return stability
        except Exception as e:
            logger.error(f"Error in stability analysis: {e}")
            return {
                'start': 0,
                'end': len(sequence),
                'score': 0.0,
                'type': 'stability_analysis_error',
                'error': str(e)
            }

    def _predict_function(self, embeddings):
        """Predict protein function based on sequence embeddings"""
        # Simple function prediction based on embedding patterns
        embedding_features = torch.cat([
            torch.mean(embeddings, dim=0),
            torch.var(embeddings, dim=0),
            torch.max(embeddings, dim=0)[0],
            torch.min(embeddings, dim=0)[0]
        ])

        # Calculate feature scores for different function categories
        enzymatic_score = torch.sigmoid(torch.sum(embedding_features[:100])).item()
        structural_score = torch.sigmoid(torch.sum(embedding_features[100:200])).item()
        regulatory_score = torch.sigmoid(torch.sum(embedding_features[200:300])).item()

        # Determine function based on highest score
        scores = [(enzymatic_score, "enzymatic activity"),
                 (structural_score, "structural role"),
                 (regulatory_score, "regulatory function")]

        max_score, predicted_function = max(scores)
        confidence = "Likely" if max_score > 0.7 else "Possible"

        return {
            'start': 0,
            'end': embeddings.shape[1],
            'score': float(max_score),
            'type': 'function_prediction',
            'prediction': {
                'start': 0,
                'end': embeddings.shape[1],
                'score': float(max_score),
                'type': 'predicted_function',
                'function': f"{confidence} {predicted_function}",
                'confidence': confidence,
                'scores': {
                    'enzymatic': {
                        'start': 0,
                        'end': embeddings.shape[1],
                        'score': float(enzymatic_score),
                        'type': 'enzymatic_score'
                    },
                    'structural': {
                        'start': 0,
                        'end': embeddings.shape[1],
                        'score': float(structural_score),
                        'type': 'structural_score'
                    },
                    'regulatory': {
                        'start': 0,
                        'end': embeddings.shape[1],
                        'score': float(regulatory_score),
                        'type': 'regulatory_score'
                    }
                }
            }
        }

    def _generate_description(self, **properties):
        """Generate natural language description of protein properties"""
        # Basic information
        description = f"""This protein sequence consists of {properties['sequence_length']} amino acids with a molecular weight of {properties['molecular_weight']:.2f} Da. """

        # Structural information
        helix, sheet, coil = properties['secondary_structure']
        description += f"""The predicted secondary structure composition includes {helix:.1%} alpha-helix, {sheet:.1%} beta-sheet, and {coil:.1%} random coil. """

        # Stability information
        if properties['stability']:
            stability_status = "stable" if properties['stability']['is_stable'] else "potentially unstable"
            description += f"""The protein is predicted to be {stability_status} with an instability index of {properties['stability']['index']:.1f}. """
            description += f"""It shows {'high' if properties['stability']['flexibility'] > 0.5 else 'moderate' if properties['stability']['flexibility'] > 0.3 else 'low'} flexibility. """

        # Chemical properties
        description += f"""It has an isoelectric point of {properties['isoelectric_point']:.2f} and an aromaticity of {properties['aromaticity']:.2f}. """

        # Domain information
        if properties['domains']:
            description += f"""The sequence contains {len(properties['domains'])} predicted domains: {', '.join(properties['domains'])}. """

        # Sequence properties
        hydro_desc = "hydrophobic" if properties['hydrophobicity'] > 0 else "hydrophilic"
        description += f"""The sequence is generally {hydro_desc} with an average hydrophobicity of {properties['hydrophobicity']:.2f}. """
        description += f"""The sequence complexity is {properties['complexity']:.1f}%, indicating {'high' if properties['complexity'] > 75 else 'moderate' if properties['complexity'] > 50 else 'low'} amino acid diversity. """

        # Functional prediction
        description += f"""Based on the sequence analysis, this protein may have {properties['function'].lower()}."""

        return description

    def compare_sequences(self, seq1: str, seq2: str) -> Dict:
        """Compare two protein sequences and analyze their similarities"""
        try:
            # Get embeddings for both sequences
            inputs1 = self.tokenizer(seq1, return_tensors="pt", padding=True, truncation=True)
            inputs2 = self.tokenizer(seq2, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                emb1 = self.model(**inputs1).logits
                emb2 = self.model(**inputs2).logits

            # Calculate similarity score
            similarity = torch.cosine_similarity(emb1.mean(dim=1), emb2.mean(dim=1)).item()

            # Calculate additional comparison metrics
            length_similarity = 1 - abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))

            return {
                'start': 0,
                'end': max(len(seq1), len(seq2)),
                'score': float(similarity),
                'type': 'sequence_comparison',
                'comparison': {
                    'start': 0,
                    'end': max(len(seq1), len(seq2)),
                    'score': float(similarity),
                    'type': 'comparison_details',
                    'metrics': {
                        'similarity': {
                            'start': 0,
                            'end': max(len(seq1), len(seq2)),
                            'score': float(similarity),
                            'type': 'similarity_score',
                            'value': float(similarity)
                        },
                        'length_comparison': {
                            'start': 0,
                            'end': max(len(seq1), len(seq2)),
                            'score': float(length_similarity),
                            'type': 'length_similarity',
                            'value': abs(len(seq1) - len(seq2))
                        }
                    }
                }
            }
        except Exception as e:
            return {
                'start': 0,
                'end': max(len(seq1), len(seq2)),
                'score': 0.0,
                'type': 'sequence_comparison_error',
                'error': str(e)
            }

    def answer_question(self, sequence: str, question: str) -> Dict:
        """Answer questions about protein sequence using NLP analysis"""
        try:
            # Analyze sequence first
            analysis = self.analyze_sequence(sequence)

            # Generate answer using NLP pipeline
            context = f"Protein sequence analysis: {analysis['description']}"
            answer = self.nlp_pipeline(f"Context: {context}\nQuestion: {question}")[0]['generated_text']

            # Calculate answer confidence based on analysis score and question relevance
            answer_confidence = float(analysis['score']) * 0.8  # Discount factor for answer uncertainty

            return {
                'start': 0,
                'end': len(sequence),
                'score': float(analysis['score']),
                'type': 'protein_qa',
                'response': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(answer_confidence),
                    'type': 'qa_response',
                    'question': {
                        'start': 0,
                        'end': len(question),
                        'score': 1.0,
                        'type': 'question_text',
                        'text': question
                    },
                    'answer': {
                        'start': 0,
                        'end': len(answer),
                        'score': float(answer_confidence),
                        'type': 'answer_text',
                        'text': answer
                    }
                }
            }
        except Exception as e:
            return {
                'start': 0,
                'end': len(sequence),
                'score': 0.0,
                'type': 'protein_qa_error',
                'error': str(e)
            }

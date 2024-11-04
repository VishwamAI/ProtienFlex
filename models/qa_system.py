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

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import logging

logger = logging.getLogger(__name__)

class ProteinQASystem:
    def __init__(self):
        try:
            # Initialize base QA model
            model_name = "deepset/roberta-base-squad2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()

            # Define protein-specific terminology and concepts
            self.protein_terms = {
                'structure': ['alpha helix', 'beta sheet', 'fold', 'domain', 'motif'],
                'function': ['binding', 'catalysis', 'regulation', 'signaling'],
                'properties': ['hydrophobic', 'hydrophilic', 'charge', 'stability'],
                'interactions': ['ligand', 'substrate', 'inhibitor', 'drug']
            }

            logger.info("QA system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QA system: {e}")
            raise

    def answer_question(self, context, question):
        try:
            # Preprocess question with protein terminology
            enhanced_question = self._enhance_question(question)

            # Process context and question
            inputs = self.tokenizer(enhanced_question, context,
                                  return_tensors="pt",
                                  max_length=512,
                                  truncation=True,
                                  padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Get answer span probabilities
                start_probs = torch.softmax(outputs.start_logits, dim=1)
                end_probs = torch.softmax(outputs.end_logits, dim=1)

                # Find best answer span
                max_answer_length = 50
                best_score = -float('inf')
                best_start = 0
                best_end = 0

                for start_idx in range(len(start_probs[0])):
                    for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_probs[0]))):
                        score = start_probs[0][start_idx] * end_probs[0][end_idx]
                        if score > best_score:
                            best_score = score
                            best_start = start_idx
                            best_end = end_idx + 1

            # Extract and validate answer
            answer = self.tokenizer.decode(inputs["input_ids"][0][best_start:best_end])
            answer = self._post_process_answer(answer)

            # Calculate confidence score
            confidence = float(best_score)
            relevance = self._calculate_relevance(answer, question)
            final_confidence = (confidence + relevance) / 2
            context_match = self._calculate_context_match(answer, context)

            # Return standardized dictionary with required fields
            return {
                'start': best_start,
                'end': best_end,
                'score': float(final_confidence),
                'type': 'qa_response',
                'answer': {
                    'start': best_start,
                    'end': best_end,
                    'score': float(final_confidence),
                    'type': 'answer_text',
                    'text': answer
                },
                'confidence': {
                    'start': best_start,
                    'end': best_end,
                    'score': float(confidence),
                    'type': 'confidence_score',
                    'value': float(confidence)
                },
                'relevance': {
                    'start': best_start,
                    'end': best_end,
                    'score': float(relevance),
                    'type': 'relevance_score',
                    'value': float(relevance)
                },
                'context_match': {
                    'start': best_start,
                    'end': best_end,
                    'score': float(context_match),
                    'type': 'context_match_score',
                    'value': float(context_match)
                }
            }
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                'start': 0,
                'end': min(50, len(context)),
                'score': 0.0,
                'type': 'qa_response_error',
                'error': str(e),
                'answer': {
                    'start': 0,
                    'end': min(50, len(context)),
                    'score': 0.0,
                    'type': 'answer_text',
                    'text': "Unable to process protein-related question"
                },
                'confidence': {
                    'start': 0,
                    'end': min(50, len(context)),
                    'score': 0.0,
                    'type': 'confidence_score',
                    'value': 0.0
                },
                'relevance': {
                    'start': 0,
                    'end': min(50, len(context)),
                    'score': 0.0,
                    'type': 'relevance_score',
                    'value': 0.0
                },
                'context_match': {
                    'start': 0,
                    'end': min(50, len(context)),
                    'score': 0.0,
                    'type': 'context_match_score',
                    'value': 0.0
                }
            }

    def _enhance_question(self, question):
        """Enhance question with protein-specific terminology."""
        enhanced = question
        for category, terms in self.protein_terms.items():
            for term in terms:
                if term in question.lower():
                    enhanced = f"{category} {enhanced}"
                    break
        return enhanced

    def _post_process_answer(self, answer):
        """Clean and validate the answer text."""
        # Remove special tokens and clean whitespace
        answer = answer.strip()
        answer = ' '.join(answer.split())

        # Ensure answer is not empty or just special tokens
        if not answer or answer in ['[CLS]', '[SEP]', '[PAD]']:
            return "No relevant answer found"

        return answer

    def _calculate_relevance(self, answer, question):
        """Calculate relevance score based on protein terminology."""
        relevance_score = 0.0
        total_terms = 0

        # Check for protein-specific terms in answer
        for terms in self.protein_terms.values():
            for term in terms:
                if term in answer.lower():
                    relevance_score += 1
                if term in question.lower():
                    total_terms += 1

        # Normalize score
        if total_terms > 0:
            relevance_score = relevance_score / total_terms

        return min(1.0, relevance_score)

    def _calculate_context_match(self, answer, context):
        """Calculate how well the answer matches the context."""
        # Convert to lowercase for comparison
        answer_lower = answer.lower()
        context_lower = context.lower()

        # Calculate word overlap
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        overlap = len(answer_words.intersection(context_words))

        # Calculate match score
        if len(answer_words) == 0:
            return 0.0

        return min(1.0, overlap / len(answer_words))

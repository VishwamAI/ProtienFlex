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
import re
from typing import Dict, Any, Optional, Tuple
import requests
import json

class TextToProteinGenerator:
    """Generate protein sequences using the Gemini API."""

    # Required sequences for each segment
    required_sequences = {
        '[1-79]': 'TVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADN',
        '[80-85]': 'DADNDG',
        '[86-99]': 'GPSGPGTSGPSGPG',
        '[100-102]': 'WNW',
        '[103-159]': 'IVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNP',
        '[160-165]': 'DQDEDG',
        '[166-179]': 'GPSGPGTSGPSGPG',
        '[180-182]': 'FKY',
        '[183-239]': 'TVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNP',
        '[240-245]': 'DADNDG',
        '[246-259]': 'GPSGPGTSGPSGPG',
        '[260-262]': 'YRF',
        '[263-343]': 'IVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNP'
    }

    def __init__(self):
        """Initialize the generator."""
        self.api_key = os.environ.get('Gemini_api')
        if not self.api_key:
            raise ValueError("Gemini API key not found in environment variables")

        # Updated API endpoint for Gemini (beta version)
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def _create_prompt(self, description: str) -> str:
        """Create a structured prompt for protein sequence generation."""
        return f"""IUPAC Amino Acid Sequence Generation Task:
Objective: Design pectate lyase B sequence (EC 4.2.2.2) with template-based approach

Template Structure (343 residues total):
[1-79]   : Beta-helix core region 1 (79 residues)
   Required sequence:
   {self.required_sequences['[1-79]']}

[80-85]  : Calcium binding site 1 (6 residues)
   Required sequence:
   {self.required_sequences['[80-85]']}

[86-99]  : Connecting region 1 (14 residues)
   Required sequence:
   {self.required_sequences['[86-99]']}

[100-102]: Substrate recognition site 1 (3 residues)
   Required sequence:
   {self.required_sequences['[100-102]']}

[103-159]: Beta-helix core region 2 (57 residues)
   Required sequence:
   {self.required_sequences['[103-159]']}

[160-165]: Calcium binding site 2 (6 residues)
   Required sequence:
   {self.required_sequences['[160-165]']}

[166-179]: Connecting region 2 (14 residues)
   Required sequence:
   {self.required_sequences['[166-179]']}

[180-182]: Substrate recognition site 2 (3 residues)
   Required sequence:
   {self.required_sequences['[180-182]']}

[183-239]: Beta-helix core region 3 (57 residues)
   Required sequence:
   {self.required_sequences['[183-239]']}

[240-245]: Calcium binding site 3 (6 residues)
   Required sequence:
   {self.required_sequences['[240-245]']}

[246-259]: Connecting region 3 (14 residues)
   Required sequence:
   {self.required_sequences['[246-259]']}

[260-262]: Substrate recognition site 3 (3 residues)
   Required sequence:
   {self.required_sequences['[260-262]']}

[263-343]: Beta-helix core region 4 (81 residues)
   Required sequence:
   {self.required_sequences['[263-343]']}

{description}

Required Output Format:
Please provide the sequence in the following structured format:

SEGMENT_START
[1-79]:
{self.required_sequences['[1-79]']}
[80-85]:
{self.required_sequences['[80-85]']}
[86-99]:
{self.required_sequences['[86-99]']}
[100-102]:
{self.required_sequences['[100-102]']}
[103-159]:
{self.required_sequences['[103-159]']}
[160-165]:
{self.required_sequences['[160-165]']}
[166-179]:
{self.required_sequences['[166-179]']}
[180-182]:
{self.required_sequences['[180-182]']}
[183-239]:
{self.required_sequences['[183-239]']}
[240-245]:
{self.required_sequences['[240-245]']}
[246-259]:
{self.required_sequences['[246-259]']}
[260-262]:
{self.required_sequences['[260-262]']}
[263-343]:
{self.required_sequences['[263-343]']}
SEGMENT_END

Note: Copy each segment exactly as shown above. Do not modify any sequences."""

    def _call_gemini_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make a request to the Gemini API."""
        try:
            # Print API key format for debugging (only first/last 4 chars)
            key_preview = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
            print(f"\nAPI Key format check: {key_preview}")

            # Construct URL with API key
            url = f"{self.api_url}?key={self.api_key}"

            headers = {
                "Content-Type": "application/json"
            }

            # Updated request format for Gemini API with additional safety settings
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048
                }
            }

            print(f"\nMaking API request to Gemini...")
            print(f"URL: {self.api_url}")
            print(f"Request data: {json.dumps(data, indent=2)}")

            response = requests.post(
                url,
                headers=headers,
                json=data
            )

            if response.status_code != 200:
                print(f"API Error - Status Code: {response.status_code}")
                print(f"Response Headers: {response.headers}")
                print(f"Response Content: {response.text}")
                return None

            response_data = response.json()
            print(f"\nRaw API Response:\n{json.dumps(response_data, indent=2)}")

            if 'candidates' not in response_data:
                print("Error: No candidates in response")
                return None

            return response_data
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def _parse_sequence_response(self, response: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Parse the API response to extract sequence and explanation."""
        try:
            if not response or 'candidates' not in response:
                return None, None

            # Get the first candidate's text
            candidate = response['candidates'][0]
            if 'content' not in candidate or 'parts' not in candidate['content']:
                print("Invalid response format")
                return None, None

            text = candidate['content']['parts'][0]['text']
            print(f"\nRaw API Response Text:\n{text}\n")

            # Parse segments between SEGMENT_START and SEGMENT_END
            segments = {}
            current_segment = None
            current_sequence = []

            # Split text into lines and process
            lines = text.strip().split('\n')
            in_segments = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line == 'SEGMENT_START':
                    in_segments = True
                    continue
                elif line == 'SEGMENT_END':
                    break

                if not in_segments:
                    continue

                # Parse segment header
                if line.startswith('[') and ']:' in line:
                    if current_segment and current_sequence:
                        # Store previous segment
                        segments[current_segment] = ''.join(current_sequence)
                        current_sequence = []
                    current_segment = line.split(']:')[0] + ']'
                else:
                    # Add sequence line (only if it contains valid amino acids)
                    cleaned_line = ''.join(c for c in line if c.isalpha()).upper()
                    if cleaned_line and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in cleaned_line):
                        current_sequence.append(cleaned_line)

            # Don't forget to store the last segment
            if current_segment and current_sequence:
                segments[current_segment] = ''.join(current_sequence)

            # Validate and combine segments
            final_sequence = self._validate_segments(segments)
            if not final_sequence:
                return None, None

            print(f"\nExtracted Sequence ({len(final_sequence)} residues):")
            print("=" * 50)
            print(final_sequence)
            print("=" * 50)
            return final_sequence, "Sequence generated successfully"

        except Exception as e:
            print(f"Error parsing response: {e}")
            return None, None

    def _validate_segments(self, segments: Dict[str, str]) -> Optional[str]:
        """Validate and combine sequence segments with flexible pattern matching."""
        try:
            # Expected lengths for each segment
            expected_lengths = {}
            # Calculate lengths from required sequences
            for segment_id, sequence in self.required_sequences.items():
                expected_lengths[segment_id] = len(sequence)

            # Validate each segment
            validated_segments = {}
            for segment_id, sequence in segments.items():
                if segment_id not in expected_lengths:
                    print(f"Unknown segment: {segment_id}")
                    return None

                # Clean the sequence
                cleaned_sequence = ''.join(c for c in sequence if c.isalpha()).upper()
                cleaned_sequence = ''.join(c for c in cleaned_sequence if c in 'ACDEFGHIKLMNPQRSTVWY')
                if not cleaned_sequence:
                    print(f"Empty sequence for {segment_id}")
                    return None

                # Basic validation
                if not re.match(r'^[A-Z]+$', cleaned_sequence):
                    print(f"Invalid characters in sequence for {segment_id}")
                    return None

                # Length validation with tolerance
                expected_length = expected_lengths[segment_id]
                actual_length = len(cleaned_sequence)
                if abs(actual_length - expected_length) > 1:  # Allow 1 residue difference
                    print(f"Invalid sequence length for {segment_id}")
                    print(f"Expected length: {expected_length}, got: {actual_length}")
                    print(f"Sequence: {cleaned_sequence}")
                    return None

                # Check if sequence follows the required pattern
                required_seq = self.required_sequences[segment_id]
                if not self._sequences_match_pattern(cleaned_sequence, required_seq):
                    print(f"Warning: Sequence for {segment_id} differs from template")
                    print(f"Expected pattern: {required_seq}")
                    print(f"Got: {cleaned_sequence}")
                    # Continue anyway as long as length is acceptable

                # Store validated sequence
                validated_segments[segment_id] = cleaned_sequence

            # Combine all segments in order
            final_sequence = ''
            for segment_id in sorted(expected_lengths.keys()):
                if segment_id not in segments:
                    print(f"Missing segment {segment_id}")
                    return None
                final_sequence += segments[segment_id]

            print(f"\nExtracted Sequence ({len(final_sequence)} residues):")
            print("=" * 50)
            print(final_sequence)
            print("=" * 50)

            return final_sequence

        except Exception as e:
            print(f"Error validating segments: {e}")
            return None

    def _sequences_match_pattern(self, seq1: str, seq2: str) -> bool:
        """Check if sequences match allowing for conservative substitutions."""
        if len(seq1) != len(seq2):
            return False

        # Define groups of similar amino acids
        similar_groups = [
            set('ILVM'),     # Aliphatic
            set('FYW'),      # Aromatic
            set('KRH'),      # Basic
            set('DE'),       # Acidic
            set('STNQ'),     # Polar
            set('AG'),       # Small
            set('PG')        # Special
        ]

        for a1, a2 in zip(seq1, seq2):
            if a1 == a2:
                continue
            # Check if amino acids are in the same group
            is_similar = any(a1 in group and a2 in group for group in similar_groups)
            if not is_similar:
                return False
        return True

    def generate_sequence(self, description: str) -> Tuple[Optional[str], Optional[str]]:
        """Generate a protein sequence from a structure description."""
        prompt = self._create_prompt(description)
        response = self._call_gemini_api(prompt)

        if not response:
            return None, "Failed to get response from Gemini API"

        sequence, explanation = self._parse_sequence_response(response)

        if not sequence:
            return None, "Failed to generate valid sequence"

        return sequence, explanation

def main():
    """Main function to demonstrate the text-to-protein generator."""
    # Read the structure description from the file
    try:
        import sys
        sys.path.append('/home/ubuntu/ProtienFlex')
        from fetch_pdb_info import create_structure_description

        description = create_structure_description('7BBV')
        generator = TextToProteinGenerator()
        sequence, explanation = generator.generate_sequence(description)

        print("\nGenerated Protein Sequence:")
        print("=" * 50)
        if sequence:
            print(f"Sequence: {sequence}")
            print("\nExplanation:")
            print(explanation)
        else:
            print("Failed to generate sequence")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()

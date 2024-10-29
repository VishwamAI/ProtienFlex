from flask import Flask
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_protein_prediction():
    # Insulin sequence (a well-known protein)
    insulin_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"

    try:
        # Make prediction request
        response = requests.post(
            'http://localhost:5000/predict',
            json={'sequence': insulin_sequence},
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("Prediction successful!")
            logger.info(f"Confidence Score: {result.get('confidence_score', 'N/A')}%")
            logger.info(f"Description: {result.get('description', 'N/A')}")
            logger.info("Contact map shape: " + str(len(result.get('contact_map', []))))
            logger.info("PDB string length: " + str(len(result.get('pdb_string', ''))))
            return True
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(response.text)
            return False

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Start the test
    logger.info("Starting protein prediction test...")
    success = test_protein_prediction()
    logger.info(f"Test {'succeeded' if success else 'failed'}")

import requests
import logging
import json
import time
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_protein_visualization():
    # Test sequence (insulin)
    test_cases = [
        {
            "name": "Insulin",
            "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
        },
        {
            "name": "Short peptide",
            "sequence": "ACDKEFGH"
        }
    ]

    base_url = "http://localhost:5000"

    # Test homepage
    try:
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        logger.info("Homepage test: SUCCESS")
    except Exception as e:
        logger.error(f"Homepage test failed: {e}")
        return False

    # Test predictions
    for test_case in test_cases:
        try:
            logger.info(f"Testing {test_case['name']}...")

            response = requests.post(
                f"{base_url}/predict",
                json={"sequence": test_case["sequence"]},
                headers={"Content-Type": "application/json"}
            )

            assert response.status_code == 200
            result = response.json()

            # Verify required fields
            assert "pdb_string" in result
            assert "confidence_score" in result
            assert "contact_map" in result
            assert "description" in result

            # Verify data types and ranges
            assert isinstance(result["confidence_score"], (int, float))
            assert 0 <= result["confidence_score"] <= 100
            assert len(result["contact_map"]) == len(test_case["sequence"])

            logger.info(f"{test_case['name']} test results:")
            logger.info(f"Confidence Score: {result['confidence_score']}%")
            logger.info(f"PDB string length: {len(result['pdb_string'])}")
            logger.info(f"Contact map dimensions: {len(result['contact_map'])}x{len(result['contact_map'][0])}")
            logger.info(f"Description: {result['description'][:100]}...")
            logger.info(f"{test_case['name']} test: SUCCESS\n")

        except Exception as e:
            logger.error(f"{test_case['name']} test failed: {e}")
            return False

    return True

if __name__ == "__main__":
    logger.info("Starting integration tests...")

    # Wait for Flask to start
    time.sleep(5)

    success = test_protein_visualization()

    if success:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)

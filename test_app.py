import requests
import json

def test_prediction():
    url = 'http://localhost:5000/predict'
    sequence = 'MVKVGVNG'  # Simple test sequence

    response = requests.post(url, json={'sequence': sequence})
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == '__main__':
    test_prediction()

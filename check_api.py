import requests

API_URL = "http://20.245.200.125:8000/v1/models"
try:
    print(f"Connecting to {API_URL}...")
    response = requests.get(API_URL, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("API is ONLINE!")
        print("Models:")
        print(response.json())
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")

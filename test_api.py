import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY")
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 5,
    "offset": 0,
    "filters[state]": "Maharashtra",
    "filters[commodity]": "Tomato"
}

print("API Key Found:", bool(API_KEY))

try:
    response = requests.get(url, params=params, timeout=60)

    print("Status Code:", response.status_code)
    print("URL:", response.url)
    print("Response Text:", response.text[:1000])

    data = response.json()
    records = data.get("records", [])

    print("Records Found:", len(records))

    if records:
        print("First Record:", records[0])

except requests.exceptions.Timeout:
    print("ERROR: API request timed out. data.gov.in server is slow.")
except Exception as e:
    print("ERROR:", e)
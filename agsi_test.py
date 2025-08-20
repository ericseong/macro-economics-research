import requests

headers = {"x-key": "AGSI_API_KEY"}
r = requests.get("https://agsi.gie.eu/api", params={"type": "eu"}, headers=headers, timeout=30)
r.raise_for_status()
print(r.json())

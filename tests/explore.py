# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from datasets import load_dataset
# from data.scraper import scrape_dtc_detail
# import requests
# from bs4 import BeautifulSoup

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
# }

# response = requests.get("https://obdguide.com/en/codes/p0301", headers=HEADERS, timeout=10)
# soup = BeautifulSoup(response.text, "html.parser")

# # Print the raw page structure
# print("H1 tags:", [h.get_text(strip=True) for h in soup.find_all("h1")])
# print("H2 tags:", [h.get_text(strip=True) for h in soup.find_all("h2")])
# print("H3 tags:", [h.get_text(strip=True) for h in soup.find_all("h3")])
# print("---")
# print("All list items:", [li.get_text(strip=True) for li in soup.find_all("li")][:20])
# result = scrape_dtc_detail("P0301")
# print(result)

# dataset = load_dataset("Epitech/obd-codes-fine-tune")
# print(dataset)
# print("---")
# print(dataset['train'][0])
# print("---")
# print(dataset['train'].column_names)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scraper import scrape_dtc_detail
import json

result = scrape_dtc_detail("P0301")
print(json.dumps(result, indent=2))
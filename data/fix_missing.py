import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from data.scraper import scrape_dtc_detail
from tqdm import tqdm
import time

def fix_missing():
    # Load current scraped data
    with open("data/scraped_obd.json", "r") as f:
        scraped = json.load(f)
    
    scraped_codes = {d["dtc_code"] for d in scraped}
    
    # Load merged data to find codes with no rich content
    with open("data/merged_obd.json", "r") as f:
        merged = json.load(f)
    
    # Find codes that exist but have no symptoms/causes
    missing = [
        d["dtc_code"] for d in merged 
        if not d["causes"] and not d["symptoms"]
    ]
    
    print(f"Found {len(missing)} codes with no rich content")
    print(f"Attempting to scrape them...")
    
    newly_scraped = []
    for code in tqdm(missing):
        result = scrape_dtc_detail(code)
        if result and (result["causes"] or result["symptoms"]):
            newly_scraped.append(result)
        time.sleep(0.5)
    
    print(f"Successfully scraped {len(newly_scraped)} previously missing codes")
    
    # Merge into scraped data
    scraped.extend(newly_scraped)
    
    with open("data/scraped_obd.json", "w") as f:
        json.dump(scraped, f, indent=2)
    
    print("Updated scraped_obd.json")
    return newly_scraped

if __name__ == "__main__":
    fix_missing()
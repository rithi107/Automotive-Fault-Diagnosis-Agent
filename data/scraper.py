import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm

BASE_URL = "https://obdguide.com/en"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_all_codes() -> list:
    """Get all DTC codes from the main page"""
    url = f"{BASE_URL}/codes"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        codes = []
        for link in soup.find_all("a", href=True):
            href = link['href']
            # Pattern is /en/p0301 or /en/b0001 etc
            parts = href.strip("/").split("/")
            if len(parts) == 2 and parts[0] == "en" and len(parts[1]) >= 5:
                code = parts[1].upper()
                codes.append(code)
        
        return list(set(codes))
    except Exception as e:
        print(f"Error fetching codes list: {e}")
        return []

def scrape_dtc_detail(code: str) -> dict:
    """Scrape detailed info for a single DTC code"""
    url = f"{BASE_URL}/{code.lower()}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        result = {
            "dtc_code": code,
            "fault_name": "",
            "symptoms": [],
            "causes": [],
            "repair_steps": []
        }
        
        # H1 is the DTC code itself — get fault name from h3
        for h3 in soup.find_all("h3"):
            text = h3.get_text(strip=True)
            if "mean" in text.lower():
                # Next sibling has the description
                next_elem = h3.find_next_sibling()
                if next_elem:
                    raw = next_elem.get_text(strip=True)
                    if "indicates:" in raw:
                        result["fault_name"] = raw.split("indicates:")[-1].strip()
                    else:
                        result["fault_name"] = raw
                break
        
        # Get sections by H2
        for h2 in soup.find_all("h2"):
            title = h2.get_text(strip=True).lower()
            next_elem = h2.find_next_sibling()
            
            if not next_elem:
                continue
            
            items = [li.get_text(strip=True) for li in next_elem.find_all("li")]
            
            if "symptom" in title:
                result["symptoms"] = items
            elif "cause" in title:
                result["causes"] = items
            elif "fix" in title or "repair" in title:
                result["repair_steps"] = items
        
        return result
        
    except Exception as e:
        print(f"Error scraping {code}: {e}")
        return None

def scrape_all(max_codes: int = None) -> list:
    """Scrape all DTC codes up to max_codes limit"""
    print("Collecting DTC code list...")
    all_codes = get_all_codes()
    
    if max_codes:
        all_codes = all_codes[:max_codes]
    
    print(f"Total codes to scrape: {len(all_codes)}")
    
    results = []
    for code in tqdm(all_codes, desc="Scraping DTC details"):
        detail = scrape_dtc_detail(code)
        if detail:
            results.append(detail)
        time.sleep(0.5)
    
    return results

def save_scraped_data(data: list, path: str = "data/scraped_obd.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} records to {path}")

if __name__ == "__main__":
    data = scrape_all()
    save_scraped_data(data)
    print("Scraping complete!")
    
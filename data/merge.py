import json
from datasets import load_dataset

def load_scraped_data(path: str = "data/scraped_obd.json") -> dict:
    """Load scraped data as a dict keyed by DTC code"""
    with open(path, "r") as f:
        data = json.load(f)
    return {item["dtc_code"]: item for item in data}

def load_huggingface_data() -> dict:
    """Load HuggingFace dataset as a dict keyed by DTC code"""
    dataset = load_dataset("Epitech/obd-codes-fine-tune")
    hf_data = {}
    for row in dataset['train']:
        code = row['instruction'].strip().split()[-1].upper()
        hf_data[code] = row['output']
    return hf_data

def merge_datasets() -> list:
    """Merge both datasets into unified format"""
    print("Loading scraped data...")
    scraped = load_scraped_data()
    
    print("Loading HuggingFace data...")
    hf_data = load_huggingface_data()
    
    all_codes = set(list(scraped.keys()) + list(hf_data.keys()))
    print(f"Total unique codes: {len(all_codes)}")
    
    merged = []
    for code in all_codes:
        record = {
            "dtc_code": code,
            "fault_name": "",
            "symptoms": [],
            "causes": [],
            "repair_steps": [],
            "text": ""
        }
        
        # HuggingFace gives us fault name
        if code in hf_data:
            record["fault_name"] = hf_data[code]
        
        # Scraped data enriches with symptoms, causes, repair steps
        if code in scraped:
            scrape = scraped[code]
            # Prefer scraped fault name if available and HF one is missing
            if not record["fault_name"] and scrape["fault_name"]:
                record["fault_name"] = scrape["fault_name"]
            record["symptoms"] = scrape["symptoms"]
            record["causes"] = scrape["causes"]
            record["repair_steps"] = scrape["repair_steps"]
        
        # Create rich text for embedding
        record["text"] = f"""DTC Code: {record['dtc_code']}
Fault: {record['fault_name']}
Symptoms: {', '.join(record['symptoms']) if record['symptoms'] else 'N/A'}
Causes: {', '.join(record['causes']) if record['causes'] else 'N/A'}
Repair Steps: {', '.join(record['repair_steps']) if record['repair_steps'] else 'N/A'}"""
        
        merged.append(record)
    
    return merged

def save_merged(data: list, path: str = "data/merged_obd.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} merged records to {path}")

if __name__ == "__main__":
    merged = merge_datasets()
    save_merged(merged)
    
    # Quick stats
    with_symptoms = sum(1 for r in merged if r['symptoms'])
    with_causes = sum(1 for r in merged if r['causes'])
    print(f"Records with symptoms: {with_symptoms}")
    print(f"Records with causes: {with_causes}")
    print(f"Sample record:")
    print(json.dumps(merged[0], indent=2))
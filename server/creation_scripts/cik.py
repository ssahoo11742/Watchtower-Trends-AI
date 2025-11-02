import requests
import json

def fetch_sec_tickers():
    url = "https://www.sec.gov/include/ticker.txt"
    
    # SEC requires a User-Agent header
    headers = {
        "User-Agent": "MyAppName (swayamsa@gmail.com)"  # replace with your info
    }

    # Download the ticker file
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises error if request fails
    data = response.text

    # Parse the file into a dictionary
    ticker_cik_dict = {}
    lines = data.splitlines()
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ticker, cik = parts
            ticker_cik_dict[ticker.lower()] = cik

    return ticker_cik_dict

def save_to_json(ticker_dict, filename="ticker_cik.json"):
    with open(filename, "w") as f:
        json.dump(ticker_dict, f, indent=4)
    print(f"Saved {len(ticker_dict)} tickers to {filename}")

if __name__ == "__main__":
    # Fetch tickers
    ticker_dict = fetch_sec_tickers()

    # Save to JSON
    save_to_json(ticker_dict)

    # Optional: print first 10 entries as a sanity check
    for i, (ticker, cik) in enumerate(ticker_dict.items()):
        print(ticker, cik)
        if i >= 9:
            break

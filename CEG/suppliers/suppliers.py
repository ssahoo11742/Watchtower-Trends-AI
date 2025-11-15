from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

def scrape_importyeti_suppliers(company_url):
    chrome_options = Options()
    
    # Better headless configuration
    chrome_options.add_argument('--headless=new')  # Use new headless mode
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-software-rasterizer')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Add user agent
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Only disable images
    prefs = {
        'profile.default_content_setting_values': {
            'images': 2,
        }
    }
    chrome_options.add_experimental_option('prefs', prefs)
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(company_url)
        
        # Wait longer and check multiple conditions
        print("Waiting for page load...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        
        # Extra wait for React to render
        time.sleep(5)
        
        # Verify data is there
        rows = driver.find_elements(By.XPATH, "//tbody[@class='text-sm']/tr")
        print(f"Initial rows loaded: {len(rows)}")
        
        if len(rows) == 0:
            print("No rows found! Page might not have loaded properly.")
            driver.quit()
            return []
        
        # Click "Show More"
        clicks = 0
        max_clicks = 20
        
        while clicks < max_clicks:
            try:
                show_more = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, 
                        "//span[contains(text(), 'Show More')]"))
                )
                
                current_rows = len(driver.find_elements(By.XPATH, "//tbody[@class='text-sm']/tr"))
                driver.execute_script("arguments[0].click();", show_more)
                clicks += 1
                print(f"Click {clicks}: {current_rows} rows")
                time.sleep(2)
                
                new_rows = len(driver.find_elements(By.XPATH, "//tbody[@class='text-sm']/tr"))
                if new_rows == current_rows:
                    break
            except:
                print(f"Finished clicking after {clicks} clicks")
                break
        
        # Parse HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        rows = soup.find_all('tr')
        
        suppliers = []
        for row in rows:
            try:
                supplier_link = row.find('a', class_='text-yeti-abominable-10')
                if not supplier_link:
                    continue
                
                location_links = row.find_all('a', class_='text-yeti-secondary-1')
                shipment_span = row.find('span', class_='text-sm font-semibold')
                product_div = row.find('div', class_='line-clamp-5')
                hs_links = row.find_all('a', href=lambda x: x and '/hs-codes/' in x)
                
                suppliers.append({
                    'supplier_name': supplier_link.text.strip(),
                    'supplier_url': "https://www.importyeti.com" + supplier_link['href'],
                    'location': ", ".join([loc.text.strip() for loc in location_links]),
                    'total_shipments': shipment_span.text.strip() if shipment_span else "",
                    'product_description': product_div.text.strip() if product_div else "",
                    'hs_codes': ", ".join([hs.text.strip() for hs in hs_links])
                })
            except:
                continue
        
        return suppliers
        
    finally:
        driver.quit()

def get_company_url(ticker):
    return f"https://www.importyeti.com/company/{ticker.lower()}"
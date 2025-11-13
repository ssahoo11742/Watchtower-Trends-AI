import yfinance as yf
from suppliers import scrape_importyeti_suppliers
from commodities import Commodities
class StaticEdges:
    def __init__(self, ticker):
        self.ticker = ticker
        self.country = self.get_location()
        self.sector = self.get_sector()
        self.url = "https://www.importyeti.com/company/quantum-scape"
        self.suppliers = self.get_suppliers()
        self.produces = Commodities(ticker).produces
        self.requires = Commodities(ticker).requires
    def get_location(self):
        info = yf.Ticker(self.ticker).info
        country = info.get("country", "Unknown")
        return [country]

    def get_sector(self):
        info = yf.Ticker(self.ticker).info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        return [sector, industry]
    
    def get_suppliers(self):
        return scrape_importyeti_suppliers(self.url)


static_edges = StaticEdges("QS")
print("Country Edge:", static_edges.country)
print("Sector and Industry Edges:", static_edges.sector)
print("Suppliers Edge:", static_edges.suppliers[0])
print("Produces Edge:", static_edges.produces[0])
print("Requires Edge:", static_edges.requires[0])


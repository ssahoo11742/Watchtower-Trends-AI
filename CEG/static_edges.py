import yfinance as yf

class StaticEdges:
    def __init__(self, ticker):
        self.ticker = ticker
        self.country = self.get_location()
        self.sector = self.get_sector()

    def get_location(self):
        info = yf.Ticker(self.ticker).info
        country = info.get("country", "Unknown")
        return [country]

    def get_sector(self):
        info = yf.Ticker(self.ticker).info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        return [sector, industry]


static_edges = StaticEdges("TSLA")
print("Country Edge:", static_edges.country)
print("Sector and Industry Edges:", static_edges.sector)


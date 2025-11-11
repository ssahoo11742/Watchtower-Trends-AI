from alpaca_trade_api.rest import REST

api = REST()

bars = api.get_bars("TSLA", "1Day")
print(bars)

from scoring import *

data = fetch_comprehensive_stock_data("RR")
sk = {
    "data": data,
    "day_trader": calculate_day_trader_score(data),
    "swing_trader": calculate_swing_trader_score(data),
    "position_trader": calculate_position_trader_score(data),
    "longterm_investor": calculate_longterm_investor_score(data),
}

print(sk)
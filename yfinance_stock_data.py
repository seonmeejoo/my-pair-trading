import yfinance as yf

aapl = yf.Ticker("AAPL")

hist = aapl.history(period="1y")
print(hist.head())

print(aapl.info)

print(aapl.financials)

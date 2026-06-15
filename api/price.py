from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO", "NG": "UNG", "BOIL": "BOIL",
}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            tk   = yf.Ticker(sym)
            hist = tk.history(period="65d")

            # yfinance 버전에 따라 MultiIndex 올 수 있음 → 평탄화
            if hasattr(hist.columns, "levels"):
                hist.columns = hist.columns.get_level_values(0)

            if "Close" not in hist.columns:
                raise ValueError(f"No Close column. Columns: {list(hist.columns)}")

            closes = hist["Close"].dropna()

            if len(closes) < 2:
                raise ValueError(f"Not enough data: {len(closes)} rows")

            current = float(closes.iloc[-1])
            prev    = float(closes.iloc[-2])
            change  = current - prev
            chg_pct = change / prev * 100
            returns = closes.pct_change().dropna().tolist()

            body = json.dumps({
                "asset":   asset,
                "price":   round(current, 2),
                "change":  round(change, 2),
                "chg_pct": round(chg_pct, 2),
                "up":      chg_pct >= 0,
                "returns": returns,
            })
        except Exception as e:
            body = json.dumps({"error": str(e), "asset": asset})

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass

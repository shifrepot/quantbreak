from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.linalg import eigh
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
# 시장 변수 5개로 줄임 (타임아웃 방지)
MARKET_TICKERS = ["^VIX", "SPY", "TLT", "GLD", "^TNX"]
MARKET_NAMES   = ["VIX", "SPY", "TLT(Bond)", "Gold", "10Y Yield"]

def fetch_yahoo(sym, days=130):
    end   = int(time.time())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    url   = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?interval=1d&period1={start}&period2={end}"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=6) as resp:
        import json as _j
        data = _j.loads(resp.read())
    result = data["chart"]["result"][0]
    ts     = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    pairs  = [(t, c) for t, c in zip(ts, closes) if c is not None]
    return [t for t, _ in pairs], [c for _, c in pairs]

def compute_sir(X, Y, h=8):
    n, p  = X.shape
    Sigma = np.cov(X.T) + 1e-6 * np.eye(p)

    quantiles = np.percentile(Y, np.linspace(0, 100, h+1))
    slice_means, slice_weights = [], []
    for j in range(h):
        lo, hi = quantiles[j], quantiles[j+1]
        mask   = (Y >= lo) & (Y <= hi) if j==h-1 else (Y >= lo) & (Y < hi)
        if mask.sum() >= 2:
            slice_means.append(X[mask].mean(axis=0))
            slice_weights.append(mask.sum())

    if len(slice_means) < 2:
        eigvals, eigvecs = eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[0]], eigvecs[:, idx[1]], "PCA"

    sm = np.array(slice_means)
    sw = np.array(slice_weights, dtype=float); sw /= sw.sum()
    gm = (sw[:, None] * sm).sum(axis=0)
    M  = sum(w * np.outer(m-gm, m-gm) for w, m in zip(sw, sm))

    A = np.linalg.inv(Sigma) @ M
    eigvals, eigvecs = eigh(A)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[0]], eigvecs[:, idx[1]], "SIR"

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # 자산 가격
            _, asset_prices = fetch_yahoo(sym, days=130)
            if len(asset_prices) < 40:
                raise ValueError(f"Not enough asset data: {len(asset_prices)}")

            # Y = 미래 20일 forward drawdown
            horizon = 20
            Y = np.array([
                min(asset_prices[t+1:t+horizon+1]) / asset_prices[t] - 1
                for t in range(len(asset_prices) - horizon)
            ])

            # 시장 변수 X (5개, 순차적으로 빠르게)
            mkt_cols = []
            for mkt_sym in MARKET_TICKERS:
                try:
                    _, mp = fetch_yahoo(mkt_sym, days=130)
                    mkt_cols.append(mp)
                except Exception:
                    continue

            if len(mkt_cols) < 2:
                raise ValueError("Not enough market data")

            # 길이 맞추기
            min_len = min(len(Y), min(len(col) for col in mkt_cols))
            mkt_cols = [col[-min_len:] for col in mkt_cols]
            Y_cut    = Y[-min_len:]

            # 수익률 계산
            X = np.column_stack([
                np.array([(col[i]-col[i-1])/col[i-1] for i in range(1, len(col))])
                for col in mkt_cols
            ])
            Y_cut = Y_cut[1:]   # 수익률과 길이 맞춤

            if len(Y_cut) < 15:
                raise ValueError(f"Not enough aligned: {len(Y_cut)}")

            # SIR
            beta1, beta2, method = compute_sir(X, Y_cut, h=8)
            projected = X @ np.column_stack([beta1, beta2])

            # Y 기준 레짐 분류
            y_q33 = float(np.percentile(Y_cut, 33))
            y_q66 = float(np.percentile(Y_cut, 66))

            points = []
            for i, (px, py) in enumerate(projected[:-1]):
                y_val = Y_cut[i]
                col   = "#F03860" if y_val <= y_q33 else "#F0A800" if y_val <= y_q66 else "#00D878"
                points.append({"x": round(float(px)*60,4), "y": round(float(py)*60,4), "col": col})

            cx = round(float(projected[-1,0])*60, 4)
            cy = round(float(projected[-1,1])*60, 4)

            recent_dd = float(Y_cut[-5:].mean())
            regime    = "CRASH ZONE" if recent_dd <= y_q33 else "ELEVATED" if recent_dd <= y_q66 else "SAFE ZONE"

            corr = float(np.corrcoef(projected[:,0], Y_cut)[0,1])

            body = json.dumps({
                "points":             points,
                "current":            {"x": cx, "y": cy},
                "regime":             regime,
                "variance_explained": round(corr**2*100, 1),
                "top_risk_factor":    MARKET_NAMES[int(np.argmax(np.abs(beta1)))] if len(beta1) <= len(MARKET_NAMES) else "Market",
                "method":             method,
                "sir_meta": {
                    "Y_definition": "20-day forward maximum drawdown",
                    "Y_mean":       round(float(Y_cut.mean())*100, 2),
                    "corr_beta1_Y": round(corr, 3),
                    "n_samples":    len(Y_cut),
                },
            })
        except Exception as e:
            body = json.dumps({"error": str(e)})

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass

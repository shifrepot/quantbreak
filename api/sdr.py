from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from scipy.linalg import eigh
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
MARKET_TICKERS = ["^VIX","SPY","TLT","GLD","UUP","USO","XLF","XLK","HYG","^TNX"]
MARKET_NAMES   = ["VIX","SPY","TLT","Gold","Dollar","Oil","Financials","Tech","HY Bond","10Y Yield"]


def download_single(sym, period="120d"):
    """단일 티커 안전 다운로드"""
    tk   = yf.Ticker(sym)
    hist = tk.history(period=period)
    if hasattr(hist.columns, "levels"):
        hist.columns = hist.columns.get_level_values(0)
    return hist["Close"].dropna()


def download_multi(tickers, period="120d"):
    """멀티 티커 — MultiIndex 평탄화 포함"""
    raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    # MultiIndex 처리
    if hasattr(raw.columns, "levels"):
        # (Price, Ticker) 형태일 때 Close만 추출
        if "Close" in raw.columns.get_level_values(0):
            raw = raw["Close"]
        else:
            raw.columns = raw.columns.get_level_values(0)
    elif "Close" in raw.columns:
        raw = raw[["Close"]]
    return raw.dropna()


def compute_sir(X, Y, h=10):
    """
    Sliced Inverse Regression
    Y = 미래 20일 forward maximum drawdown
    β = eigenvec(Σ⁻¹M)
    """
    n, p  = X.shape
    Sigma = np.cov(X.T)

    quantiles = np.percentile(Y, np.linspace(0, 100, h+1))
    slice_means, slice_weights = [], []

    for j in range(h):
        lo, hi = quantiles[j], quantiles[j+1]
        mask   = (Y >= lo) & (Y <= hi) if j == h-1 else (Y >= lo) & (Y < hi)
        if mask.sum() >= 2:
            slice_means.append(X[mask].mean(axis=0))
            slice_weights.append(mask.sum())

    if len(slice_means) < 2:
        # fallback: PCA
        eigvals, eigvecs = eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[0]], eigvecs[:, idx[1]], "PCA fallback"

    slice_means   = np.array(slice_means)
    slice_weights = np.array(slice_weights, dtype=float)
    slice_weights /= slice_weights.sum()

    grand_mean = (slice_weights[:, None] * slice_means).sum(axis=0)
    M = np.zeros((p, p))
    for w, m in zip(slice_weights, slice_means):
        d  = (m - grand_mean)[:, None]
        M += w * (d @ d.T)

    Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(p))
    eigvals, eigvecs = eigh(Sigma_inv @ M)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[0]], eigvecs[:, idx[1]], "SIR"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # ── 자산 가격 (Y 계산용)
            asset_prices = download_single(sym, "120d")
            if len(asset_prices) < 40:
                raise ValueError(f"Not enough price data: {len(asset_prices)}")

            # ── Y = 미래 20일 forward maximum drawdown
            prices  = asset_prices.values
            horizon = 20
            Y = np.array([
                (prices[t+1:t+horizon+1].min() / prices[t]) - 1
                for t in range(len(prices) - horizon)
            ])

            # ── 시장 변수 X (10개)
            try:
                mkt_raw = download_multi(MARKET_TICKERS, "120d")
                # 각 티커 컬럼 확인
                available = [t for t in MARKET_TICKERS if t in mkt_raw.columns]
                if len(available) < 5:
                    raise ValueError("Not enough market tickers")
                mkt_closes = mkt_raw[available].dropna()
                mkt_rets   = mkt_closes.pct_change().dropna().values
            except Exception:
                # 시장 변수 다운로드 실패 → 자산 수익률만으로 fallback
                asset_rets = asset_prices.pct_change().dropna().values
                mkt_rets   = np.column_stack([asset_rets] * 5)

            # 길이 맞추기
            n_xy = min(len(Y), len(mkt_rets))
            X    = mkt_rets[:n_xy]
            Y    = Y[:n_xy]

            if len(Y) < 15:
                raise ValueError(f"Not enough aligned data: {len(Y)}")

            # ── SIR
            beta1, beta2, method = compute_sir(X, Y, h=8)

            # ── 2D 투영
            projected = X @ np.column_stack([beta1, beta2])

            # ── β₁과 Y의 상관
            corr = float(np.corrcoef(projected[:, 0], Y)[0, 1])
            var_explained = round(corr**2 * 100, 1)

            # ── 레짐 분류 (Y 기준: 작을수록 crash)
            y_q33 = float(np.percentile(Y, 33))
            y_q66 = float(np.percentile(Y, 66))

            points = []
            for i, (px, py) in enumerate(projected[:-1]):
                y_val = Y[i]
                if y_val <= y_q33:   col = "#F03860"  # crash
                elif y_val <= y_q66: col = "#F0A800"  # elevated
                else:                col = "#00D878"  # safe
                points.append({
                    "x": round(float(px) * 60, 4),
                    "y": round(float(py) * 60, 4),
                    "col": col
                })

            cx = round(float(projected[-1, 0]) * 60, 4)
            cy = round(float(projected[-1, 1]) * 60, 4)

            recent_dd = float(Y[-5:].mean())
            if recent_dd <= y_q33:   regime = "CRASH ZONE"
            elif recent_dd <= y_q66: regime = "ELEVATED"
            else:                    regime = "SAFE ZONE"

            top_idx    = int(np.argmax(np.abs(beta1)))
            top_factor = MARKET_NAMES[top_idx] if top_idx < len(MARKET_NAMES) else "Market"

            body = json.dumps({
                "points":             points,
                "current":            {"x": cx, "y": cy},
                "regime":             regime,
                "variance_explained": var_explained,
                "top_risk_factor":    top_factor,
                "method":             method,
                "sir_meta": {
                    "Y_definition": "20-day forward maximum drawdown",
                    "Y_mean":       round(float(Y.mean()) * 100, 2),
                    "Y_cvar5":      round(float(np.percentile(Y, 5)) * 100, 2),
                    "corr_beta1_Y": round(corr, 3),
                    "n_samples":    len(Y),
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

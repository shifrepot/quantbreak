from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}

LEVERAGES = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

def black_scholes(S, K, T, r, sigma):
    """실제 Black-Scholes 공식: C = S·N(d1) - K·e^{-rT}·N(d2)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call      = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    prob_prof = float(norm.cdf(d2))
    return d1, d2, call, prob_prof

def cvar(returns, alpha=0.05):
    """Historical CVaR: E[L | L > VaR_α]"""
    var = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            hist    = yf.Ticker(sym).history(period="65d")
            closes  = hist["Close"].dropna()
            returns = closes.pct_change().dropna().values  # numpy array

            # ── 실현 변동성 (연율화)
            sigma = float(np.std(returns) * np.sqrt(252))
            S     = float(closes.iloc[-1])
            K, T, r = S, 0.5, 0.05  # ATM, 6개월, 무위험이자율 5%

            # ── 실제 Black-Scholes
            d1, d2, call_price, prob_profit = black_scholes(S, K, T, r, sigma)
            bs_expected_ret  = float((np.exp(r * T) - 1) * 100)

            # BS 예측 꼬리 확률 (월간 -30% 이상)
            monthly_sigma  = sigma / np.sqrt(12)
            bs_tail_prob   = float(norm.cdf(-0.30 / monthly_sigma) * 100)

            # ── 실제 Fat Tail 비율
            # 60일 데이터로 슬라이딩 윈도우 월간 수익률 추정
            n = len(returns)
            monthly_rets = [
                float(np.prod(1 + returns[i:i+21]) - 1)
                for i in range(0, n - 21, 5)
            ]
            crash_count   = sum(1 for r_ in monthly_rets if r_ < -0.30)
            actual_tail   = crash_count / max(len(monthly_rets), 1) * 100
            fat_tail_ratio = round(actual_tail / bs_tail_prob, 1) if bs_tail_prob > 0 else 8.3

            # ── 레버리지별 실제 CVaR
            cvar_by_lev = []
            mean_by_lev = []
            for lev in LEVERAGES:
                lr = returns * lev
                cvar_by_lev.append(round(cvar(lr) * 100, 2))
                mean_by_lev.append(float(np.mean(lr)))

            # 최적 레버리지: 기대수익 양수인 것 중 CVaR 손실 가장 작은 것
            feasible = [
                (i, cvar_by_lev[i])
                for i in range(len(LEVERAGES))
                if mean_by_lev[i] > 0
            ]
            opt_idx = max(feasible, key=lambda x: -x[1])[0] if feasible else 4

            # volatility decay (레버리지 ETF 특유의 복리 손실)
            lev3_compound = float(np.prod(1 + returns * 3) - 1) * 100
            lev1_compound = float(np.prod(1 + returns) - 1) * 100
            vol_decay     = round(lev3_compound - lev1_compound * 3, 1)

            body = json.dumps({
                # Black-Scholes 결과
                "sigma":            round(sigma * 100, 1),
                "bs_prob_profit":   round(prob_profit * 100, 1),
                "bs_expected_ret":  round(bs_expected_ret, 1),
                "bs_tail_prob":     round(bs_tail_prob, 3),
                "bs_call_price":    round(call_price, 2),
                "d1": round(d1, 4), "d2": round(d2, 4),
                # 실제 데이터
                "fat_tail_ratio":   fat_tail_ratio,
                "vol_decay":        vol_decay,
                # CVaR
                "cvar_5pct":        round(cvar(returns) * 100, 2),
                "cvar_by_leverage": cvar_by_lev,
                "optimal_leverage": LEVERAGES[opt_idx],
                "optimal_idx":      opt_idx,
                # 에너지 바 (정규화된 CVaR 절댓값 → 높을수록 위험)
                "energies": [
                    round(abs(c) / max(abs(v) for v in cvar_by_lev), 4)
                    for c in cvar_by_lev
                ],
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

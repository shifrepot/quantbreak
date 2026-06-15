from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
LEVERAGES = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

def fetch_yahoo(sym, days=120):
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
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    prices = [c for c in closes if c is not None]
    if len(prices) < 10:
        raise ValueError(f"Not enough data: {len(prices)}")
    return prices

def cvar(returns, alpha=0.05):
    var  = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            prices  = fetch_yahoo(sym, days=120)   # 120일로 늘림
            returns = np.array([
                (prices[i]-prices[i-1])/prices[i-1]
                for i in range(1, len(prices))
            ])

            sigma = float(np.std(returns) * np.sqrt(252))
            S     = float(prices[-1])
            K, T, r = S, 0.5, 0.05

            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            prob_profit     = float(norm.cdf(d2))
            bs_expected_ret = float((np.exp(r*T)-1)*100)
            monthly_sigma   = sigma / np.sqrt(12)
            bs_tail_prob    = float(norm.cdf(-0.30/monthly_sigma)*100)

            n = len(returns)

            # Fat Tail Ratio
            # 임계값: -15% (TQQQ 같은 3× ETF 기준, -30%는 최근 상승장에서 안 나옴)
            # 기간: 21일 rolling, 3일 간격으로 더 많은 샘플
            monthly_rets = [
                float(np.prod(1+returns[i:i+21])-1)
                for i in range(0, n-21, 3)
            ]
            # -15% 기준 실제 발생 빈도
            threshold   = -0.15
            crash_count = sum(1 for r_ in monthly_rets if r_ < threshold)
            actual_tail = crash_count / max(len(monthly_rets),1) * 100

            # GBM이 예측하는 -15% 이하 확률
            gbm_tail_prob_15 = float(norm.cdf(threshold / monthly_sigma) * 100)
            fat_tail_ratio   = round(actual_tail / gbm_tail_prob_15, 1) if gbm_tail_prob_15 > 0 else 1.0

            # 원래 -30% 기준도 유지 (화면 표시용)
            crash_count_30 = sum(1 for r_ in monthly_rets if r_ < -0.30)
            actual_tail_30 = crash_count_30 / max(len(monthly_rets),1) * 100
            fat_tail_ratio_30 = round(actual_tail_30 / bs_tail_prob, 1) if bs_tail_prob > 0 and actual_tail_30 > 0 else fat_tail_ratio

            lev3      = float(np.prod(1+returns*3)-1)*100
            lev1      = float(np.prod(1+returns)-1)*100
            vol_decay = round(lev3 - lev1*3, 1)

            cvar_by_lev = []
            mean_by_lev = []
            for lev in LEVERAGES:
                lr = returns * lev
                cvar_by_lev.append(round(cvar(lr)*100, 2))
                mean_by_lev.append(float(np.mean(lr)))

            # 최적 레버리지:
            # 기대수익 양수인 후보 중 CVaR 절댓값이 가장 작은 것
            # (손실 위험이 가장 적은 것)
            feasible = [(i, cvar_by_lev[i]) for i in range(len(LEVERAGES)) if mean_by_lev[i] > 0]
            if feasible:
                # CVaR가 가장 덜 음수 = 절댓값 가장 작음 = 가장 안전
                opt_idx = min(feasible, key=lambda x: abs(x[1]))[0]
            else:
                opt_idx = 2  # 기본값 1.0×

            max_abs  = max(abs(v) for v in cvar_by_lev) or 1
            energies = [round(abs(c)/max_abs, 4) for c in cvar_by_lev]

            body = json.dumps({
                "sigma":            round(sigma*100, 1),
                "bs_prob_profit":   round(prob_profit*100, 1),
                "bs_expected_ret":  round(bs_expected_ret, 1),
                "bs_tail_prob":     round(bs_tail_prob, 3),
                "fat_tail_ratio":   fat_tail_ratio,
                "fat_tail_ratio_30": fat_tail_ratio_30,
                "vol_decay":        vol_decay,
                "cvar_5pct":        round(cvar(returns)*100, 2),
                "cvar_by_leverage": cvar_by_lev,
                "optimal_leverage": LEVERAGES[opt_idx],
                "optimal_idx":      opt_idx,
                "energies":         energies,
                "returns":          [round(float(r),6) for r in returns],
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

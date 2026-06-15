from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI":  "USO",  "NG":   "UNG",  "BOIL": "BOIL",
}

# 자산별 위기 케이스 정의
CASES = {
    "TQQQ": [
        {"name": "COVID Crash",      "train_end": "2020-02-01", "crash_end": "2020-03-23"},
        {"name": "Rate Shock 2022",  "train_end": "2022-01-03", "crash_end": "2022-06-16"},
        {"name": "Yen Carry 2024",   "train_end": "2024-08-01", "crash_end": "2024-08-05"},
    ],
    "SOXL": [
        {"name": "COVID Crash",      "train_end": "2020-02-01", "crash_end": "2020-03-23"},
        {"name": "Rate Shock 2022",  "train_end": "2022-01-03", "crash_end": "2022-06-16"},
        {"name": "Yen Carry 2024",   "train_end": "2024-08-01", "crash_end": "2024-08-05"},
    ],
    "SQQQ": [
        {"name": "COVID Recovery",   "train_end": "2020-03-23", "crash_end": "2020-06-08"},
        {"name": "2023 AI Rally",    "train_end": "2023-01-02", "crash_end": "2023-07-19"},
        {"name": "Nov 2024 Rally",   "train_end": "2024-11-04", "crash_end": "2024-11-29"},
    ],
    "USO": [
        {"name": "COVID Oil Crash",  "train_end": "2020-02-01", "crash_end": "2020-04-20"},
        {"name": "2022 Peak&Crash",  "train_end": "2022-06-01", "crash_end": "2022-12-09"},
        {"name": "2023 OPEC Shock",  "train_end": "2023-09-01", "crash_end": "2023-10-06"},
    ],
    "UNG": [
        {"name": "2021 Winter Freeze","train_end":"2021-02-01", "crash_end": "2021-03-01"},
        {"name": "2022 Europe Crisis","train_end":"2022-08-01", "crash_end": "2022-12-01"},
        {"name": "2024 Supply Glut", "train_end": "2024-02-01", "crash_end": "2024-04-15"},
    ],
    "BOIL": [
        {"name": "2021 Winter Freeze","train_end":"2021-02-01", "crash_end": "2021-03-01"},
        {"name": "2022 Europe Crisis","train_end":"2022-08-01", "crash_end": "2022-12-01"},
        {"name": "2024 Supply Glut", "train_end": "2024-02-01", "crash_end": "2024-04-15"},
    ],
}

def date_to_ts(date_str):
    """YYYY-MM-DD → unix timestamp"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())

def fetch_range(sym, start_str, end_str):
    """특정 기간 주가 데이터 가져오기"""
    start = date_to_ts(start_str)
    end   = date_to_ts(end_str) + 86400  # 하루 추가
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
    return prices

def calc_bs_cvar(returns, alpha=0.05):
    """Black-Scholes 가정 하의 CVaR (정규분포)"""
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma < 1e-9:
        return 0.0
    z_alpha = norm.ppf(alpha)
    # CVaR = mu - sigma * phi(z_alpha) / alpha
    cvar = mu - sigma * norm.pdf(z_alpha) / alpha
    return float(cvar * np.sqrt(21))  # 월간 스케일

def calc_regime_cvar(returns, alpha=0.05):
    """
    Regime-Weighted CVaR
    수익률을 3개 레짐으로 나눠 가중 CVaR 계산
    """
    returns = np.array(returns)
    n = len(returns)
    if n < 6:
        return float(np.percentile(returns, alpha*100) * np.sqrt(21))

    # 레짐 분류 (수익률 분위수 기준)
    q33 = np.percentile(returns, 33)
    q66 = np.percentile(returns, 66)

    crash_r = returns[returns <= q33]
    elev_r  = returns[(returns > q33) & (returns <= q66)]
    safe_r  = returns[returns > q66]

    p_crash = len(crash_r) / n
    p_elev  = len(elev_r)  / n
    p_safe  = len(safe_r)  / n

    def regime_cvar(r):
        if len(r) < 2:
            return 0.0
        v = np.percentile(r, alpha*100)
        t = r[r <= v]
        return float(t.mean()) if len(t) > 0 else float(v)

    cvar_crash = regime_cvar(crash_r)
    cvar_elev  = regime_cvar(elev_r)
    cvar_safe  = regime_cvar(safe_r)

    weighted = p_crash*cvar_crash + p_elev*cvar_elev + p_safe*cvar_safe
    return float(weighted * np.sqrt(21))  # 월간 스케일

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)
        cases = CASES.get(sym, CASES.get(asset, CASES["TQQQ"]))

        results = []
        for case in cases:
            try:
                # ── 훈련 데이터: 위기 직전 180일
                train_end = case["train_end"]
                train_dt  = datetime.strptime(train_end, "%Y-%m-%d")
                train_start = (train_dt - timedelta(days=180)).strftime("%Y-%m-%d")

                train_prices = fetch_range(sym, train_start, train_end)
                if len(train_prices) < 20:
                    raise ValueError(f"Not enough train data: {len(train_prices)}")

                train_returns = np.array([
                    (train_prices[i]-train_prices[i-1])/train_prices[i-1]
                    for i in range(1, len(train_prices))
                ])

                # ── BS CVaR (위기 전 데이터 기준)
                bs_cvar = calc_bs_cvar(train_returns)

                # ── Regime-Weighted CVaR (위기 전 데이터 기준)
                reg_cvar = calc_regime_cvar(train_returns)

                # ── 실제 손실: 위기 구간 최대 낙폭
                crash_prices = fetch_range(sym, train_end, case["crash_end"])
                if len(crash_prices) >= 2:
                    actual_dd = (min(crash_prices) / crash_prices[0] - 1) * 100
                else:
                    actual_dd = None

                # ── 오차 계산
                actual_pct    = round(actual_dd, 1) if actual_dd is not None else None
                bs_cvar_pct   = round(bs_cvar  * 100, 1)
                reg_cvar_pct  = round(reg_cvar * 100, 1)

                bs_err  = round(abs(actual_dd - bs_cvar_pct),  1) if actual_dd else None
                reg_err = round(abs(actual_dd - reg_cvar_pct), 1) if actual_dd else None
                captured = round((bs_err - reg_err) / abs(bs_err) * 100, 1) if (bs_err and bs_err != 0) else None

                results.append({
                    "name":        case["name"],
                    "train_end":   train_end,
                    "crash_end":   case["crash_end"],
                    "bs_cvar":     bs_cvar_pct,
                    "regime_cvar": reg_cvar_pct,
                    "actual":      actual_pct,
                    "bs_underestimate":     bs_err,
                    "regime_underestimate": reg_err,
                    "additional_captured":  captured,
                    "note": "All CVaR values monthly-scaled. actual = max drawdown in crash window."
                })

            except Exception as e:
                results.append({
                    "name":  case["name"],
                    "error": str(e),
                })

        # 요약 통계
        valid = [r for r in results if "bs_underestimate" in r and r["bs_underestimate"]]
        if valid:
            avg_bs  = round(np.mean([r["bs_underestimate"]  for r in valid]), 1)
            avg_reg = round(np.mean([r["regime_underestimate"] for r in valid]), 1)
            avg_cap = round((avg_bs - avg_reg) / avg_bs * 100, 1) if avg_bs else None
        else:
            avg_bs = avg_reg = avg_cap = None

        body = json.dumps({
            "asset":   asset,
            "cases":   results,
            "summary": {
                "avg_bs_underestimate":     avg_bs,
                "avg_regime_underestimate": avg_reg,
                "avg_additional_captured":  avg_cap,
            }
        })

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass

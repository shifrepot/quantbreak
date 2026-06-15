from http.server import BaseHTTPRequestHandler
import json, numpy as np
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}

def fetch_yahoo(sym, days=70):
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
    if len(prices) < 5:
        raise ValueError(f"Not enough data: {len(prices)}")
    return prices

def rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def run_qaoa(costs, p=2):
    """
    QAOA with one-hot penalty

    H_cost = Σᵢ hᵢσᵢᶻ + A·(Σᵢxᵢ - 1)²

    one-hot penalty가 있어야 정확히 1개 레버리지만 선택됨.
    penalty 없으면 항상 h=0인 첫 번째 후보(0.5×)로 수렴.
    """
    n   = len(costs)
    dim = 2**n

    # 비용 정규화 [0,1]
    h = np.array(costs, dtype=float)
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    # one-hot penalty 계수 (비용보다 충분히 크게)
    A = 3.0

    # 각 상태(비트스트링)의 전체 비용 계산
    # H_cost = Σᵢ hᵢxᵢ + A·(Σᵢxᵢ - 1)²
    cost_diag = np.zeros(dim)
    for x in range(dim):
        bits     = [(x >> (n-1-i)) & 1 for i in range(n)]
        n_ones   = sum(bits)
        cvr_cost = sum(h[i] * bits[i] for i in range(n))
        penalty  = A * (n_ones - 1)**2   # one-hot: 정확히 1개일 때 0
        cost_diag[x] = cvr_cost + penalty

    # 초기 균등 중첩
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # 자산별 최적 파라미터 탐색 (grid search, 간단 버전)
    best_energy = float('inf')
    best_probs  = None

    gamma_list = [0.2, 0.4, 0.6, 0.8]
    beta_list  = [0.3, 0.5, 0.7, 0.9]

    for g in gamma_list:
        for b in beta_list:
            psi_try = np.ones(dim, dtype=complex) / np.sqrt(dim)

            for layer in range(p):
                # Cost layer: e^{-iγH_cost}
                psi_try *= np.exp(-1j * g * cost_diag)

                # Mixer layer: ⊗ᵢ Rx(2β)
                for i in range(n):
                    gate = rx(2 * b)
                    I2   = np.eye(2, dtype=complex)
                    ops  = [I2] * n
                    ops[i] = gate
                    full = ops[0]
                    for m in ops[1:]:
                        full = np.kron(full, m)
                    psi_try = full @ psi_try

            probs_try  = np.abs(psi_try)**2
            energy_try = float(np.dot(probs_try, cost_diag))

            if energy_try < best_energy:
                best_energy = energy_try
                best_probs  = probs_try.copy()
                best_gamma  = g
                best_beta   = b

    # 각 레버리지 후보 확률 집계
    # one-hot 상태만 유효 → 정확히 1개 비트가 1인 상태들
    cp = np.zeros(n)
    for state in range(dim):
        bits   = [(state >> (n-1-i)) & 1 for i in range(n)]
        n_ones = sum(bits)
        if n_ones == 1:   # one-hot 상태만
            idx = bits.index(1)
            cp[idx] += best_probs[state]

    total = cp.sum()
    if total > 0:
        cp /= total
    else:
        # fallback: 비용 최소 후보
        cp[np.argmin(h)] = 1.0

    return cp, best_gamma, best_beta

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            prices    = fetch_yahoo(sym, days=70)
            returns   = np.array([
                (prices[i]-prices[i-1])/prices[i-1]
                for i in range(1, len(prices))
            ])
            alpha     = 0.05
            leverages = [0.5, 1.0, 1.4, 2.0, 3.0]

            # 레버리지별 Historical CVaR + 기대수익 반영
            #
            # 목적함수: |CVaR(L)| / max(E[R(L)], ε)
            # = 꼬리위험 / 기대수익 비율
            # → 낮을수록 좋음 (위험 대비 수익이 좋음)
            # → CVaR(L) ≈ L×CVaR(1) 이지만
            #    E[R(L)] ≈ L×μ 도 함께 증가하므로
            #    단순 CVaR 최소화(= 항상 0.5×)와 달리
            #    시장 상황에 따라 중간 레버리지가 선택됨
            raw_cvars = []
            costs = []
            for lev in leverages:
                lr    = returns * lev
                var   = np.percentile(lr, alpha*100)
                t     = lr[lr <= var]
                cvar_val  = float(t.mean()) if len(t) > 0 else float(var)
                exp_ret   = float(np.mean(lr))
                raw_cvars.append(round(cvar_val*100, 2))

                # 기대수익이 음수면 penalty 추가 (손실 구간에서 레버리지 금지)
                if exp_ret <= 0:
                    cost = abs(cvar_val) * 10  # 큰 penalty
                else:
                    cost = abs(cvar_val) / (exp_ret + 1e-6)

                costs.append(cost)

            # QAOA 실행 (one-hot penalty + grid search)
            probs, best_g, best_b = run_qaoa(costs, p=2)

            opt_idx = int(np.argmax(probs))
            opt_lev = leverages[opt_idx]
            naive   = raw_cvars[-1]   # 3× CVaR (음수, 표시용)
            opt_cvar = raw_cvars[opt_idx]
            impr    = abs((naive - opt_cvar) / (naive+1e-9)) * 100

            body = json.dumps({
                "optimal_leverage": opt_lev,
                "optimal_idx":      opt_idx,
                "cvar_optimal":     opt_cvar,
                "cvar_naive_3x":    naive,
                "improvement_pct":  round(impr, 1),
                "probabilities":    [round(float(p_), 4) for p_ in probs],
                "leverage_labels":  [f"{l}×" for l in leverages],
                "costs":            raw_cvars,   # 음수 CVaR 표시용
                "costs_abs":        [round(c*100,2) for c in costs],  # 절댓값 (QAOA 내부용)
                "n_qubits":         len(leverages),
                "p_layers":         2,
                "best_gamma":       round(best_g, 3),
                "best_beta":        round(best_b, 3),
                "circuit_info": {
                    "ansatz":       "|ψ(γ,β)⟩ = ∏ e^{-iβH_M} e^{-iγH_C} |s⟩",
                    "H_cost":       "Σᵢ |CVaR(Lᵢ)|·σᵢᶻ + A·(Σxᵢ-1)² (abs CVaR + one-hot)",
                    "H_mixer":      "Σᵢ σᵢˣ",
                    "penalty":      "A=3.0",
                    "cost_direction": "higher leverage → larger |CVaR| → higher cost → QAOA avoids",
                }
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

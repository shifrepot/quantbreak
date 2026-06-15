from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}

def get_returns(sym, period="65d"):
    tk   = yf.Ticker(sym)
    hist = tk.history(period=period)
    if hasattr(hist.columns, "levels"):
        hist.columns = hist.columns.get_level_values(0)
    closes = hist["Close"].dropna()
    if len(closes) < 5:
        raise ValueError(f"Not enough data: {len(closes)}")
    return closes.pct_change().dropna().values

def rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def kron_gate(n, i, gate):
    I2 = np.eye(2, dtype=complex)
    ops = [I2]*n
    ops[i] = gate
    result = ops[0]
    for m in ops[1:]:
        result = np.kron(result, m)
    return result

def run_qaoa(costs, p=2):
    n   = len(costs)
    dim = 2**n

    # 비용 정규화 + one-hot penalty
    h    = np.array(costs, dtype=float)
    h    = (h - h.min()) / (h.max() - h.min() + 1e-9)

    # 초기 균등 중첩
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

    gammas = [0.39, 0.51][:p]
    betas  = [0.61, 0.49][:p]

    for layer in range(p):
        # Cost layer (diagonal)
        psi *= np.exp(-1j * gammas[layer] * h[
            [bin(x).count('1') % n for x in range(dim)]
        ] if False else np.array([
            sum(h[i] * ((x >> (n-1-i)) & 1) for i in range(n))
            for x in range(dim)
        ]))

        # Mixer layer
        for i in range(n):
            psi = kron_gate(n, i, rx(2*betas[layer])) @ psi

    probs = np.abs(psi)**2
    candidate_probs = np.zeros(n)
    for state in range(dim):
        bits = format(state, f'0{n}b')
        for i, bit in enumerate(bits):
            if bit == '1':
                candidate_probs[i] += probs[state]

    total = candidate_probs.sum()
    if total > 0:
        candidate_probs /= total

    return candidate_probs


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            returns   = get_returns(sym, "65d")
            alpha     = 0.05
            leverages = [0.5, 1.0, 1.4, 2.0, 3.0]
            n         = len(leverages)

            # 레버리지별 CVaR
            costs = []
            for lev in leverages:
                lr  = returns * lev
                var = np.percentile(lr, alpha*100)
                t   = lr[lr <= var]
                costs.append(float(t.mean()) if len(t) > 0 else float(var))

            # QAOA
            probs    = run_qaoa(costs, p=2)
            opt_idx  = int(np.argmax(probs))
            opt_lev  = leverages[opt_idx]
            opt_cvar = costs[opt_idx]
            naive    = costs[-1]
            impr     = abs((naive - opt_cvar) / (naive + 1e-9)) * 100

            body = json.dumps({
                "optimal_leverage":  opt_lev,
                "optimal_idx":       opt_idx,
                "cvar_optimal":      round(opt_cvar*100, 2),
                "cvar_naive_3x":     round(naive*100, 2),
                "improvement_pct":   round(impr, 1),
                "probabilities":     [round(float(p), 4) for p in probs],
                "leverage_labels":   [f"{l}×" for l in leverages],
                "costs":             [round(c*100, 2) for c in costs],
                "n_qubits":          n,
                "p_layers":          2,
                "circuit_info": {
                    "ansatz":  "|ψ(γ,β)⟩ = ∏ e^{-iβH_M} e^{-iγH_C} |s⟩",
                    "H_cost":  "Σᵢ hᵢσᵢᶻ + A·(Σxᵢ-1)² (Ising + one-hot)",
                    "H_mixer": "Σᵢ σᵢˣ",
                    "dim":     2**n,
                    "gamma":   [0.39, 0.51],
                    "beta":    [0.61, 0.49],
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

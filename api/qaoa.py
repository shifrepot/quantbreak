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
        import json as _j
        data = _j.loads(resp.read())
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
    n   = len(costs)
    dim = 2**n
    h   = np.array(costs, dtype=float)
    h   = (h - h.min()) / (h.max() - h.min() + 1e-9)

    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    gammas = [0.39, 0.51][:p]
    betas  = [0.61, 0.49][:p]

    for layer in range(p):
        # Cost layer
        cost_diag = np.array([
            sum(h[i]*((x >> (n-1-i)) & 1) for i in range(n))
            for x in range(dim)
        ])
        psi *= np.exp(-1j * gammas[layer] * cost_diag)

        # Mixer layer
        for i in range(n):
            gate = rx(2*betas[layer])
            I2   = np.eye(2, dtype=complex)
            ops  = [I2]*n; ops[i] = gate
            full = ops[0]
            for m in ops[1:]: full = np.kron(full, m)
            psi = full @ psi

    probs = np.abs(psi)**2
    cp    = np.zeros(n)
    for state in range(dim):
        bits = format(state, f'0{n}b')
        for i, bit in enumerate(bits):
            if bit == '1': cp[i] += probs[state]
    total = cp.sum()
    if total > 0: cp /= total
    return cp

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            prices    = fetch_yahoo(sym, days=70)
            returns   = np.array([(prices[i]-prices[i-1])/prices[i-1] for i in range(1,len(prices))])
            alpha     = 0.05
            leverages = [0.5, 1.0, 1.4, 2.0, 3.0]

            costs = []
            for lev in leverages:
                lr  = returns * lev
                var = np.percentile(lr, alpha*100)
                t   = lr[lr <= var]
                costs.append(float(t.mean()) if len(t) > 0 else float(var))

            probs   = run_qaoa(costs, p=2)
            opt_idx = int(np.argmax(probs))
            opt_lev = leverages[opt_idx]
            naive   = costs[-1]
            impr    = abs((naive - costs[opt_idx]) / (naive+1e-9)) * 100

            body = json.dumps({
                "optimal_leverage": opt_lev,
                "optimal_idx":      opt_idx,
                "cvar_optimal":     round(costs[opt_idx]*100, 2),
                "cvar_naive_3x":    round(naive*100, 2),
                "improvement_pct":  round(impr, 1),
                "probabilities":    [round(float(p_), 4) for p_ in probs],
                "leverage_labels":  [f"{l}×" for l in leverages],
                "costs":            [round(c*100, 2) for c in costs],
                "n_qubits":         len(leverages),
                "p_layers":         2,
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

from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from scipy.linalg import eigh
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}

# X: 10개 시장 변수
MARKET_TICKERS = ["^VIX","SPY","TLT","GLD","UUP","USO","XLF","XLK","HYG","^TNX"]
MARKET_NAMES   = ["VIX","SPY","TLT(Bond)","Gold","Dollar","Oil",
                  "Financials","Tech","HY Bond","10Y Yield"]


def compute_sir(X, Y, h=10):
    """
    Sliced Inverse Regression (SIR)
    
    Y = 미래 20일 최대 손실 (forward drawdown)
    
    알고리즘:
    1. Y를 h개 슬라이스로 나눔
    2. 각 슬라이스에서 E[X|Y∈슬라이스] 계산
    3. 슬라이스 평균들의 가중 공분산 행렬 M 계산
    4. Σ⁻¹M 의 고유벡터 → SDR 방향 β
    
    이론적 근거:
    E[X|Y]가 βᵀX의 선형 함수 (Linear Conditional Mean) 이면,
    Cov(E[X|Y]) = M 의 column space ⊆ Σβ 의 column space
    → Σ⁻¹M 의 top eigenvector = β (SDR 방향)
    """
    n, p = X.shape
    Sigma = np.cov(X.T)                          # (p, p)

    # Y를 h개 분위수 슬라이스로 분할
    quantiles = np.percentile(Y, np.linspace(0, 100, h + 1))

    slice_means   = []
    slice_weights = []

    for j in range(h):
        lo, hi = quantiles[j], quantiles[j + 1]
        # 마지막 슬라이스는 상한 포함
        if j < h - 1:
            mask = (Y >= lo) & (Y < hi)
        else:
            mask = (Y >= lo) & (Y <= hi)

        if mask.sum() < 2:
            continue
        slice_means.append(X[mask].mean(axis=0))
        slice_weights.append(mask.sum())

    if len(slice_means) < 2:
        # 슬라이스가 충분하지 않으면 PCA fallback
        eigvals, eigvecs = eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[0]], eigvecs[:, idx[1]], Sigma, "PCA fallback"

    slice_means   = np.array(slice_means)         # (h, p)
    slice_weights = np.array(slice_weights, dtype=float)
    slice_weights /= slice_weights.sum()           # 정규화

    # M = 슬라이스 평균들의 가중 공분산
    # M = Σₕ wₕ (m̄ₕ - m̄)(m̄ₕ - m̄)ᵀ
    grand_mean = (slice_weights[:, None] * slice_means).sum(axis=0)
    M = np.zeros((p, p))
    for w, m in zip(slice_weights, slice_means):
        diff = (m - grand_mean)[:, None]
        M += w * (diff @ diff.T)

    # Σ⁻¹M 의 고유벡터 → SDR 방향
    # (수치 안정성을 위해 regularization 추가)
    Sigma_reg = Sigma + 1e-6 * np.eye(p)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    A = Sigma_inv @ M

    eigvals, eigvecs = eigh(A)
    idx = np.argsort(eigvals)[::-1]
    beta1 = eigvecs[:, idx[0]]
    beta2 = eigvecs[:, idx[1]]

    return beta1, beta2, Sigma, "SIR"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # ── 데이터 다운로드 (더 긴 기간 — Y 계산에 20일 필요)
            all_tickers = [sym] + MARKET_TICKERS
            raw = yf.download(all_tickers, period="120d", progress=False)["Close"].dropna()

            if raw.shape[0] < 40:
                raise ValueError("데이터 부족 (최소 40일 필요)")

            # ── Y 생성: 미래 20일 최대 손실 (forward drawdown)
            #
            # Y[t] = min(price[t+1..t+20]) / price[t] - 1
            #      = t 시점 이후 20일간 최악의 손실
            #
            # 이게 우리가 SDR로 "설명하고 싶은" 꼬리 위험
            target_prices = raw[sym].values          # 자산 가격
            n_total = len(target_prices)
            horizon = 20                             # 20일 forward

            Y = np.array([
                (target_prices[t+1:t+horizon+1].min() / target_prices[t]) - 1
                for t in range(n_total - horizon)
            ])                                       # (n_total - horizon,)

            # ── X 생성: 시장 변수 수익률 (Y와 길이 맞춤)
            rets_all = raw.pct_change().dropna().values  # (n_days, 11)
            # X[t]는 t시점 정보 → Y[t]와 대응
            # rets_all은 1일 지연되어 있으므로 길이 조정
            n_xy = min(len(Y), len(rets_all))
            X = rets_all[:n_xy, 1:]                  # 자산 제외, 시장변수만 (10개)
            Y = Y[:n_xy]

            if len(Y) < 20:
                raise ValueError("정렬 후 데이터 부족")

            # ── SIR 실행
            beta1, beta2, Sigma, method = compute_sir(X, Y, h=10)

            # ── Rᵖ(Σ) Hilbert Space 투영
            # ⟨u,v⟩_Σ = uᵀΣv 내적을 사용하는 공간에서의 투영
            projected = X @ np.column_stack([beta1, beta2])  # (n, 2)

            # ── 분산 설명 비율 (Y 기준)
            # SIR에서 β 방향이 Y를 얼마나 설명하는지
            proj_1d = X @ beta1
            corr_with_Y = float(np.corrcoef(proj_1d, Y)[0, 1])
            var_explained = round(corr_with_Y ** 2 * 100, 1)

            # ── 레짐 분류 (Y 기준 — drawdown이 클수록 crash)
            # Y가 작을수록(더 음수) crash → β₁ 투영값으로 분류
            y_q33 = float(np.percentile(Y, 33))
            y_q66 = float(np.percentile(Y, 66))

            # β₁ 투영값과 Y의 상관 방향 파악
            flip = -1 if corr_with_Y > 0 else 1   # β₁이 Y와 음의 상관이면 flip

            points = []
            for i, (px, py) in enumerate(projected[:-1]):
                y_val = Y[i]
                # Y (drawdown)가 작을수록 crash
                if y_val <= y_q33:
                    col = "#F03860"   # crash (가장 큰 손실)
                elif y_val <= y_q66:
                    col = "#F0A800"   # elevated
                else:
                    col = "#00D878"   # safe (손실 적음)
                points.append({
                    "x": round(float(px) * 60, 4),
                    "y": round(float(py) * 60, 4),
                    "col": col
                })

            # 현재 상태 (마지막 날)
            cx = round(float(projected[-1, 0]) * 60, 4)
            cy = round(float(projected[-1, 1]) * 60, 4)

            # 현재 레짐: 마지막 X로 Y 예측 (최근 20일 평균 drawdown 기준)
            recent_dd = float(Y[-5:].mean())     # 최근 5일 drawdown 평균
            if recent_dd <= y_q33:
                regime = "CRASH ZONE"
            elif recent_dd <= y_q66:
                regime = "ELEVATED"
            else:
                regime = "SAFE ZONE"

            # β₁ 중 가장 중요한 시장 변수
            top_idx = int(np.argmax(np.abs(beta1)))
            top_factor = MARKET_NAMES[top_idx] if top_idx < len(MARKET_NAMES) else "Market"

            # Y 분포 통계 (CVaR 연결용)
            y_cvar5 = float(np.percentile(Y, 5))       # 하위 5% drawdown

            body = json.dumps({
                "points":             points,
                "current":            {"x": cx, "y": cy},
                "regime":             regime,
                "variance_explained": var_explained,
                "top_risk_factor":    top_factor,
                "method":             method,
                # SIR 관련 메타데이터 (발표용)
                "sir_meta": {
                    "Y_definition":   "20-day forward maximum drawdown",
                    "Y_mean":         round(float(Y.mean()) * 100, 2),
                    "Y_cvar5":        round(y_cvar5 * 100, 2),
                    "n_slices":       10,
                    "corr_beta1_Y":   round(float(np.corrcoef(X @ beta1, Y)[0,1]), 3),
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

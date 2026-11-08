"""v4 실험 B: 드리프트와 재학습 주기.

데이터: gen_data_v4.generate_drift (2025-01 ~ 2026-06, 2026-01부터 신종 수법 + 간편송금 확산)
운영 가정
- 평가 기간: 2025-10 ~ 2026-06 (9개월), 월 단위
- 라벨 지연 30일: 시점 T에 학습할 때 T-30일 이전 거래의 라벨만 안다
- 학습 데이터: 2025-04-01 ~ T-30일 (누적)
- IF는 라벨이 필요 없어 T 직전 90일 거래로 다시 fit
- 알림 컷: 재학습 시점 직전 30일 거래를 새 모델로 채점해, 월 예산(가구당 0.1건)이 되도록 정한다 (스태킹 70%, IF 30%)
  다음 재학습까지 그 컷을 고정해서 쓴다 → 행동이 바뀌면 알림 수가 예산을 넘는다
정책: 고정(2025-10에 한 번), 분기(10·1·4월), 매월
"""
import numpy as np, pandas as pd, json, time, sys
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from gen_data_v4 import generate_drift
from features_v4 import build_features_v4, FEATURES_V4
from exp_seeds import DEFAULT

START = pd.Timestamp("2025-04-01")
MONTHS = pd.date_range("2025-10-01", "2026-06-01", freq="MS")
DELAY = pd.Timedelta(days=30)
B, N_FAM, ISO_SHARE = 0.1, 4000, 0.3
POLICIES = {"고정": [MONTHS[0]], "분기": [m for m in MONTHS if m.month in (10, 1, 4)], "매월": list(MONTHS)}
F = FEATURES_V4


def fit_iso(X, seed):
    return IsolationForest(n_estimators=200, random_state=seed).fit(X.sample(min(200_000, len(X)), random_state=seed))


def train_at(ev, T, seed=0):
    """시점 T에 쓸 수 있는 정보로 IF + 스태킹 모델과 알림 컷을 만든다."""
    lab = ev[(ev.ts >= START) & (ev.ts < T - DELAY)]
    recent = ev[(ev.ts >= T - pd.Timedelta(days=90)) & (ev.ts < T)]
    iso = fit_iso(recent[F], seed)
    # 학습 행의 IF 점수: 가구 3-fold로 그 행을 안 본 IF가 매김 (nested)
    s_tr = np.full(len(lab), np.nan)
    for j, (a, b) in enumerate(GroupKFold(3).split(lab, groups=lab.pid)):
        s_tr[b] = -fit_iso(lab.iloc[a][F], seed + 10 + j).score_samples(lab.iloc[b][F])
    X = lab[F].copy(); X["s_iso"] = s_tr
    hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **DEFAULT).fit(X, lab.is_fraud)
    last = ev[(ev.ts >= T - pd.Timedelta(days=30)) & (ev.ts < T)]
    si, ss = score(iso, hgb, last)
    k = B * N_FAM
    cut_s = np.sort(ss)[::-1][int(k * (1 - ISO_SHARE)) - 1]
    cut_i = np.sort(si)[::-1][int(k * ISO_SHARE) - 1]
    return dict(iso=iso, hgb=hgb, cut_s=cut_s, cut_i=cut_i, n_lab_eps=int(lab.episode[lab.is_fraud == 1].nunique()))


def score(iso, hgb, X):
    si = -iso.score_samples(X[F])
    Z = X[F].copy(); Z["s_iso"] = si
    return si, hgb.predict_proba(Z)[:, 1]


def psi(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if len(np.unique(ref)) <= 2:
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        # 0이 많은 피처는 '0' 구간을 따로 두고, 나머지를 분위수로 나눈다
        pos = ref[ref > 0]
        q = np.unique(np.quantile(pos if (ref == 0).mean() > 0.3 else ref, np.linspace(0, 1, bins + 1)))
        edges = np.unique(np.concatenate([[-np.inf, 1e-12] if (ref == 0).mean() > 0.3 else [-np.inf], q[1:-1], [np.inf]]))
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def run(seed=42):
    t0 = time.time()
    df_raw, eps = generate_drift(seed=seed)
    ev = build_features_v4(df_raw)
    ev = ev[ev.ts >= START].merge(eps[["episode", "scenario"]], on="episode", how="left").reset_index(drop=True)
    first = ev[ev.is_fraud == 1].groupby("episode").ts.min()
    eps = eps.set_index("episode").assign(start=first).dropna(subset=["start"]).reset_index()
    eps["month"] = eps.start.dt.to_period("M").dt.to_timestamp()
    ev["month"] = ev.ts.dt.to_period("M").dt.to_timestamp()

    models = {}
    for T in sorted(set(sum(POLICIES.values(), []))):
        models[T] = train_at(ev, T, seed)
        print("trained", T.date(), "labeled eps", models[T]["n_lab_eps"], round(time.time() - t0), flush=True)

    rows, alert_rows = [], []
    for pol, Ts in POLICIES.items():
        caught_all, caught_sup, caught_iso = set(), set(), set()
        for M in MONTHS:
            T = max(t for t in Ts if t <= M)
            m = models[T]
            cur = ev[ev.month == M]
            si, ss = score(m["iso"], m["hgb"], cur)
            a_sup = ss >= m["cut_s"]; a_iso = si >= m["cut_i"]
            al = a_sup | a_iso
            fr = cur.is_fraud.to_numpy() == 1
            caught_all |= set(cur.episode[al & fr]); caught_sup |= set(cur.episode[a_sup & fr]); caught_iso |= set(cur.episode[a_iso & fr])
            alert_rows.append(dict(정책=pol, 월=M, 알림수=int(al.sum()), 헛알림=int((al & ~fr).sum()),
                                   스태킹알림=int(a_sup.sum()), IF알림=int(a_iso.sum())))
        for M in MONTHS:
            e = eps[eps.month == M]
            for grp, sub in [("기존 수법", e[~e.scenario.isin(["invest_scam", "giftcard_scam"])]),
                             ("신종: 투자사기", e[e.scenario == "invest_scam"]), ("신종: 상품권 결제", e[e.scenario == "giftcard_scam"])]:
                if len(sub):
                    rows.append(dict(정책=pol, 월=M, 구분=grp, 에피소드=len(sub),
                                     탐지율=sub.episode.isin(caught_all).mean(), 스태킹만=sub.episode.isin(caught_sup).mean(),
                                     IF만=sub.episode.isin(caught_iso).mean()))
    rec = pd.DataFrame(rows); alr = pd.DataFrame(alert_rows)

    # PSI: 기준 = 2025-04~08 (고정 모델의 학습 구간)
    ref = ev[(ev.ts >= START) & (ev.ts < MONTHS[0] - DELAY)]
    psi_feats = ["new_payee", "newpayee_n_7d", "z_shrink", "out_24h_ratio", "out_72h_ratio", "log_amt", "t_card"]
    m0 = models[MONTHS[0]]
    _, ss_ref = score(m0["iso"], m0["hgb"], ref.sample(200_000, random_state=0))
    prow = []
    for M in pd.date_range("2025-06-01", "2026-06-01", freq="MS"):
        cur = ev[ev.month == M]
        r = {"월": M}
        for f in psi_feats:
            r[f] = psi(ref[f], cur[f])
        r["점수(고정 모델)"] = psi(ss_ref, score(m0["iso"], m0["hgb"], cur)[1])
        prow.append(r)
    ps = pd.DataFrame(prow)
    return dict(rec=rec, alr=alr, psi=ps, eps=eps, n_lab={str(T.date()): m["n_lab_eps"] for T, m in models.items()}, sec=round(time.time() - t0))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    out = run(seed)
    pd.to_pickle(out, f"drift_seed{seed}.pkl")
    print(out["rec"].groupby(["정책", "구분"]).apply(lambda d: np.average(d.탐지율, weights=d.에피소드)).round(3))
    print(out["alr"].pivot(index="월", columns="정책", values="알림수"))
    print(out["psi"].round(3).to_string())
    print(out["sec"])

"""v6 실험: v5의 판단력 저하 감지 결과가 '가정'에 얼마나 기대고 있는지 확인한다.

문헌이 말하는 것 두 가지가 v5 가정과 다르다.
 (1) 변화는 훨씬 느리다 (진단 5~10년 전부터)        → ramp를 3~5년으로 늘린 4년 관측 실험
 (2) 집단 평균 차이는 작다 (예: 연체 7.7% vs 7.3%)   → 신호 크기 배율 effect를 1 → 0.03까지 낮춘 실험
그리고 문헌에서 가져온 신호(여가 소비 감소, 접속 감소, 비밀번호 재설정·카드 분실)를 더했을 때의 효과를 본다.

방법 3가지 (컷은 모두 '저하가 아닌 가구 중 관측 기간에 한 번이라도 알림을 받는 비율 = 5%'로 맞춤)
- 규칙+맥락        : v5의 최종 규칙 (거래 신호만)
- 규칙+맥락+문헌신호: 위에 문헌 신호 3개를 더한 규칙
- 지도학습(상한)    : 모든 피처 + 진단 라벨이 있다고 가정한 5-fold
"""
import sys, os, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from gen_data_v6 import generate_v6
from decline_features import monthly_raw, change_features, CHG, BASE_M

UP = ["chg_card_dup", "chg_atm_n", "chg_night", "chg_top_mer", "chg_bill_missed", "chg_bill_dup"]
EXTRA = ["chg_leisure", "chg_login", "svc_6m"]
FA = 0.05


def extra_features(df2, events, checkpoints, n_months):
    """문헌 신호: 여가·외출 결제 비중 변화, 접속 횟수 변화, 최근 6개월 비밀번호 재설정·카드 분실 건수(기준선 대비)."""
    c = df2[df2.type == "card"]
    mid = pd.to_numeric(c.payee.str[1:], errors="coerce")
    c = c.assign(leis=(mid >= 200).astype(float))
    ls = c.groupby(["pid", "m"]).leis.mean().unstack()
    ev = events[events.m < n_months]
    lg = ev.pivot(index="pid", columns="m", values="logins")
    sv = ev.assign(svc=ev.pin_reset + ev.card_lost).pivot(index="pid", columns="m", values="svc")
    base = lambda w: w[[m for m in w.columns if m < BASE_M]]
    ls = ls.reindex(lg.index)
    ls = ls.apply(lambda col: col.fillna(base(ls).median(axis=1)))
    rows = []
    for cp in checkpoints:
        rec = [m for m in lg.columns if cp - 3 < m <= cp]
        r = pd.DataFrame(index=lg.index)
        r["chg_leisure"] = (ls[rec].mean(axis=1) - base(ls).mean(axis=1)) / base(ls).std(axis=1).clip(lower=0.08)
        r["chg_login"] = (lg[rec].mean(axis=1) - base(lg).mean(axis=1)) / base(lg).std(axis=1).clip(lower=1.5)
        r["svc_6m"] = sv[[m for m in sv.columns if cp - 6 < m <= cp]].sum(axis=1) - base(sv).sum(axis=1)
        r["c"] = cp
        rows.append(r.reset_index())
    return pd.concat(rows, ignore_index=True).fillna(0)


def build(seed, n_months, **gen):
    df, eps, dec, lc, events = generate_v6(seed=seed, days=n_months * 30 + 10, **gen)
    cps = range(BASE_M, n_months)
    raw, d2 = monthly_raw(df, n_months)
    X = change_features(raw, d2, cps).merge(extra_features(d2, events, cps, n_months), on=["pid", "c"])
    X = X.merge(dec, on="pid", how="left").merge(lc, on="pid", how="left")
    X["grp"] = np.where(X.onset.notna(), "decline", np.where(X.life.notna(), X.life, "normal"))
    X["since"] = X.c + 1 - X.onset / 30
    return X.sort_values(["pid", "c"]).reset_index(drop=True)


def scores(X):
    S = pd.DataFrame(index=X.index)
    bill = X[["chg_bill_missed", "chg_bill_dup"]].clip(0, 5).sum(axis=1) + (X.bill_jitter / 1.5).clip(0, 5)
    r = X[["chg_card_dup", "chg_atm_n", "chg_night", "chg_top_mer"]].clip(0, 5).sum(axis=1) + (-X.chg_atm_amt).clip(0, 5) \
        + (X.psi_hour * 5).clip(0, 5) + bill * (1 - X.gap_recent)
    S["규칙+맥락"] = r
    lit = (-X.chg_leisure).clip(0, 5) + (-X.chg_login).clip(0, 5)
    S["규칙+맥락+문헌신호"] = r + lit * (1 - X.gap_recent) + (2.5 * X.svc_6m).clip(0, 5)
    F = CHG + EXTRA
    Xs = X[F].assign(rule=S["규칙+맥락+문헌신호"])
    y = ((X.grp == "decline") & (X.since > 0)).astype(int)
    sup = np.full(len(X), np.nan)
    for k, (a, b) in enumerate(GroupKFold(5).split(Xs, groups=X.pid)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=k).fit(Xs.iloc[a], y.iloc[a])
        sup[b] = m.predict_proba(Xs.iloc[b])[:, 1]
    S["지도학습(상한)"] = sup
    return S


def evaluate(X, S, n_months, horizons):
    hh = X.groupby("pid").agg(grp=("grp", "first"), onset=("onset", "first"))
    neg = hh.index[hh.grp != "decline"]
    out = {}
    for col in S.columns:
        mx = S[col].groupby(X.pid).max()
        cut = np.quantile(mx.loc[neg], 1 - FA)
        flag = X[S[col] > cut].groupby("pid").c.min()
        d = hh[hh.grp == "decline"].copy()
        d["lead"] = (flag.reindex(d.index) + 1) - d.onset / 30
        r = {"fa_hospital": float(flag.reindex(hh.index[hh.grp == "hospital"]).notna().mean()),
             "det_ever": float((d.lead > 0).mean()), "lead_median": float(d.lead[d.lead > 0].median())}
        for k in horizons:
            ok = d[d.onset / 30 + k <= n_months]
            r[f"det_{k}m"] = float(((ok.lead > 0) & (ok.lead <= k)).mean()) if len(ok) else None
        out[col] = r
    return out


CONFIGS = {
    "기본(v5 가정)": dict(n_months=24, effect=1.0),
    "신호 0.5배": dict(n_months=24, effect=0.5),
    "신호 0.25배": dict(n_months=24, effect=0.25),
    "신호 0.1배": dict(n_months=24, effect=0.1),
    "신호 0.03배(문헌의 연체 차이 수준)": dict(n_months=24, effect=0.03),
    "느린 진행(3~5년), 4년 관측": dict(n_months=48, effect=1.0, ramp=(1100, 1800)),
}

if __name__ == "__main__":
    seed = int(sys.argv[1]); names = sys.argv[2:] or list(CONFIGS)
    path = f"results_v6_seed{seed}.json"
    R = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        cfg = dict(CONFIGS[name]); nm = cfg.pop("n_months"); t = time.time()
        X = build(seed, nm, **cfg)
        res = evaluate(X, scores(X), nm, [6, 12, 18] if nm == 24 else [12, 24, 36])
        R[name] = res
        json.dump(R, open(path, "w"), ensure_ascii=False, indent=1)
        print(seed, name, round(time.time() - t), json.dumps(res, ensure_ascii=False), flush=True)

"""v5 실험: 판단력 저하를 가구 단위 장기 변화로 감지.

체크포인트: 매월(7~24개월째). 각 체크포인트에서 가구마다 점수 하나.
알림 규칙: 점수가 컷을 넘으면 그 가구에 '확인 권유' 알림 1회(이후 같은 가구는 다시 세지 않음).
컷은 '판단력 저하가 아닌 가구 중 18개월 동안 한 번이라도 알림을 받는 비율 = 5%'가 되도록 정한다.

모델
- 규칙: 나빠지는 방향의 변화(0~5로 자름)를 더한 점수
- 규칙+지속: 최근 3개월 규칙 점수의 평균 (일시적 변화를 걸러내려는 시도)
- 규칙+맥락: 활동 공백(입원 등) 직후의 공과금 신호를 빼고, 납부일 '평균 이동' 대신 '흐트러짐'을 씀
- IF: 변화 피처로 학습한 Isolation Forest (라벨 없음)
- 지도학습(참고용 상한): 가구 단위 5-fold, 라벨=판단력 저하 시작 이후. 실제로는 진단 라벨을 얻기 어렵다.
"""
import numpy as np, pandas as pd, json, sys, os, time
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from gen_data_v5 import generate_decline
from decline_features import monthly_raw, change_features, CHG

UP = ["chg_bill_dev", "chg_bill_missed", "chg_bill_dup", "chg_card_dup", "chg_atm_n", "chg_night", "chg_top_mer"]
BILL = ["chg_bill_dev", "chg_bill_missed", "chg_bill_dup"]
DOWN = ["chg_atm_amt"]
FA = 0.05
CHECK = range(6, 24)


def build(seed):
    df, eps, dec, lc = generate_decline(seed=seed)
    raw, d2 = monthly_raw(df, 24)
    X = change_features(raw, d2, CHECK)
    X = X.merge(dec, on="pid", how="left").merge(lc, on="pid", how="left")
    X["grp"] = np.where(X.onset.notna(), "decline", np.where(X.life.notna(), X.life, "normal"))
    X["since"] = X.c + 1 - X.onset / 30
    return X.sort_values(["pid", "c"]).reset_index(drop=True)


def scores(X, seed):
    S = pd.DataFrame(index=X.index)
    r = X[UP].clip(0, 5).sum(axis=1) + (-X[DOWN]).clip(0, 5).sum(axis=1) + (X.psi_hour * 5).clip(0, 5)
    S["규칙"] = r
    S["규칙+지속"] = r.groupby(X.pid).transform(lambda s: s.rolling(3, min_periods=1).mean())
    # 맥락 보정: 활동 공백(입원 등)이 있었던 기간의 공과금 신호는 빼고, 납부일 '평균 이동' 대신 '흐트러짐'을 쓴다
    up2 = [f for f in UP if f not in BILL] + ["chg_bill_missed", "chg_bill_dup"]
    r2 = X[up2].clip(0, 5).sum(axis=1) + (X.bill_jitter / 1.5).clip(0, 5) + (-X[DOWN]).clip(0, 5).sum(axis=1) + (X.psi_hour * 5).clip(0, 5)
    r2 = r2 - X.gap_recent * (X[["chg_bill_missed", "chg_bill_dup"]].clip(0, 5).sum(axis=1) + (X.bill_jitter / 1.5).clip(0, 5))
    S["규칙+맥락"] = r2
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(X[CHG].sample(20_000, random_state=seed))
    s_iso = -iso.score_samples(X[CHG])
    S["IF"] = pd.Series(s_iso, index=X.index).groupby(X.pid).transform(lambda s: s.rolling(3, min_periods=1).mean())
    # 지도학습 상한
    Xs = X[CHG].assign(rule=r, rule_lag3=r.groupby(X.pid).shift(3).fillna(0), rule_avg3=S["규칙+지속"])
    y = ((X.grp == "decline") & (X.since > 0)).astype(int)
    sup = np.full(len(X), np.nan)
    for k, (a, b) in enumerate(GroupKFold(5).split(Xs, groups=X.pid)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=k).fit(Xs.iloc[a], y.iloc[a])
        sup[b] = m.predict_proba(Xs.iloc[b])[:, 1]
    S["지도학습(상한)"] = sup
    return S


def evaluate(X, S, fa=FA):
    hh = X.groupby("pid").agg(grp=("grp", "first"), onset=("onset", "first"))
    neg = hh.index[hh.grp != "decline"]
    out = {}
    for col in S.columns:
        mx = S[col].groupby(X.pid).max()
        cut = np.quantile(mx.loc[neg], 1 - fa)  # 비저하 가구 중 fa만큼만 한 번이라도 넘음
        flag = X[S[col] > cut].groupby("pid").c.min()  # 첫 알림 체크포인트
        r = {"cut": float(cut)}
        for g in ["normal", "hospital", "move_in", "new_routine"]:
            ids = hh.index[hh.grp == g]
            r[f"fa_{g}"] = float(flag.reindex(ids).notna().mean())
        d = hh[hh.grp == "decline"].copy()
        d["first"] = flag.reindex(d.index)
        d["lead"] = (d["first"] + 1) - d.onset / 30  # 알림 시점 - 시작 시점(개월)
        r["pre_onset"] = float((d.lead <= 0).mean())
        for k in [3, 6, 9, 12]:
            ok = d[d.onset / 30 + k <= 24]  # k개월 뒤까지 관측되는 가구만
            r[f"det_{k}m"] = float(((ok.lead > 0) & (ok.lead <= k)).mean())
            r[f"n_{k}m"] = int(len(ok))
        r["det_ever"] = float((d.lead > 0).mean())
        r["lead_median"] = float(d.lead[d.lead > 0].median())
        out[col] = r
    return out


def run(seed):
    t = time.time()
    X = build(seed)
    S = scores(X, seed)
    res = {str(fa): evaluate(X, S, fa) for fa in [0.02, 0.05, 0.10]}
    if seed == 42:
        X.join(S).to_pickle("decline_seed42.pkl")
    return {"seed": seed, "n_decline": int((X.groupby("pid").grp.first() == "decline").sum()), "res": res, "sec": round(time.time() - t)}


if __name__ == "__main__":
    path = "results_decline.json"
    R = json.load(open(path)) if os.path.exists(path) else []
    for sd in [int(s) for s in sys.argv[1:]] or [42, 1, 2]:
        o = run(sd)
        R = [r for r in R if r["seed"] != sd] + [o]
        json.dump(R, open(path, "w"), ensure_ascii=False, indent=1)
        print(sd, o["sec"], pd.DataFrame(o["res"]["0.05"]).round(3).to_string(), flush=True)

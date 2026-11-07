"""v3 실험 2: 데이터 시드 5개 반복 + 시간순 검증.
seed마다 데이터를 새로 만들고 아래를 계산해 results_seeds.json에 쌓는다.
- 가구 CV(평가 2025-04~12): v2 방식 스태킹 vs v3(z_shrink+hist_n, nested CV, 튜닝)
- LOSO(처음 보는 사기)와 2트랙(IF 30%)
- 시간순 검증: 학습 가구의 2025-04~08로 학습 → 평가 가구의 2025-09~12로 평가
  비교용으로 '미래 기간까지 학습에 쓴' 경우도 같은 평가 구간에서 계산"""
import numpy as np, pandas as pd, json, time, sys, os
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from gen_data import generate
from features import FEATURES
from features_v3 import build_features_v3, FEATURES_V3

DEFAULT = dict(max_iter=300, learning_rate=0.05)
TUNED = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20)
SC = ["impersonation", "cash_pickup", "loan_scam", "exploitation"]
B, SPLIT, START = 0.1, pd.Timestamp("2025-09-01"), pd.Timestamp("2025-04-01")


def iso(X, seed):
    return IsolationForest(n_estimators=200, random_state=seed).fit(X.sample(min(200_000, len(X)), random_state=seed))


def hgb(X, y, p, seed):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **p).fit(X, y)


def nested_iso(tr, te, F, k):
    s_tr = np.full(len(tr), np.nan)
    for j, (ia, ib) in enumerate(GroupKFold(3).split(tr, groups=tr.pid)):
        s_tr[ib] = -iso(tr.iloc[ia][F], 100 + 10 * k + j).score_samples(tr.iloc[ib][F])
    return s_tr, -iso(tr[F], k).score_samples(te[F])


def with_s(X, s):
    X = X.copy(); X["s_iso"] = s; return X


def recall(frame, score, k, ep_ids):
    idx = np.argsort(-np.asarray(score))[:k]
    sel = frame.iloc[idx]
    caught = set(sel.episode[sel.is_fraud == 1])
    return float(np.mean([e in caught for e in ep_ids])) if len(ep_ids) else np.nan


def two_track(frame, s_stack, s_iso, k, ep_ids, share=0.3):
    k2 = int(k * share); k1 = k - k2
    i1 = np.argsort(-np.asarray(s_stack))[:k1]; i2 = np.argsort(-np.asarray(s_iso))[:k2]
    sel = frame.iloc[np.union1d(i1, i2)]
    caught = set(sel.episode[sel.is_fraud == 1])
    return float(np.mean([e in caught for e in ep_ids]))


def run(seed):
    t0 = time.time()
    df_raw, eps = generate(n_people=4000, seed=seed)
    df = build_features_v3(df_raw)
    ev = df[df.ts >= START].merge(eps[["episode", "scenario"]], on="episode", how="left").reset_index(drop=True)
    y = ev.is_fraud
    folds = list(GroupKFold(5).split(ev, groups=ev.pid))
    n = len(ev)
    S = {c: np.full(n, np.nan) for c in ["iso_v2", "iso_v3", "stack_v2", "stackN_v3_def", "stackN_v3_tun"] + [f"loso_{s}" for s in SC]}
    for k, (a, b) in enumerate(folds):
        tr, te = ev.iloc[a], ev.iloc[b]
        # v2 방식: IF는 바깥 fold OOF 점수를 그대로 사용
        S["iso_v2"][b] = -iso(tr[FEATURES], k).score_samples(te[FEATURES])
        S["iso_v3"][b] = -iso(tr[FEATURES_V3], k).score_samples(te[FEATURES_V3])
        s_tr, s_te = nested_iso(tr, te, FEATURES_V3, k)
        S["stackN_v3_def"][b] = hgb(with_s(tr[FEATURES_V3], s_tr), tr.is_fraud, DEFAULT, k).predict_proba(with_s(te[FEATURES_V3], s_te))[:, 1]
        S["stackN_v3_tun"][b] = hgb(with_s(tr[FEATURES_V3], s_tr), tr.is_fraud, TUNED, k).predict_proba(with_s(te[FEATURES_V3], s_te))[:, 1]
        for sc in SC:
            m = (tr.scenario != sc).to_numpy()
            S[f"loso_{sc}"][b] = hgb(with_s(tr[FEATURES_V3], s_tr)[m], tr.is_fraud[m], TUNED, k).predict_proba(with_s(te[FEATURES_V3], s_te))[:, 1]
    # v2 스태킹(비nested): 학습 행도 OOF iso_v2 점수 사용
    for k, (a, b) in enumerate(folds):
        tr, te = ev.iloc[a], ev.iloc[b]
        S["stack_v2"][b] = hgb(with_s(tr[FEATURES], S["iso_v2"][a]), tr.is_fraud, DEFAULT, k).predict_proba(with_s(te[FEATURES], S["iso_v2"][b]))[:, 1]

    K = int(B * ev.pid.nunique() * 9)
    out = {"seed": seed, "n_tx": n, "n_eps": len(eps), "cv": {}}
    for c in ["iso_v2", "iso_v3", "stack_v2", "stackN_v3_def", "stackN_v3_tun"]:
        r = {"pr_auc": average_precision_score(y, S[c]), "recall": recall(ev, S[c], K, eps.episode)}
        for sc in SC: r[sc] = recall(ev, S[c], K, eps[eps.scenario == sc].episode)
        out["cv"][c] = r
    unseen = {sc: recall(ev, S[f"loso_{sc}"], K, eps[eps.scenario == sc].episode) for sc in SC}
    w = eps.scenario.value_counts()
    out["unseen_stack"] = sum(unseen[s] * w[s] for s in SC) / w.sum()
    out["unseen_by_sc"] = unseen
    for name, s_iso in [("v2", S["iso_v2"]), ("v3", S["iso_v3"])]:
        st = S["stack_v2"] if name == "v2" else S["stackN_v3_tun"]
        out[f"twotrack_known_{name}"] = two_track(ev, st, s_iso, K, eps.episode)
        un = sum(two_track(ev, S[f"loso_{sc}"] if name == "v3" else S[f"loso_{sc}"], s_iso, K, eps[eps.scenario == sc].episode) * w[sc] for sc in SC) / w.sum()
        out[f"twotrack_unseen_{name}"] = un

    # 시간순 검증
    te_mask = ev.ts >= SPLIT
    sc_time = {"iso": np.full(n, np.nan), "time": np.full(n, np.nan), "leak": np.full(n, np.nan)}
    for k, (a, b) in enumerate(folds):
        trall = ev.iloc[a]
        trp = trall[trall.ts < SPLIT]
        bb = b[te_mask.iloc[b].to_numpy()]
        te = ev.iloc[bb]
        for tag, tr in [("time", trp), ("leak", trall)]:
            s_tr, s_te = nested_iso(tr, te, FEATURES_V3, k)
            sc_time[tag][bb] = hgb(with_s(tr[FEATURES_V3], s_tr), tr.is_fraud, TUNED, k).predict_proba(with_s(te[FEATURES_V3], s_te))[:, 1]
            if tag == "time": sc_time["iso"][bb] = s_te
    tm = te_mask.to_numpy()
    evt = ev[tm]
    ep_t = evt.episode[evt.is_fraud == 1].dropna().unique()
    Kt = int(B * ev.pid.nunique() * 4)
    out["time"] = {"n_eps": int(len(ep_t))}
    for tag in sc_time:
        s = sc_time[tag][tm]
        out["time"][tag] = {"pr_auc": average_precision_score(evt.is_fraud, s), "recall": recall(evt, s, Kt, ep_t)}
    out["time"]["twotrack"] = two_track(evt, sc_time["time"][tm], sc_time["iso"][tm], Kt, ep_t)
    out["sec"] = round(time.time() - t0)
    return out, ev, S, sc_time


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 1, 2, 3, 4]
    path = "results_seeds.json"
    res = json.load(open(path)) if os.path.exists(path) else []
    for sd in seeds:
        o, ev, S, st = run(sd)
        res = [r for r in res if r["seed"] != sd] + [o]
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1, default=float)
        if sd == 42:
            pd.DataFrame(S).to_pickle("scores_seed42.pkl")
        print(json.dumps(o, ensure_ascii=False, default=float), flush=True)

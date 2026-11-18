"""v7 실험: 상대 계좌 피처의 효과. 시드별로 v4 피처 vs v4 + 상대 계좌 피처.
가구 5-fold, nested IF 스태킹, LOSO(처음 보는 사기), 2트랙(IF 30%). 데이터는 대포통장·사업자·경조사 계좌 구조를 입힌 1년 치."""
import numpy as np, pandas as pd, json, time, sys, os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from gen_data import generate
from features_v4 import build_features_v4, FEATURES_V4
from counterparty import add_counterparties, add_cp_features, CP
from exp_seeds import hgb, nested_iso, with_s, recall, two_track, DEFAULT, SC

B = 0.1


def run(seed):
    t0 = time.time()
    df_raw, eps = generate(n_people=4000, seed=seed)
    df_raw = add_counterparties(df_raw, eps, seed)
    df = add_cp_features(build_features_v4(df_raw))
    ev = df[df.ts >= pd.Timestamp("2025-04-01")].merge(eps[["episode", "scenario"]], on="episode", how="left").reset_index(drop=True)
    folds = list(GroupKFold(5).split(ev, groups=ev.pid))
    K = int(B * ev.pid.nunique() * 9)
    w = eps.scenario.value_counts()
    evt = ev.payee.str.startswith("EVT").to_numpy()
    out = {"seed": seed, "n_mule": int(df_raw.payee[df_raw.payee.str.startswith("MULE")].nunique()), "n_evt_tx": int(evt.sum())}
    for tag, F in [("v4", FEATURES_V4), ("v7", FEATURES_V4 + CP)]:
        n = len(ev)
        s_iso, s_st = np.full(n, np.nan), np.full(n, np.nan)
        lo = {sc: np.full(n, np.nan) for sc in SC}
        for k, (a, b) in enumerate(folds):
            tr, te = ev.iloc[a], ev.iloc[b]
            s_tr, s_te = nested_iso(tr, te, F, k)
            s_iso[b] = s_te
            Xtr, Xte = with_s(tr[F], s_tr), with_s(te[F], s_te)
            s_st[b] = hgb(Xtr, tr.is_fraud, DEFAULT, k).predict_proba(Xte)[:, 1]
            for sc in SC:
                m = (tr.scenario != sc).to_numpy()
                lo[sc][b] = hgb(Xtr[m], tr.is_fraud[m], DEFAULT, k).predict_proba(Xte)[:, 1]
        r = {"pr_auc": average_precision_score(ev.is_fraud, s_st), "recall": recall(ev, s_st, K, eps.episode), "iso_recall": recall(ev, s_iso, K, eps.episode)}
        k1 = K - int(K * 0.3)
        top = np.union1d(np.argsort(-s_st)[:k1], np.argsort(-s_iso)[:K - k1])
        r["evt_false_alerts"] = int(evt[top].sum())           # 경조사 계좌 송금에 간 헛알림 수
        first_tx = ev[ev.is_fraud == 1].sort_values("ts").groupby("episode").head(1).index.to_numpy()
        r["first_tx_caught"] = float(np.isin(first_tx, np.argsort(-s_st)[:K]).mean())  # 사건의 '첫 거래'에서 잡힌 비율
        for sc in SC:
            e = eps[eps.scenario == sc].episode
            r[f"known_{sc}"] = recall(ev, s_st, K, e); r[f"unseen_{sc}"] = recall(ev, lo[sc], K, e)
            r[f"iso_{sc}"] = recall(ev, s_iso, K, e); r[f"tt_unseen_{sc}"] = two_track(ev, lo[sc], s_iso, K, e)
        r["unseen"] = sum(r[f"unseen_{sc}"] * w[sc] for sc in SC) / w.sum()
        r["tt_known"] = two_track(ev, s_st, s_iso, K, eps.episode)
        r["tt_unseen"] = sum(r[f"tt_unseen_{sc}"] * w[sc] for sc in SC) / w.sum()
        out[tag] = r
    out["sec"] = round(time.time() - t0)
    return out


if __name__ == "__main__":
    path = "results_counterparty.json"
    for sd in [int(s) for s in sys.argv[1:]]:
        o = run(sd)
        res = json.load(open(path)) if os.path.exists(path) else []
        res = [r for r in res if r["seed"] != sd] + [o]
        json.dump(res, open(path, "w"), ensure_ascii=False, indent=1, default=float)
        print(json.dumps(o, ensure_ascii=False, default=float), flush=True)

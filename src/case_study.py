"""사례 연구: 가상의 인물 한 명의 2년을 두 모델에 통과시킨다.

인물(가상): 박순자, 76세, 혼자 거주. 국민연금 월 90만 원. 지인 3명에게 가끔 송금.
2년(2025-01 ~ 2026-12) 동안 일어나는 일 — 모델은 이 표를 모른다.

  2025-05-31  경조사 송금 30만 원, 처음 보는 계좌            (정상)
  2025-07-20  집수리 업체에 120만 원, 처음 보는 사업자 계좌   (정상)
  2025-08 무렵 부고 계좌에 10만 원, 며칠 새 여러 명이 보낸 계좌 (정상)
  2025-09-03  딸 사칭 메신저피싱: 새 계좌로 280만 + 150만 원  (사기 1, 다른 피해자도 쓴 대포통장)
  2025-10-28  판단력 저하 시작, 약 13개월에 걸쳐 진행
  2025-11-27  자녀에게 300만 원 송금                          (정상)
  2026-02-05  병원비 카드 180만 원                            (정상)
  2026-05-16  대면편취: 이틀 동안 현금 4번 인출, 275만 원     (사기 2)
  2026-08-04~ 지인의 착취: 매주 20~30만 원씩 10번, 아는 계좌  (사기 3)

모델은 이 인물이 없는 4,000가구로 만든다. 3,200가구로 학습하고, 800가구로 알림 컷을 정한다
(가구당 월 0.1건, 스태킹 70% + IF 30%). 판단력 저하 컷은 정상 가구의 5%만 2년 중 한 번 알림을 받게 맞춘다.

v7부터: 상대 계좌 구조(대포통장·사업자·경조사 계좌)를 입힌 데이터로, 상대 계좌 피처를 넣은 모델과 넣지 않은 모델을
같은 조건에서 비교한다. 개인 프로필 카드와 송금 상대 판단 문구도 함께 만든다.
"""
import numpy as np, pandas as pd, json, time
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from gen_data_v5 import generate_decline
from features_v4 import build_features_v4, FEATURES_V4
from decline_features import monthly_raw, change_features
from counterparty import add_counterparties, add_cp_features, CP, verdict, others_new
from profile import build_profile, drift, to_markdown

PID, START, EVAL0 = 4000, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")
DAYS, ONSET, RAMP = 730, 300, 400
FSETS = {"v4": FEATURES_V4, "v7": FEATURES_V4 + CP}


def persona(seed=7, mule="F9001", biz="BIZ0", evt=None):
    rng = np.random.default_rng(seed)
    rows = []
    add = lambda t, typ, amt, payee, fraud=0, ep=-1, note="": rows.append((PID, t, typ, float(round(amt, -2)), payee, fraud, ep, note))
    sev = lambda d: float(np.clip((d - ONSET) / RAMP, 0, 1))
    known = [f"P{PID}_k{i}" for i in range(3)]
    bill_day = [8, 15, 22]

    def ts(day, hour=None):
        day = int(min(day, DAYS - 1))
        if hour is None:
            hour = rng.integers(0, 6) if rng.random() < 0.02 + 0.13 * sev(day) else int(np.clip(rng.normal(13, 3), 6, 21))
        return START + pd.Timedelta(days=day, hours=int(hour), minutes=int(rng.integers(0, 60)))

    for m in range(DAYS // 30):
        d0 = m * 30
        add(ts(d0 + 24), "pension_in", 900_000, "NPS")
        for b in range(3):
            d = d0 + bill_day[b]; s = sev(d)
            if rng.random() < 0.03 + 0.32 * s: continue
            d += int(rng.integers(-2, 3)) + int(rng.integers(0, int(15 * s) + 1))
            amt = 60_000 * rng.uniform(0.6, 1.4)
            if d < DAYS:
                add(ts(d), "bill", amt, f"BILL{b}")
                if rng.random() < 0.01 + 0.24 * s: add(ts(d + int(rng.integers(0, 4))), "bill", amt, f"BILL{b}")
        s = sev(d0 + 15)
        for _ in range(rng.poisson(2.5 * (1 + 1.5 * s))):
            add(ts(d0 + int(rng.integers(0, 30))), "atm", rng.lognormal(np.log(200_000) - 0.6 * s, 0.4), "ATM")
        for _ in range(rng.poisson(1.5)):
            add(ts(d0 + int(rng.integers(0, 30))), "transfer_out", rng.lognormal(np.log(150_000), 0.6), rng.choice(known))
        for _ in range(rng.poisson(3 * s)):
            add(ts(d0 + int(rng.integers(0, 30))), "card", rng.uniform(50_000, 200_000), "HOMESHOP")
    for d in range(DAYS):
        s = sev(d)
        for _ in range(rng.poisson(0.5)):
            t = ts(d); mer = f"M{rng.integers(0, 300)}"; amt = rng.lognormal(np.log(25_000), 0.8)
            add(t, "card", amt, mer)
            if rng.random() < 0.005 + 0.195 * s: add(t + pd.Timedelta(minutes=int(rng.integers(10, 180))), "card", amt, mer)

    # 정해 둔 사건
    add(ts(150, 11), "transfer_out", 300_000, f"P{PID}_g150", note="경조사 송금 (정상)")
    t0 = ts(245, 14)
    add(t0, "transfer_out", 2_800_000, mule, 1, 9001, "사기 1: 딸 사칭 메신저피싱 ①")
    add(t0 + pd.Timedelta(minutes=40), "transfer_out", 1_500_000, mule, 1, 9001, "사기 1: 딸 사칭 메신저피싱 ②")
    add(ts(200, 11), "transfer_out", 1_200_000, biz, note="집수리 업체 (정상)")
    if evt is not None:
        add(evt[1], "transfer_out", 100_000, evt[0], note="부고 계좌 (정상)")
    add(ts(330, 10), "transfer_out", 3_000_000, known[0], note="자녀에게 큰 송금 (정상)")
    add(ts(400, 15), "card", 1_800_000, "HOSP", note="병원비 (정상)")
    for k, (dd, hh, a) in enumerate([(500, 10, 700_000), (500, 15, 650_000), (501, 9, 800_000), (501, 14, 600_000)]):
        add(ts(dd, hh), "atm", a, "ATM", 1, 9002, f"사기 2: 대면편취 현금 인출 {k + 1}/4")
    for k in range(10):
        add(ts(580 + 7 * k, 12), "transfer_out", rng.lognormal(np.log(250_000), 0.2), known[2], 1, 9003, f"사기 3: 지인의 착취 {k + 1}/10")
    df = pd.DataFrame(rows, columns=["pid", "ts", "type", "amount", "payee", "is_fraud", "episode", "note"])
    return df.sort_values("ts").reset_index(drop=True)


def reasons(r):
    out = []
    typ = {"transfer_out": "송금", "atm": "현금 인출", "card": "카드 결제"}.get(r.type, r.type)
    if r.loan_48h: out.append("48시간 안에 대출 입금이 있었음")
    if r.new_payee: out.append("처음 보는 계좌로 송금")
    if r.type == "atm" and r.atm_n_72h >= 3: out.append(f"최근 3일 현금 인출 {int(r.atm_n_72h)}회")
    if r.z_shrink > 2.5: out.append(f"평소 {typ} 금액보다 크게 높음")
    if r.out_24h_ratio > 15: out.append(f"하루 출금이 평소의 약 {r.out_24h_ratio:.0f}배")
    if r.type == "transfer_out" and r.payee_trend > 1 and not r.new_payee: out.append("이 계좌로 보내는 돈이 최근 4주 크게 늘어남")
    if r.type == "transfer_out" and r.cp_age < 45 and others_new(r) >= 1 and np.expm1(r.cp_avg_in_30d) >= 500_000:
        out.insert(0, f"생긴 지 {int(r.cp_age)}일 된 계좌에 다른 고객 {others_new(r)}명도 최근 처음 큰돈을 보냄")
    if r.out_72h_ratio > 20 and len(out) < 2: out.append(f"3일 출금이 평소 하루의 약 {r.out_72h_ratio:.0f}배")
    return " · ".join(out[:2]) or "여러 신호가 조금씩 평소와 다름"


def fit_iso(X, seed):
    return IsolationForest(n_estimators=200, random_state=seed).fit(X.sample(min(200_000, len(X)), random_state=seed))


if __name__ == "__main__":
    t0 = time.time()
    pop, eps, dec, lc = generate_decline(seed=42)
    pop = add_counterparties(pop, eps, 42)
    # 인물의 사기 1은 다른 피해자가 며칠 전에 이미 당한 대포통장으로 간다. 부고 계좌는 실제로 여러 명이 보낸 계좌를 고른다.
    D1 = START + pd.Timedelta(days=245)
    mu = pop[pop.payee.str.startswith("MULE")].groupby("payee").ts.min()
    lag = (D1 - mu).dt.days
    mule = lag[(lag >= 1) & (lag <= 20)].idxmin() if ((lag >= 1) & (lag <= 20)).any() else lag.abs().idxmin()
    ev_first = pop[pop.payee.str.startswith("EVT")].groupby("payee").ts.min()
    e = (ev_first - (START + pd.Timedelta(days=225))).abs().idxmin()
    me = persona(mule=mule, evt=(e, ev_first[e] + pd.Timedelta(days=2, hours=3)))
    print("mule", mule, "first used", mu[mule], "| evt", e, flush=True)
    df = pd.concat([pop.assign(note=""), me], ignore_index=True)
    feat = add_cp_features(build_features_v4(df.drop(columns="note")))
    feat = feat.merge(df[["pid", "ts", "type", "amount", "payee", "note"]].drop_duplicates(["pid", "ts", "type", "amount", "payee"]),
                      on=["pid", "ts", "type", "amount", "payee"], how="left")
    ev = feat[feat.ts >= EVAL0].reset_index(drop=True)
    print("features", round(time.time() - t0), flush=True)

    rng = np.random.default_rng(0)
    hh = rng.permutation(4000); cal_h = set(hh[:800])
    tr = ev[(ev.pid != PID) & ~ev.pid.isin(cal_h)]; cal = ev[ev.pid.isin(cal_h)]; mine = ev[ev.pid == PID].copy()
    months = (ev.ts.max() - EVAL0).days / 30
    K = int(0.1 * 800 * months); k_iso = int(K * 0.3)
    meta = dict(K=K, mule=mule, mule_first=str(mu[mule]), onset_month=ONSET / 30, n_tx=int(len(mine)))
    for tag, F in FSETS.items():
        s_tr = np.full(len(tr), np.nan)
        for j, (a, b) in enumerate(GroupKFold(3).split(tr, groups=tr.pid)):
            s_tr[b] = -fit_iso(tr.iloc[a][F], 10 + j).score_samples(tr.iloc[b][F])
        iso = fit_iso(tr[F], 0)
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0).fit(tr[F].assign(s_iso=s_tr), tr.is_fraud)
        def score(X):
            si = -iso.score_samples(X[F]); return si, hgb.predict_proba(X[F].assign(s_iso=si))[:, 1]
        ci, cs = score(cal)
        cut_s = np.sort(cs)[::-1][K - k_iso - 1]; cut_i = np.sort(ci)[::-1][k_iso - 1]
        si, ss = score(mine)
        mine[f"s_iso_{tag}"], mine[f"s_stack_{tag}"] = si, ss
        mine[f"alert_{tag}"] = (ss >= cut_s) | (si >= cut_i)
        al = (cs >= cut_s) | (ci >= cut_i)
        ep_cal = cal.assign(alert=al)[cal.is_fraud == 1].groupby("episode").alert.max()
        meta[tag] = dict(cut_stack=float(cut_s), cut_iso=float(cut_i), cal_recall=float(ep_cal.mean()), cal_eps=int(len(ep_cal)), cal_alerts=int(al.sum()))
        print(tag, round(time.time() - t0), meta[tag], flush=True)
    mine["alert"] = mine.alert_v7
    mine["verdict"] = [verdict(r) for r in mine.itertuples()]
    mine["reason"] = [reasons(r) if r.alert else "" for r in mine.itertuples()]

    # 프로필 카드 (첫 사기 직전까지의 '평소'), 변화 표 (대면편취 직전)
    allmine = feat[feat.pid == PID].copy()
    allmine["verdict"] = [verdict(r) for r in allmine.itertuples()]
    prof = build_profile(allmine, "2025-09-01", "박순자(가상)", verdict_fn=lambda r: r.verdict)
    dr = drift(allmine, "2026-05-01")
    open("case_profile.md", "w").write(to_markdown(prof) + "\n\n**최근 3개월 변화 (2026-05-01 기준)**\n\n" + dr.to_markdown(index=False) + "\n")

    raw, d2 = monthly_raw(df.drop(columns="note"), 24)
    X = change_features(raw, d2, range(6, 24)).sort_values(["pid", "c"]).reset_index(drop=True)
    bill = X[["chg_bill_missed", "chg_bill_dup"]].clip(0, 5).sum(axis=1) + (X.bill_jitter / 1.5).clip(0, 5)
    X["score"] = X[["chg_card_dup", "chg_atm_n", "chg_night", "chg_top_mer"]].clip(0, 5).sum(axis=1) + (-X.chg_atm_amt).clip(0, 5) \
        + (X.psi_hour * 5).clip(0, 5) + bill * (1 - X.gap_recent)
    neg = ~X.pid.isin(dec.pid) & (X.pid != PID)
    meta["cut_decline"] = float(np.quantile(X[neg].groupby("pid").score.max(), 0.95))
    X[X.pid == PID].assign(cut=meta["cut_decline"]).to_csv("case_decline.csv", index=False)

    cols = ["ts", "type", "amount", "payee", "is_fraud", "episode", "note", "s_stack_v4", "s_iso_v4", "alert_v4", "s_stack_v7", "s_iso_v7", "alert_v7",
            "alert", "reason", "verdict", "z_shrink", "new_payee", "out_24h_ratio", "atm_n_72h", "payee_trend"] + CP
    mine[cols].to_csv("case_tx.csv", index=False)
    json.dump(meta, open("case_meta.json", "w"), indent=1, ensure_ascii=False)
    print("done", round(time.time() - t0))

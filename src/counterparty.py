"""상대 계좌 판단 (v7).

지금까지는 '이 사람이 이 계좌로 보낸 적이 있나'만 봤다. 실제 은행은 그 계좌로 '다른 고객들이' 어떻게 보내는지도 안다.
사기에 쓰이는 계좌(대포통장)는 생긴 지 얼마 안 됐고, 짧은 기간에 여러 사람이 처음으로 큰돈을 보낸다.

1) add_counterparties : 가상 데이터에 현실적인 상대 계좌 구조를 입힌다
   - 대포통장 MULE : 사칭·대출사기 송금의 수취 계좌. 한 계좌를 3주 안에 피해자 1~5명이 같이 쓴다
   - 사업자 BIZ   : 이사·수리 같은 큰 송금의 절반은 여러 고객이 오래 이용해 온 80개 계좌로 간다
   - 경조사 EVT   : 며칠 사이 5~12명이 처음으로 5~30만 원을 보내는 계좌 (대포통장과 겉모습이 닮은 정상 계좌)
   주변인 착취의 수취 계좌는 그대로 둔다(개인 한 명, 다른 고객과 무관).

2) add_cp_features : 송금 거래마다 상대 계좌 피처를 만든다. 모두 그 거래 이전 정보만 쓴다
   - cp_age            : 이 계좌가 은행 데이터에 처음 나타난 뒤 지난 일수
   - cp_senders_total  : 지금까지 이 계좌로 보낸 적 있는 고객 수
   - cp_new_senders_30d: 최근 30일에 이 계좌로 '처음' 보낸 고객 수
   - cp_tx_30d, cp_amt_30d : 최근 30일 이 계좌가 받은 건수, 금액(log)
   - cp_burst          : 최근 30일 신규 송금인 ÷ (누적 송금인 + 1). 갑자기 몰리는 정도
   - cp_avg_in_30d     : 최근 30일 한 건 평균 입금액(log)
   - payee_tenure      : 이 사람이 이 계좌와 거래해 온 일수
   - payee_amt_ratio   : 이번 금액 ÷ 이 사람이 이 계좌로 보내 온 평균 금액 (처음이면 1)

3) verdict : 사람이 읽는 상대 판단 문구
"""
import numpy as np
import pandas as pd
from features import _window

CP = ["cp_age", "cp_senders_total", "cp_new_senders_30d", "cp_tx_30d", "cp_amt_30d", "cp_burst", "cp_avg_in_30d",
      "payee_tenure", "payee_amt_ratio"]


def add_counterparties(df, eps, seed=0, n_biz=80, events_per_year=60):
    rng = np.random.default_rng(seed + 1000)
    df = df.copy()
    sc = df.episode.map(eps.set_index("episode").scenario)
    # 1) 대포통장: 사칭·대출사기 송금
    m = (df.type == "transfer_out") & (df.is_fraud == 1) & sc.isin(["impersonation", "loan_scam"])
    first = df[m].groupby("episode").ts.min().sort_values()
    mule, k, left, t_start = {}, 0, 0, None
    for ep, t in first.items():
        if left == 0 or (t - t_start).days > 21:
            k += 1; left = 1 + rng.poisson(2); t_start = t
        mule[ep] = f"MULE{k}"; left -= 1
    df.loc[m, "payee"] = df.loc[m, "episode"].map(mule)
    # 2) 사업자 계좌: 큰 새 송금의 절반
    big = df.payee.str.match(r"P\d+_n\d+") & (df.is_fraud == 0)
    pick = big & (rng.random(len(df)) < 0.5)
    df.loc[pick, "payee"] = [f"BIZ{j}" for j in rng.integers(0, n_biz, pick.sum())]
    # 3) 경조사 계좌: 며칠 사이 여러 사람이 처음으로 소액
    t0, t1 = df.ts.min(), df.ts.max()
    days = (t1 - t0).days
    pids = df.pid.unique()
    rows = []
    for e in range(int(events_per_year * days / 365)):
        d = rng.integers(20, days - 5)
        for p in rng.choice(pids, rng.integers(5, 13), replace=False):
            t = t0.normalize() + pd.Timedelta(days=int(d + rng.integers(0, 4)), hours=int(rng.integers(8, 21)), minutes=int(rng.integers(0, 60)))
            rows.append((p, t, "transfer_out", float(rng.choice([50_000, 100_000, 100_000, 200_000, 300_000])), f"EVT{e}", 0, -1))
    add = pd.DataFrame(rows, columns=["pid", "ts", "type", "amount", "payee", "is_fraud", "episode"])
    return pd.concat([df, add], ignore_index=True).sort_values(["pid", "ts"]).reset_index(drop=True)


def add_cp_features(f):
    """f: build_features_v4 결과 (new_payee 등 포함)."""
    f = f.copy()
    tr = (f["type"] == "transfer_out").to_numpy()
    sec = ((f["ts"] - pd.Timestamp("2025-01-01")).dt.total_seconds()).to_numpy().astype(np.int64)
    pk = pd.factorize(f["payee"])[0].astype(np.int64)
    first = f["new_payee"].to_numpy().astype(float)
    amt = f["amount"].to_numpy() * tr
    W, ALL = 30 * 86400, 20 * 365 * 86400
    f["cp_new_senders_30d"] = _window(pk, sec, first, W) * tr
    f["cp_tx_30d"] = _window(pk, sec, tr.astype(float), W) * tr
    a30 = _window(pk, sec, amt, W) * tr
    f["cp_amt_30d"] = np.log1p(a30)
    f["cp_avg_in_30d"] = np.log1p(a30 / np.maximum(f["cp_tx_30d"], 1))
    f["cp_senders_total"] = _window(pk, sec, first, ALL) * tr
    f["cp_burst"] = f["cp_new_senders_30d"] / (f["cp_senders_total"] + 1)
    seen = f[tr].groupby("payee").ts.min()
    f["cp_age"] = ((f["ts"] - f["payee"].map(seen)).dt.total_seconds() / 86400).where(tr, 0).fillna(0)
    mine = f.groupby(["pid", "payee"]).ts.transform("min")
    f["payee_tenure"] = ((f["ts"] - mine).dt.total_seconds() / 86400) * tr
    g = f.groupby(["pid", "payee"])["amount"]
    n = g.cumcount()
    past_mean = (g.cumsum() - f["amount"]) / n.where(n > 0)
    f["payee_amt_ratio"] = (f["amount"] / past_mean).where(tr & (n > 0), 1.0).fillna(1.0)
    return f


def others_new(r):
    """최근 30일 이 계좌로 처음 보낸 '다른' 고객 수. 본인이 30일 안에 처음 보낸 경우 본인 몫 1명을 뺀다."""
    own = 0 if r.new_payee else int(r.payee_tenure < 30)
    return max(int(r.cp_new_senders_30d) - own, 0)


def verdict(r):
    """송금 한 건의 상대 계좌를 한 줄로 판단한다. r: 피처가 들어 있는 행."""
    if r.type != "transfer_out":
        return ""
    avg_in = np.expm1(r.cp_avg_in_30d)
    if not r.new_payee and r.payee_tenure >= 90:
        s = f"오래 거래해 온 상대 ({int(r.payee_tenure // 30)}개월째)"
        if r.payee_amt_ratio >= 3: s += f", 다만 이번 금액이 평소 이 상대에게 보내던 금액의 {r.payee_amt_ratio:.0f}배"
        elif r.payee_trend > 1: s += ", 다만 최근 4주 보내는 돈이 크게 늘었음"
        return s
    if r.cp_senders_total >= 5 and r.cp_age >= 180 and r.cp_burst < 0.5:
        return f"여러 고객이 오래 이용해 온 계좌 (누적 {int(r.cp_senders_total)}명, 사업자·단체로 추정)"
    others = others_new(r)
    if r.cp_age < 45 and others >= 1 and avg_in >= 500_000:
        return f"위험: 생긴 지 {int(r.cp_age)}일 된 계좌에 최근 다른 고객 {others}명도 처음으로 큰돈을 보냄 (건당 평균 {avg_in / 10_000:.0f}만 원)"
    if r.cp_age < 45 and others >= 2:
        return f"최근 생긴 계좌에 {others}명이 소액을 보냄 (경조사·모임으로 추정)"
    if r.new_payee:
        return "처음 보는 계좌, 다른 고객의 이용 기록도 거의 없음"
    return f"가끔 거래하는 상대 ({int(r.payee_tenure)}일째)"

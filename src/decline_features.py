"""가구-월 단위 '장기 변화' 피처 (판단력 저하 감지용).

거래 하나가 아니라 '이 사람의 최근 3개월'을 '이 사람의 첫 6개월(기준선)'과 비교한다.
피처는 모두 그 시점까지의 정보만 쓴다.

월별 원재료 (가구 × 월)
- bill_dev    : 공과금 납부일이 그 공과금의 평소 납부일(기준선 중앙값)에서 벗어난 평균 일수
- bill_missed : 그달 내지 않은 공과금 종류 수 (0~3)
- bill_dup    : 그달 같은 공과금을 두 번 이상 낸 종류 수
- card_dup    : 같은 가맹점·같은 금액 카드 결제를 3시간 안에 또 한 건수
- atm_n, atm_amt : ATM 인출 횟수, 한 번 평균 금액(log)
- night       : 0~6시 거래 비중
- top_mer     : 카드 결제액 중 가장 많이 쓴 가맹점 비중
- card_n      : 카드 결제 건수
- act         : 그달 전체 거래 건수 (입원·여행 같은 활동 공백을 알아보기 위함)

변화 피처 (체크포인트 월 c 기준)
- 최근 3개월(c-2 ~ c) 평균을 기준선 6개월 평균·표준편차로 표준화한 값: chg_<원재료>
- bill_jitter : 최근 3개월 공과금 납부일의 들쭉날쭉함(공과금별 표준편차 평균) − 기준선의 들쭉날쭉함.
                납부일이 '한 날짜로 옮겨간 것'(자동이체 변경)과 '흐트러진 것'(판단력 저하)을 구분한다
- gap_recent  : 최근 4개월 중 거래 건수가 평소의 30% 아래인 달이 있었는지 (0/1)
- 개인 PSI: 거래 시각 분포(6구간)와 카드 금액 분포(5구간)를 최근 90일 vs 기준선으로 비교
"""
import numpy as np
import pandas as pd

BASE_M = 6
RAW = ["bill_dev", "bill_missed", "bill_dup", "card_dup", "atm_n", "atm_amt", "night", "top_mer", "card_n", "act"]
CHG = [f"chg_{r}" for r in RAW] + ["bill_jitter", "gap_recent", "psi_hour", "psi_card_amt"]
HOUR_BINS = [-1, 5, 8, 11, 14, 17, 24]


def _psi(p, q):
    p, q = np.clip(p, 1e-3, None), np.clip(q, 1e-3, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def monthly_raw(df, n_months):
    start = pd.Timestamp("2025-01-01")
    df = df.copy()
    df["day"] = ((df.ts - start).dt.total_seconds() // 86400).astype(int)
    df["m"] = df.day // 30
    df = df[df.m < n_months]
    df["dom"] = df.day % 30
    df["hour"] = df.ts.dt.hour
    pids = df.pid.unique()
    idx = pd.MultiIndex.from_product([pids, range(n_months)], names=["pid", "m"])
    out = pd.DataFrame(index=idx)

    b = df[df.type == "bill"]
    base_dom = b[b.m < BASE_M].groupby(["pid", "payee"]).dom.median().rename("base_dom")
    b = b.join(base_dom, on=["pid", "payee"])
    b["dev"] = (b.dom - b.base_dom).abs()
    b["dev"] = np.minimum(b.dev, 30 - b.dev)  # 월 경계를 넘는 경우
    out["bill_dev"] = b.groupby(["pid", "m"]).dev.mean()
    paid = b.groupby(["pid", "m"]).payee.nunique()
    out["bill_missed"] = (3 - paid.reindex(idx).fillna(0)).clip(lower=0)
    cnt = b.groupby(["pid", "m", "payee"]).size()
    out["bill_dup"] = (cnt > 1).groupby(["pid", "m"]).sum()

    c = df[df.type == "card"].sort_values(["pid", "ts"])
    same = (c.pid.eq(c.pid.shift())) & (c.payee.eq(c.payee.shift())) & (c.amount.eq(c.amount.shift())) & \
           ((c.ts - c.ts.shift()).dt.total_seconds() < 3 * 3600)
    c = c.assign(dup=same.astype(int))
    out["card_dup"] = c.groupby(["pid", "m"]).dup.sum()
    out["card_n"] = c.groupby(["pid", "m"]).size()
    ms = c.groupby(["pid", "m", "payee"]).amount.sum()
    out["top_mer"] = ms.groupby(["pid", "m"]).max() / ms.groupby(["pid", "m"]).sum()

    a = df[df.type == "atm"]
    out["atm_n"] = a.groupby(["pid", "m"]).size()
    out["atm_amt"] = np.log1p(a.groupby(["pid", "m"]).amount.mean())
    out["night"] = df.groupby(["pid", "m"]).hour.apply(lambda h: (h < 6).mean())
    out["act"] = df.groupby(["pid", "m"]).size()

    fill0 = ["bill_missed", "bill_dup", "card_dup", "card_n", "atm_n", "night", "act"]
    out[fill0] = out[fill0].fillna(0)
    out["bill_dev"] = out.bill_dev.fillna(out.groupby("pid").bill_dev.transform("median"))
    for col in ["top_mer", "atm_amt"]:
        out[col] = out[col].fillna(out.groupby("pid")[col].transform("median"))
    return out.fillna(0), df


def change_features(raw, df, checkpoints):
    """체크포인트 월 c마다 최근 3개월 vs 기준선 변화. 반환: (pid, c) 행."""
    base = raw[raw.index.get_level_values("m") < BASE_M].groupby("pid")
    mu, sd = base.mean(), base.std()
    floor = {"bill_dev": 1.0, "bill_missed": 0.3, "bill_dup": 0.3, "card_dup": 0.3, "atm_n": 0.7,
             "atm_amt": 0.15, "night": 0.02, "top_mer": 0.05, "card_n": 2.0, "act": 3.0}
    sd = sd.clip(lower=pd.Series(floor), axis=1)

    # 개인 PSI용 기준선 분포
    hb = df.assign(hb=pd.cut(df.hour, HOUR_BINS, labels=False))
    base_h = hb[hb.m < BASE_M].groupby(["pid", "hb"]).size().unstack(fill_value=0)
    cd = df[df.type == "card"].copy()
    cd["la"] = np.log1p(cd.amount)
    qs = cd[cd.m < BASE_M].groupby("pid").la.quantile([0.2, 0.4, 0.6, 0.8]).unstack()
    cd = cd.join(qs, on="pid")
    cd["ab"] = (cd.la > cd[0.2]).astype(int) + (cd.la > cd[0.4]) + (cd.la > cd[0.6]) + (cd.la > cd[0.8])
    base_a = cd[cd.m < BASE_M].groupby(["pid", "ab"]).size().unstack(fill_value=0)
    hcnt = hb.groupby(["pid", "m", "hb"]).size().unstack(fill_value=0)
    acnt = cd.groupby(["pid", "m", "ab"]).size().unstack(fill_value=0)

    bl = df[df.type == "bill"]
    base_jit = bl[bl.m < BASE_M].groupby(["pid", "payee"]).dom.std().groupby("pid").mean()
    act = raw["act"].unstack()
    gap = act.lt(0.3 * mu["act"], axis=0)

    rows = []
    for c in checkpoints:
        rec = raw[(raw.index.get_level_values("m") > c - 3) & (raw.index.get_level_values("m") <= c)].groupby("pid").mean()
        z = (rec - mu) / sd
        z.columns = [f"chg_{x}" for x in z.columns]
        hr = hcnt[(hcnt.index.get_level_values("m") > c - 3) & (hcnt.index.get_level_values("m") <= c)].groupby("pid").sum()
        ar = acnt[(acnt.index.get_level_values("m") > c - 3) & (acnt.index.get_level_values("m") <= c)].groupby("pid").sum()
        hr = hr.reindex(index=base_h.index, columns=base_h.columns, fill_value=0)
        ar = ar.reindex(index=base_a.index, columns=base_a.columns, fill_value=0)
        z["psi_hour"] = [_psi(base_h.loc[p].values, hr.loc[p].values) for p in base_h.index]
        z["psi_card_amt"] = pd.Series([_psi(base_a.loc[p].values, ar.loc[p].values) for p in base_a.index], index=base_a.index)
        w = bl[(bl.m > c - 3) & (bl.m <= c)]
        jit = w.groupby(["pid", "payee"]).dom.std().groupby("pid").mean()
        z["bill_jitter"] = (jit - base_jit).reindex(z.index).fillna(0)
        z["gap_recent"] = gap.loc[:, [m for m in gap.columns if c - 4 < m <= c]].any(axis=1).astype(int).reindex(z.index).fillna(0)
        z["c"] = c
        rows.append(z.reset_index())
    return pd.concat(rows, ignore_index=True).fillna(0)

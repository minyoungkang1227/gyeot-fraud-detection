"""거래 단위 피처 생성. 모든 피처는 '그 거래 이전' 정보만 사용한다(누수 방지)."""
import numpy as np
import pandas as pd

OUT_TYPES = ["card", "atm", "transfer_out", "bill"]


def _window(keys, ts, vals, window_sec):
    """같은 key 그룹 안에서, 현재 행 '이전' 행들 중 [t-window, t] 구간의 vals 합 (현재 행 제외)."""
    order = np.lexsort((np.arange(len(ts)), ts, keys))
    k = keys[order].astype(np.int64)
    t = ts[order]
    v = vals[order].astype(float)
    comp = k * 10**9 + t
    left = np.searchsorted(comp, comp - window_sec, side="left")
    cs = np.concatenate([[0.0], np.cumsum(v)])
    pos = np.arange(len(t))
    res = cs[pos] - cs[left]
    out = np.empty(len(t))
    out[order] = res
    return out


def build_features(df):
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True).copy()
    df["log_amt"] = np.log1p(df["amount"])
    df["is_out"] = df["type"].isin(OUT_TYPES).astype(int)
    df["out_amt"] = df["amount"] * df["is_out"]
    df["is_loan"] = (df["type"] == "loan_in").astype(int)
    df["hour"] = df["ts"].dt.hour
    df["night"] = (df["hour"] < 6).astype(int)

    # 1) 같은 유형 거래 대비 평소와 얼마나 다른가 (개인별 z-score)
    g = df.groupby(["pid", "type"])["log_amt"]
    df["past_mean"] = g.transform(lambda s: s.shift().expanding().mean())
    df["past_std"] = g.transform(lambda s: s.shift().expanding().std())
    df["past_n"] = df.groupby(["pid", "type"]).cumcount()
    df["z_type"] = (df["log_amt"] - df["past_mean"]) / df["past_std"].clip(lower=0.3)
    df["z_type"] = df["z_type"].where(df["past_n"] >= 3, 0.0).fillna(0.0)

    # 2) 처음 보는 수취인 / 최근 30일 그 수취인에게 보낸 횟수
    tr = df["type"] == "transfer_out"
    first_seen = df.groupby(["pid", "payee"]).cumcount() == 0
    df["new_payee"] = (first_seen & tr).astype(int)
    sec = ((df["ts"] - pd.Timestamp("2025-01-01")).dt.total_seconds()).to_numpy().astype(np.int64)
    pid = df["pid"].to_numpy()
    payee_key = df.groupby(["pid", "payee"]).ngroup().to_numpy()
    ones = np.ones(len(df))
    df["payee_cnt_30d"] = _window(payee_key, sec, ones, 30 * 86400) * tr

    # 3) 시간 창 집계: 직전 24시간 출금 건수·금액, 48시간 내 대출 입금, 30일 출금 합
    df["n_out_24h"] = _window(pid, sec, df["is_out"].to_numpy(), 86400)
    df["sum_out_24h"] = _window(pid, sec, df["out_amt"].to_numpy(), 86400) + df["out_amt"]
    df["loan_48h"] = (_window(pid, sec, df["is_loan"].to_numpy(), 2 * 86400) > 0).astype(int)
    df["sum_out_30d"] = _window(pid, sec, df["out_amt"].to_numpy(), 30 * 86400) + df["out_amt"]

    # 4) 수취인 추세 (v2): 이 수취인에게 최근 28일 보낸 돈이 그 이전 28일보다 얼마나 늘었나
    amt_tr = (df["amount"] * tr).to_numpy()
    s28 = _window(payee_key, sec, amt_tr, 28 * 86400)
    s56 = _window(payee_key, sec, amt_tr, 56 * 86400)
    df["payee_28d_amt"] = (s28 + amt_tr) * tr
    df["payee_trend"] = np.log1p((s28 + amt_tr) / (s56 - s28 + 100_000)) * tr
    df["payee_cnt_90d"] = _window(payee_key, sec, ones, 90 * 86400) * tr

    # 평소 하루 출금액 (지금까지 누적 출금 / 경과 일수)
    cum_out = df.groupby("pid")["out_amt"].cumsum() - df["out_amt"]
    days = (df["ts"] - df.groupby("pid")["ts"].transform("min")).dt.total_seconds() / 86400
    daily = (cum_out / days.clip(lower=1)).clip(lower=10_000)
    df["out_24h_ratio"] = df["sum_out_24h"] / daily
    df["out_30d_ratio"] = df["sum_out_30d"] / (daily * 30)
    df["payee_28d_ratio"] = df["payee_28d_amt"] / (daily * 28)   # 이 수취인 몫이 평소 한 달 지출의 몇 배인지

    for t in ["atm", "transfer_out", "card", "loan_in"]:
        df[f"t_{t}"] = (df["type"] == t).astype(int)
    return df


FEATURES_V1 = ["log_amt", "z_type", "new_payee", "payee_cnt_30d", "n_out_24h", "out_24h_ratio",
               "out_30d_ratio", "loan_48h", "t_atm", "t_transfer_out", "t_card", "t_loan_in"]
FEATURES = FEATURES_V1 + ["payee_trend", "payee_28d_ratio", "payee_cnt_90d"]

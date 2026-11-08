"""v4 피처: v3 + '최근 며칠을 묶어서 보는' 시간창 피처.

거래 한 건씩 보면 정상처럼 보이는 사기(대면편취: 현금 여러 번 인출, 쪼개기 송금)를
최근 72시간·7일 단위로 묶어 사건처럼 보게 한다. 모두 그 거래 시점까지의 정보만 쓴다.

- atm_n_72h      : 직전 72시간 ATM 인출 건수 (현재 포함)
- atm_72h_ratio  : 직전 72시간 ATM 인출 합계 / 평소 하루 출금액
- out_n_72h      : 직전 72시간 출금 건수 (현재 포함)
- out_72h_ratio  : 직전 72시간 출금 합계 / 평소 하루 출금액
- newpayee_n_7d  : 직전 7일 처음 보는 수취인 송금 건수 (현재 포함)
- atm_72h_vs_hist: 72시간 ATM 건수가 이 사람의 평소 72시간 ATM 건수보다 몇 배인지
"""
import numpy as np
import pandas as pd
from features import _window
from features_v3 import build_features_v3, FEATURES_V3


def add_window_features(df):
    sec = ((df["ts"] - pd.Timestamp("2025-01-01")).dt.total_seconds()).to_numpy().astype(np.int64)
    pid = df["pid"].to_numpy()
    is_atm = (df["type"] == "atm").to_numpy().astype(float)
    atm_amt = df["amount"].to_numpy() * is_atm
    is_out = df["is_out"].to_numpy().astype(float)
    out_amt = df["out_amt"].to_numpy().astype(float)
    newp = df["new_payee"].to_numpy().astype(float)

    W3 = 3 * 86400
    df["atm_n_72h"] = _window(pid, sec, is_atm, W3) + is_atm
    df["out_n_72h"] = _window(pid, sec, is_out, W3) + is_out
    atm_sum = _window(pid, sec, atm_amt, W3) + atm_amt
    out_sum = _window(pid, sec, out_amt, W3) + out_amt
    df["newpayee_n_7d"] = _window(pid, sec, newp, 7 * 86400) + newp

    # 평소 하루 출금액 (features.py와 같은 정의)
    cum_out = df.groupby("pid")["out_amt"].cumsum() - df["out_amt"]
    days = (df["ts"] - df.groupby("pid")["ts"].transform("min")).dt.total_seconds() / 86400
    daily = (cum_out / days.clip(lower=1)).clip(lower=10_000)
    df["atm_72h_ratio"] = atm_sum / daily
    df["out_72h_ratio"] = out_sum / daily
    # 평소 72시간당 ATM 건수 (지금까지 ATM 건수 / 경과 72시간 수), 최소 0.2
    cum_atm = df.groupby("pid")["t_atm"].cumsum() - df["t_atm"]
    base = (cum_atm / (days.clip(lower=3) / 3)).clip(lower=0.2)
    df["atm_72h_vs_hist"] = df["atm_n_72h"] / base
    return df


def build_features_v4(df_raw):
    return add_window_features(build_features_v3(df_raw))


WINDOW = ["atm_n_72h", "atm_72h_ratio", "out_n_72h", "out_72h_ratio", "newpayee_n_7d", "atm_72h_vs_hist"]
FEATURES_V4 = FEATURES_V3 + WINDOW

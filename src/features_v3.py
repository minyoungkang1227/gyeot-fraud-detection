"""v3 피처: v2 피처 + 기록 부족 문제 보완.

v2의 z_type은 같은 유형 과거 거래가 3건 미만이면 0으로 둔다.
그러면 '기록이 없어서 0'과 '평소와 같아서 0'이 구분되지 않는다.
v3는 두 가지를 더한다.

1) hist_n: 같은 유형 과거 거래 건수 (log1p). 모델이 '기록이 적다'는 사실 자체를 보게 한다.
2) z_shrink: 개인 평균·분산을 집단 평균·분산 쪽으로 당겨서(수축 추정) 계산한 z-score.
   mu_hat  = (n * mu_personal + K * mu_group) / (n + K)
   var_hat = ((n-1) * s2_personal + K * var_group) / (n - 1 + K)
   기록이 0건이면 집단 기준으로, 많을수록 개인 기준으로 계산된다. 기록 0건에도 값이 나온다.
   집단 평균·분산은 평가 시작 전 3개월(학습용 기간) 거래로만 계산해 누수를 막는다.
"""
import numpy as np
import pandas as pd
from features import build_features, FEATURES

K_SHRINK = 5
PRIOR_END = pd.Timestamp("2025-04-01")


def build_features_v3(df_raw, k=K_SHRINK, prior_end=PRIOR_END):
    df = build_features(df_raw)
    prior = df[df.ts < prior_end].groupby("type")["log_amt"].agg(["mean", "var"])
    mu_g = df["type"].map(prior["mean"])
    var_g = df["type"].map(prior["var"])

    n = df["past_n"]
    mu_p = df["past_mean"].fillna(0.0)
    s2_p = (df["past_std"] ** 2).fillna(0.0)
    mu_hat = (n * mu_p + k * mu_g) / (n + k)
    var_hat = ((n - 1).clip(lower=0) * s2_p + k * var_g) / ((n - 1).clip(lower=0) + k)
    df["z_shrink"] = (df["log_amt"] - mu_hat) / np.sqrt(var_hat).clip(lower=0.3)
    df["hist_n"] = np.log1p(n)
    return df


FEATURES_V3 = [f for f in FEATURES if f != "z_type"] + ["z_shrink", "hist_n"]

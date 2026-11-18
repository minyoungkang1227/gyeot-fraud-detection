"""누수 검증: '미래 거래를 지워도 과거 시점의 피처가 바뀌지 않는가'.

전체 데이터로 만든 피처와, 기준 시점 T 이후 거래를 지운 데이터로 만든 피처를 비교한다.
T 이전 행의 피처가 한 칸이라도 다르면 그 피처는 미래 정보를 쓰고 있다는 뜻이다.

실행: python tests/test_no_leakage.py   (약 1분)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd

from gen_data import generate
from features import build_features, FEATURES
from features_v3 import build_features_v3, FEATURES_V3
from features_v4 import build_features_v4, FEATURES_V4
from gen_data_v5 import generate_decline
from decline_features import monthly_raw, change_features, CHG
from counterparty import add_counterparties, add_cp_features, CP

KEY = ["pid", "ts", "type", "amount", "payee"]


def check_tx(build, feats, df, T, name):
    full = build(df)
    cut = build(df[df.ts < T])
    a = full[full.ts < T].sort_values(KEY).reset_index(drop=True)
    b = cut.sort_values(KEY).reset_index(drop=True)
    assert len(a) == len(b), name
    bad = [f for f in feats if not np.allclose(a[f].to_numpy(float), b[f].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)]
    print(f"[{'OK' if not bad else 'LEAK'}] {name}: 피처 {len(feats)}개, T 이전 {len(a):,}행" + (f" → 다른 피처 {bad}" if bad else ""))
    return bad


def check_decline(df, c, name):
    raw, d2 = monthly_raw(df, 24)
    full = change_features(raw, d2, [c]).set_index("pid")
    start = pd.Timestamp("2025-01-01")
    dcut = df[df.ts < start + pd.Timedelta(days=30 * (c + 1))]
    raw2, d3 = monthly_raw(dcut, c + 1)
    cut = change_features(raw2, d3, [c]).set_index("pid").reindex(full.index)
    bad = [f for f in CHG if not np.allclose(full[f].to_numpy(float), cut[f].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)]
    print(f"[{'OK' if not bad else 'LEAK'}] {name}: 변화 피처 {len(CHG)}개, 체크포인트 {c + 1}개월째, 가구 {len(full):,}" + (f" → 다른 피처 {bad}" if bad else ""))
    return bad


if __name__ == "__main__":
    df, eps = generate(n_people=300, seed=7)
    T = pd.Timestamp("2025-08-15")
    bad = []
    bad += check_tx(build_features, FEATURES, df, T, "거래 피처 v2")
    bad += check_tx(build_features_v3, FEATURES_V3, df, T, "거래 피처 v3")
    bad += check_tx(build_features_v4, FEATURES_V4, df, T, "거래 피처 v4")
    dcp = add_counterparties(df, eps, 7)
    bad += check_tx(lambda d: add_cp_features(build_features_v4(d)), CP, dcp, T, "상대 계좌 피처 v7")
    d5, *_ = generate_decline(n_people=300, seed=7)
    bad += check_decline(d5, 11, "가구-월 변화 피처 v5")
    print("\n결과:", "누수 없음" if not bad else f"누수 의심 피처 {sorted(set(bad))}")
    sys.exit(1 if bad else 0)

"""가상 고령자 거래 데이터 생성기 v4 — 드리프트 포함 (곁 GYEOT 이상거래 탐지용)

기본 생성기(gen_data.py)와 같고, 다음 두 가지 변화가 drift_day(기본 2026-01-01)부터 생긴다.
- 신종 수법(invest_scam, 투자 리딩방 사기): 처음 보는 계좌로 소액 송금 후 3~5일 간격으로 금액을 키워 5~9회 송금
- 신종 수법(giftcard_scam, 메신저피싱 상품권 결제): 몇 시간 안에 처음 보는 가맹점에서 카드로 30~80만 원 결제를 3~6회.
  기존 사기 4종은 카드를 쓰지 않으므로, 학습 라벨에 전혀 없던 형태다
- 정상 행동 변화(간편송금 확산): 처음 보는 계좌로 송금(중고거래·더치페이)이 월 평균 4건 생긴다
기간은 2025-01-01부터 days일(기본 546일, 2026-06 말까지).

- 개인마다 '평소 패턴'(연금 입금, 공과금, 카드, ATM 인출, 지인 송금)을 따로 갖는다.
- 정상이지만 튀는 거래(병원비, 경조사 송금, 자녀 증여)를 섞어 오탐 상황을 만든다.
- 일부 개인에게 사기/착취 에피소드를 주입하고 거래 단위 라벨(is_fraud)과 episode_id를 붙인다.
"""
import numpy as np
import pandas as pd

TYPES = ["pension_in", "bill", "card", "atm", "transfer_out", "loan_in"]


def generate_drift(n_people=4000, days=546, fraud_rate=0.06, new_rate=0.02, gift_rate=0.03, drift_day=365, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows = []
    ep_meta = []
    ep_id = 0

    def add(pid, t, typ, amt, payee, fraud=0, ep=-1):
        rows.append((pid, t, typ, float(round(amt, -2)), payee, fraud, ep))

    for pid in range(n_people):
        # 개인 특성: 소득·지출 규모
        wealth = rng.lognormal(0, 0.5)
        pension = 900_000 * wealth
        card_mu = np.log(25_000 * wealth)
        atm_mu = np.log(200_000 * wealth)
        tr_mu = np.log(150_000 * wealth)
        known = [f"P{pid}_k{i}" for i in range(rng.integers(2, 5))]
        active_h = rng.integers(8, 11)  # 주로 활동 시작 시각

        def ts(day, night=False):
            h = rng.integers(0, 6) if night else int(np.clip(rng.normal(active_h + 4, 3), 6, 21))
            return start + pd.Timedelta(days=int(day), hours=int(h), minutes=int(rng.integers(0, 60)))

        for m in range(days // 30):
            d0 = m * 30
            add(pid, ts(d0 + 24), "pension_in", pension, "NPS")
            for b in range(3):  # 공과금·통신비·보험료
                add(pid, ts(d0 + 10 + b * 5), "bill", 60_000 * wealth * rng.uniform(0.6, 1.4), f"BILL{b}")
            for _ in range(rng.poisson(2.5)):
                add(pid, ts(d0 + rng.integers(0, 30)), "atm", rng.lognormal(atm_mu, 0.4), "ATM")
            for _ in range(rng.poisson(1.5)):
                add(pid, ts(d0 + rng.integers(0, 30)), "transfer_out", rng.lognormal(tr_mu, 0.6), rng.choice(known))
        for d in range(days):
            for _ in range(rng.poisson(0.5)):
                add(pid, ts(d), "card", rng.lognormal(card_mu, 0.8), f"M{rng.integers(0, 300)}")

        # 정상 행동 변화: drift_day 이후 간편송금 확산 (처음 보는 계좌로 소액 송금)
        for d in range(drift_day, days):
            if rng.random() < 4 / 30:
                add(pid, ts(d), "transfer_out", rng.lognormal(np.log(150_000), 0.8), f"P{pid}_m{d}")

        # 정상이지만 튀는 거래 (오탐 유발): 사기와 겉모습이 비슷한 진짜 거래
        for _ in range(rng.poisson(3)):
            kind = rng.choice(["hospital", "gift", "family", "big_new", "holiday_cash"],
                              p=[0.2, 0.25, 0.2, 0.15, 0.2])
            d = int(rng.integers(20, days - 3))
            if kind == "hospital":
                add(pid, ts(d), "card", rng.lognormal(np.log(1_500_000 * wealth), 0.5), "HOSP")
            elif kind == "gift":  # 경조사: 처음 보는 계좌로 송금
                add(pid, ts(d), "transfer_out", rng.choice([100_000, 200_000, 300_000, 500_000]), f"P{pid}_g{d}")
            elif kind == "family":  # 자녀에게 큰 돈
                add(pid, ts(d), "transfer_out", rng.lognormal(np.log(3_000_000 * wealth), 0.6), known[0])
            elif kind == "big_new":  # 이사·수리·중고거래: 처음 보는 계좌로 큰 돈
                add(pid, ts(d), "transfer_out", rng.lognormal(np.log(2_000_000 * wealth), 0.7), f"P{pid}_n{d}")
            else:  # 명절·경조사 현금: 이틀 동안 여러 번 인출
                for k in range(rng.integers(2, 4)):
                    add(pid, ts(d + k // 2), "atm", rng.lognormal(np.log(600_000 * wealth), 0.4), "ATM")
        if rng.random() < 0.02:  # 정상 대출 후 자녀에게 이체
            d = int(rng.integers(30, days - 2))
            loan = rng.choice([5_000_000, 10_000_000, 30_000_000])
            add(pid, ts(d), "loan_in", loan, "LOAN")
            add(pid, ts(d + 1), "transfer_out", loan * rng.uniform(0.5, 1.0), known[0])

        # 사기/착취 에피소드 주입 (4~12개월 사이)
        if rng.random() < fraud_rate:
            sc = rng.choice(["impersonation", "cash_pickup", "loan_scam", "exploitation"],
                            p=[0.35, 0.25, 0.2, 0.2])
            d = int(rng.integers(100, days - 40))
            ep_id += 1
            total = 0.0
            if sc == "impersonation":  # 가족·기관 사칭: 새 계좌로 몇 시간 안에 연속 송금 (금액 편차 큼)
                t0 = ts(d)
                for k in range(rng.integers(1, 4)):
                    a = rng.lognormal(np.log(2_500_000 * wealth), 0.9)
                    add(pid, t0 + pd.Timedelta(minutes=int(40 * k)), "transfer_out", a, f"F{ep_id}", 1, ep_id)
                    total += a
            elif sc == "cash_pickup":  # 대면편취: 1~3일 사이 현금 여러 번 인출
                for k in range(rng.integers(2, 6)):
                    a = rng.lognormal(np.log(700_000 * wealth), 0.5)
                    add(pid, ts(d + k // 2), "atm", a, "ATM", 1, ep_id)
                    total += a
            elif sc == "loan_scam":  # 비대면 대출 실행 → 하루 안에 새 계좌 이체 또는 현금 인출
                loan = rng.choice([5_000_000, 10_000_000, 20_000_000])
                t0 = ts(d)
                add(pid, t0, "loan_in", loan, "LOAN", 1, ep_id)
                if rng.random() < 0.5:
                    a = loan * rng.uniform(0.9, 1.0)
                    add(pid, t0 + pd.Timedelta(hours=int(rng.integers(1, 20))), "transfer_out", a, f"F{ep_id}", 1, ep_id)
                    total += a
                else:
                    for k in range(rng.integers(2, 5)):
                        a = rng.lognormal(np.log(1_000_000), 0.3)
                        add(pid, t0 + pd.Timedelta(hours=int(2 + 3 * k)), "atm", a, "ATM", 1, ep_id)
                        total += a
            else:  # 주변인 착취: 매주 중간 금액, 수개월. 절반은 이미 알던 수취인
                payee = f"F{ep_id}" if rng.random() < 0.5 else known[-1]
                for k in range(rng.integers(6, 14)):
                    a = rng.lognormal(np.log(250_000 * wealth), 0.4)
                    add(pid, ts(min(d + 7 * k, days - 1)), "transfer_out", a, payee, 1, ep_id)
                    total += a
            ep_meta.append((ep_id, pid, sc, total))

        # 신종 수법: drift_day 이후에만 등장
        if rng.random() < new_rate:
            d = int(rng.integers(drift_day + 5, days - 45))
            ep_id += 1
            total = 0.0
            payee = f"F{ep_id}"
            a = 300_000 * wealth * rng.uniform(0.7, 1.3)
            for k in range(rng.integers(5, 10)):
                if k == 4 and rng.random() < 0.5:  # 중간에 '수익금 출금 수수료' 명목으로 다른 계좌
                    payee = f"F{ep_id}b"
                add(pid, ts(min(d, days - 1)), "transfer_out", a, payee, 1, ep_id)
                total += a
                a *= rng.uniform(1.3, 1.7)
                d += int(rng.integers(3, 6))
            ep_meta.append((ep_id, pid, "invest_scam", total))

        # 신종 수법 2: 메신저피싱 상품권 카드 결제 (drift_day 이후)
        if rng.random() < gift_rate:
            d = int(rng.integers(drift_day + 5, days - 5))
            ep_id += 1
            total = 0.0
            t0 = ts(d)
            for k in range(rng.integers(3, 7)):
                a = rng.uniform(300_000, 800_000)
                add(pid, t0 + pd.Timedelta(minutes=int(25 * k + rng.integers(0, 15))), "card", a, f"GIFT{ep_id}_{k}", 1, ep_id)
                total += a
            ep_meta.append((ep_id, pid, "giftcard_scam", total))

    df = pd.DataFrame(rows, columns=["pid", "ts", "type", "amount", "payee", "is_fraud", "episode"])
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    eps = pd.DataFrame(ep_meta, columns=["episode", "pid", "scenario", "loss"])
    return df, eps


if __name__ == "__main__":
    df, eps = generate_drift()
    print(df.shape, df.is_fraud.mean())
    print(df.type.value_counts())
    print(eps.scenario.value_counts())

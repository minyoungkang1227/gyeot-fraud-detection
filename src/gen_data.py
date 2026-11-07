"""가상 고령자 거래 데이터 생성기 (곁 GYEOT 이상거래 탐지용)

- 개인마다 '평소 패턴'(연금 입금, 공과금, 카드, ATM 인출, 지인 송금)을 따로 갖는다.
- 정상이지만 튀는 거래(병원비, 경조사 송금, 자녀 증여)를 섞어 오탐 상황을 만든다.
- 일부 개인에게 사기/착취 에피소드를 주입하고 거래 단위 라벨(is_fraud)과 episode_id를 붙인다.
"""
import numpy as np
import pandas as pd

TYPES = ["pension_in", "bill", "card", "atm", "transfer_out", "loan_in"]


def generate(n_people=1500, days=365, fraud_rate=0.06, seed=42):
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

        for m in range(12):
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

    df = pd.DataFrame(rows, columns=["pid", "ts", "type", "amount", "payee", "is_fraud", "episode"])
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    eps = pd.DataFrame(ep_meta, columns=["episode", "pid", "scenario", "loss"])
    return df, eps


if __name__ == "__main__":
    df, eps = generate()
    print(df.shape, df.is_fraud.mean())
    print(df.type.value_counts())
    print(eps.scenario.value_counts())

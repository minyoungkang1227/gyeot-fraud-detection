"""가상 고령자 거래 데이터 생성기 v5 — 판단력 저하(인지 저하) 시나리오 (곁 GYEOT)

기본 생성기(gen_data.py)의 평소 패턴·튀는 정상 거래·사기 4종을 그대로 두고, 기간을 2년으로 늘린 뒤 두 가지를 더한다.

1) 판단력 저하 가구 (decline_rate, 기본 5%)
   - 시작 시점 onset: 7~15개월째 중 무작위. 첫 6개월은 모든 가구가 '평소'다(기준선 기간).
   - 진행도 s(t) = (t - onset) / ramp 를 0~1로 자른 값. ramp는 300~500일(1년 안팎에 걸쳐 서서히).
   - s가 커질수록:
     · 공과금: 납부일이 밀리거나(최대 15일) 빠뜨림(최대 35%), 같은 공과금을 두 번 냄(최대 25%)
     · 카드: 같은 가맹점·같은 금액 결제를 몇 시간 안에 또 함(최대 20%)
     · 카드: 특정 가맹점(TV홈쇼핑 등) 반복 결제가 생김(최대 월 3회)
     · ATM: 인출 횟수가 늘고(최대 2.5배) 한 번 금액은 줄어듦
     · 거래 시각: 밤 시간(0~6시) 거래 비중이 늘어남(최대 15%)
   정상 가구도 가끔은 같은 일을 한다: 공과금 ±2일 변동·3% 빠뜨림·1% 중복, 카드 중복 0.5%, 밤 거래 2%.
   판단력 저하는 사기와 달리 라벨이 '거래'가 아니라 '가구와 시점'에 붙는다(decline 표).

2) 생활 변화 가구 (life_rate, 기본 8%) — 판단력 저하로 오해할 수 있는 정상 변화
   - hospital: 1~2개월 입원. 그 기간 거래가 거의 없고 공과금이 밀렸다가 퇴원 후 몰아서 냄
   - move_in: 자녀 집으로 합가. 그 뒤 ATM 인출이 크게 줄고 카드 결제가 늘며, 공과금이 한 날짜에 자동이체로 바뀜
   - new_routine: 손주 돌봄 등으로 생활 리듬 변화. 카드 결제가 늘고 활동 시각이 앞당겨짐
"""
import numpy as np
import pandas as pd


def generate_decline(n_people=4000, days=730, fraud_rate=0.06, decline_rate=0.05, life_rate=0.08, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows, ep_meta, dec_meta, life_meta = [], [], [], []
    ep_id = 0

    def add(pid, t, typ, amt, payee, fraud=0, ep=-1):
        rows.append((pid, t, typ, float(round(amt, -2)), payee, fraud, ep))

    for pid in range(n_people):
        wealth = rng.lognormal(0, 0.5)
        pension = 900_000 * wealth
        card_mu = np.log(25_000 * wealth)
        atm_mu = np.log(200_000 * wealth)
        tr_mu = np.log(150_000 * wealth)
        known = [f"P{pid}_k{i}" for i in range(rng.integers(2, 5))]
        active_h = rng.integers(8, 11)
        bill_day = [int(rng.integers(5, 25)) for _ in range(3)]  # 공과금별 평소 납부일

        # 판단력 저하
        dec = rng.random() < decline_rate
        onset = int(rng.integers(180, 450)) if dec else 10**9
        ramp = int(rng.integers(300, 500))
        sev = lambda d: float(np.clip((d - onset) / ramp, 0, 1))
        if dec:
            dec_meta.append((pid, onset, ramp))

        # 생활 변화
        life, lday, ldur = None, 10**9, 0
        if not dec and rng.random() < life_rate:
            life = rng.choice(["hospital", "move_in", "new_routine"], p=[0.4, 0.3, 0.3])
            lday = int(rng.integers(200, days - 120))
            ldur = int(rng.integers(30, 61)) if life == "hospital" else 10**9
            life_meta.append((pid, life, lday))
        in_hosp = lambda d: life == "hospital" and lday <= d < lday + ldur
        after = lambda d: life in ("move_in", "new_routine") and d >= lday

        def ts(day, night=False):
            day = int(min(day, days - 1))
            if not night and rng.random() < 0.02 + 0.13 * sev(day):  # 누구나 가끔 밤 거래, 저하 시 늘어남
                night = True
            h0 = active_h - 2 if (life == "new_routine" and day >= lday) else active_h
            h = rng.integers(0, 6) if night else int(np.clip(rng.normal(h0 + 4, 3), 6, 21))
            return start + pd.Timedelta(days=day, hours=int(h), minutes=int(rng.integers(0, 60)))

        for m in range(days // 30):
            d0 = m * 30
            add(pid, ts(d0 + 24), "pension_in", pension, "NPS")
            for b in range(3):
                d = d0 + bill_day[b]
                s = sev(d)
                if after(d) and life == "move_in":
                    d = d0 + 25  # 자동이체로 한 날짜에
                if in_hosp(d):
                    d = lday + ldur + int(rng.integers(0, 4))  # 퇴원 후 몰아서
                if rng.random() < 0.03 + 0.32 * s:
                    continue  # 빠뜨림 (누구나 가끔, 저하 시 늘어남)
                d += int(rng.integers(-2, 3)) + int(rng.integers(0, int(15 * s) + 1))  # 평소 ±2일, 저하 시 밀림
                amt = 60_000 * wealth * rng.uniform(0.6, 1.4)
                if d < days:
                    add(pid, ts(d), "bill", amt, f"BILL{b}")
                    if rng.random() < 0.01 + 0.24 * s:  # 같은 공과금 두 번
                        add(pid, ts(d + int(rng.integers(0, 4))), "bill", amt, f"BILL{b}")
            s = sev(d0 + 15)
            atm_rate = 2.5 * (1 + 1.5 * s)
            if after(d0) and life == "move_in":
                atm_rate = 0.8
            for _ in range(rng.poisson(atm_rate)):
                d = d0 + int(rng.integers(0, 30))
                if in_hosp(d):
                    continue
                add(pid, ts(d), "atm", rng.lognormal(atm_mu - 0.6 * s, 0.4), "ATM")
            for _ in range(rng.poisson(1.5)):
                d = d0 + int(rng.integers(0, 30))
                if in_hosp(d):
                    continue
                add(pid, ts(d), "transfer_out", rng.lognormal(tr_mu, 0.6), rng.choice(known))
            for _ in range(rng.poisson(3 * s)):  # TV홈쇼핑 반복 결제
                add(pid, ts(d0 + int(rng.integers(0, 30))), "card", rng.uniform(50_000, 200_000), "HOMESHOP")

        for d in range(days):
            if in_hosp(d):
                continue
            rate = 0.5 * (1.8 if after(d) else 1.0)
            s = sev(d)
            for _ in range(rng.poisson(rate)):
                t = ts(d); mer = f"M{rng.integers(0, 300)}"; amt = rng.lognormal(card_mu, 0.8)
                add(pid, t, "card", amt, mer)
                if rng.random() < 0.005 + 0.195 * s:  # 같은 결제를 또 함
                    add(pid, t + pd.Timedelta(minutes=int(rng.integers(10, 180))), "card", amt, mer)

        # 정상이지만 튀는 거래 (기본 생성기와 같음)
        for _ in range(rng.poisson(3 * days / 365)):
            kind = rng.choice(["hospital", "gift", "family", "big_new", "holiday_cash"], p=[0.2, 0.25, 0.2, 0.15, 0.2])
            d = int(rng.integers(20, days - 3))
            if kind == "hospital":
                add(pid, ts(d), "card", rng.lognormal(np.log(1_500_000 * wealth), 0.5), "HOSP")
            elif kind == "gift":
                add(pid, ts(d), "transfer_out", rng.choice([100_000, 200_000, 300_000, 500_000]), f"P{pid}_g{d}")
            elif kind == "family":
                add(pid, ts(d), "transfer_out", rng.lognormal(np.log(3_000_000 * wealth), 0.6), known[0])
            elif kind == "big_new":
                add(pid, ts(d), "transfer_out", rng.lognormal(np.log(2_000_000 * wealth), 0.7), f"P{pid}_n{d}")
            else:
                for k in range(rng.integers(2, 4)):
                    add(pid, ts(d + k // 2), "atm", rng.lognormal(np.log(600_000 * wealth), 0.4), "ATM")

        # 사기 에피소드 (기본 생성기와 같은 4종)
        if rng.random() < fraud_rate:
            sc = rng.choice(["impersonation", "cash_pickup", "loan_scam", "exploitation"], p=[0.35, 0.25, 0.2, 0.2])
            d = int(rng.integers(100, days - 40)); ep_id += 1; total = 0.0
            if sc == "impersonation":
                t0 = ts(d)
                for k in range(rng.integers(1, 4)):
                    a = rng.lognormal(np.log(2_500_000 * wealth), 0.9)
                    add(pid, t0 + pd.Timedelta(minutes=int(40 * k)), "transfer_out", a, f"F{ep_id}", 1, ep_id); total += a
            elif sc == "cash_pickup":
                for k in range(rng.integers(2, 6)):
                    a = rng.lognormal(np.log(700_000 * wealth), 0.5)
                    add(pid, ts(d + k // 2), "atm", a, "ATM", 1, ep_id); total += a
            elif sc == "loan_scam":
                loan = rng.choice([5_000_000, 10_000_000, 20_000_000]); t0 = ts(d)
                add(pid, t0, "loan_in", loan, "LOAN", 1, ep_id)
                a = loan * rng.uniform(0.9, 1.0)
                add(pid, t0 + pd.Timedelta(hours=int(rng.integers(1, 20))), "transfer_out", a, f"F{ep_id}", 1, ep_id); total += a
            else:
                payee = f"F{ep_id}" if rng.random() < 0.5 else known[-1]
                for k in range(rng.integers(6, 14)):
                    a = rng.lognormal(np.log(250_000 * wealth), 0.4)
                    add(pid, ts(min(d + 7 * k, days - 1)), "transfer_out", a, payee, 1, ep_id); total += a
            ep_meta.append((ep_id, pid, sc, total))

    df = pd.DataFrame(rows, columns=["pid", "ts", "type", "amount", "payee", "is_fraud", "episode"])
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    eps = pd.DataFrame(ep_meta, columns=["episode", "pid", "scenario", "loss"])
    decline = pd.DataFrame(dec_meta, columns=["pid", "onset", "ramp"])
    lifechg = pd.DataFrame(life_meta, columns=["pid", "life", "day"])
    return df, eps, decline, lifechg

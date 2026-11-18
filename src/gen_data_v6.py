"""가상 고령자 거래 데이터 생성기 v6 — 판단력 저하 시나리오를 연구 결과에 맞춰 조절할 수 있게 한 버전 (곁 GYEOT)

gen_data_v5.py와 구조는 같다. 다음을 바꿀 수 있다.

- effect : 판단력 저하 신호의 크기 배율(1.0 = v5 가정). 문헌이 보고하는 집단 평균 차이는 v5 가정보다 훨씬 작아서,
           배율을 낮췄을 때 감지율이 어떻게 무너지는지 보려는 것이다.
- ramp   : 진행 기간(일) 범위. v5는 300~500일. 문헌은 진단 5~10년 전부터 변화가 보인다고 하므로 1100~1800일도 실험한다.
- days   : 관측 기간. 느린 진행은 4년(1460일)으로 본다.

문헌(Trendl 외 2025, 영국 은행 데이터)에서 가져온 신호 세 가지를 더했다. 방향만 문헌을 따르고, 크기는 가정이다.
- 여가·외출 소비 감소: 카드 가맹점 M200~M299를 '여가·외출'(여행, 취미, 의류, 외식)로 두고, 진행도 s일 때 이 결제의 60%·s를 건너뛴다
- 디지털 뱅킹 접속 감소: 월 접속 횟수 Poisson(8) → 8·(1 − 0.5·s)
- 비밀번호 재설정·카드 분실 신고 증가: 월 확률 1% → 1% + 15%·s, 0.4% → 0.4% + 6%·s
거래가 아닌 이 기록들은 events 표(pid, m, logins, pin_reset, card_lost)로 따로 돌려준다.
"""
import numpy as np
import pandas as pd


def generate_v6(n_people=4000, days=730, fraud_rate=0.06, decline_rate=0.05, life_rate=0.08, seed=42,
                effect=1.0, ramp=(300, 500), onset=(180, 450)):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    rows, ep_meta, dec_meta, life_meta, ev_rows = [], [], [], [], []
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
        on = int(rng.integers(*onset)) if dec else 10**9
        rp = int(rng.integers(*ramp))
        sev = lambda d: effect * float(np.clip((d - on) / rp, 0, 1))
        if dec:
            dec_meta.append((pid, on, rp))

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
            # 거래가 아닌 기록: 접속 횟수, 비밀번호 재설정, 카드 분실 신고
            hosp_m = in_hosp(d0 + 15)
            ev_rows.append((pid, m, int(rng.poisson(8 * (1 - 0.5 * s) * (0.2 if hosp_m else 1.0))),
                            int(rng.random() < 0.01 + 0.15 * s), int(rng.random() < 0.004 + 0.06 * s)))
            for _ in range(rng.poisson(3 * s)):  # TV홈쇼핑 반복 결제
                add(pid, ts(d0 + int(rng.integers(0, 30))), "card", rng.uniform(50_000, 200_000), "HOMESHOP")

        for d in range(days):
            if in_hosp(d):
                continue
            rate = 0.5 * (1.8 if after(d) else 1.0)
            s = sev(d)
            for _ in range(rng.poisson(rate)):
                t = ts(d); mid = int(rng.integers(0, 300)); mer = f"M{mid}"; amt = rng.lognormal(card_mu, 0.8)
                if mid >= 200 and rng.random() < 0.6 * s:
                    continue  # 여가·외출 소비 감소
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
    events = pd.DataFrame(ev_rows, columns=["pid", "m", "logins", "pin_reset", "card_lost"])
    return df, eps, decline, lifechg, events

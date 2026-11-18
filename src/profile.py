"""개인 프로필: 한 사람의 거래 기록에서 '이 사람의 평소'를 사람이 읽을 수 있게 정리한다.

모델 피처(z-score, 새 수취인 등)는 이미 '그 사람의 평소'와 비교하지만 숫자로만 남는다.
프로필은 같은 정보를 가족·상담원이 읽는 문장과 표로 바꾼 것이다. 알림이 왔을 때 "평소엔 어땠는데?"에 답한다.

build_profile(tx, asof)   : asof 시점까지의 거래로 프로필(dict)을 만든다
drift(tx, asof, months=3) : 최근 몇 달이 평소(그 이전)와 얼마나 다른지
to_markdown(profile)      : 프로필 카드
"""
import numpy as np
import pandas as pd

TYPE_KO = {"card": "카드 결제", "atm": "현금 인출", "transfer_out": "송금", "bill": "공과금", "pension_in": "연금 입금", "loan_in": "대출 입금"}
OUT = ["card", "atm", "transfer_out", "bill"]
won = lambda x: f"{x / 10_000:,.0f}만 원" if x >= 10_000 else f"{x:,.0f}원"


def _stats(t):
    months = max((t.ts.max() - t.ts.min()).days / 30, 1)
    out = t[t.type.isin(OUT)]
    by = {}
    for ty in OUT:
        a = out[out.type == ty].amount
        if len(a):
            by[ty] = dict(per_month=len(a) / months, median=float(a.median()), p90=float(a.quantile(0.9)), max=float(a.max()),
                          share=float(a.sum() / out.amount.sum()))
    h = t[t.type.isin(OUT)].ts.dt.hour
    return dict(months=months, income_month=float(t[t.type.str.endswith("_in")].amount.sum() / months),
                out_month=float(out.amount.sum() / months), by_type=by,
                hour_lo=int(h.quantile(0.1)), hour_hi=int(h.quantile(0.9)), night_share=float((h < 6).mean()),
                big=float(out.amount.quantile(0.95)), biggest=float(out.amount.max()))


def build_profile(tx, asof, name="", verdict_fn=None):
    t = tx[tx.ts < asof].sort_values("ts")
    p = _stats(t); p["name"], p["asof"], p["n_tx"] = name, str(pd.Timestamp(asof).date()), int(len(t))
    b = t[t.type == "bill"].assign(dom=lambda d: d.ts.dt.day)
    p["bill_days"] = {k: int(v) for k, v in b.groupby("payee").dom.median().items()}
    tr = t[t.type == "transfer_out"]
    cps = []
    for payee, g in tr.groupby("payee"):
        last = g.iloc[-1]
        cps.append(dict(payee=payee, n=int(len(g)), median=float(g.amount.median()), total=float(g.amount.sum()),
                        first=str(g.ts.min().date()), last=str(g.ts.max().date()),
                        verdict=verdict_fn(last) if verdict_fn else ""))
    p["counterparties"] = sorted(cps, key=lambda c: -c["total"])
    c = t[t.type == "card"]
    p["merchants"] = int(c.payee.nunique())
    p["top_merchant_share"] = float(c.groupby("payee").amount.sum().max() / c.amount.sum()) if len(c) else 0.0
    return p


def drift(tx, asof, months=3):
    """최근 months개월 vs 그 이전. 값이 크게 다르면 생활이 바뀌고 있다는 뜻이다."""
    asof = pd.Timestamp(asof); cut = asof - pd.Timedelta(days=30 * months)
    a, b = _stats(tx[tx.ts < cut]), _stats(tx[(tx.ts >= cut) & (tx.ts < asof)])
    rows = [("월 지출", won(a["out_month"]), won(b["out_month"])),
            ("현금 인출 횟수(월)", f"{a['by_type'].get('atm', {}).get('per_month', 0):.1f}회", f"{b['by_type'].get('atm', {}).get('per_month', 0):.1f}회"),
            ("현금 인출 한 번 금액", won(a["by_type"].get("atm", {}).get("median", 0)), won(b["by_type"].get("atm", {}).get("median", 0))),
            ("공과금 납부 건수(월)", f"{a['by_type'].get('bill', {}).get('per_month', 0):.1f}건", f"{b['by_type'].get('bill', {}).get('per_month', 0):.1f}건"),
            ("송금 횟수(월)", f"{a['by_type'].get('transfer_out', {}).get('per_month', 0):.1f}회", f"{b['by_type'].get('transfer_out', {}).get('per_month', 0):.1f}회"),
            ("밤(0~6시) 거래 비중", f"{a['night_share']:.0%}", f"{b['night_share']:.0%}")]
    return pd.DataFrame(rows, columns=["항목", "평소", f"최근 {months}개월"])


def to_markdown(p, top=6):
    L = [f"### {p['name']} 프로필 ({p['asof']} 기준, 거래 {p['n_tx']:,}건 / {p['months']:.0f}개월)", "",
         f"- **들어오는 돈**: 월 평균 {won(p['income_month'])}",
         f"- **나가는 돈**: 월 평균 {won(p['out_month'])}",
         f"- **이 사람에게 큰돈이란**: 한 번에 {won(p['big'])} 이상이면 상위 5% (지금까지 최대 {won(p['biggest'])})",
         f"- **거래 시간**: 주로 {p['hour_lo']}시~{p['hour_hi']}시, 밤(0~6시) 거래 {p['night_share']:.0%}",
         f"- **공과금 납부일**: " + ", ".join(f"{k} 매월 {v}일 무렵" for k, v in p["bill_days"].items()),
         f"- **카드**: 가맹점 {p['merchants']}곳, 가장 많이 쓴 곳의 비중 {p['top_merchant_share']:.0%}", "",
         "| 거래 종류 | 월 횟수 | 보통 금액 | 큰 편(상위 10%) | 지출 비중 |", "| --- | --- | --- | --- | --- |"]
    for ty, s in p["by_type"].items():
        L.append(f"| {TYPE_KO[ty]} | {s['per_month']:.1f}회 | {won(s['median'])} | {won(s['p90'])} | {s['share']:.0%} |")
    L += ["", "| 송금 상대 | 횟수 | 보통 금액 | 거래 기간 | 상대 판단 |", "| --- | --- | --- | --- | --- |"]
    for c in p["counterparties"][:top]:
        L.append(f"| {c['payee']} | {c['n']}회 | {won(c['median'])} | {c['first']} ~ {c['last']} | {c['verdict']} |")
    return "\n".join(L)

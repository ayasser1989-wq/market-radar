"""منطق المتداول — يحوّل المؤشرات إلى خطة صفقة (شراء فقط).

لكل سهم:
  - يحدد نوع الفرصة: اختراق / ارتداد من الدعم / استمرار الاتجاه / ارتداد من تشبّع بيعي
  - يعطي درجة من 100 (اتجاه + زخم + حجم + تقلب + جودة النمط)
  - يحسب الدخول، وقف الخسارة (2×ATR)، الهدف الأول (1.5R) والثاني (3R)
  - يحسب حجم المركز حسب رأس مالك ونسبة المخاطرة
  - يختبر نفس القواعد على آخر سنتين ليعطي نسبة النجاح التاريخية
"""

import math

import numpy as np
import pandas as pd

from watchlist import ACCOUNT, RULES

SETUP_NAMES = {
    "breakout": "اختراق",
    "pullback": "ارتداد من الدعم",
    "trend": "استمرار الاتجاه",
    "oversold": "ارتداد من تشبّع بيعي",
}
QUALITY = {"breakout": 15, "pullback": 12, "trend": 8, "oversold": 5}

STOP_ATR = 2.0      # وقف الخسارة = الدخول − 2×ATR
T1_R = 1.5          # الهدف الأول = 1.5 ضعف المخاطرة
T2_R = 3.0          # الهدف الثاني = 3 أضعاف المخاطرة
HOLD_DAYS = 20      # أقصى مدة للصفقة في الاختبار التاريخي


def indicators(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["c"], out["h"], out["l"] = c, h, l
    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(100)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    out["volx"] = v / v.shift().rolling(20).mean()
    out["hi20"] = h.shift().rolling(20).max()
    return out


def evaluate(x, i):
    """يقيّم اليوم i. يعيد (نوع الفرصة أو None، الدرجة، تفاصيل الدرجة)."""
    r = x.iloc[i]
    if any(pd.isna(r[k]) for k in ("sma50", "sma200", "atr", "hi20", "volx")):
        return None, 0, []
    c, s20, s50, s200, rsi, atr = r["c"], r["sma20"], r["sma50"], r["sma200"], r["rsi"], r["atr"]
    hist, hist_prev, volx = r["hist"], x["hist"].iloc[i - 1], r["volx"]

    setup = None
    if c > r["hi20"] and volx >= 1.5 and c > s50:
        setup = "breakout"
    elif c > s200 and s50 > s200 and abs(c / s50 - 1) <= 0.03 and 38 <= rsi <= 55:
        setup = "pullback"
    elif c > s20 > s50 > s200 and 50 <= rsi <= 68 and hist > 0:
        setup = "trend"
    elif rsi < 32 and c > s200 * 0.9:
        setup = "oversold"

    parts = []
    def add(ok, pts, label):
        if ok:
            parts.append((label, pts))
    add(c > s200, 15, "فوق متوسط 200 يوم")
    add(s50 > s200, 10, "متوسط 50 فوق متوسط 200")
    add(c > s50, 10, "فوق متوسط 50 يوم")
    add(hist > 0, 10, "MACD إيجابي")
    add(hist > hist_prev, 5, "الزخم يتحسن")
    add(45 <= rsi <= 68 or (setup == "oversold" and rsi < 32), 10, "RSI في منطقة صحية")
    add(volx >= 1.5, 15, "حجم تداول مرتفع يؤكد الحركة")
    add(1.1 <= volx < 1.5, 8, "حجم تداول فوق المتوسط")
    add(0.01 <= atr / c <= 0.04, 10, "تقلب مناسب للمخاطرة")
    if setup:
        parts.append(("نمط " + SETUP_NAMES[setup], QUALITY[setup]))
    return setup, min(100, sum(p for _, p in parts)), parts


def levels(entry, atr):
    risk = STOP_ATR * atr
    return {"entry": entry, "stop": entry - risk, "t1": entry + T1_R * risk,
            "t2": entry + T2_R * risk, "risk": risk}


def size(entry, risk, fx):
    """عدد الأسهم حسب نسبة المخاطرة، بحد أقصى لحجم المركز من رأس المال."""
    cap = ACCOUNT["capital_sar"]
    by_risk = cap * ACCOUNT["risk_pct"] / 100 / (risk * fx)
    by_cap = cap * ACCOUNT["max_position_pct"] / 100 / (entry * fx)
    shares = max(0, math.floor(min(by_risk, by_cap)))
    return {"shares": shares, "cost_sar": round(shares * entry * fx),
            "max_loss_sar": round(shares * risk * fx), "capped": by_cap < by_risk}


def backtest(x, min_score):
    """يطبق نفس القواعد على كل يوم سابق ويحاكي الصفقة 20 يومًا للأمام."""
    trades, i, n = [], 201, len(x)
    while i < n - 1:
        setup, score, _ = evaluate(x, i)
        if not setup or score < min_score:
            i += 1
            continue
        lv = levels(float(x["c"].iloc[i]), float(x["atr"].iloc[i]))
        result, j = None, i + 1
        while j < min(n, i + 1 + HOLD_DAYS):
            lo, hi = float(x["l"].iloc[j]), float(x["h"].iloc[j])
            if lo <= lv["stop"]:          # إن لمس الوقف والهدف في نفس اليوم نحسبها خسارة (تحفّظ)
                result = -1.0
                break
            if hi >= lv["t1"]:
                result = T1_R
                break
            j += 1
        if result is None:
            if j >= n:                     # صفقة لم تكتمل بعد — لا تُحسب
                break
            result = (float(x["c"].iloc[j - 1]) - lv["entry"]) / lv["risk"]
        trades.append((setup, result))
        i = j + 1                          # لا صفقات متداخلة على نفس السهم
    return trades


def summarise(trades):
    if not trades:
        return {"n": 0, "win": None, "avg_r": None}
    rs = [r for _, r in trades]
    return {"n": len(rs), "win": round(100 * sum(r > 0 for r in rs) / len(rs)),
            "avg_r": round(sum(rs) / len(rs), 2)}


def plan(df, mkt):
    """خطة السهم الحالية + نتائج الاختبار التاريخي."""
    x = indicators(df)
    min_score = RULES["min_setup_score"]
    setup, score, parts = evaluate(x, len(x) - 1)
    r = x.iloc[-1]
    if r["c"] > r["sma50"] and r["sma50"] > r["sma200"]:
        bias = "up"
    elif r["c"] < r["sma50"] and (pd.isna(r["sma200"]) or r["sma50"] < r["sma200"]):
        bias = "down"
    else:
        bias = "flat"

    trades = backtest(x, min_score)
    out = {
        "score": score, "bias": bias, "atr": round(float(r["atr"]), 2),
        "why": [lbl for lbl, _ in parts],
        "hist": summarise(trades),
        "trades": trades,
        "setup": None,
    }
    if setup and score >= min_score:
        fx = ACCOUNT["usd_sar"] if mkt == "US" else 1.0
        lv = levels(float(r["c"]), float(r["atr"]))
        out["setup"] = {
            "type": setup, "name": SETUP_NAMES[setup],
            **{k: round(v, 2) for k, v in lv.items()},
            "rr": T1_R, **size(lv["entry"], lv["risk"], fx),
            "hist": summarise([t for t in trades if t[0] == setup]),
        }
    return out

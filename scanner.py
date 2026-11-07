"""رادار الأسواق — ماسح للأسهم الأمريكية والسعودية.

في كل تشغيل:
  1. يسحب بيانات سنة يومية لكل سهم من Yahoo Finance (متأخرة ~15 دقيقة).
  2. يحسب المؤشرات ويبحث عن الإشارات.
  3. يرسل التنبيهات الجديدة على تيليجرام (للسوق المفتوح فقط، ولا يكرر التنبيه في نفس اليوم).
  4. يكتب لوحة الجوال في docs/index.html

التشغيل محليًا:   python scanner.py
تشغيل تجريبي بلا إنترنت:   DEMO=1 python scanner.py
"""

import json
import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from watchlist import RULES, SA, US

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
STATE_FILE = ROOT / "state.json"
TEMPLATE = ROOT / "template.html"

RIYADH = ZoneInfo("Asia/Riyadh")
NEW_YORK = ZoneInfo("America/New_York")

MARKETS = {
    "US": {"name": "السوق الأمريكي", "tz": NEW_YORK, "days": {0, 1, 2, 3, 4},
           "open": time(9, 30), "close": time(16, 0), "currency": "$"},
    "SA": {"name": "السوق السعودي", "tz": RIYADH, "days": {6, 0, 1, 2, 3},
           "open": time(10, 0), "close": time(15, 0), "currency": "ر.س"},
}


# ---------------------------------------------------------------- البيانات

def fetch_history(tickers):
    """يعيد dict: الرمز -> DataFrame فيه Close / High / Low / Volume."""
    if os.environ.get("DEMO"):
        return demo_history(tickers)
    import yfinance as yf

    raw = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                      auto_adjust=False, threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            df = raw[t][["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) >= 30:
            out[t] = df
    return out


def demo_history(tickers):
    """بيانات عشوائية لاختبار الكود فقط — ليست أسعارًا حقيقية."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=datetime.now(), periods=260)
    out = {}
    for i, t in enumerate(tickers):
        drift = rng.normal(0.0004, 0.001)
        rets = rng.normal(drift, 0.018, len(idx))
        rets[-1] = rng.normal(0, 0.03)            # يوم أخير أكثر تقلبًا لإظهار إشارات
        close = 50 * (1 + i % 7) * np.exp(np.cumsum(rets))
        vol = rng.lognormal(15, 0.35, len(idx))
        vol[-1] *= rng.choice([1, 1, 1, 2.6])
        out[t] = pd.DataFrame({"Close": close, "High": close * 1.01,
                               "Low": close * 0.99, "Volume": vol}, index=idx)
    return out


# ---------------------------------------------------------------- المؤشرات

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def analyse(sym, name, mkt, df):
    c, v = df["Close"], df["Volume"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    r = rsi(c)
    price, prev = float(c.iloc[-1]), float(c.iloc[-2])
    chg = (price / prev - 1) * 100
    avg_vol = float(v.iloc[-21:-1].mean())
    volx = float(v.iloc[-1]) / avg_vol if avg_vol else 0.0
    prior = df.iloc[-253:-1]
    hi52, lo52 = float(prior["High"].max()), float(prior["Low"].min())

    def last(s, k=-1):
        x = s.iloc[k]
        return None if pd.isna(x) else float(x)

    s50, s50p, s200, s200p = last(sma50), last(sma50, -2), last(sma200), last(sma200, -2)
    rnow = float(r.iloc[-1])

    sig = []
    if abs(chg) >= RULES["big_move_pct"]:
        sig.append(("move", f"حركة قوية \u2066{chg:+.1f}%\u2069", "up" if chg > 0 else "down"))
    if rnow <= RULES["rsi_low"]:
        sig.append(("rsi_low", f"تشبّع بيعي RSI {rnow:.0f}", "up"))
    elif rnow >= RULES["rsi_high"]:
        sig.append(("rsi_high", f"تشبّع شرائي RSI {rnow:.0f}", "down"))
    if None not in (s50, s50p, s200, s200p):
        if s50p <= s200p and s50 > s200:
            sig.append(("golden", "تقاطع ذهبي: متوسط 50 فوق 200", "up"))
        elif s50p >= s200p and s50 < s200:
            sig.append(("death", "تقاطع الموت: متوسط 50 تحت 200", "down"))
    if s50 and s50p:
        if prev <= s50p and price > s50:
            sig.append(("x50_up", "اخترق متوسط 50 يوم صعودًا", "up"))
        elif prev >= s50p and price < s50:
            sig.append(("x50_down", "كسر متوسط 50 يوم هبوطًا", "down"))
    if price > hi52:
        sig.append(("hi52", "قمة جديدة لـ 52 أسبوعًا", "up"))
    elif price < lo52:
        sig.append(("lo52", "قاع جديد لـ 52 أسبوعًا", "down"))
    if volx >= RULES["volume_x"]:
        sig.append(("vol", f"حجم تداول \u2066×{volx:.1f}\u2069 من المتوسط", "up" if chg >= 0 else "down"))

    return {
        "sym": sym.replace(".SR", ""), "ticker": sym, "name": name, "mkt": mkt,
        "price": round(price, 2), "chg": round(chg, 2), "rsi": round(rnow, 1),
        "pct50": round((price / s50 - 1) * 100, 1) if s50 else None,
        "volx": round(volx, 2), "hi52": round(hi52, 2), "lo52": round(lo52, 2),
        "date": c.index[-1].strftime("%Y-%m-%d"),
        "spark": [round(float(x), 2) for x in c.iloc[-30:]],
        "signals": [{"k": k, "label": lbl, "dir": d} for k, lbl, d in sig],
    }


# ---------------------------------------------------------------- السوق والتنبيهات

def market_open(key, now_utc):
    m = MARKETS[key]
    local = now_utc.astimezone(m["tz"])
    # نعتبر السوق "مفتوحًا للتنبيهات" حتى 20 دقيقة بعد الإغلاق بسبب تأخر البيانات
    close_plus = (datetime.combine(local.date(), m["close"]) + timedelta(minutes=20)).time()
    return local.weekday() in m["days"] and m["open"] <= local.time() <= close_plus


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sent": {}}


def send_telegram(text):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram] لم تُضبط بيانات تيليجرام — تخطّي الإرسال")
        return
    import requests
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=20)
    if not r.ok:
        print("[telegram] فشل الإرسال:", r.status_code, r.text[:200])


def notify(stocks, open_now, state, now_utc):
    today = now_utc.astimezone(RIYADH).strftime("%Y-%m-%d")
    sent = {k: v for k, v in state["sent"].items() if v >= (now_utc - timedelta(days=3)).strftime("%Y-%m-%d")}
    lines = []
    for s in stocks:
        if not (open_now[s["mkt"]] or os.environ.get("FORCE_ALERTS")):
            continue
        for g in s["signals"]:
            key = f"{today}:{s['ticker']}:{g['k']}"
            if key in sent:
                continue
            sent[key] = today
            arrow = "🟢" if g["dir"] == "up" else "🔴"
            cur = MARKETS[s["mkt"]]["currency"]
            lines.append(f"{arrow} <b>{s['sym']}</b> {s['name']} — {g['label']}\n"
                         f"    السعر {s['price']} {cur} (\u2066{s['chg']:+.2f}%\u2069)")
    state["sent"] = sent
    if lines:
        page = os.environ.get("DASHBOARD_URL", "")
        msg = "📡 <b>رادار الأسواق</b>\n\n" + "\n\n".join(lines[:30])
        if page:
            msg += f"\n\n<a href=\"{page}\">افتح اللوحة</a>"
        send_telegram(msg)
    print(f"[alerts] تنبيهات جديدة: {len(lines)}")


# ---------------------------------------------------------------- اللوحة

def render(stocks, open_now, now_utc):
    data = {
        "updated": now_utc.astimezone(RIYADH).isoformat(timespec="minutes"),
        "demo": bool(os.environ.get("DEMO")),
        "markets": {k: {"name": m["name"], "open": open_now[k], "cur": m["currency"]}
                    for k, m in MARKETS.items()},
        "stocks": stocks,
    }
    DOCS.mkdir(exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / "data.json").write_text(payload, encoding="utf-8")


def main():
    now_utc = datetime.now(timezone.utc)
    universe = [(t, n, "US") for t, n in US] + [(t, n, "SA") for t, n in SA]
    hist = fetch_history([t for t, _, _ in universe])
    stocks = []
    for t, n, m in universe:
        if t in hist:
            try:
                stocks.append(analyse(t, n, m, hist[t]))
            except Exception as e:  # سهم واحد لا يوقف الماسح
                print(f"[skip] {t}: {e}")
        else:
            print(f"[skip] {t}: لا توجد بيانات")
    if not stocks:
        print("لم تصل أي بيانات — توقف.")
        sys.exit(1)

    open_now = {k: market_open(k, now_utc) for k in MARKETS}
    state = load_state()
    notify(stocks, open_now, state, now_utc)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    render(stocks, open_now, now_utc)
    n_sig = sum(len(s["signals"]) for s in stocks)
    print(f"[done] {len(stocks)} سهم، {n_sig} إشارة، الأسواق المفتوحة: {[k for k, v in open_now.items() if v]}")


if __name__ == "__main__":
    main()

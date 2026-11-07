"""
Nifty Live Option Chain (Upstox) - Streamlit page
- रंगीन चेन (Calls लाल / Puts हरे) + % बार
- AI सपोर्ट / रेजिस्टेंस / रिवर्सल प्राइस
- OI अलर्ट (ऊपर/नीचे OI% बढ़ रहा है या नहीं)
- अलग चार्ट टैब

इस्तेमाल: इस फ़ाइल को अपने repo में डालें (या pages/ फ़ोल्डर में),
Streamlit secrets में UPSTOX_TOKEN डालें (या साइडबार में पेस्ट करें).
requirements.txt: streamlit, requests, pandas, plotly, streamlit-autorefresh
"""
import json
import time
from collections import deque
from datetime import date, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BASE = "https://api.upstox.com/v2"
INDEX_KEY = "NSE_INDEX|Nifty 50"
N_STRIKES = 10  # ATM के ऊपर और नीचे


def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ---------------------------------------------------------------- API
def _hdr(token):
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


@st.cache_data(ttl=600, show_spinner=False)
def get_expiries(token):
    r = requests.get(f"{BASE}/option/contract", params={"instrument_key": INDEX_KEY},
                     headers=_hdr(token), timeout=15)
    r.raise_for_status()
    ex = sorted({str(c["expiry"])[:10] for c in r.json()["data"]})
    return [e for e in ex if e >= date.today().isoformat()]


def get_chain(token, expiry):
    r = requests.get(f"{BASE}/option/chain",
                     params={"instrument_key": INDEX_KEY, "expiry_date": expiry},
                     headers=_hdr(token), timeout=15)
    r.raise_for_status()
    return r.json()["data"]


def get_candles(token):
    url = f"{BASE}/historical-candle/intraday/{quote(INDEX_KEY, safe='')}/1minute"
    r = requests.get(url, headers=_hdr(token), timeout=15)
    r.raise_for_status()
    c = r.json()["data"]["candles"]
    df = pd.DataFrame(c, columns=["time", "open", "high", "low", "close", "vol", "oi"])
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")



@st.cache_data(ttl=300, show_spinner=False)
def get_hist_candles(token, days=6):
    to_d = date.today()
    from_d = to_d - timedelta(days=days)
    url = (f"{BASE}/historical-candle/{quote(INDEX_KEY, safe='')}/1minute/"
           f"{to_d.isoformat()}/{from_d.isoformat()}")
    r = requests.get(url, headers=_hdr(token), timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["candles"],
                      columns=["time", "open", "high", "low", "close", "vol", "oi"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_candles(token):
    """पिछले कुछ दिन (history) + आज (intraday) के 1-मिनट कैंडल."""
    parts = []
    try:
        parts.append(get_hist_candles(token))
    except Exception:
        pass
    parts.append(get_candles(token))
    df = pd.concat(parts).drop_duplicates("time").sort_values("time")
    return df.reset_index(drop=True)


def resample(df, minutes):
    if minutes == 1:
        return df.reset_index(drop=True)
    o = df.set_index("time").resample(f"{minutes}min", origin="start_day", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
    return o.dropna(subset=["open"]).reset_index()


# ---------------------------------------------------------------- सप्लाई / डिमांड ज़ोन
def _zone_candidates(df, spot, atr_mult=1.5, min_width=0.0, widen=False):
    """Base + Impulse नियम (RBR/DBR/RBD/DBD), साथ में Imbalance (FVG), liquidity grab और fresh/tested जाँच."""
    if len(df) < 30:
        return []
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values
    n = len(df)
    cand = []
    for i in range(15, n):
        a = atr[i - 1]
        if not a == a or a <= 0:
            continue
        body = abs(c[i] - o[i])
        if body < atr_mult * a or (h[i - 1] - l[i - 1]) > a:
            continue
        base = [i - 1]
        while len(base) < 3 and base[-1] >= 1 and (h[base[-1] - 1] - l[base[-1] - 1]) <= 0.8 * a:
            base.append(base[-1] - 1)
        b0 = min(base)
        zl, zh = min(l[b] for b in base), max(h[b] for b in base)
        demand = c[i] > o[i]
        if widen:  # ज़ोन को impulse कैंडल की open तक बढ़ाओ
            if demand:
                zh = max(zh, o[i])
            else:
                zl = min(zl, o[i])
        if zh - zl < min_width:
            continue
        touches, inside, broken = 0, False, False
        for k in range(i + 1, n):
            if demand:
                if c[k] < zl:
                    broken = True
                    break
                now = l[k] <= zh
            else:
                if c[k] > zh:
                    broken = True
                    break
                now = h[k] >= zl
            if now and not inside:
                touches += 1
            inside = now
        if broken:
            continue
        if (demand and zl > spot) or ((not demand) and zh < spot):
            continue
        dist = max(0.0, spot - zh) if demand else max(0.0, zl - spot)

        before = "flat"
        if b0 >= 6:
            pre = c[b0 - 1] - c[b0 - 6]
            before = "drop" if pre < -a else "rally" if pre > a else "flat"
        if demand:
            pattern = {"drop": "DBR (रिवर्सल)", "rally": "RBR (कंटिन्यूएशन)", "flat": "बेस-रैली"}[before]
        else:
            pattern = {"rally": "RBD (रिवर्सल)", "drop": "DBD (कंटिन्यूएशन)", "flat": "बेस-ड्रॉप"}[before]
        fvg = bool(i + 1 < n and ((demand and l[i + 1] > h[i - 1]) or ((not demand) and h[i + 1] < l[i - 1])))
        grab = False
        if b0 > 0:
            prior = slice(max(0, b0 - 10), b0)
            grab = bool((demand and zl < l[prior].min()) or ((not demand) and zh > h[prior].max()))
        cand.append(dict(kind="demand" if demand else "supply", low=float(zl), high=float(zh), start=b0, i=i,
                         touches=touches, impulse=float(body / a), pattern=pattern, fvg=fvg, grab=grab, dist=dist))
    return cand


def ai_zones(df, spot, ana, lc=None, chain=None, htf=None, n_each=2, min_width=0.0, widen=False):
    """
    सारे जाने-माने तरीकों को जोड़कर ज़ोन और स्कोर (0-100):
    Impulse ताकत 25 + Fresh/Tested 20 + Imbalance(FVG) 10 + Liquidity grab 10 + ऊँचे टाइमफ्रेम से मेल 10
    + नयापन 5 + ऑप्शन चेन OI मेल 20. ऐसे OI वॉल जिन पर प्राइस-एक्शन ज़ोन नहीं, वे अलग 'OI ज़ोन' बनते हैं.
    """
    cand = _zone_candidates(df, spot, min_width=min_width, widen=widen)
    htf_c = _zone_candidates(htf, spot, min_width=min_width, widen=widen) if htf is not None else []
    n = len(df)
    cmax = max(chain["c_oi"].max(), 1e-9) if chain is not None else 1.0
    pmax = max(chain["p_oi"].max(), 1e-9) if chain is not None else 1.0

    for z in cand:
        sc = min(z["impulse"] / 3, 1) * 25
        why = [f"Impulse ATR का {z['impulse']:.1f} गुना"]
        if z["touches"] == 0:
            sc += 20
            why.append("Fresh (अभी तक टच नहीं)")
        elif z["touches"] == 1:
            sc += 10
            why.append("1 बार टच हुआ")
        else:
            why.append(f"{z['touches']} बार टच (कमज़ोर)")
        if z["fvg"]:
            sc += 10
            why.append("Imbalance (FVG) छोड़ा")
        if z["grab"]:
            sc += 10
            why.append("बेस ने पिछला low/high sweep किया")
        if any(x["kind"] == z["kind"] and x["low"] <= z["high"] and x["high"] >= z["low"] for x in htf_c):
            sc += 10
            why.append("15 मिनट ज़ोन से मेल")
        sc += 5 * z["i"] / n
        if chain is not None:
            sub = chain[(chain["strike"] >= z["low"] - 25) & (chain["strike"] <= z["high"] + 25)]
            if not sub.empty:
                if z["kind"] == "supply":
                    r = sub.loc[sub["c_oi"].idxmax()]
                    pct, chg, side = r["c_oi"] / cmax * 100, r["c_chg"], "Call"
                else:
                    r = sub.loc[sub["p_oi"].idxmax()]
                    pct, chg, side = r["p_oi"] / pmax * 100, r["p_chg"], "Put"
                if pct >= 50:
                    sc += 15 + (5 if chg > 0 else 0)
                    why.append(f"{side} OI {r['strike']:.0f} पर {pct:.0f}% ({'बढ़ रहा' if chg > 0 else 'घट रहा'})")
        z["score"] = int(round(min(sc, 100)))
        z["reasons"] = why
        z["src"] = "price"

    out = []
    for kind, name in (("supply", "Supply"), ("demand", "Demand")):
        pool = [z for z in cand if z["kind"] == kind]
        pool = [z for z in pool if z["score"] >= 35] or pool
        pool.sort(key=lambda z: (z["dist"], -z["score"]))
        picked = []
        for z in pool:
            if any(z["low"] <= p["high"] and z["high"] >= p["low"] for p in picked):
                continue
            picked.append(z)
            if len(picked) == n_each:
                break
        for k, z in enumerate(picked, 1):
            z["label"] = f"{name} {k} Zone"
        out += picked

    if lc:
        for kind, walls, side in (("supply", lc.get("call_walls", []), "Call"), ("demand", lc.get("put_walls", []), "Put")):
            if not walls:
                continue
            w = walls[0]
            if any(z["kind"] == kind and z["low"] <= w["strike"] + 30 and z["high"] >= w["strike"] - 30 for z in out):
                continue
            out.append(dict(kind=kind, low=w["strike"] - 20, high=w["strike"] + 20, start=max(n - 80, 0), i=n - 1,
                            touches=0, pattern="OI वॉल", src="OI", fvg=False, grab=False,
                            label="OI Supply" if kind == "supply" else "OI Demand",
                            score=int(min(90, round(30 + 0.5 * w["pct"] + (10 if w["chg"] > 0 else 0)))),
                            reasons=[f"{side} OI {w['oi']:.1f} लाख ({w['pct']:.0f}%): सबसे बड़ी वॉल"]))
    return out


def build_ai_payload(zones, ana, lc, spot, liq=None):
    data = {
        "nifty_spot": round(float(spot), 2),
        "zones": [dict(label=z["label"], kind=z["kind"], low=round(z["low"]), high=round(z["high"]),
                       score=z["score"], pattern=z["pattern"], source=z["src"], touches=z["touches"],
                       reasons=z["reasons"]) for z in zones],
        "option_chain": dict(
            support=ana.get("support"), resistance=ana.get("resistance"),
            support_reversal=ana.get("support_rev"), resistance_reversal=ana.get("resistance_rev"),
            max_pain=ana.get("max_pain"), pcr=ana.get("pcr"), bias=ana.get("bias"),
            call_walls=[dict(strike=w["strike"], oi_lakh=round(w["oi"], 1), pct=round(w["pct"])) for w in (lc or {}).get("call_walls", [])],
            put_walls=[dict(strike=w["strike"], oi_lakh=round(w["oi"], 1), pct=round(w["pct"])) for w in (lc or {}).get("put_walls", [])]),
        "chart_liquidity": [dict(kind=x["kind"], price=round(x["price"]), status=x["status"]) for x in (liq or [])],
        "oi_alerts": [m for _, m in ana.get("alerts", [])][:8],
    }
    return json.dumps(data, ensure_ascii=False, indent=1, default=float)


def ai_commentary(api_key, payload):
    """ऊपर के नियम-आधारित ज़ोन को Claude को भेजकर रैंकिंग/व्याख्या. नए लेवल बनाने की मनाही है."""
    system = ("तुम Nifty के सप्लाई-डिमांड ज़ोन और ऑप्शन चेन के अनुभवी विश्लेषक हो. सिर्फ़ दिए गए JSON के स्तर इस्तेमाल करो, "
              "अपने से कोई नया प्राइस लेवल मत बनाओ. हिंदी में 6-8 छोटी पंक्तियों में बताओ: "
              "1) सबसे अहम सप्लाई और डिमांड ज़ोन और क्यों, 2) कौन सा ज़ोन कमज़ोर या शक वाला, "
              "3) प्राइस अभी कहाँ है और ज़ोन व OI वॉल के हिसाब से आगे क्या देखना चाहिए, "
              "4) किस स्तर के टूटने पर यह विश्लेषण गलत माना जाए. आख़िर में एक पंक्ति: यह ट्रेड की सलाह नहीं है.")
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": "claude-sonnet-5-5", "max_tokens": 900, "system": system,
                            "messages": [{"role": "user", "content": payload}]}, timeout=60)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


# ---------------------------------------------------------------- डेटा
def build_df(data):
    rows = []
    for d in data:
        co, po = d.get("call_options", {}), d.get("put_options", {})
        cm, pm = co.get("market_data", {}), po.get("market_data", {})
        cg, pg = co.get("option_greeks", {}), po.get("option_greeks", {})
        rows.append(dict(
            strike=d["strike_price"],
            c_ltp=cm.get("ltp", 0), c_vol=cm.get("volume", 0),
            c_oi=cm.get("oi", 0) / 1e5, c_chg=(cm.get("oi", 0) - cm.get("prev_oi", 0)) / 1e5,
            c_iv=cg.get("iv", 0), c_cl=cm.get("close_price", 0), p_cl=pm.get("close_price", 0),
            c_bid=cm.get("bid_price", 0), c_ask=cm.get("ask_price", 0),
            p_bid=pm.get("bid_price", 0), p_ask=pm.get("ask_price", 0),
            p_ltp=pm.get("ltp", 0), p_vol=pm.get("volume", 0),
            p_oi=pm.get("oi", 0) / 1e5, p_chg=(pm.get("oi", 0) - pm.get("prev_oi", 0)) / 1e5,
            p_iv=pg.get("iv", 0),
        ))
    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    spot = data[0].get("underlying_spot_price", 0)
    return df, spot


def window(df, spot, n=N_STRIKES):
    atm_i = (df["strike"] - spot).abs().idxmin()
    lo, hi = max(atm_i - n, 0), min(atm_i + n, len(df) - 1)
    return df.loc[lo:hi].reset_index(drop=True), df.loc[atm_i, "strike"]



# ---------------------------------------------------------------- 5 मिनट OI बदलाव
@st.cache_resource
def history_store():
    return {}  # expiry -> deque[(timestamp, {strike: (call_oi, put_oi)})]


def record_and_baseline(expiry, full, minutes=5):
    """अभी का snapshot रखता है, और ~5 मिनट पुराना snapshot (base) लौटाता है."""
    dq = history_store().setdefault(expiry, deque(maxlen=900))
    now = time.time()
    if not dq or now - dq[-1][0] >= 5:
        dq.append((now, {r.strike: (r.c_oi, r.p_oi, r.c_ltp, r.p_ltp) for r in full.itertuples()}))
    target = now - minutes * 60
    base = None
    for ts, snap in dq:
        if ts <= target:
            base = (ts, snap)
    if base is None and len(dq) > 1:
        base = dq[0]
    if base is None or now - base[0] < 20:
        return None, 0.0
    return base[1], (now - base[0]) / 60


def add_5m(df, base):
    df = df.copy()
    nan = float("nan")
    if not base:
        for col in ("c_chg5", "p_chg5", "c_chg5p", "p_chg5p", "c_dl5", "p_dl5"):
            df[col] = nan
        return df
    g = lambda i: df["strike"].map(lambda k: base.get(k, (nan, nan, nan, nan))[i])
    cb, pb = g(0).where(lambda x: x > 0), g(1).where(lambda x: x > 0)
    df["c_chg5"], df["p_chg5"] = df["c_oi"] - cb, df["p_oi"] - pb
    df["c_chg5p"], df["p_chg5p"] = df["c_chg5"] / cb * 100, df["p_chg5"] / pb * 100
    df["c_dl5"], df["p_dl5"] = df["c_ltp"] - g(2), df["p_ltp"] - g(3)
    return df


def _sig(oi_pct, dltp, is_call, thr):
    """OI% और प्रीमियम बदलाव से सिग्नल. Call: OI घटे + प्रीमियम बढ़े = Break Out.
    Put: OI घटे + प्रीमियम बढ़े = Break Down."""
    if oi_pct != oi_pct or dltp != dltp:
        return "-", "inherit"
    up_oi, dn_oi = oi_pct >= thr, oi_pct <= -thr
    up_p, dn_p = dltp > 0, dltp < 0
    if is_call:
        if dn_oi and up_p:
            return "Break Out", "#15803d"
        if up_oi and dn_p:
            return "Resistance", "#dc2626"
        if up_oi and up_p:
            return "Call Buy", "#16a34a"
    else:
        if dn_oi and up_p:
            return "Break Down", "#b91c1c"
        if up_oi and dn_p:
            return "Support", "#16a34a"
        if up_oi and up_p:
            return "Put Buy", "#dc2626"
    if dn_oi and dn_p:
        return "Unwind", "#6b7280"
    return "-", "inherit"


def add_signals(df, base):
    """हर स्ट्राइक पर Break Out / Break Down वगैरह. base न हो तो दिन भर के बदलाव से."""
    df = df.copy()
    five = bool(base)
    thr = 0.5 if five else 2.0
    nan = float("nan")
    for side, is_call in (("c", True), ("p", False)):
        sigs, cols, subs = [], [], []
        for _, r in df.iterrows():
            if five:
                pct, dl = r[f"{side}_chg5p"], r[f"{side}_dl5"]
            else:
                prev = r[f"{side}_oi"] - r[f"{side}_chg"]
                pct = r[f"{side}_chg"] / prev * 100 if prev > 0 else nan
                dl = r[f"{side}_ltp"] - r[f"{side}_cl"] if r[f"{side}_cl"] else nan
            sg, cl = _sig(pct, dl, is_call, thr)
            sigs.append(sg)
            cols.append(cl)
            subs.append(f"OI {pct:+.1f}%" if pct == pct else "")
        df[f"{side}_sig"], df[f"{side}_sigcol"], df[f"{side}_sigsub"] = sigs, cols, subs
    return df


# ---------------------------------------------------------------- AI विश्लेषण
def _wavg(sub, col, k=2):
    top = sub.nlargest(k, col)
    if top[col].sum() == 0:
        return float(top["strike"].mean())
    return float((top["strike"] * top[col]).sum() / top[col].sum())


def max_pain(df):
    strikes = df["strike"].values
    best, best_pain = None, None
    for k in strikes:
        pain = ((k - strikes).clip(min=0) * df["c_oi"].values).sum() + \
               ((strikes - k).clip(min=0) * df["p_oi"].values).sum()
        if best_pain is None or pain < best_pain:
            best, best_pain = k, pain
    return float(best)


def analyze(df, spot, prev_snap, age_min=0.0):
    below = df[df["strike"] <= spot]
    above = df[df["strike"] >= spot]
    out = {"alerts": [], "spot": spot}
    if below.empty or above.empty:
        return out

    sup = below.loc[below["p_oi"].idxmax()]
    res = above.loc[above["c_oi"].idxmax()]
    out.update(
        support=float(sup["strike"]), resistance=float(res["strike"]),
        support_rev=_wavg(below, "p_oi"), resistance_rev=_wavg(above, "c_oi"),
        max_pain=max_pain(df),
        pcr=df["p_oi"].sum() / max(df["c_oi"].sum(), 1e-9),
    )
    # नज़दीकी रिवर्सल प्राइस
    d_sup, d_res = spot - out["support_rev"], out["resistance_rev"] - spot
    out["near_rev"] = out["support_rev"] if d_sup < d_res else out["resistance_rev"]
    out["near_side"] = "सपोर्ट" if d_sup < d_res else "रेजिस्टेंस"

    cmax, pmax = max(df["c_oi"].max(), 1e-9), max(df["p_oi"].max(), 1e-9)
    A = out["alerts"]

    # 1) रेजिस्टेंस / सपोर्ट की ताकत
    if res["c_chg"] < 0:
        A.append(("warn", f"रेजिस्टेंस {res['strike']:.0f} कमजोर: Call OI घट रहा है -> ब्रेकआउट संभव"))
    else:
        A.append(("ok", f"रेजिस्टेंस {res['strike']:.0f} मजबूत: Call OI बढ़ रहा है"))
    if sup["p_chg"] < 0:
        A.append(("warn", f"सपोर्ट {sup['strike']:.0f} कमजोर: Put OI घट रहा है -> ब्रेकडाउन संभव"))
    else:
        A.append(("ok", f"सपोर्ट {sup['strike']:.0f} मजबूत: Put OI बढ़ रहा है"))

    if "c_sig" in df.columns:
        if df.loc[df["strike"] == res["strike"], "c_sig"].iloc[0] == "Break Out":
            A.append(("warn", f"रेजिस्टेंस {res['strike']:.0f} पर Break Out सिग्नल: Call OI घट रहा, प्रीमियम बढ़ रहा"))
        if df.loc[df["strike"] == sup["strike"], "p_sig"].iloc[0] == "Break Down":
            A.append(("warn", f"सपोर्ट {sup['strike']:.0f} पर Break Down सिग्नल: Put OI घट रहा, प्रीमियम बढ़ रहा"))

    # 2) रेजिस्टेंस के ऊपर % बढ़ा क्या? / सपोर्ट के नीचे % बढ़ा क्या?
    for _, r in df[df["strike"] > res["strike"]].iterrows():
        if r["c_chg"] > 0 and r["c_oi"] / cmax >= 0.5:
            A.append(("warn", f"{r['strike']:.0f} पर Call OI {r['c_oi']/cmax:.0%} और बढ़ रहा है -> रेजिस्टेंस ऊपर खिसक रहा"))
    for _, r in df[df["strike"] < sup["strike"]].iterrows():
        if r["p_chg"] > 0 and r["p_oi"] / pmax >= 0.5:
            A.append(("warn", f"{r['strike']:.0f} पर Put OI {r['p_oi']/pmax:.0%} और बढ़ रहा है -> सपोर्ट नीचे खिसक रहा"))

    # 3) पिछले रिफ्रेश से OI में बदलाव (लाइव)
    if prev_snap:
        for _, r in df.iterrows():
            k = r["strike"]
            if k in prev_snap:
                pc, pp = prev_snap[k][:2]
                if pc > 0 and (r["c_oi"] - pc) / pc >= 0.05 and r["c_oi"] / cmax >= 0.3:
                    A.append(("info", f"लाइव: {k:.0f} Call OI पिछले {age_min:.0f} मिनट में +{(r['c_oi']-pc)/pc:.1%}"))
                if pp > 0 and (r["p_oi"] - pp) / pp >= 0.05 and r["p_oi"] / pmax >= 0.3:
                    A.append(("info", f"लाइव: {k:.0f} Put OI पिछले {age_min:.0f} मिनट में +{(r['p_oi']-pp)/pp:.1%}"))

    # 4) कुल झुकाव
    if out["pcr"] >= 1.2:
        out["bias"] = "तेज़ी (Bullish) - Put OI ज्यादा"
    elif out["pcr"] <= 0.8:
        out["bias"] = "मंदी (Bearish) - Call OI ज्यादा"
    else:
        out["bias"] = "साइडवेज़ / न्यूट्रल"
    return out



# ---------------------------------------------------------------- Liquidity (ऑप्शन चेन)
def add_liquidity(df):
    """हर स्ट्राइक का liquidity स्कोर: 40% OI% + 40% Volume% + 20% bid-ask spread (5%+ spread = 0)."""
    df = df.copy()
    for s_ in ("c", "p"):
        oi = df[f"{s_}_oi"] / max(df[f"{s_}_oi"].max(), 1e-9) * 100
        vol = df[f"{s_}_vol"] / max(df[f"{s_}_vol"].max(), 1) * 100
        bid, ask = df[f"{s_}_bid"], df[f"{s_}_ask"]
        ok = (bid > 0) & (ask > 0) & (ask >= bid)
        spr = ((ask - bid) / ((ask + bid) / 2) * 100).where(ok)
        spr_score = (1 - spr / 5).clip(0, 1) * 100
        score = (0.4 * oi + 0.4 * vol + 0.2 * spr_score).where(ok, 0.5 * oi + 0.5 * vol)
        df[f"{s_}_spr"], df[f"{s_}_liq"] = spr, score.round()
    return df


def _pick(a, b):
    return a if a == a else b


def liquidity_chain(df, spot, ana):
    """OI वॉल (ऊपर/नीचे की liquidity), वॉल sweep, max-pain pinning, और सबसे अच्छी liquidity वाले स्ट्राइक."""
    out = {"alerts": [], "call_walls": [], "put_walls": [], "best_ce": None, "best_pe": None}
    if "support" not in ana:
        return out
    cmax, pmax = max(df["c_oi"].max(), 1e-9), max(df["p_oi"].max(), 1e-9)
    nan = float("nan")

    def mk(sub, col, chg, mx):
        return [dict(strike=float(r["strike"]), oi=float(r[col]), pct=float(r[col] / mx * 100), chg=float(r[chg]))
                for _, r in sub.iterrows()]

    cw = df[df["strike"] >= spot].nlargest(2, "c_oi").sort_values("strike")
    pw = df[df["strike"] <= spot].nlargest(2, "p_oi").sort_values("strike", ascending=False)
    out["call_walls"], out["put_walls"] = mk(cw, "c_oi", "c_chg", cmax), mk(pw, "p_oi", "p_chg", pmax)
    A = out["alerts"]

    near = spot * 0.0015
    for w in out["call_walls"]:
        if abs(w["strike"] - spot) <= near:
            A.append(("info", f"Call वॉल {w['strike']:.0f} के पास: ऊपर की liquidity, यहाँ रुकावट/रिवर्सल या sweep संभव"))
    for w in out["put_walls"]:
        if abs(w["strike"] - spot) <= near:
            A.append(("info", f"Put वॉल {w['strike']:.0f} के पास: नीचे की liquidity, यहाँ सपोर्ट/रिवर्सल या sweep संभव"))

    for _, r in df.iterrows():
        k = r["strike"]
        cchg = _pick(r.get("c_chg5", nan), r["c_chg"])
        pchg = _pick(r.get("p_chg5", nan), r["p_chg"])
        if k < spot <= k + 100 and r["c_oi"] / cmax >= 0.6 and cchg < 0:
            A.append(("warn", f"Call वॉल {k:.0f} के ऊपर प्राइस निकल गया और वहाँ Call OI घट रहा: ऊपर की liquidity sweep / ब्रेकआउट"))
        if k - 100 <= spot < k and r["p_oi"] / pmax >= 0.6 and pchg < 0:
            A.append(("warn", f"Put वॉल {k:.0f} के नीचे प्राइस निकल गया और वहाँ Put OI घट रहा: नीचे की liquidity sweep / ब्रेकडाउन"))

    mp = ana.get("max_pain")
    if mp and abs(spot - mp) <= 40:
        A.append(("info", f"प्राइस Max Pain {mp:.0f} के पास है: expiry के आसपास pinning (वहीं अटकना) संभव"))

    near_atm = df[(df["strike"] - spot).abs() <= 250]
    if not near_atm.empty and "c_liq" in df.columns:
        out["best_ce"] = near_atm.loc[near_atm["c_liq"].idxmax()]
        out["best_pe"] = near_atm.loc[near_atm["p_liq"].idxmax()]
    return out


# ---------------------------------------------------------------- Liquidity (चार्ट)
def _clusters(idx, vals, tol):
    pts = sorted(idx, key=lambda i: vals[i])
    groups, cur = [], []
    for i in pts:
        if cur and abs(vals[i] - vals[cur[-1]]) > tol:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return [g for g in groups if len(g) >= 2]


def _level_status(price, start, h, l, c, n, high=True):
    """start के बाद: close पार = broken, सिर्फ़ wick पार और close वापस = swept, वरना untaken."""
    for m in range(start, n):
        if high:
            if c[m] > price:
                return "broken", m
            if h[m] > price:
                return "swept", m
        else:
            if c[m] < price:
                return "broken", m
            if l[m] < price:
                return "swept", m
    return "untaken", None


def detect_liquidity(df, spot, tf=1, recent=40):
    """Equal Highs/Lows (EQH/EQL) और पिछले दिन का High/Low (PDH/PDL), और उन पर sweep."""
    n = len(df)
    if n < 40:
        return []
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    tol = max(0.3 * atr, 1.0)
    k = 5 if tf == 1 else 3

    def swings(vals, high):
        out, last = [], -10 ** 9
        for i in range(k, n - k):
            w = vals[i - k:i + k + 1]
            if (vals[i] >= w.max() if high else vals[i] <= w.min()) and i - last >= k:
                out.append(i)
                last = i
        return out

    levels = []
    for high, kind in ((True, "EQH"), (False, "EQL")):
        vals = h if high else l
        found = []
        for g in _clusters(swings(vals, high), vals, tol):
            price = float(max(vals[i] for i in g) if high else min(vals[i] for i in g))
            status, mi = _level_status(price, max(g) + 1, h, l, c, n, high)
            if status == "broken" or (status == "swept" and n - mi > recent):
                continue
            found.append(dict(kind=kind, price=price, count=len(g), status=status, sweep_i=mi))
        levels += sorted(found, key=lambda x: abs(x["price"] - spot))[:3]

    dates = df["time"].dt.date.values
    days = sorted(set(dates))
    if len(days) >= 2:
        prev_mask = dates == days[-2]
        today_start = int(np.argmax(dates == days[-1]))
        for kind, price, high in (("PDH", float(h[prev_mask].max()), True), ("PDL", float(l[prev_mask].min()), False)):
            status, mi = _level_status(price, today_start, h, l, c, n, high)
            levels.append(dict(kind=kind, price=price, count=1, status=status, sweep_i=mi))
    return levels


def liquidity_messages(levels, zones, ana, df):
    msgs = []
    sup, res = ana.get("support"), ana.get("resistance")
    for lv in levels:
        p, kind = lv["price"], lv["kind"]
        up_side = kind in ("EQH", "PDH")
        if lv["status"] == "swept":
            t = df["time"].iloc[lv["sweep_i"]].strftime("%H:%M")
            if up_side:
                txt = (f"ऊपर की liquidity ({kind} {p:.0f}) sweep हुई ({t}): wick ऊपर गई, close वापस नीचे. "
                       "ऊपर के stops उड़े, नीचे रिवर्सल का संकेत (पक्का नहीं)")
            else:
                txt = (f"नीचे की liquidity ({kind} {p:.0f}) sweep हुई ({t}): wick नीचे गई, close वापस ऊपर. "
                       "नीचे के stops उड़े, ऊपर रिवर्सल का संकेत (पक्का नहीं)")
            extra = []
            if (up_side and res and abs(p - res) <= 30) or ((not up_side) and sup and abs(p - sup) <= 30):
                extra.append("OI सपोर्ट/रेजिस्टेंस")
            for z in zones:
                if z["low"] - 5 <= p <= z["high"] + 5:
                    extra.append(z["label"])
            if extra:
                txt += " | मेल: " + ", ".join(extra)
            msgs.append(("warn", txt))
        elif lv["status"] == "broken":
            msgs.append(("info", f"{kind} {p:.0f} के पार close हुआ: उस तरफ़ breakout (sweep नहीं)"))
        elif kind in ("EQH", "EQL") and abs(p - ana["spot"]) <= ana["spot"] * 0.003:
            side = "ऊपर" if up_side else "नीचे"
            msgs.append(("info", f"{kind} {p:.0f} नज़दीक ({lv['count']} बार बना): {side} की liquidity अभी बाकी है"))
    return msgs

# ---------------------------------------------------------------- HTML टेबल
def _cell(text, pct=None, color=None, side="right", extra=""):
    style = ""
    if pct is not None and color:
        direction = "to left" if side == "left" else "to right"
        w = max(min(pct, 100), 0)
        style = f"background:linear-gradient({direction},{color} {w}%,transparent {w}%);"
    sub = f"<div class='pc'>({pct:.1f}%)</div>" if pct is not None else ""
    if pct is not None and pct >= 99.5:
        text = f"<b>{text}</b>"
    return f"<td style='{style}{extra}'>{text}{sub}</td>"


def _chg(v):
    col = "#16a34a" if v > 0 else "#dc2626" if v < 0 else "inherit"
    return f"<span style='color:{col}'>{v:+.1f}</span>"


def _sigcell(sig, col, sub):
    if sig == "-":
        return "<td>-</td>"
    big = sig in ("Break Out", "Break Down")
    bg = f"background:{col};" if big else ""
    fg = "#fff" if big else col
    sfg = "#fff" if big else "#555"
    return (f"<td style='{bg}'><span style='color:{fg};font-weight:700'>{sig}</span>"
            f"<div class='pc' style='color:{sfg}'>({sub})</div></td>")


def _ltpcell(r, side):
    liq, spr = r.get(f"{side}_liq"), r.get(f"{side}_spr")
    sub = ""
    if liq is not None and liq == liq:
        sub = f"L{liq:.0f}" + (f" · {spr:.1f}%" if spr is not None and spr == spr else "")
        sub = f"<div class='pc'>{sub}</div>"
    return f"<td>{r[f'{side}_ltp']:.2f}{sub}</td>"


def table_html(df, atm, ana, age_min=0.0):
    cv, pv = max(df["c_vol"].max(), 1), max(df["p_vol"].max(), 1)
    co, po = max(df["c_oi"].max(), 1e-9), max(df["p_oi"].max(), 1e-9)
    RED, GRN = "rgba(239,68,68,.35)", "rgba(34,197,94,.35)"
    h = """<style>
    .oc{width:100%;border-collapse:collapse;font-size:12px;text-align:center}
    .oc th{background:#f3f4f6;color:#111;padding:4px;position:sticky;top:0}
    .oc td{padding:3px 4px;border-bottom:1px solid #e5e7eb;color:#111}
    .oc .pc{font-size:9px;color:#555}
    </style><div style='overflow-x:auto'><table class='oc'><tr>
    <th>LTP</th><th>Signal</th><th>OI-lakh</th><th>Vol</th><th>Strike</th>
    <th>Vol</th><th>OI-lakh</th><th>Signal</th><th>LTP</th></tr>"""
    for _, r in df.iterrows():
        k = r["strike"]
        tag = ""
        if k == ana.get("support"):
            tag = " <b style='color:#16a34a'>S</b>"
        if k == ana.get("resistance"):
            tag += " <b style='color:#dc2626'>R</b>"
        kst = "background:#bfdbfe;font-weight:700;" if k == atm else "font-weight:600;"
        h += "<tr>"
        h += _ltpcell(r, "c")
        h += _sigcell(r["c_sig"], r["c_sigcol"], r["c_sigsub"])
        h += _cell(f"{r['c_oi']:.1f}", r["c_oi"] / co * 100, RED, "left")
        h += _cell(f"{r['c_vol']:,.0f}", r["c_vol"] / cv * 100, RED, "left")
        h += f"<td style='{kst}'>{k:.0f}{tag}</td>"
        h += _cell(f"{r['p_vol']:,.0f}", r["p_vol"] / pv * 100, GRN, "right")
        h += _cell(f"{r['p_oi']:.1f}", r["p_oi"] / po * 100, GRN, "right")
        h += _sigcell(r["p_sig"], r["p_sigcol"], r["p_sigsub"])
        h += _ltpcell(r, "p")
        h += "</tr>"
    return h + "</table></div>"


# ---------------------------------------------------------------- चार्ट
def _levels(fig, ana, horizontal=False):
    lines = [("स्पॉट", ana["spot"], "#2563eb"), ("सपोर्ट", ana.get("support"), "#16a34a"),
             ("रेजिस्टेंस", ana.get("resistance"), "#dc2626")]
    for name, v, c in lines:
        if v is None:
            continue
        if horizontal:
            fig.add_hline(y=v, line_dash="dash", line_color=c, annotation_text=f"{name} {v:.0f}")
        else:
            fig.add_vline(x=v, line_dash="dash", line_color=c, annotation_text=name)


def oi_chart(df, ana):
    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["c_oi"], name="Call OI", marker_color="#ef4444")
    fig.add_bar(x=df["strike"], y=df["p_oi"], name="Put OI", marker_color="#22c55e")
    _levels(fig, ana)
    fig.update_layout(barmode="group", title="OI (लाख) - Call vs Put", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def chg_chart(df, ana):
    fig = go.Figure()
    fig.add_bar(x=df["strike"], y=df["c_chg"], name="Call OI Chg", marker_color="#ef4444")
    fig.add_bar(x=df["strike"], y=df["p_chg"], name="Put OI Chg", marker_color="#22c55e")
    _levels(fig, ana)
    fig.update_layout(barmode="group", title="OI Change (लाख)", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def price_chart(cdf, ana, zones, show_bars=300, liq=None, walls=None):
    d = cdf.tail(show_bars).reset_index(drop=True)
    off = len(cdf) - len(d)
    x = list(range(len(d)))
    ext = max(int(len(d) * 0.25), 10)
    fig = go.Figure(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"],
                                   close=d["close"], name="Nifty"))
    col = {"supply": ("rgba(239,68,68,.25)", "#dc2626"), "demand": ("rgba(34,197,94,.25)", "#16a34a")}
    for z in zones:
        fill, line = col[z["kind"]]
        x0 = max(z["start"] - off, 0)
        fig.add_shape(type="rect", x0=x0, x1=len(d) + ext - 1, y0=z["low"], y1=z["high"],
                      fillcolor=fill, line=dict(color=line, width=1, dash="dash" if z.get("src") == "OI" else "solid"), layer="below")
        fig.add_annotation(x=len(d) + ext - 1, y=(z["low"] + z["high"]) / 2, xanchor="right",
                           text=f"{z['label']} {z['low']:.0f}-{z['high']:.0f} | {z['high'] - z['low']:.0f}pt ({z['score']})", showarrow=False,
                           font=dict(size=10, color="white"), bgcolor=line)
    for name, v, c in [("OI सपोर्ट", ana.get("support"), "#16a34a"), ("OI रेजिस्टेंस", ana.get("resistance"), "#dc2626")]:
        if v:
            fig.add_hline(y=v, line_dash="dash", line_color=c, annotation_text=f"{name} {v:.0f}", annotation_position="left")
    for lv in (liq or []):
        txt = f"{lv['kind']} {lv['price']:.0f}" + (f" (x{lv['count']})" if lv["count"] > 1 else "")
        if lv["status"] == "broken":
            txt += " टूटा"
        fig.add_hline(y=lv["price"], line_dash="dot", line_color="#7c3aed", line_width=1,
                      annotation_text=txt, annotation_position="top left", annotation_font_size=9)
        if lv["status"] == "swept" and lv["sweep_i"] is not None and lv["sweep_i"] - off >= 0:
            up = lv["kind"] in ("EQH", "PDH")
            fig.add_scatter(x=[lv["sweep_i"] - off], y=[lv["price"]], mode="markers",
                            marker=dict(symbol="triangle-down" if up else "triangle-up", size=12, color="#7c3aed"),
                            showlegend=False, hoverinfo="skip")
    for w in (walls or {}).get("call_walls", []):
        fig.add_hline(y=w["strike"], line_dash="dot", line_color="#ef4444", line_width=1,
                      annotation_text=f"Call वॉल {w['strike']:.0f}", annotation_position="bottom left", annotation_font_size=9)
    for w in (walls or {}).get("put_walls", []):
        fig.add_hline(y=w["strike"], line_dash="dot", line_color="#22c55e", line_width=1,
                      annotation_text=f"Put वॉल {w['strike']:.0f}", annotation_position="bottom left", annotation_font_size=9)
    step = max(len(d) // 8, 1)
    fig.update_xaxes(tickvals=x[::step], ticktext=[t.strftime("%d %H:%M") for t in d["time"].iloc[::step]],
                     range=[-1, len(d) + ext])
    fig.update_layout(title="Nifty + सप्लाई/डिमांड ज़ोन", height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


# ---------------------------------------------------------------- UI
def _show(kind, msg):
    {"ok": st.success, "warn": st.warning, "info": st.info}[kind](msg)


# ---------------------------------------------------------------- फोकस मोड के फ़िल्टर
def focus_view(df, atm, k=4):
    """टेबल में ATM के ऊपर-नीचे सिर्फ़ k स्ट्राइक (विश्लेषण पूरी 10-10 विंडो पर ही चलता है)."""
    i = int((df["strike"] - atm).abs().idxmin())
    return df.iloc[max(i - k, 0): i + k + 1].reset_index(drop=True)


def focus_zones(zones, spot):
    """हर तरफ़ से स्पॉट का सबसे नज़दीकी 1 ज़ोन (1 Supply + 1 Demand)."""
    out = []
    for kind in ("supply", "demand"):
        zs = [z for z in zones if z["kind"] == kind]
        if zs:
            out.append(min(zs, key=lambda z: max(0.0, z["low"] - spot) if kind == "supply" else max(0.0, spot - z["high"])))
    return out


def focus_liq(levels, spot):
    """ऊपर और नीचे का सबसे नज़दीकी liquidity लेवल, और हाल के सारे sweep."""
    above = [x for x in levels if x["price"] >= spot and x["status"] != "swept"]
    below = [x for x in levels if x["price"] < spot and x["status"] != "swept"]
    out = [x for x in levels if x["status"] == "swept"]
    if above:
        out.append(min(above, key=lambda x: x["price"] - spot))
    if below:
        out.append(min(below, key=lambda x: spot - x["price"]))
    return out


def focus_walls(lc):
    return {"call_walls": lc.get("call_walls", [])[:1], "put_walls": lc.get("put_walls", [])[:1]}


def ai_block(akey, zones, ana, lc, spot, liq):
    if st.button("AI से इन ज़ोन का विश्लेषण कराओ"):
        if not akey:
            st.warning("साइडबार में Anthropic API Key डालें.")
        else:
            with st.spinner("AI विश्लेषण कर रहा है..."):
                try:
                    txt = ai_commentary(akey, build_ai_payload(zones, ana, lc, spot, liq))
                    st.session_state["ai_text"] = (time.strftime("%H:%M:%S"), txt)
                except Exception as e:
                    st.error(f"AI से जवाब नहीं आया: {e}")
    if "ai_text" in st.session_state:
        ts_, txt_ = st.session_state["ai_text"]
        st.info(txt_)
        st.caption(f"यह विश्लेषण {ts_} के डेटा पर है (ऑटो रिफ्रेश पर अपने आप नहीं बदलता). "
                   "AI सिर्फ़ ऊपर के ज़ोन/लेवल की व्याख्या करता है, नए लेवल नहीं बनाता.")


def main():
    st.set_page_config(page_title="Nifty Option Chain", layout="wide")
    st.title("Nifty लाइव ऑप्शन चेन")

    token = st.sidebar.text_input("Upstox Access Token", type="password", value=_secret("UPSTOX_TOKEN"))
    akey = st.sidebar.text_input("Anthropic API Key (AI विश्लेषण के लिए, वैकल्पिक)", type="password",
                                 value=_secret("ANTHROPIC_API_KEY"))
    focus = st.sidebar.radio("मोड", ["फोकस", "फुल"], index=0,
                             help="फोकस: सिर्फ़ ज़रूरी चीज़ें (ATM के पास 4-4 स्ट्राइक, सबसे नज़दीकी ज़ोन/लेवल, सिर्फ़ ज़रूरी अलर्ट). "
                                  "फुल: सब कुछ.") == "फोकस"
    min_w = st.sidebar.number_input("ज़ोन की न्यूनतम चौड़ाई (पॉइंट)", 0, 50, 8,
                                    help="इससे पतले ज़ोन छोड़ दिए जाते हैं. 5/15 मिनट में 8 ठीक है, 1 मिनट में 3-5 रखें. 0 = बंद")
    widen = st.sidebar.checkbox("ज़ोन को impulse की open तक बढ़ाओ", value=False,
                                help="ज़ोन थोड़ा चौड़ा हो जाता है (Demand का ऊपरी और Supply का निचला किनारा)")
    secs = st.sidebar.selectbox("ऑटो रिफ्रेश", [0, 10, 30, 60], index=2, format_func=lambda s: "बंद" if s == 0 else f"{s} सेकंड")
    if secs:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=secs * 1000, key="auto")
        except ImportError:
            st.sidebar.warning("ऑटो रिफ्रेश के लिए: pip install streamlit-autorefresh")

    if not token:
        st.info("साइडबार में Upstox token डालें (token रोज़ expire होता है).")
        return

    try:
        expiries = get_expiries(token)
        expiry = st.sidebar.selectbox("Expiry", expiries)
        raw = get_chain(token, expiry)
    except Exception as e:
        st.error(f"Upstox से डेटा नहीं आया: {e}")
        return

    full, spot = build_df(raw)
    df, atm = window(full, spot)
    base, age = record_and_baseline(expiry, full)
    df = add_liquidity(add_signals(add_5m(df, base), base))
    ana = analyze(df, spot, base, age)
    lc = liquidity_chain(df, spot, ana)

    m = st.columns(4)
    m[0].metric("Nifty", f"{spot:,.2f}")
    m[1].metric("PCR", f"{ana.get('pcr', 0):.2f}")
    m[2].metric("Max Pain", f"{ana.get('max_pain', 0):.0f}")
    m[3].metric("झुकाव", ana.get("bias", "-"))

    t1, t2 = st.tabs(["ऑप्शन चेन", "चार्ट"])
    with t1:
        if focus:
            if "support" in ana:
                st.markdown(f"**सपोर्ट {ana['support']:.0f}** (रिवर्सल {ana['support_rev']:.0f}) | "
                            f"**रेजिस्टेंस {ana['resistance']:.0f}** (रिवर्सल {ana['resistance_rev']:.0f}) | "
                            f"नज़दीकी रिवर्सल **{ana['near_rev']:.0f}** ({ana['near_side']})")
                for msg in [m_ for k_, m_ in ana["alerts"] + lc["alerts"] if k_ == "warn"][:3]:
                    st.warning(msg)
                if lc["call_walls"]:
                    line = ("Call वॉल " + ", ".join(f"{x['strike']:.0f}" for x in lc["call_walls"]) + " | Put वॉल "
                            + ", ".join(f"{x['strike']:.0f}" for x in lc["put_walls"]))
                    if lc["best_ce"] is not None:
                        line += (f" | बेस्ट liquidity: CE {lc['best_ce']['strike']:.0f}, "
                                 f"PE {lc['best_pe']['strike']:.0f}")
                    st.caption(line)
            st.markdown(table_html(focus_view(df, atm), atm, ana, age), unsafe_allow_html=True)
            st.caption("फोकस मोड: ATM के पास 4-4 स्ट्राइक. पूरी चेन, सारे अलर्ट और समझाने वाले नोट के लिए साइडबार में 'फुल' चुनें.")
        else:
            if "support" in ana:
                c = st.columns(3)
                c[0].success(f"सपोर्ट: {ana['support']:.0f}\n\nरिवर्सल ज़ोन: {ana['support_rev']:.0f}")
                c[1].error(f"रेजिस्टेंस: {ana['resistance']:.0f}\n\nरिवर्सल ज़ोन: {ana['resistance_rev']:.0f}")
                c[2].info(f"नज़दीकी रिवर्सल प्राइस: {ana['near_rev']:.0f} ({ana['near_side']})")
                for kind, msg in ana["alerts"]:
                    _show(kind, msg)
            if lc["call_walls"]:
                st.markdown("**ऑप्शन चेन liquidity**")
                w = st.columns(3)
                w[0].error("Call वॉल (ऊपर की liquidity)\n\n" + "\n\n".join(
                    f"{x['strike']:.0f}: OI {x['oi']:.1f} लाख ({x['pct']:.0f}%)" for x in lc["call_walls"]))
                w[1].success("Put वॉल (नीचे की liquidity)\n\n" + "\n\n".join(
                    f"{x['strike']:.0f}: OI {x['oi']:.1f} लाख ({x['pct']:.0f}%)" for x in lc["put_walls"]))
                if lc["best_ce"] is not None:
                    w[2].info(f"ट्रेड के लिए सबसे अच्छी liquidity (ATM के पास)\n\n"
                              f"CE {lc['best_ce']['strike']:.0f} (स्कोर {lc['best_ce']['c_liq']:.0f})\n\n"
                              f"PE {lc['best_pe']['strike']:.0f} (स्कोर {lc['best_pe']['p_liq']:.0f})")
                for kind, msg in lc["alerts"]:
                    _show(kind, msg)
            st.markdown(table_html(df, atm, ana, age), unsafe_allow_html=True)
            st.caption("LTP के नीचे L = स्ट्राइक liquidity स्कोर (40% OI% + 40% Volume% + 20% bid-ask spread), "
                       "बगल का % = bid-ask spread (कम बेहतर).")
            basis = "पिछले ~5 मिनट" if base else "दिन भर (5 मिनट का डेटा बनने तक)"
            st.caption(f"Signal आधार: {basis} का OI% और प्रीमियम बदलाव. "
                       "Call: OI घटे + प्रीमियम बढ़े = Break Out, OI बढ़े + प्रीमियम घटे = Resistance. "
                       "Put: OI घटे + प्रीमियम बढ़े = Break Down, OI बढ़े + प्रीमियम घटे = Support.")
            st.caption("Volume/OI के नीचे (%) = उसी तरफ (Call या Put) के सबसे ज्यादा वाले स्ट्राइक के मुकाबले.")
            st.caption("S = सपोर्ट (सबसे ज्यादा Put OI, स्पॉट के नीचे), R = रेजिस्टेंस (सबसे ज्यादा Call OI, स्पॉट के ऊपर). "
                       "रिवर्सल ज़ोन = टॉप-2 OI स्ट्राइक का OI-वेटेड औसत. यह सिर्फ़ विश्लेषण है, ट्रेड सलाह नहीं.")

    with t2:
        if not focus:
            st.plotly_chart(oi_chart(df, ana), use_container_width=True)
            st.plotly_chart(chg_chart(df, ana), use_container_width=True)
        tf = st.radio("टाइमफ्रेम", [1, 5, 15], index=1 if focus else 0, horizontal=True,
                      format_func=lambda m: f"{m} मिनट")
        try:
            raw_c = load_candles(token)
            cdf = resample(raw_c, tf)
            htf = resample(raw_c, 15) if tf < 15 else None
            zones = ai_zones(cdf, spot, ana, lc, df, htf, min_width=min_w, widen=widen)
            liq = detect_liquidity(cdf, spot, tf)
            if focus:
                zv, lv, wv = focus_zones(zones, spot), focus_liq(liq, spot), focus_walls(lc)
            else:
                zv, lv, wv = zones, liq, lc
            st.plotly_chart(price_chart(cdf, ana, zv, show_bars=120 if focus else 300, liq=lv, walls=wv),
                            use_container_width=True)
            for kind, msg in liquidity_messages(lv, zones, ana, cdf):
                if focus and kind != "warn":
                    continue
                _show(kind, msg)
            for z in zv:
                if z["low"] <= spot <= z["high"]:
                    st.warning(f"प्राइस अभी {z['label']} ({z['low']:.0f}-{z['high']:.0f}) के अंदर है")
            if not zones:
                st.info("अभी कोई वैध ज़ोन नहीं मिला.")
            elif focus:
                for z in zv:
                    st.caption(f"{z['label']} {z['low']:.0f}-{z['high']:.0f} ({z['high'] - z['low']:.0f} पॉइंट, स्कोर {z['score']}): "
                               + " | ".join(z["reasons"][:3]))
                with st.expander("AI विश्लेषण (Claude)"):
                    ai_block(akey, zones, ana, lc, spot, liq)
            else:
                st.caption("बैंगनी बिंदुदार लाइन = चार्ट liquidity: EQH/EQL (बराबर high/low, ऊपर-नीचे stops जमा), "
                           "PDH/PDL (पिछले दिन का high/low). त्रिकोण = sweep हुआ. लाल/हरी बिंदुदार = Call/Put OI वॉल. "
                           "यह अंदाज़ा है, पक्का रिवर्सल नहीं.")
                st.dataframe(pd.DataFrame([{
                    "ज़ोन": z["label"], "Low": round(z["low"]), "High": round(z["high"]),
                    "चौड़ाई (पॉइंट)": round(z["high"] - z["low"]), "स्कोर": z["score"], "पैटर्न": z["pattern"], "टच": z["touches"],
                    "कारण": " | ".join(z["reasons"])} for z in zones]),
                    hide_index=True, use_container_width=True)
                st.caption("स्कोर (0-100) = Impulse ताकत 25 + Fresh/Tested 20 + Imbalance 10 + Liquidity grab 10 "
                           "+ 15 मिनट ज़ोन से मेल 10 + नयापन 5 + ऑप्शन चेन OI मेल 20. "
                           "डैश वाला बॉक्स = सिर्फ़ OI वॉल से बना ज़ोन (प्राइस-एक्शन ज़ोन नहीं मिला). "
                           "यह मेरा अपना नियम-सेट है, किसी इंडिकेटर की कॉपी नहीं.")
                st.markdown("**AI विश्लेषण (Claude)**")
                ai_block(akey, zones, ana, lc, spot, liq)
        except Exception as e:
            st.warning(f"प्राइस चार्ट नहीं आया: {e}")


if __name__ == "__main__":
    main()

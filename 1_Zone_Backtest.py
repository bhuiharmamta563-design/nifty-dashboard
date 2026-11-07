"""
Nifty Supply/Demand Zone Backtest (Streamlit page)
नियम (बिना future देखे):
 1. ज़ोन तभी बनता है जब impulse कैंडल बंद हो जाए (ATR x multiplier से बड़ी, बेस के बाद).
 2. ज़ोन में पहली बार प्राइस लौटे तो एंट्री (Demand = Buy, Supply = Sell).
 3. स्टॉप = ज़ोन के दूसरी तरफ + थोड़ा बफर, टारगेट = RR x रिस्क.
 4. स्टॉप और टारगेट एक ही कैंडल में हों तो स्टॉप माना जाता है (सख़्त नियम).
 5. दिन के आख़िर तक न स्टॉप न टारगेट लगे तो दिन के आख़िरी कैंडल पर exit.
 6. एक समय में एक ही ट्रेड.
नतीजे Nifty इंडेक्स के पॉइंट्स/R में हैं, ऑप्शन प्रीमियम में नहीं.

इस फ़ाइल को nifty_chain.py वाले फ़ोल्डर में रखें.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nifty_chain import _secret, get_hist_candles, resample


def zone_backtest(df, atr_mult=1.5, rr=2.0, cost_pts=2.0, buf=0.1, max_wait=150, min_risk=5.0,
                  min_width=0.0, widen=False):
    if len(df) < 40:
        return pd.DataFrame()
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    day = df["time"].dt.date.values
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values
    n = len(df)
    last_of_day = {}
    for i, d in enumerate(day):
        last_of_day[d] = i

    trades = []
    for i in range(15, n):
        a = atr[i - 1]
        if not a == a or a <= 0:
            continue
        if abs(c[i] - o[i]) < atr_mult * a or (h[i - 1] - l[i - 1]) > a:
            continue
        base = [i - 1]
        while len(base) < 3 and base[-1] >= 1 and (h[base[-1] - 1] - l[base[-1] - 1]) <= 0.8 * a:
            base.append(base[-1] - 1)
        zl, zh = min(l[b] for b in base), max(h[b] for b in base)
        demand = c[i] > o[i]
        if widen:
            if demand:
                zh = max(zh, o[i])
            else:
                zl = min(zl, o[i])
        if zh - zl < min_width:
            continue

        k = None
        for j in range(i + 1, min(n, i + 1 + max_wait)):
            if (demand and l[j] <= zh) or ((not demand) and h[j] >= zl):
                k = j
                break
        if k is None:
            continue

        entry = min(o[k], zh) if demand else max(o[k], zl)
        stop = zl - buf * a if demand else zh + buf * a
        risk = (entry - stop) if demand else (stop - entry)
        if risk < min_risk:
            continue
        target = entry + rr * risk if demand else entry - rr * risk

        end = last_of_day[day[k]]
        exit_i, exit_p, res = end, c[end], "EOD"
        for m in range(k, end + 1):
            if demand:
                if l[m] <= stop:
                    exit_i, exit_p, res = m, stop, "SL"
                    break
                if h[m] >= target:
                    exit_i, exit_p, res = m, target, "TP"
                    break
            else:
                if h[m] >= stop:
                    exit_i, exit_p, res = m, stop, "SL"
                    break
                if l[m] <= target:
                    exit_i, exit_p, res = m, target, "TP"
                    break
        pnl = (exit_p - entry) if demand else (entry - exit_p)
        trades.append(dict(entry_i=k, exit_i=exit_i, side="BUY" if demand else "SELL",
                           entry_time=df["time"].iloc[k], exit_time=df["time"].iloc[exit_i],
                           entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
                           exit=round(exit_p, 2), result=res, risk=round(risk, 2),
                           R=round((pnl - cost_pts) / risk, 3)))

    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame(trades).sort_values("entry_i")
    keep, last_exit = [], -1
    for _, r in t.iterrows():
        if r["entry_i"] > last_exit:
            keep.append(r)
            last_exit = r["exit_i"]
    return pd.DataFrame(keep).reset_index(drop=True)


def stats(t):
    if t.empty:
        return dict(trades=0, win=0.0, total_R=0.0, avg_R=0.0, max_dd=0.0, pf=0.0)
    eq = t["R"].cumsum()
    gp, gl = t.loc[t["R"] > 0, "R"].sum(), -t.loc[t["R"] < 0, "R"].sum()
    return dict(trades=len(t), win=round((t["R"] > 0).mean() * 100, 1), total_R=round(t["R"].sum(), 1),
                avg_R=round(t["R"].mean(), 3), max_dd=round((eq.cummax() - eq).max(), 1),
                pf=round(gp / gl, 2) if gl > 0 else float("inf"))


def main():
    st.set_page_config(page_title="Nifty Zone Backtest", layout="wide")
    st.title("Nifty सप्लाई/डिमांड ज़ोन - बैकटेस्ट")

    token = st.sidebar.text_input("Upstox Access Token", type="password",
                                  value=_secret("UPSTOX_TOKEN"))
    days = st.sidebar.slider("कितने दिन (1-मिनट डेटा)", 5, 28, 28)
    tf = st.sidebar.radio("टाइमफ्रेम", [1, 5, 15], index=1, format_func=lambda m: f"{m} मिनट")
    mult = st.sidebar.select_slider("Impulse (ATR x)", [1.2, 1.5, 2.0, 2.5, 3.0], value=1.5)
    rr = st.sidebar.select_slider("Risk:Reward", [1.0, 1.5, 2.0, 3.0], value=2.0)
    cost = st.sidebar.number_input("खर्च/स्लिपेज (पॉइंट प्रति ट्रेड)", 0.0, 20.0, 2.0, 0.5)
    min_w = st.sidebar.number_input("ज़ोन की न्यूनतम चौड़ाई (पॉइंट)", 0, 50, 0)
    widen = st.sidebar.checkbox("ज़ोन को impulse की open तक बढ़ाओ", value=False)

    if not token:
        st.info("साइडबार में Upstox token डालें.")
        return
    if not st.sidebar.button("बैकटेस्ट चलाओ", type="primary"):
        st.caption("सेटिंग चुनकर 'बैकटेस्ट चलाओ' दबाएँ.")
        return

    try:
        raw = get_hist_candles(token, days=days)
    except Exception as e:
        st.error(f"डेटा नहीं आया: {e}")
        return
    df = resample(raw.sort_values("time").drop_duplicates("time"), tf)
    st.write(f"{len(df)} कैंडल | {df['time'].min():%d %b} से {df['time'].max():%d %b}")

    t = zone_backtest(df, atr_mult=mult, rr=rr, cost_pts=cost, min_width=min_w, widen=widen)
    s = stats(t)
    m = st.columns(5)
    m[0].metric("ट्रेड", s["trades"])
    m[1].metric("जीत %", s["win"])
    m[2].metric("कुल R", s["total_R"])
    m[3].metric("औसत R/ट्रेड", s["avg_R"])
    m[4].metric("Max Drawdown (R)", s["max_dd"])
    st.caption(f"Profit factor: {s['pf']}. RR {rr} पर सिर्फ़ जीत % देखकर नहीं, कुल R और drawdown देखें.")

    if t.empty:
        st.warning("इन सेटिंग पर कोई ट्रेड नहीं बना.")
        return

    fig = go.Figure(go.Scatter(x=t["exit_time"], y=t["R"].cumsum(), mode="lines+markers"))
    fig.update_layout(title="Equity curve (R में)", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    by_side = t.groupby("side")["R"].agg(["count", "sum", "mean"]).round(2)
    by_side.columns = ["ट्रेड", "कुल R", "औसत R"]
    st.write("Buy (Demand) बनाम Sell (Supply)")
    st.dataframe(by_side, use_container_width=True)

    st.write("सेटिंग तुलना (कुल R) - Impulse x RR")
    grid = {}
    for mu in [1.2, 1.5, 2.0, 2.5, 3.0]:
        grid[f"ATR x{mu}"] = {f"RR {r}": stats(zone_backtest(df, atr_mult=mu, rr=r, cost_pts=cost, min_width=min_w, widen=widen))["total_R"]
                              for r in [1.0, 1.5, 2.0, 3.0]}
    st.dataframe(pd.DataFrame(grid).T, use_container_width=True)
    st.caption("सावधान: बेस्ट सेटिंग चुनना in-sample है, यानी उसी डेटा पर फिट. जो सेटिंग सिर्फ़ एक कोने में चले "
               "और बाकी में न चले, उस पर भरोसा न करें. असली परख नए दिनों (out-of-sample) पर होती है.")

    st.write("चौड़ाई सेटिंग की तुलना (अभी के ATR x और RR पर)")
    cmp_rows = []
    wmin = min_w if min_w > 0 else 8
    for label, mw, wd in [("सामान्य", 0, False), (f"न्यूनतम {wmin}pt", wmin, False),
                          ("open तक बढ़ाया", 0, True), (f"दोनों ({wmin}pt + open)", wmin, True)]:
        r_ = stats(zone_backtest(df, atr_mult=mult, rr=rr, cost_pts=cost, min_width=mw, widen=wd))
        cmp_rows.append({"सेटिंग": label, "ट्रेड": r_["trades"], "जीत %": r_["win"],
                         "कुल R": r_["total_R"], "Max DD (R)": r_["max_dd"]})
    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)

    st.write("सारे ट्रेड")
    st.dataframe(t.drop(columns=["entry_i", "exit_i"]), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

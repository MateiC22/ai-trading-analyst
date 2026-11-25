import os
import json
import time
import html
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from google import genai
except ImportError:
    genai = None

# ==========================================
# 1. LISTELE DE COMPANII (SUA + WORLDWIDE)
# ==========================================
TOP_50_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "LLY", "AVGO",
    "JPM", "V", "WMT", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ",
    "NFLX", "BAC", "ORCL", "CRM", "ABBV", "CVX", "KO", "MRK", "AMD", "PEP",
    "TMO", "LIN", "ACN", "MCD", "ADBE", "CSCO", "ABT", "GE", "QCOM", "INTU",
    "IBM", "TXN", "AMGN", "ISRG", "NOW", "CAT", "UBER", "GS", "DIS", "PLTR"
]

TOP_WORLDWIDE = [
    "ASML.AS",    # ASML Holding (Olanda)
    "NOVO-B.CO",  # Novo Nordisk (Danemarca)
    "MC.PA",      # LVMH (Franta)
    "SAP.DE",     # SAP (Germania)
    "SIE.DE",     # Siemens (Germania)
    "AZN.L",      # AstraZeneca (UK)
    "SHEL.L",     # Shell (UK)
    "HSBA.L",     # HSBC (UK)
    "NESN.SW",    # Nestle (Elvetia)
    "ROG.SW",     # Roche (Elvetia)
    "NOVN.SW",    # Novartis (Elvetia)
    "AIR.PA",     # Airbus (Franta)
    "RACE",       # Ferrari (Italia/US)
    "SPOT",       # Spotify (Suedia/US)
    "TSM",        # TSMC (Taiwan/US)
    "BABA",       # Alibaba (China/US)
    "MELI",       # MercadoLibre (America Latina/US)
    "SHOP",       # Shopify (Canada/US)
    "TM",         # Toyota (Japonia/US)
    "SONY"        # Sony (Japonia/US)
]

T212_EXCHANGE_SUFFIX = {
    "_US_EQ": "",
    "_DE_EQ": ".DE",
    "_NL_EQ": ".AS",
    "_FR_EQ": ".PA",
    "_UK_EQ": ".L",
    "_CH_EQ": ".SW",
    "_DK_EQ": ".CO",
    "_IT_EQ": ".MI",
    "_ES_EQ": ".MC",
    "_BE_EQ": ".BR",
    "_PT_EQ": ".LS",
    "_AT_EQ": ".VI",
    "_SE_EQ": ".ST",
    "_CA_EQ": ".TO"
}

TZ_RO = ZoneInfo("Europe/Bucharest")
TZ_NY = ZoneInfo("America/New_York")


def detect_scan_phase():
    now_ro = datetime.now(TZ_RO)
    now_ny = datetime.now(TZ_NY)
    event_name = os.getenv("GITHUB_EVENT_NAME", "manual")

    if event_name == "schedule":
        mins_ro = now_ro.hour * 60 + now_ro.minute
        in_prep = (15 * 60 + 5) <= mins_ro <= (15 * 60 + 50)
        in_pre = (15 * 60 + 55) <= mins_ro <= (16 * 60 + 32)
        in_open = (16 * 60 + 35) <= mins_ro <= (17 * 60 + 25)
        if not (in_prep or in_pre or in_open):
            return None, None, now_ro, now_ny

    ny_mins = now_ny.hour * 60 + now_ny.minute
    if ny_mins < 9 * 60:
        phase = "PREP"
        status_label = "15:30 • Pregătire Piață"
    elif ny_mins < 9 * 60 + 32:
        phase = "PREMARKET"
        status_label = "16:15 • Pre-Market Scan"
    elif ny_mins <= 11 * 60:
        phase = "OPEN_CONFIRM"
        status_label = "16:45 • Open Confirm"
    else:
        phase = "INTRADAY"
        status_label = "Live Market Scan"

    return phase, status_label, now_ro, now_ny


def load_custom_watchlist(filepath="watchlist.txt"):
    custom = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip().upper()
                if clean and not clean.startswith("#"):
                    custom.append(clean)
    return list(dict.fromkeys(custom))


def load_previous_scan(filepath="data.json"):
    """Citeste scanarea anterioara pentru a evita notificari duplicate in aceeasi zi."""
    if not os.path.exists(filepath):
        return "", {}, {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        updated_at = old_data.get("updatedAt", "")
        prev_date = updated_at.split(",")[0].strip() if "," in updated_at else ""
        prev_sig_map = {
            item["ticker"]: item.get("signal", "WAIT")
            for item in old_data.get("opportunities", [])
            if "ticker" in item
        }
        prev_exit_map = {
            item["ticker"]: item.get("exitSignal", "HOLD")
            for item in old_data.get("portfolioPositions", [])
            if "ticker" in item
        }
        return prev_date, prev_sig_map, prev_exit_map
    except Exception:
        return "", {}, {}


def t212_raw_to_yf(raw_ticker, short_name, known_tickers_by_base):
    """Mapeaza simbolul din Trading 212 (ex: PLTR_US_EQ, ASML_NL_EQ) la simbolul Yahoo Finance."""
    raw = (raw_ticker or "").upper()
    base = (short_name or raw.split("_")[0]).upper().replace(".", "-")

    if base in known_tickers_by_base:
        return known_tickers_by_base[base]

    for suffix, yf_ext in T212_EXCHANGE_SUFFIX.items():
        if raw.endswith(suffix):
            return f"{base}{yf_ext}"

    return base


def get_trading212_data(known_tickers_by_base):
    """
    Se conecteaza la Trading 212 API (Live sau Demo) pentru a extrage:
    1. Lista instrumentelor disponibile pe platforma.
    2. Pozitiile reale din portofoliu (cantitate, pret mediu, P&L) + moneda contului.
    """
    api_key = os.getenv("T212_API_KEY", "").strip()
    api_secret = os.getenv("T212_API_SECRET", "").strip()

    if not api_key:
        print("Info: T212_API_KEY nu este setat. Rulez analiza pe baza listelor standard.")
        return set(), {}, "USD"

    base_urls = ["https://live.trading212.com/api/v0", "https://demo.trading212.com/api/v0"]
    available_tickers = set()
    portfolio_map = {}
    account_currency = "USD"

    for base in base_urls:
        try:
            auth = (api_key, api_secret) if api_secret else None
            headers = {} if api_secret else {"Authorization": api_key}

            resp = requests.get(f"{base}/equity/metadata/instruments", auth=auth, headers=headers, timeout=15)
            if resp.status_code == 200:
                instruments = resp.json()
                raw_to_short = {}
                for item in instruments:
                    short_name = item.get("shortName", "").upper()
                    ticker_raw = item.get("ticker", "").upper()
                    if short_name:
                        available_tickers.add(short_name)
                    if ticker_raw:
                        available_tickers.add(ticker_raw.split("_")[0])
                        raw_to_short[ticker_raw] = short_name

                # Aflam moneda contului Trading 212 (RON, EUR, USD, GBP)
                acc_resp = requests.get(f"{base}/equity/account/info", auth=auth, headers=headers, timeout=15)
                if acc_resp.status_code == 200:
                    account_currency = acc_resp.json().get("currencyCode", "USD")

                # Citim pozitiile deschise din portofoliu
                port_resp = requests.get(f"{base}/equity/portfolio", auth=auth, headers=headers, timeout=15)
                if port_resp.status_code == 200:
                    for pos in port_resp.json():
                        raw_t = pos.get("ticker", "").upper()
                        if not raw_t:
                            continue
                        short_n = raw_to_short.get(raw_t, raw_t.split("_")[0])
                        yf_sym = t212_raw_to_yf(raw_t, short_n, known_tickers_by_base)
                        qty = float(pos.get("quantity") or 0)
                        avg_p = float(pos.get("averagePrice") or 0)
                        cur_p = float(pos.get("currentPrice") or 0)
                        ppl = float(pos.get("ppl") or 0)

                        portfolio_map[yf_sym] = {
                            "rawTicker": raw_t,
                            "yfTicker": yf_sym,
                            "quantity": round(qty, 4),
                            "avgPrice": round(avg_p, 2),
                            "t212CurrentPrice": round(cur_p, 2),
                            "ppl": round(ppl, 2),
                            "accountCurrency": account_currency
                        }

                print(
                    f"Conectat cu succes la Trading 212 ({base}). "
                    f"Instrumente: {len(available_tickers)} | Pozitii in portofoliu: {len(portfolio_map)}"
                )
                break
        except Exception as e:
            print(f"Eroare conexiune {base}: {e}")

    return available_tickers, portfolio_map, account_currency


def analyze_intraday_and_premarket(stock, prev_close, is_us, now_ny, now_ro):
    result = {
        "live_price": prev_close,
        "change_pct": 0.0,
        "premarket_pct": None,
        "open_vol_ratio": None,
        "above_vwap": None,
        "has_premarket": False,
        "has_open_15m": False
    }

    try:
        intra = stock.history(period="5d", interval="15m", prepost=True)
        if intra is None or intra.empty:
            return result

        target_tz = TZ_NY if is_us else TZ_RO
        if intra.index.tz is None:
            intra.index = intra.index.tz_localize("UTC").tz_convert(target_tz)
        else:
            intra.index = intra.index.tz_convert(target_tz)

        today_date = now_ny.date() if is_us else now_ro.date()
        today_bars = intra[intra.index.date == today_date].copy()

        if today_bars.empty:
            return result

        live_price = float(today_bars["Close"].iloc[-1])
        result["live_price"] = live_price
        if prev_close > 0:
            result["change_pct"] = round(((live_price - prev_close) / prev_close) * 100, 2)

        if is_us:
            mins_of_day = today_bars.index.hour * 60 + today_bars.index.minute
            pre_bars = today_bars[mins_of_day < (9 * 60 + 30)]
            reg_bars = today_bars[mins_of_day >= (9 * 60 + 30)]

            if not pre_bars.empty and prev_close > 0:
                last_pre_price = float(pre_bars["Close"].iloc[-1])
                result["premarket_pct"] = round(((last_pre_price - prev_close) / prev_close) * 100, 2)
                result["has_premarket"] = True

            if not reg_bars.empty:
                result["has_open_15m"] = True
                typical_price = (reg_bars["High"] + reg_bars["Low"] + reg_bars["Close"]) / 3.0
                total_vol = reg_bars["Volume"].sum()
                if total_vol > 0:
                    vwap = float((typical_price * reg_bars["Volume"]).sum() / total_vol)
                    result["above_vwap"] = live_price >= vwap

                first_bar_vol = float(reg_bars["Volume"].iloc[0])
                past_bars = intra[intra.index.date < today_date]
                if not past_bars.empty and first_bar_vol > 0:
                    past_open_bars = past_bars[
                        (past_bars.index.hour == 9) & (past_bars.index.minute == 30)
                    ]
                    if not past_open_bars.empty:
                        avg_open_vol = float(past_open_bars["Volume"].mean())
                        if avg_open_vol > 0:
                            result["open_vol_ratio"] = round(first_bar_vol / avg_open_vol, 2)
        else:
            typical_price = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3.0
            total_vol = today_bars["Volume"].sum()
            if total_vol > 0:
                vwap = float((typical_price * today_bars["Volume"]).sum() / total_vol)
                result["above_vwap"] = live_price >= vwap

    except Exception as e:
        print(f"Info intraday indisponibil: {e}")

    return result


def compute_indicators_and_backtest(df, is_us, now_ny, now_ro, intra_info, phase):
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_local = df.tz_convert(TZ_NY if is_us else TZ_RO)
    today_date = now_ny.date() if is_us else now_ro.date()

    closed_df = df_local[df_local.index.date < today_date].copy()
    if len(closed_df) < 220:
        closed_df = df_local.copy()
    if len(closed_df) < 220:
        return None

    close = closed_df["Close"]
    high = closed_df["High"]
    low = closed_df["Low"]
    volume = closed_df["Volume"]

    closed_df["SMA20"] = close.rolling(20).mean()
    closed_df["SMA50"] = close.rolling(50).mean()
    closed_df["SMA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    closed_df["RSI"] = 100 - (100 / (1 + rs))

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    closed_df["ATR"] = tr.rolling(14).mean()

    closed_df["VolRatio"] = volume / volume.rolling(20).mean()

    closed_df = closed_df.dropna().copy()
    if len(closed_df) < 200:
        return None

    latest = closed_df.iloc[-1]
    prev_close = float(latest["Close"])
    live_price = intra_info["live_price"] if intra_info["live_price"] > 0 else prev_close

    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])
    sma50 = float(latest["SMA50"])
    sma200 = float(latest["SMA200"])
    vol_ratio = float(latest["VolRatio"])

    tp_pct = min(max((2.2 * atr) / live_price, 0.025), 0.12)
    sl_pct = min(max((1.2 * atr) / live_price, 0.015), 0.06)
    rr_ratio = round(tp_pct / sl_pct, 2)

    above_200 = live_price > sma200
    above_50 = live_price > sma50
    rsi_low = rsi < 45
    rsi_mid = 45 <= rsi <= 65

    wins = 0
    losses = 0
    closes_arr = closed_df["Close"].values
    highs_arr = closed_df["High"].values
    lows_arr = closed_df["Low"].values
    rsi_arr = closed_df["RSI"].values
    sma200_arr = closed_df["SMA200"].values

    for i in range(len(closed_df) - 16):
        hist_above_200 = closes_arr[i] > sma200_arr[i]
        hist_rsi = rsi_arr[i]

        same_regime = (hist_above_200 == above_200) and (
            (rsi_low and hist_rsi < 45) or
            (rsi_mid and 45 <= hist_rsi <= 65) or
            (not rsi_low and not rsi_mid and hist_rsi > 65)
        )

        if same_regime:
            entry = closes_arr[i]
            tp_level = entry * (1 + tp_pct)
            sl_level = entry * (1 - sl_pct)

            for j in range(i + 1, i + 16):
                if lows_arr[j] <= sl_level:
                    losses += 1
                    break
                if highs_arr[j] >= tp_level:
                    wins += 1
                    break

    total_trades = wins + losses
    if total_trades >= 5:
        win_chance = int(round((wins / total_trades) * 100))
    else:
        win_chance = 55 if above_200 else 42

    win_chance = max(25, min(88, win_chance))
    loss_risk = 100 - win_chance

    score = int(win_chance * 0.62)
    if above_200:
        score += 14
    if above_50:
        score += 7
    if 34 <= rsi <= 60:
        score += 9
    elif rsi > 74:
        score -= 10
    if vol_ratio > 1.15:
        score += 4

    pre_pct = intra_info.get("premarket_pct")
    open_vol = intra_info.get("open_vol_ratio")
    above_vwap = intra_info.get("above_vwap")
    change_pct = intra_info.get("change_pct", 0.0)

    if phase == "PREMARKET" and pre_pct is not None:
        if 0.6 <= pre_pct <= 6.5 and above_200:
            score += 8
        elif pre_pct > 9.0:
            score -= 6
        elif pre_pct < -2.5:
            score -= 8

    elif phase in ("OPEN_CONFIRM", "INTRADAY"):
        if open_vol is not None and open_vol >= 1.35 and change_pct > 0.3:
            score += 8
        if above_vwap is True and change_pct > 0:
            score += 6
        elif above_vwap is False and pre_pct is not None and pre_pct > 1.0:
            score -= 9

    score = max(15, min(96, score))

    return {
        "price": live_price,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "premarket_pct": pre_pct,
        "open_vol_ratio": open_vol,
        "above_vwap": above_vwap,
        "rsi": round(rsi, 1),
        "sma50_above": above_50,
        "sma200_above": above_200,
        "vol_ratio": round(vol_ratio, 2),
        "tp_pct": round(tp_pct * 100, 1),
        "sl_pct": round(sl_pct * 100, 1),
        "rr_ratio": rr_ratio,
        "wins": wins,
        "losses": losses,
        "win_chance": win_chance,
        "loss_risk": loss_risk,
        "score": score
    }


def evaluate_portfolio_exit(ticker, name, pos_info, stats, sym):
    """
    Evalueaza o pozitie detinuta in Trading 212 si decide daca merita
    marcat profitul (Take Profit) sau protejata pozitia (Stop Loss / Trend rupt).
    """
    avg_p = pos_info["avgPrice"]
    cur_p = stats["price"]
    qty = pos_info["quantity"]
    ppl = pos_info["ppl"]
    acc_curr = pos_info["accountCurrency"]

    if avg_p > 0:
        pnl_pct = round(((cur_p - avg_p) / avg_p) * 100, 2)
    else:
        pnl_pct = 0.0

    tp_target = max(stats["tp_pct"], 5.5)
    sl_limit = max(stats["sl_pct"], 4.0)

    if pnl_pct >= tp_target:
        exit_signal = "TAKE PROFIT"
        is_alert = True
        reason = f"Profit +{pnl_pct}% — a atins ținta de Take Profit (+{stats['tp_pct']}%). Poți marca câștigul."
    elif stats["rsi"] >= 74 and pnl_pct >= 2.0:
        exit_signal = "TAKE PROFIT / TRIM"
        is_alert = True
        reason = f"Supracumpărat (RSI {stats['rsi']}) la +{pnl_pct}% profit. Risc crescut de corecție scurtă."
    elif pnl_pct <= -sl_limit:
        exit_signal = "SELL / STOP LOSS"
        is_alert = True
        reason = f"Pierdere {pnl_pct}% — a depășit pragul de protecție (-{stats['sl_pct']}%)."
    elif not stats["sma50_above"] and stats["rsi"] < 44:
        exit_signal = "SELL / PROTECT"
        is_alert = True
        reason = f"Trend mediu pierdut (sub SMA50, RSI {stats['rsi']}). Risc de continuare a scăderii."
    elif stats["above_vwap"] is False and stats["change_pct"] <= -2.2:
        exit_signal = "WATCH EXIT"
        is_alert = True
        reason = f"Presiune de vânzare intraday ({stats['change_pct']}%) sub VWAP."
    else:
        exit_signal = "HOLD"
        is_alert = False
        reason = f"Trend stabil ({'peste' if stats['sma50_above'] else 'sub'} SMA50, RSI {stats['rsi']}). Păstrează poziția."

    pnl_sign = "+" if pnl_pct >= 0 else ""
    ppl_sign = "+" if ppl >= 0 else ""

    return {
        "ticker": ticker,
        "name": name,
        "quantity": qty,
        "avgPrice": f"{sym}{avg_p:.2f}",
        "currentPrice": f"{sym}{cur_p:.2f}",
        "pnlPct": pnl_pct,
        "pnlPctFormatted": f"{pnl_sign}{pnl_pct}%",
        "pplFormatted": f"{ppl_sign}{ppl:.2f} {acc_curr}",
        "exitSignal": exit_signal,
        "isAlert": is_alert,
        "reason": reason
    }


def get_ai_news_insight(ticker, name, headlines, stats, phase):
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    live_context = ""
    if stats["premarket_pct"] is not None:
        sign = "+" if stats["premarket_pct"] >= 0 else ""
        live_context += f" Pre-market: {sign}{stats['premarket_pct']}%."
    if stats["open_vol_ratio"] is not None:
        live_context += f" Volum deschidere: {stats['open_vol_ratio']}x."
    if stats["above_vwap"] is not None:
        live_context += f" {'Peste' if stats['above_vwap'] else 'Sub'} VWAP."

    fallback_text = (
        f"Win istoric {stats['win_chance']}% ({stats['wins']}W/{stats['losses']}L). "
        f"RSI {stats['rsi']}, {'peste' if stats['sma200_above'] else 'sub'} SMA200.{live_context}"
    )

    if not gemini_key or genai is None or not headlines:
        return fallback_text

    try:
        client = genai.Client(api_key=gemini_key)
        prompt = (
            f"Esti un analist bursier concis. Analizeaza {name} ({ticker}) in maximum 2 propozitii scurte in romana (max 190 caractere). "
            f"Faza zilei: {phase}. Scor: {stats['score']}/100, Win istoric: {stats['win_chance']}%, "
            f"variatie azi: {stats['change_pct']}%, pre-market: {stats['premarket_pct']}%, volum open: {stats['open_vol_ratio']}x. "
            f"Stiri recente: {'; '.join(headlines[:4])}. "
            f"Spune direct daca miscarea este confirmata de stiri sau prezinta risc. Fara introduceri."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip().replace("\n", " ")
    except Exception as e:
        print(f"Gemini info pentru {ticker}: {e}")

    return fallback_text


def send_notifications(results, portfolio_positions, status_label, phase, today_str, prev_date, prev_sig_map, prev_exit_map):
    """
    Trimite carduri vizuale Discord Embeds (si/sau Telegram) pentru:
    1. Alerte de iesire / Take Profit pe pozitiile din portofoliul Trading 212
    2. Semnale noi BUY
    3. Miscari importante pe actiunile din watchlist.txt (CUSTOM)
    """
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    dashboard_url = os.getenv("DASHBOARD_URL", "").strip()
    event_name = os.getenv("GITHUB_EVENT_NAME", "manual")

    if not ((tg_token and tg_chat_id) or discord_webhook):
        print("Info: Nu sunt configurate chei Discord sau Telegram. Sar peste notificari.")
        return

    is_new_day = (today_str != prev_date)
    exit_alerts = []
    buy_alerts = []
    custom_alerts = []

    for pos in portfolio_positions:
        if pos["isAlert"]:
            prev_ex = prev_exit_map.get(pos["ticker"], "HOLD")
            if is_new_day or prev_ex != pos["exitSignal"] or phase == "OPEN_CONFIRM" or event_name != "schedule":
                exit_alerts.append(pos)

    for item in results:
        t = item["ticker"]
        sig = item["signal"]
        prev_sig = prev_sig_map.get(t, "WAIT")
        chg = abs(item.get("_changePct", 0.0))
        pm_chg = abs(item.get("_premarketPct") or 0.0)
        strong_move = max(chg, pm_chg) >= 2.0

        if sig == "BUY":
            if is_new_day or prev_sig != "BUY" or phase == "OPEN_CONFIRM" or event_name != "schedule":
                buy_alerts.append(item)
        elif item["isCustom"] and (item["score"] >= 62 or strong_move):
            if is_new_day or phase in ("PREMARKET", "OPEN_CONFIRM") or event_name != "schedule":
                custom_alerts.append(item)

    if not exit_alerts and not buy_alerts and not custom_alerts:
        print("Nu exista alerte noi (Portofoliu / BUY / Watchlist) pentru notificare.")
        return

    # 1. Trimitere pe DISCORD folosind Embeds colorate
    if discord_webhook:
        embeds = []

        if exit_alerts:
            desc_lines = []
            for p in exit_alerts[:6]:
                desc_lines.append(
                    f"• **{p['ticker']}** — **`{p['exitSignal']}`** ({p['pnlPctFormatted']} | {p['pplFormatted']})\n"
                    f"  Preț: `{p['currentPrice']}` (Mediu: `{p['avgPrice']}`) — {p['reason']}"
                )
            embeds.append({
                "title": f"🚨 Alerte Portofoliu Trading 212 • {status_label}",
                "description": "\n".join(desc_lines),
                "color": 15680580,  # Coral Red (#ef4444)
                "url": dashboard_url if dashboard_url else None
            })

        if buy_alerts:
            desc_lines = []
            for x in buy_alerts[:6]:
                badge = " ⭐" if x["isCustom"] else ""
                desc_lines.append(
                    f"• **{x['ticker']}**{badge} | Scor: **{x['score']}/100** (Win **{x['winChance']}%**)\n"
                    f"  Preț: `{x['price']}` | TP: `{x['takeProfit']}` | SL: `{x['stopLoss']}`"
                )
            embeds.append({
                "title": f"🟢 Semnale BUY Detectate • {status_label}",
                "description": "\n".join(desc_lines),
                "color": 1096065,  # Emerald Green (#10b981)
                "url": dashboard_url if dashboard_url else None
            })

        if custom_alerts:
            desc_lines = []
            for x in custom_alerts[:5]:
                desc_lines.append(
                    f"• **{x['ticker']}** [`{x['signal']}`] — Scor **{x['score']}/100**\n"
                    f"  Preț: `{x['price']}` — {x['summary']}"
                )
            embeds.append({
                "title": f"⭐ Alerte Watchlist (CUSTOM) • {status_label}",
                "description": "\n".join(desc_lines),
                "color": 3899638,  # Sky Blue (#3b82f6)
                "url": dashboard_url if dashboard_url else None
            })

        # Curatam cheile None din embeds
        for emb in embeds:
            if emb.get("url") is None:
                emb.pop("url", None)

        payload = {
            "username": "AI Trading Analyst",
            "embeds": embeds
        }
        if dashboard_url:
            payload["content"] = f"📊 **Dashboard Live:** {dashboard_url}"

        try:
            r = requests.post(discord_webhook, json=payload, timeout=15)
            if r.status_code in (200, 204):
                print("Notificare Discord Embed trimisa cu succes!")
            else:
                print(f"Eroare Discord ({r.status_code}): {r.text}")
        except Exception as e:
            print(f"Exceptie Discord: {e}")

    # 2. Trimitere optionala pe TELEGRAM (daca e configurat)
    if tg_token and tg_chat_id:
        lines = [f"📊 <b>AI Trading Analyst</b> — {html.escape(status_label)}"]

        if exit_alerts:
            lines.append("\n🚨 <b>Alerte Portofoliu Trading 212:</b>")
            for p in exit_alerts[:6]:
                lines.append(
                    f"• <b>{html.escape(p['ticker'])}</b> — <b>{html.escape(p['exitSignal'])}</b> ({html.escape(p['pnlPctFormatted'])})\n"
                    f"  {html.escape(p['reason'])}"
                )

        if buy_alerts:
            lines.append("\n🟢 <b>Semnale BUY detectate:</b>")
            for x in buy_alerts[:6]:
                lines.append(
                    f"• <b>{html.escape(x['ticker'])}</b> | Scor: <b>{x['score']}/100</b> (Win {x['winChance']}%)\n"
                    f"  Preț: {html.escape(x['price'])} | TP: {html.escape(x['takeProfit'])} | SL: {html.escape(x['stopLoss'])}"
                )

        if custom_alerts:
            lines.append("\n⭐ <b>Alerte Watchlist:</b>")
            for x in custom_alerts[:5]:
                lines.append(f"• <b>{html.escape(x['ticker'])}</b> [{x['signal']}] — {html.escape(x['price'])}")

        if dashboard_url:
            lines.append(f"\n🔗 <a href=\"{html.escape(dashboard_url)}\">Deschide Dashboard-ul Live</a>")

        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={
                    "chat_id": tg_chat_id,
                    "text": "\n".join(lines),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=15
            )
        except Exception as e:
            print(f"Exceptie Telegram: {e}")


def main():
    phase, status_label, now_ro, now_ny = detect_scan_phase()
    if phase is None:
        print(f"Ora curenta RO ({now_ro.strftime('%H:%M')}) este in afara ferestrelor 15:30/16:15/16:45. Iesire rapida.")
        return

    prev_date, prev_sig_map, prev_exit_map = load_previous_scan("data.json")
    print(f"Pornire scanare in faza: {phase} ({status_label}) | RO: {now_ro.strftime('%H:%M')} | NY: {now_ny.strftime('%H:%M')}")

    custom_tickers = load_custom_watchlist("watchlist.txt")
    base_list = list(dict.fromkeys(custom_tickers + TOP_50_US + TOP_WORLDWIDE))
    known_by_base = {t.split(".")[0]: t for t in base_list}

    t212_available, t212_portfolio_map, _ = get_trading212_data(known_by_base)

    # Includem automat si orice actiune detinuta in portofoliu care nu era in listele standard
    portfolio_yf_tickers = list(t212_portfolio_map.keys())
    all_tickers = list(dict.fromkeys(portfolio_yf_tickers + custom_tickers + TOP_50_US + TOP_WORLDWIDE))
    print(f"Scanez {len(all_tickers)} companii...")

    results = []
    portfolio_positions = []
    signals_count = 0

    for ticker in all_tickers:
        try:
            is_us = "." not in ticker
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3y", interval="1d")
            if hist.empty or len(hist) < 220:
                continue

            temp_df = hist.copy()
            if temp_df.index.tz is None:
                temp_df.index = temp_df.index.tz_localize("UTC")
            temp_df = temp_df.tz_convert(TZ_NY if is_us else TZ_RO)
            today_d = now_ny.date() if is_us else now_ro.date()
            prev_rows = temp_df[temp_df.index.date < today_d]
            prev_close = float(prev_rows["Close"].iloc[-1]) if not prev_rows.empty else float(temp_df["Close"].iloc[-1])

            intra_info = analyze_intraday_and_premarket(stock, prev_close, is_us, now_ny, now_ro)
            stats = compute_indicators_and_backtest(hist, is_us, now_ny, now_ro, intra_info, phase)
            if not stats:
                continue

            info = {}
            try:
                info = stock.info or {}
            except Exception:
                pass

            company_name = info.get("shortName") or info.get("longName") or ticker
            currency = info.get("currency", "USD")
            sym = "$" if currency == "USD" else ("€" if currency == "EUR" else f"{currency} ")

            headlines = []
            try:
                news_items = stock.news or []
                for n in news_items[:4]:
                    title = n.get("title") or (n.get("content", {}) or {}).get("title")
                    if title:
                        headlines.append(title)
            except Exception:
                pass

            is_custom = ticker in custom_tickers
            base_sym = ticker.split(".")[0]
            pos_info = t212_portfolio_map.get(ticker)
            in_portfolio = pos_info is not None
            on_t212 = (base_sym in t212_available) if t212_available else True

            if t212_available and not on_t212 and not is_custom and not in_portfolio:
                continue

            if stats["score"] >= 68 and stats["win_chance"] >= 54:
                signal = "BUY"
                signals_count += 1
            elif stats["score"] >= 54:
                signal = "WATCH"
            else:
                signal = "WAIT"

            summary = get_ai_news_insight(ticker, company_name, headlines, stats, phase)

            portfolio_eval = None
            if in_portfolio:
                portfolio_eval = evaluate_portfolio_exit(ticker, company_name, pos_info, stats, sym)
                portfolio_positions.append(portfolio_eval)

            region = "WORLDWIDE" if not is_us or ticker in TOP_WORLDWIDE else "US"
            tags = []
            if in_portfolio:
                tags.append("IN PORTFOLIO")
            if is_custom:
                tags.append("CUSTOM")
            tags.append(region)

            indicators_list = [
                f"RSI(14): {stats['rsi']}",
                f"SMA200: {'Deasupra' if stats['sma200_above'] else 'Sub'}"
            ]
            if stats["premarket_pct"] is not None:
                s_pm = "+" if stats["premarket_pct"] >= 0 else ""
                indicators_list.append(f"Pre-Market: {s_pm}{stats['premarket_pct']}%")
            if stats["open_vol_ratio"] is not None:
                indicators_list.append(f"Vol Open 15m: {stats['open_vol_ratio']}x")
            if stats["above_vwap"] is not None:
                indicators_list.append(f"VWAP: {'Deasupra' if stats['above_vwap'] else 'Sub'}")

            chg = stats["change_pct"]
            chg_str = f" ({'+' if chg >= 0 else ''}{chg}%)" if chg != 0 else ""

            results.append({
                "ticker": ticker,
                "name": company_name,
                "region": " • ".join(tags),
                "isCustom": is_custom,
                "inPortfolio": in_portfolio,
                "portfolio": portfolio_eval,
                "signal": signal,
                "score": stats["score"],
                "winChance": stats["win_chance"],
                "lossRisk": stats["loss_risk"],
                "riskReward": f"1 : {stats['rr_ratio']}",
                "price": f"{sym}{stats['price']:.2f}{chg_str}",
                "takeProfit": f"+{stats['tp_pct']}% ({sym}{stats['price'] * (1 + stats['tp_pct']/100):.2f})",
                "stopLoss": f"-{stats['sl_pct']}% ({sym}{stats['price'] * (1 - stats['sl_pct']/100):.2f})",
                "backtestStats": f"{stats['wins']} Wins vs {stats['losses']} Losses (ultimii 3 ani)",
                "indicators": indicators_list,
                "summary": summary,
                "_changePct": stats["change_pct"],
                "_premarketPct": stats["premarket_pct"]
            })

            time.sleep(0.25)

        except Exception as e:
            print(f"Eroare la {ticker}: {e}")

    # Ordonam: intai pozitiile din portofoliu, apoi custom watchlist, apoi scorul descrescator
    results.sort(key=lambda x: (not x["inPortfolio"], not x["isCustom"], -x["score"]))
    portfolio_positions.sort(key=lambda x: (not x["isAlert"], -abs(x["pnlPct"])))

    tz_abbr = now_ro.tzname() or "EET"
    today_str = now_ro.strftime("%d %b %Y")
    ro_time = now_ro.strftime(f"%d %b %Y, %H:%M {tz_abbr}")

    send_notifications(
        results, portfolio_positions, status_label, phase,
        today_str, prev_date, prev_sig_map, prev_exit_map
    )

    for item in results:
        item.pop("_changePct", None)
        item.pop("_premarketPct", None)

    output_data = {
        "marketStatus": status_label,
        "stocksScanned": len(results),
        "signalsToday": signals_count,
        "updatedAt": ro_time,
        "portfolioPositions": portfolio_positions,
        "opportunities": results
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(
        f"Gata! Salvat {len(results)} oportunitati si {len(portfolio_positions)} "
        f"pozitii din portofoliu in data.json ({status_label})."
    )


if __name__ == "__main__":
    main()

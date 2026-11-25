import os
import json
import time
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

TZ_RO = ZoneInfo("Europe/Bucharest")
TZ_NY = ZoneInfo("America/New_York")


def detect_scan_phase():
    """
    Identifica faza curenta a zilei pe baza orei din Bucuresti si New York:
    - PREP (in jur de 15:30 RO): Pregatire si analiza context
    - PREMARKET (in jur de 16:15 RO): Scanare miscari Pre-Market inainte de deschiderea SUA
    - OPEN_CONFIRM (in jur de 16:45 RO): Confirmare directie si volum in primele 15 min de la deschidere
    """
    now_ro = datetime.now(TZ_RO)
    now_ny = datetime.now(TZ_NY)
    event_name = os.getenv("GITHUB_EVENT_NAME", "manual")

    # Daca rularea este automata (schedule), verificam sa fie in una din cele 3 ferestre din Bucuresti
    # (Acest filtru rezolva automat trecerea la ora de vara/iarna fara a modifica cron-ul)
    if event_name == "schedule":
        mins_ro = now_ro.hour * 60 + now_ro.minute
        in_prep = (15 * 60 + 5) <= mins_ro <= (15 * 60 + 50)
        in_pre = (15 * 60 + 55) <= mins_ro <= (16 * 60 + 32)
        in_open = (16 * 60 + 35) <= mins_ro <= (17 * 60 + 25)
        if not (in_prep or in_pre or in_open):
            return None, None, now_ro, now_ny

    # Determinam faza activa in functie de ora bursei din New York (deschidere la 09:30 ET)
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


def get_trading212_data():
    """
    Se conecteaza la Trading 212 API (Live sau Demo) pentru a verifica:
    1. Lista instrumentelor disponibile pe platforma.
    2. Pozitiile detinute in portofoliu.
    """
    api_key = os.getenv("T212_API_KEY", "").strip()
    api_secret = os.getenv("T212_API_SECRET", "").strip()

    if not api_key:
        print("Info: T212_API_KEY nu este setat. Rulez analiza pe baza listelor standard.")
        return set(), set()

    base_urls = ["https://live.trading212.com/api/v0", "https://demo.trading212.com/api/v0"]
    available_tickers = set()
    portfolio_tickers = set()

    for base in base_urls:
        try:
            auth = (api_key, api_secret) if api_secret else None
            headers = {} if api_secret else {"Authorization": api_key}

            resp = requests.get(f"{base}/equity/metadata/instruments", auth=auth, headers=headers, timeout=15)
            if resp.status_code == 200:
                instruments = resp.json()
                for item in instruments:
                    short_name = item.get("shortName", "").upper()
                    ticker_raw = item.get("ticker", "").upper()
                    if short_name:
                        available_tickers.add(short_name)
                    if ticker_raw:
                        available_tickers.add(ticker_raw.split("_")[0])

                port_resp = requests.get(f"{base}/equity/portfolio", auth=auth, headers=headers, timeout=15)
                if port_resp.status_code == 200:
                    for pos in port_resp.json():
                        t = pos.get("ticker", "").upper().split("_")[0]
                        if t:
                            portfolio_tickers.add(t)

                print(f"Conectat cu succes la Trading 212 ({base}). Instrumente gasite: {len(available_tickers)}")
                break
        except Exception as e:
            print(f"Eroare conexiune {base}: {e}")

    return available_tickers, portfolio_tickers


def analyze_intraday_and_premarket(stock, prev_close, is_us, now_ny, now_ro):
    """
    Extrage datele intraday pe 15 minute (inclusiv Pre-Market) pentru a masura:
    - Pretul live exact (inclusiv in Pre-Market)
    - Variatia procentuala fata de inchiderea de ieri
    - Raportul de volum la deschidere (primele 15 min vs media ultimelor 4 zile)
    - Pozitia fata de VWAP intraday
    """
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

        # Convertim indexul in fusul orar al pietei relevante
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
            # Separam barele din Pre-Market (< 09:30 ET) de cele din sesiunea normala (>= 09:30 ET)
            mins_of_day = today_bars.index.hour * 60 + today_bars.index.minute
            pre_bars = today_bars[mins_of_day < (9 * 60 + 30)]
            reg_bars = today_bars[mins_of_day >= (9 * 60 + 30)]

            if not pre_bars.empty and prev_close > 0:
                last_pre_price = float(pre_bars["Close"].iloc[-1])
                result["premarket_pct"] = round(((last_pre_price - prev_close) / prev_close) * 100, 2)
                result["has_premarket"] = True

            if not reg_bars.empty:
                result["has_open_15m"] = True
                # Calculam VWAP pentru sesiunea de azi
                typical_price = (reg_bars["High"] + reg_bars["Low"] + reg_bars["Close"]) / 3.0
                total_vol = reg_bars["Volume"].sum()
                if total_vol > 0:
                    vwap = float((typical_price * reg_bars["Volume"]).sum() / total_vol)
                    result["above_vwap"] = live_price >= vwap

                # Comparam volumul primei lumanari de 15m (09:30) cu media primelor 15m din zilele trecute
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
            # Pentru actiuni europene/worldwide, calculam VWAP pe sesiunea curenta
            typical_price = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3.0
            total_vol = today_bars["Volume"].sum()
            if total_vol > 0:
                vwap = float((typical_price * today_bars["Volume"]).sum() / total_vol)
                result["above_vwap"] = live_price >= vwap

    except Exception as e:
        print(f"Info intraday indisponibil: {e}")

    return result


def compute_indicators_and_backtest(df, is_us, now_ny, now_ro, intra_info, phase):
    """
    Combina analiza istorica pe 3 ani (backtest Win/Loss) cu semnalele live
    din Pre-Market si din primele 15 minute de la deschidere.
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_local = df.tz_convert(TZ_NY if is_us else TZ_RO)
    today_date = now_ny.date() if is_us else now_ro.date()

    # Pastram doar zilele inchise pentru calculul indicatorilor zilnici si al backtest-ului
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

    # Praguri dinamice Take Profit / Stop Loss bazate pe volatilitatea ATR
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

    # 1. Scorul de baza (tehnic + statistic)
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

    # 2. Ajustari specifice pentru fazele 16:15 (Pre-Market) si 16:45 (Open Confirm)
    pre_pct = intra_info.get("premarket_pct")
    open_vol = intra_info.get("open_vol_ratio")
    above_vwap = intra_info.get("above_vwap")
    change_pct = intra_info.get("change_pct", 0.0)

    if phase == "PREMARKET" and pre_pct is not None:
        if 0.6 <= pre_pct <= 6.5 and above_200:
            score += 8   # Gap pozitiv sanatos sustinut de trend
        elif pre_pct > 9.0:
            score -= 6   # Risc de marcari de profit la deschidere (gap prea mare)
        elif pre_pct < -2.5:
            score -= 8   # Presiune de vanzare in pre-market

    elif phase in ("OPEN_CONFIRM", "INTRADAY"):
        if open_vol is not None and open_vol >= 1.35 and change_pct > 0.3:
            score += 8   # Explozie de volum la deschidere pe crestere
        if above_vwap is True and change_pct > 0:
            score += 6   # Cumparatorii apara pretul peste VWAP
        elif above_vwap is False and pre_pct is not None and pre_pct > 1.0:
            score -= 9   # Capcana (Gap & Fade): a deschis sus, dar a cazut sub VWAP

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


def get_ai_news_insight(ticker, name, headlines, stats, phase):
    """
    Genereaza rezumatul scurt in romana folosind Gemini AI (daca exista GEMINI_API_KEY)
    sau construieste automat rezumatul tehnic + intraday.
    """
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


def main():
    phase, status_label, now_ro, now_ny = detect_scan_phase()
    if phase is None:
        print(f"Ora curenta RO ({now_ro.strftime('%H:%M')}) este in afara ferestrelor 15:30/16:15/16:45. Iesire rapida.")
        return

    print(f"Pornire scanare in faza: {phase} ({status_label}) | RO: {now_ro.strftime('%H:%M')} | NY: {now_ny.strftime('%H:%M')}")

    custom_tickers = load_custom_watchlist("watchlist.txt")
    t212_available, t212_portfolio = get_trading212_data()

    all_tickers = list(dict.fromkeys(custom_tickers + TOP_50_US + TOP_WORLDWIDE))
    print(f"Scanez {len(all_tickers)} companii...")

    results = []
    signals_count = 0

    for ticker in all_tickers:
        try:
            is_us = "." not in ticker
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3y", interval="1d")
            if hist.empty or len(hist) < 220:
                continue

            # Luam ultimul pret de inchidere anterior zilei de azi pentru calculul exact al variatiei
            temp_df = hist.copy()
            if temp_df.index.tz is None:
                temp_df.index = temp_df.index.tz_localize("UTC")
            temp_df = temp_df.tz_convert(TZ_NY if is_us else TZ_RO)
            today_d = now_ny.date() if is_us else now_ro.date()
            prev_rows = temp_df[temp_df.index.date < today_d]
            prev_close = float(prev_rows["Close"].iloc[-1]) if not prev_rows.empty else float(temp_df["Close"].iloc[-1])

            # Analizam miscarile live din Pre-Market si primele 15 min
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
            in_portfolio = base_sym in t212_portfolio
            on_t212 = (base_sym in t212_available) if t212_available else True

            if t212_available and not on_t212 and not is_custom:
                continue

            # Praguri semnal calibrate
            if stats["score"] >= 68 and stats["win_chance"] >= 54:
                signal = "BUY"
                signals_count += 1
            elif stats["score"] >= 54:
                signal = "WATCH"
            else:
                signal = "WAIT"

            summary = get_ai_news_insight(ticker, company_name, headlines, stats, phase)

            region = "WORLDWIDE" if not is_us or ticker in TOP_WORLDWIDE else "US"
            tags = []
            if is_custom:
                tags.append("CUSTOM")
            if in_portfolio:
                tags.append("IN PORTFOLIO")
            tags.append(region)

            # Construim badge-urile de indicatori inclusiv datele live Pre-Market / Open
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

            # Formatam pretul cu variatia de azi intre paranteze
            chg = stats["change_pct"]
            chg_str = f" ({'+' if chg >= 0 else ''}{chg}%)" if chg != 0 else ""

            results.append({
                "ticker": ticker,
                "name": company_name,
                "region": " • ".join(tags),
                "isCustom": is_custom,
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
                "summary": summary
            })

            time.sleep(0.25)

        except Exception as e:
            print(f"Eroare la {ticker}: {e}")

    results.sort(key=lambda x: (not x["isCustom"], -x["score"]))

    tz_abbr = now_ro.tzname() or "EET"
    ro_time = now_ro.strftime(f"%d %b %Y, %H:%M {tz_abbr}")

    output_data = {
        "marketStatus": status_label,
        "stocksScanned": len(results),
        "signalsToday": signals_count,
        "updatedAt": ro_time,
        "opportunities": results
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Gata! Salvat {len(output_data['opportunities'])} oportunitati in data.json ({status_label}).")


if __name__ == "__main__":
    main()

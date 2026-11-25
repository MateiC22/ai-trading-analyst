import os
import json
import time
from datetime import datetime, timezone, timedelta
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
    2. Pozitiile pe care le detii deja in portofoliu.
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

                # Preluam si portofoliul curent (optional)
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


def compute_indicators_and_backtest(df):
    """
    Calculeaza indicatorii tehnici si simuleaza statistic pe ultimii 3 ani
    de cate ori configuratia actuala a atins Take Profit inainte de Stop Loss.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Medii mobile
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # RSI (14 zile)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR (14 zile) pentru calculul dinamic de risc/recompensa
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volum relativ
    df["VolRatio"] = volume / volume.rolling(20).mean()

    df = df.dropna().copy()
    if len(df) < 220:
        return None

    latest = df.iloc[-1]
    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    atr = float(latest["ATR"])
    sma50 = float(latest["SMA50"])
    sma200 = float(latest["SMA200"])
    vol_ratio = float(latest["VolRatio"])

    # Definim praguri dinamice bazate pe volatilitatea reala (ATR)
    tp_pct = min(max((2.2 * atr) / price, 0.025), 0.12)   # Take Profit intre +2.5% si +12%
    sl_pct = min(max((1.2 * atr) / price, 0.015), 0.06)   # Stop Loss intre -1.5% si -6%
    rr_ratio = round(tp_pct / sl_pct, 2)

    # Identificam regimul curent al actiunii pentru a cauta situatii similare in trecut
    above_200 = price > sma200
    rsi_low = rsi < 45
    rsi_mid = 45 <= rsi <= 65

    wins = 0
    losses = 0
    closes_arr = df["Close"].values
    highs_arr = df["High"].values
    lows_arr = df["Low"].values
    rsi_arr = df["RSI"].values
    sma200_arr = df["SMA200"].values

    # Backtest pe ultimii ~3 ani (lasam ultimele 15 zile pentru evaluarea rezultatului)
    for i in range(len(df) - 16):
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

            # Verificam urmatoarele 15 zile de tranzactionare
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

    # Scorul tehnic compus (0 - 100)
    score = int(win_chance * 0.65)
    if above_200:
        score += 15
    if price > sma50:
        score += 8
    if 32 <= rsi <= 58:
        score += 10
    elif rsi > 74:
        score -= 10  # Supracumparat
    if vol_ratio > 1.15:
        score += 5

    score = max(15, min(96, score))

    return {
        "price": price,
        "rsi": round(rsi, 1),
        "sma50_above": price > sma50,
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


def get_ai_news_insight(ticker, name, headlines, stats):
    """
    Foloseste Gemini AI (daca GEMINI_API_KEY este setat) pentru a interpreta stirile
    si a genera un rezumat scurt in romana. Altfel genereaza un rezumat tehnic clar.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    fallback_text = (
        f"Probabilitate istorică {stats['win_chance']}% ({stats['wins']}W/{stats['losses']}L pe 3 ani). "
        f"RSI la {stats['rsi']}, preț {'peste' if stats['sma200_above'] else 'sub'} SMA200, "
        f"raport Risk/Reward 1:{stats['rr_ratio']}."
    )

    if not gemini_key or genai is None or not headlines:
        return fallback_text

    try:
        client = genai.Client(api_key=gemini_key)
        prompt = (
            f"Esti un analist financiar concis. Analizeaza compania {name} ({ticker}) in maximum 2 propozitii scurte in romana (max 180 caractere). "
            f"Date tehnice: scor {stats['score']}/100, sanse statistice Win {stats['win_chance']}%, RSI {stats['rsi']}, "
            f"trend {'ascendent' if stats['sma200_above'] else 'descendent'}. "
            f"Ultimele stiri: {'; '.join(headlines[:4])}. "
            f"Mentioneaza daca stirile sustin miscarea sau aduc risc de volatilitate. Fara introduceri."
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
    custom_tickers = load_custom_watchlist("watchlist.txt")
    t212_available, t212_portfolio = get_trading212_data()

    # Combinam listele fara duplicate, pastrand custom_tickers primele
    all_tickers = list(dict.fromkeys(custom_tickers + TOP_50_US + TOP_WORLDWIDE))
    print(f"Incep scanarea pentru {len(all_tickers)} companii...")

    results = []
    signals_count = 0

    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3y", interval="1d")
            if hist.empty or len(hist) < 220:
                continue

            stats = compute_indicators_and_backtest(hist)
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

            # Extragem ultimele titluri de stiri
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

            # Daca API-ul T212 este activ si compania nu e in watchlist-ul custom si nu e pe T212, o sarim
            if t212_available and not on_t212 and not is_custom:
                continue

            # Stabilim semnalul
            if stats["score"] >= 74 and stats["win_chance"] >= 60:
                signal = "BUY"
                signals_count += 1
            elif stats["score"] >= 58:
                signal = "WATCH"
            else:
                signal = "WAIT"

            summary = get_ai_news_insight(ticker, company_name, headlines, stats)

            region = "WORLDWIDE" if "." in ticker or ticker in TOP_WORLDWIDE else "US"
            tags = []
            if is_custom:
                tags.append("CUSTOM")
            if in_portfolio:
                tags.append("IN PORTFOLIO")
            tags.append(region)

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
                "price": f"{sym}{stats['price']:.2f}",
                "takeProfit": f"+{stats['tp_pct']}% ({sym}{stats['price'] * (1 + stats['tp_pct']/100):.2f})",
                "stopLoss": f"-{stats['sl_pct']}% ({sym}{stats['price'] * (1 - stats['sl_pct']/100):.2f})",
                "backtestStats": f"{stats['wins']} Wins vs {stats['losses']} Losses (ultimii 3 ani)",
                "indicators": [
                    f"RSI(14): {stats['rsi']}",
                    f"SMA200: {'Deasupra' if stats['sma200_above'] else 'Sub'}",
                    f"Volum: {stats['vol_ratio']}x",
                    f"Trading 212: {'Activ' if on_t212 else 'Verifica simbol'}"
                ],
                "summary": summary
            })

            time.sleep(0.25)  # Pauza scurta pentru a evita blocarea de catre Yahoo Finance

        except Exception as e:
            print(f"Eroare la {ticker}: {e}")

    # Sortam astfel incat companiile tale din watchlist.txt sa fie primele, apoi dupa scor descrescator
    results.sort(key=lambda x: (not x["isCustom"], -x["score"]))

    # Ora Romaniei (UTC+2 / UTC+3)
    ro_time = datetime.now(timezone(timedelta(hours=2))).strftime("%d %b %Y, %H:%M EET")

    output_data = {
        "marketStatus": "Active Scan",
        "stocksScanned": len(results),
        "signalsToday": signals_count,
        "updatedAt": ro_time,
        "opportunities": results[:35]  # Pastram top 35 cele mai relevante oportunitati in dashboard
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Gata! Salvat {len(output_data['opportunities'])} oportunitati in data.json.")


if __name__ == "__main__":
    main()

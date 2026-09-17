import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

# ============================================================
# GPT SIGNAL SCANNER
# 기존 app.py와 완전히 독립적으로 실행
# ============================================================

GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/"
    "pub?output=csv"
)

LOOKBACK_DAYS = 5


def is_valid_ticker(ticker):
    return bool(re.match(r"^[A-Za-z0-9.-]+$", str(ticker)))


def is_golden_alignment(ma5, ma10, ma20, ma50, idx):
    try:
        return (
            ma5.iloc[idx]
            > ma10.iloc[idx]
            > ma20.iloc[idx]
            > ma50.iloc[idx]
        )
    except (IndexError, KeyError):
        return False


def find_signals(data_df, nasdaq_tickers):
    passed_tickers = []

    for ticker in nasdaq_tickers:
        try:
            if isinstance(data_df.columns, pd.MultiIndex):
                if ticker not in data_df.columns.levels[0]:
                    continue
                hist = data_df[ticker].dropna(subset=["Close"])
            else:
                if ticker not in data_df:
                    continue
                hist = data_df[ticker].dropna(subset=["Close"])

            if len(hist) < 60:
                continue

            close = hist["Close"]

            ma5 = close.rolling(window=5).mean()
            ma10 = close.rolling(window=10).mean()
            ma20 = close.rolling(window=20).mean()
            ma50 = close.rolling(window=50).mean()

            x_day = None
            score = 0
            near_ma = None

            # 기존 app.py와 동일
            for back in range(3, 11):
                i = -back
                price = close.iloc[i]

                for ma, pts in [
                    (ma20, 3),
                    (ma10, 2),
                    (ma5, 1)
                ]:
                    ma_val = ma.iloc[i]

                    if ma_val * 0.995 <= price <= ma_val * 1.005:
                        x_day = i
                        score = pts
                        near_ma = ma
                        break

                if x_day is not None:
                    break

            if x_day is None:
                continue

            cond1 = (
                is_golden_alignment(ma5, ma10, ma20, ma50, x_day - 1)
                or
                is_golden_alignment(ma5, ma10, ma20, ma50, x_day - 2)
            )

            cond2 = (
                is_golden_alignment(ma5, ma10, ma20, ma50, x_day + 1)
                or
                is_golden_alignment(ma5, ma10, ma20, ma50, x_day + 2)
            )

            cond3 = (
                close.iloc[x_day - 1] > near_ma.iloc[x_day - 1]
                or
                close.iloc[x_day - 2] > near_ma.iloc[x_day - 2]
            )

            cond4 = (
                close.iloc[x_day + 1] > near_ma.iloc[x_day + 1]
                or
                close.iloc[x_day + 2] > near_ma.iloc[x_day + 2]
            )

            cond5 = (
                near_ma.iloc[-1] < close.iloc[-1]
                and
                close.iloc[x_day + 1] < close.iloc[-1]
            )

            if cond1 and cond2 and cond3 and cond4 and cond5:

                signal_date = hist.index[x_day].strftime("%Y-%m-%d")

                passed_tickers.append({
                    "Ticker": ticker,
                    "Close": round(float(close.iloc[-1]), 2),
                    "눌림목 점수": score,
                    "시그널 발생일자": signal_date
                })

        except Exception:
            continue

    return passed_tickers


def main():

    print("=" * 60)
    print("GPT SIGNAL SCANNER")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Google Sheets에서 ticker universe 가져오기
    # --------------------------------------------------------

    tickers_df = pd.read_csv(GOOGLE_SHEET_URL)

    nasdaq_tickers = [
        t
        for t in tickers_df["Symbol"].tolist()[:800]
        if is_valid_ticker(t)
    ]

    print(f"Ticker universe: {len(nasdaq_tickers)}")

    # --------------------------------------------------------
    # 2. 최근 6개월 데이터 다운로드
    # --------------------------------------------------------

    data = yf.download(
        nasdaq_tickers,
        period="6mo",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False
    )

    print("Market data downloaded.")

    # --------------------------------------------------------
    # 3. 기존 app.py와 동일한 Signal 계산
    # --------------------------------------------------------

    signals = find_signals(
        data,
        nasdaq_tickers
    )

    if not signals:
        print("최근 Signal 없음.")
        return

    # --------------------------------------------------------
    # 4. 최근 5일 이내 Signal만 추출
    # --------------------------------------------------------

    today = datetime.now().date()
    cutoff_date = today - timedelta(days=LOOKBACK_DAYS)

    recent_signals = []

    for item in signals:

        signal_date = datetime.strptime(
            item["시그널 발생일자"],
            "%Y-%m-%d"
        ).date()

        if signal_date >= cutoff_date:
            recent_signals.append(item)

    # 최신 Signal부터 정렬
    recent_signals.sort(
        key=lambda x: (
            x["시그널 발생일자"],
            x["눌림목 점수"]
        ),
        reverse=True
    )

    # --------------------------------------------------------
    # 5. 결과 출력
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"최근 {LOOKBACK_DAYS}일 Signal")
    print("=" * 60)

    if not recent_signals:
        print("최근 5일 이내 Signal 없음.")
        return

    for item in recent_signals:
        print(
            f'{item["시그널 발생일자"]} | '
            f'{item["Ticker"]} | '
            f'점수 {item["눌림목 점수"]} | '
            f'현재가 {item["Close"]}'
        )

    print()
    print("=" * 60)
    print("GPT_SIGNAL_RESULT")
    print("=" * 60)

    # GPT가 읽기 쉬운 형태
    for item in recent_signals:
        print(
            f'{item["Ticker"]},'
            f'{item["시그널 발생일자"]},'
            f'{item["눌림목 점수"]}'
        )


if __name__ == "__main__":
    main()
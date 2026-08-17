#!/usr/bin/env python3
"""
hynix_adr_comparison.py
------------------------
SK하이닉스(KOSPI: 000660.KS)와 나스닥에 상장된 ADR(SKHY)의 가격을 비교하는 스크립트.

- 000660.KS / SKHY 모두 배당락·액면분할/병합이 반영된 수정주가(adjusted price)를 사용해
  캔들차트와 프리미엄/디스카운트를 계산합니다 (장기간 비교 시 필수).
  단, 시가배당율(왼쪽 y축) 계산에는 "그 날의 실제 시장가"가 필요하므로 원시(raw) 종가를 별도로 사용합니다.
- 상단 차트: 000660.KS 캔들차트 + SKHY(원화, 보통주 1주 환산) 캔들차트를 오른쪽 y축(KRW)에 표시하고,
  000660.KS의 시가배당율(연환산, trailing 12M) 라인을 왼쪽 y축(%)에 함께 표시
  * SKHY는 "해당 날짜"의 USD/KRW 환율을 곱하고, ADR 비율(1 ADR = 보통주 10주)을 반영해
    "보통주 환산 원화가"로 변환해 000660.KS와 같은 축(KRW)에서 직접 비교 가능
  * 두 캔들차트 모두 상승일=빨강, 하락일=파랑 (한국 관례)
- 하단 차트: (SKHY * 10 * 해당 날짜 USD/KRW환율) 이 000660.KS 대비 몇 % 차이 나는지 (%)를
  일별 막대그래프(bar)로 표시
  * 하단 차트 x축에는 range slider(슬라이드 바)가 있어 원하는 구간을 드래그로 선택/확대 가능
  * 마우스 스크롤로도 확대/축소 가능 (scrollZoom)
- 오늘 날짜 처리:
  * 일간 다운로드에 아직 오늘 데이터가 없으면(장중 등) 실시간 현재가를 가져와 보강합니다.
  * 000660.KS / SKHY 둘 다 오늘 가격을 가져올 수 없으면 오늘은 차트에 표시하지 않습니다.
  * 둘 중 하나만 오늘 가격이 없으면, 하단 %차이 계산에서는 forward-fill(직전 값 사용)로
    오늘의 차이를 계산해 표시합니다. (상단 캔들차트는 가격이 있는 종목만 오늘 캔들이 나타납니다.)

사용법:
    python hynix_adr_comparison.py --months 6
    python hynix_adr_comparison.py --months 360   # 야후 파이낸스에 있는 전체 이력

결과:
    hynix_adr_comparison.html 파일로 저장되고, 기본 브라우저에서 바로 열립니다.
"""

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit(
        "yfinance 패키지가 설치되어 있지 않습니다. "
        "먼저 `pip install yfinance` 로 설치해 주세요."
    )

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    sys.exit(
        "plotly 패키지가 설치되어 있지 않습니다. "
        "먼저 `pip install plotly` 로 설치해 주세요."
    )

KOSPI_TICKER = "000660.KS"   # SK하이닉스 (KOSPI)
ADR_TICKER = "SKHY"          # SK하이닉스 ADR (Nasdaq)
FX_TICKER = "KRW=X"          # USD/KRW 환율
ADR_RATIO = 10                # 1 ADR = 보통주 10주

# 한국식 캔들 색상: 상승 = 빨강, 하락 = 파랑
INCREASING_COLOR = "#d62728"
DECREASING_COLOR = "#1f77b4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SK Hynix (KOSPI) vs SKHY (Nasdaq ADR) 가격 비교 차트를 생성합니다."
    )
    parser.add_argument(
        "--months",
        "-n",
        type=int,
        required=True,
        help="조회할 개월 수 (오늘로부터 n개월 전까지의 일간 데이터를 사용, "
        "야후 파이낸스에 존재하는 전체 이력을 보려면 충분히 큰 값(예: 360)을 지정)",
    )
    return parser.parse_args()


def fetch_ohlc(ticker: str, start: datetime, end: datetime, adjusted: bool = True) -> pd.DataFrame:
    """
    지정한 티커의 일간 OHLC(Open, High, Low, Close)를 DataFrame으로 반환합니다.

    adjusted=True (기본값): 배당락 및 액면분할/병합이 반영된 수정주가(adjusted price)를 반환합니다.
        장기간(수십 년) 비교 시 분할/배당 이력이 있는 종목은 반드시 이 값을 사용해야 정확합니다.
    adjusted=False: 실제 거래되었던 원시(raw) 가격을 반환합니다.
        시가배당율처럼 "그 날의 실제 시장가"가 필요한 계산에만 사용합니다.
    """
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=adjusted,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"'{ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요.")

    # yfinance가 MultiIndex 컬럼을 반환하는 경우 처리
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)

    df = df[["Open", "High", "Low", "Close"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def fetch_dividend_yield(ticker: str, price_series: pd.Series) -> pd.Series:
    """
    티커의 배당 이력을 가져와 price_series의 각 날짜에 대해
    trailing 12-month(최근 1년치) 배당금 합계를 그 날짜의 가격으로 나눈
    시가배당율(%)을 계산해 반환합니다.
    """
    div = yf.Ticker(ticker).dividends

    if div.empty:
        print(f"  ({ticker}: 배당 이력이 없어 배당율을 0으로 처리합니다.)")
        return pd.Series(0.0, index=price_series.index, name="dividend_yield_pct")

    div.index = pd.to_datetime(div.index).tz_localize(None)
    # 같은 날짜에 배당이 여러 건이면 합산 후 오름차순 정렬
    div_sorted = div.groupby(div.index).sum().sort_index()
    cum_div = div_sorted.cumsum()

    idx = price_series.index
    # 각 날짜까지의 누적 배당금 (직전 배당일 값으로 forward-fill)
    cum_now = cum_div.reindex(idx, method="ffill").fillna(0)
    # 1년 전 시점까지의 누적 배당금
    lag_idx = idx - pd.Timedelta(days=365)
    cum_prior = cum_div.reindex(lag_idx, method="ffill").fillna(0)
    cum_prior.index = idx  # 인덱스를 원래 날짜로 맞춰서 뺄셈 가능하도록 정렬

    trailing_div = (cum_now - cum_prior).clip(lower=0)
    yield_pct = (trailing_div / price_series) * 100
    yield_pct.name = "dividend_yield_pct"
    return yield_pct


def get_current_price(ticker: str):
    """
    해당 티커의 실시간(또는 가장 최근) 현재가를 가져옵니다.
    여러 방법을 순서대로 시도하고, 모두 실패하면 None을 반환합니다.
    """
    try:
        fi = yf.Ticker(ticker).fast_info
        for key in ("last_price", "lastPrice", "regular_market_price"):
            try:
                val = fi[key]
            except Exception:
                val = getattr(fi, key, None)
            if val:
                return float(val)
    except Exception:
        pass
    try:
        info = yf.Ticker(ticker).info
        val = info.get("regularMarketPrice") or info.get("currentPrice")
        if val:
            return float(val)
    except Exception:
        pass
    try:
        intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not intraday.empty:
            return float(intraday["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def ensure_today_row(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    df(OHLC, 날짜 인덱스)에 오늘 날짜 데이터가 없으면 현재가를 가져와
    O=H=L=C가 모두 현재가인 "오늘" 행을 추가합니다.
    현재가를 가져올 수 없으면 df를 그대로 반환합니다 (오늘 데이터 없이 = 표시하지 않음).
    """
    today = pd.Timestamp(datetime.now().date())
    if today in df.index:
        return df

    price = get_current_price(ticker)
    if price is None:
        print(f"  ({ticker}: 오늘({today.date()}) 현재가를 가져오지 못했습니다 — 오늘 데이터 제외)")
        return df

    print(f"  ({ticker}: 오늘({today.date()}) 현재가 {price:,.2f} 사용)")
    today_row = pd.DataFrame(
        {"Open": [price], "High": [price], "Low": [price], "Close": [price]},
        index=[today],
    )
    return pd.concat([df, today_row]).sort_index()


def ensure_today_fx(fx_close: pd.Series, fx_ticker: str) -> pd.Series:
    """fx_close(환율 Close 시계열)에 오늘 환율이 없으면 현재 환율을 가져와 추가합니다."""
    today = pd.Timestamp(datetime.now().date())
    if today in fx_close.index:
        return fx_close

    price = get_current_price(fx_ticker)
    if price is None:
        return fx_close

    today_row = pd.Series([price], index=[today])
    return pd.concat([fx_close, today_row]).sort_index()


def build_data(months: int):
    end_date = datetime.today() + timedelta(days=1)  # 오늘자 데이터까지 포함
    start_date = end_date - timedelta(days=int(months * 30.44) + 5)

    print(f"데이터 조회 기간: {start_date.date()} ~ {end_date.date()}")

    print(f"[1/5] {KOSPI_TICKER} (KOSPI) 수정주가(adjusted) 데이터 가져오는 중...")
    kospi = fetch_ohlc(KOSPI_TICKER, start_date, end_date, adjusted=True)

    print(f"[2/5] {ADR_TICKER} (Nasdaq ADR) 수정주가(adjusted) 데이터 가져오는 중...")
    adr = fetch_ohlc(ADR_TICKER, start_date, end_date, adjusted=True)

    print(f"[3/5] {FX_TICKER} (USD/KRW 환율) 데이터 가져오는 중...")
    # 참고: Yahoo Finance의 USD/KRW 환율 데이터는 통상 2000년대 초반부터 제공됩니다.
    # --months 값이 매우 커서 그 이전 구간을 요청하면, 아래 fx_aligned_*에서
    # 가장 오래된 환율값으로 backward-fill 되어 근사치로 처리됩니다.
    fx_close = fetch_ohlc(FX_TICKER, start_date, end_date, adjusted=False)["Close"]

    # 오늘 데이터가 일간 다운로드에 아직 없다면(장중이거나 반영 지연 등) 현재가로 보강
    print("오늘 날짜 데이터 확인 중...")
    kospi = ensure_today_row(kospi, KOSPI_TICKER)
    adr = ensure_today_row(adr, ADR_TICKER)
    fx_close = ensure_today_fx(fx_close, FX_TICKER)

    print(f"[4/5] {KOSPI_TICKER} 원시(raw) 종가 데이터 가져오는 중 (시가배당율 계산용)...")
    # 시가배당율은 "그 날 실제 거래된 시장가" 기준이어야 하므로 수정주가가 아닌 raw 종가를 사용
    kospi_raw_close = fetch_ohlc(KOSPI_TICKER, start_date, end_date, adjusted=False)["Close"]

    print(f"[5/5] {KOSPI_TICKER} 배당 이력(시가배당율 계산용) 가져오는 중...")
    dividend_yield = fetch_dividend_yield(KOSPI_TICKER, kospi_raw_close)

    # SKHY 거래일에 맞춰 일별 환율을 정렬(직전 영업일 환율로 forward-fill)
    fx_aligned_adr = fx_close.sort_index().reindex(adr.index, method="ffill")
    fx_aligned_adr = fx_aligned_adr.bfill()  # 맨 앞부분에 결측치가 있으면 뒤 값으로 채움

    # SKHY(USD, ADR) -> 보통주 환산 원화가 (KRW) : 일별 환율 반영
    skhy_krw = adr.mul(fx_aligned_adr * ADR_RATIO, axis=0)

    # 종가 기준으로 날짜 정렬 (한국/미국 거래일 차이는 forward-fill) - 하단 % 차이 계산용
    # (오늘 날짜에 둘 중 하나만 값이 있으면, 여기서 ffill()로 없는 쪽을 직전 값으로 채워
    #  "차이"를 계산합니다. 둘 다 없으면 애초에 오늘 인덱스 자체가 없으므로 표시되지 않습니다.)
    close_df = pd.concat(
        [kospi["Close"].rename(KOSPI_TICKER), adr["Close"].rename(ADR_TICKER)], axis=1
    ).sort_index()
    close_df = close_df.ffill().dropna()

    # close_df 날짜에 맞춰서도 일별 환율을 별도로 정렬 (해당 날짜의 환율 사용)
    fx_aligned_close = fx_close.sort_index().reindex(close_df.index, method="ffill").bfill()

    # SKHY(ADR)를 "해당 날짜의" 환율로 KRW 환산 후, 000660.KS 대비 몇 % 차이 나는지 계산
    adr_equiv_krw_daily = close_df[ADR_TICKER] * ADR_RATIO * fx_aligned_close
    pct_diff = (adr_equiv_krw_daily / close_df[KOSPI_TICKER] - 1) * 100
    pct_diff.name = "pct_diff"

    return kospi, skhy_krw, pct_diff, dividend_yield


def build_figure(kospi, skhy_krw, pct_diff, dividend_yield, months):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            f"{KOSPI_TICKER} vs {ADR_TICKER}×{ADR_RATIO} (KRW, 오른쪽) "
            f"+ Yearly Dividend Rate (%, 왼쪽) — Candlestick",
            f"Premium/Discount: {ADR_TICKER}×{ADR_RATIO}×USD/KRW(해당일 환율 반영) "
            f"vs {KOSPI_TICKER} (%)",
        ),
    )

    # --- 상단 차트: 캔들차트 2개 (오른쪽 y축, KRW), 시가배당율 라인 (왼쪽 y축, %) ---
    fig.add_trace(
        go.Scatter(
            x=dividend_yield.index,
            y=dividend_yield,
            name=f"{KOSPI_TICKER} 시가배당율 (%, 왼쪽)",
            mode="lines",
            line=dict(color="#8c564b", width=2, dash="dot"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Candlestick(
            x=kospi.index,
            open=kospi["Open"],
            high=kospi["High"],
            low=kospi["Low"],
            close=kospi["Close"],
            name=f"{KOSPI_TICKER} (KRW)",
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Candlestick(
            x=skhy_krw.index,
            open=skhy_krw["Open"],
            high=skhy_krw["High"],
            low=skhy_krw["Low"],
            close=skhy_krw["Close"],
            name=f"{ADR_TICKER}×{ADR_RATIO} 환산 (KRW)",
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Yearly Dividend Rate (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price (KRW)", row=1, col=1, secondary_y=True)
    # 캔들차트 기본 range slider는 끄고, 아래쪽 차트에만 range slider를 사용
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # --- 하단 차트: % 차이 (프리미엄/디스카운트) - 일별 막대그래프 ---
    bar_colors = [INCREASING_COLOR if v >= 0 else DECREASING_COLOR for v in pct_diff]
    fig.add_trace(
        go.Bar(
            x=pct_diff.index,
            y=pct_diff,
            name="Premium/Discount (%)",
            marker_color=bar_colors,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="Premium / Discount (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # 두 번째 차트에만 range slider(슬라이드 바) 적용 → 원하는 x 구간 드래그로 선택 가능
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider_thickness=0.08,
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"SK Hynix (000660.KS) vs SKHY ADR — 최근 {months}개월",
        template="plotly_white",
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        height=950,
        margin=dict(t=90, b=40),
        dragmode="zoom",
    )

    return fig


def main():
    args = parse_args()
    if args.months <= 0:
        sys.exit("--months 값은 1 이상이어야 합니다.")

    kospi, skhy_krw, pct_diff, dividend_yield = build_data(args.months)
    fig = build_figure(kospi, skhy_krw, pct_diff, dividend_yield, args.months)

    output_path = "hynix_adr_comparison.html"
    fig.write_html(
        output_path,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["v1hovermode", "toggleSpikelines"],
        },
    )
    print(f"\n차트가 저장되었습니다: {output_path}")

    try:
        fig.show(config={"scrollZoom": True})
    except Exception:
        print("브라우저를 자동으로 열 수 없습니다. 저장된 HTML 파일을 직접 열어 확인해 주세요.")


if __name__ == "__main__":
    main()

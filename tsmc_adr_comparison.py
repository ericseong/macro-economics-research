#!/usr/bin/env python3
"""
tsmc_adr_comparison.py
------------------------
대만 증권거래소(TWSE)에 상장된 TSMC(2330.TW)와 NYSE에 상장된 ADR(TSM)의 가격을 비교하는 스크립트.

- 2330.TW / TSM 모두 배당락·액면분할/병합이 반영된 수정주가(adjusted price)를 사용해
  캔들차트와 프리미엄/디스카운트를 계산합니다 (TSM ADR 상장일인 1997년까지 장기간 비교 시 필수).
  단, 시가배당율(왼쪽 y축) 계산에는 "그 날의 실제 시장가"가 필요하므로 원시(raw) 종가를 별도로 사용합니다.
- 상단 차트: 2330.TW 캔들차트 + TSM(대만달러, 원주 1주 환산) 캔들차트를 오른쪽 y축(TWD)에 표시하고,
  2330.TW의 시가배당율(연환산, trailing 12M) 라인을 왼쪽 y축(%)에 함께 표시
  * 1 TSM ADR = 2330.TW 원주 5주 이므로, TSM(USD) 가격에 "해당 날짜"의 USD/TWD 환율을 곱한 뒤
    ADR 비율(5)로 나눠 "원주 1주 환산 대만달러가"로 변환 -> 2330.TW와 같은 축(TWD)에서 직접 비교 가능
  * 두 캔들차트 모두 상승일=빨강, 하락일=파랑
- 하단 차트: (TSM * 해당 날짜 USD/TWD환율 / 5) 이 2330.TW 대비 몇 % 차이 나는지 (%)를
  일별 막대그래프(bar)로 표시
  * 하단 차트 x축에는 range slider(슬라이드 바)가 있어 원하는 구간을 드래그로 선택/확대 가능
  * 마우스 스크롤로도 확대/축소 가능 (scrollZoom)

사용법:
    python tsmc_adr_comparison.py --months 6

결과:
    tsmc_adr_comparison.html 파일로 저장되고, 기본 브라우저에서 바로 열립니다.
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

LOCAL_TICKER = "2330.TW"     # TSMC (대만 TWSE)
ADR_TICKER = "TSM"           # TSMC ADR (NYSE)
FX_TICKER = "TWD=X"          # USD/TWD 환율
ADR_RATIO = 5                 # 1 ADR = 보통주 5주 (TSMC 기준)

# 캔들 색상: 상승 = 빨강, 하락 = 파랑
INCREASING_COLOR = "#d62728"
DECREASING_COLOR = "#1f77b4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSMC (TWSE) vs TSM (NYSE ADR) 가격 비교 차트를 생성합니다."
    )
    parser.add_argument(
        "--months",
        "-n",
        type=int,
        required=True,
        help="조회할 개월 수 (오늘로부터 n개월 전까지의 일간 데이터를 사용, "
        "예: TSM ADR 상장 시점인 1997년까지 전체를 보려면 약 360)",
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


def build_data(months: int):
    end_date = datetime.today() + timedelta(days=1)  # 오늘자 데이터까지 포함
    start_date = end_date - timedelta(days=int(months * 30.44) + 5)

    print(f"데이터 조회 기간: {start_date.date()} ~ {end_date.date()}")

    print(f"[1/5] {LOCAL_TICKER} (TWSE) 수정주가(adjusted) 데이터 가져오는 중...")
    local = fetch_ohlc(LOCAL_TICKER, start_date, end_date, adjusted=True)

    print(f"[2/5] {ADR_TICKER} (NYSE ADR) 수정주가(adjusted) 데이터 가져오는 중...")
    adr = fetch_ohlc(ADR_TICKER, start_date, end_date, adjusted=True)

    print(f"[3/5] {FX_TICKER} (USD/TWD 환율) 데이터 가져오는 중...")
    # 참고: Yahoo Finance의 USD/TWD 환율 데이터는 통상 2000년대 초반부터 제공됩니다.
    # --months 값이 매우 커서(예: 30년) 그 이전 구간을 요청하면, 아래 fx_aligned_*에서
    # 가장 오래된 환율값으로 backward-fill 되어 근사치로 처리됩니다.
    fx_close = fetch_ohlc(FX_TICKER, start_date, end_date, adjusted=False)["Close"]

    print(f"[4/5] {LOCAL_TICKER} 원시(raw) 종가 데이터 가져오는 중 (시가배당율 계산용)...")
    # 시가배당율은 "그 날 실제 거래된 시장가" 기준이어야 하므로 수정주가가 아닌 raw 종가를 사용
    local_raw_close = fetch_ohlc(LOCAL_TICKER, start_date, end_date, adjusted=False)["Close"]

    print(f"[5/5] {LOCAL_TICKER} 배당 이력(시가배당율 계산용) 가져오는 중...")
    dividend_yield = fetch_dividend_yield(LOCAL_TICKER, local_raw_close)

    # 참고용: 가장 최근 USD/TWD 환율 (차트 제목 표시용)
    current_usd2twd = float(fx_close.iloc[-1])
    print(f"참고 - 최근 USD/TWD 환율(최근 종가 기준): {current_usd2twd:,.2f}")

    # TSM 거래일에 맞춰 일별 환율을 정렬(직전 영업일 환율로 forward-fill)
    fx_aligned_adr = fx_close.sort_index().reindex(adr.index, method="ffill")
    fx_aligned_adr = fx_aligned_adr.bfill()  # 맨 앞부분에 결측치가 있으면 뒤 값으로 채움

    # TSM(USD, ADR) -> 원주 1주 환산 대만달러가 (TWD) : 일별 환율 반영
    # 1 ADR = 원주 5주이므로, ADR가(USD)*환율 은 "원주 5주"의 가치를 나타냄
    # -> 원주 1주 기준으로 비교하려면 ADR_RATIO(5)로 나눠야 함
    tsm_twd = adr.mul(fx_aligned_adr / ADR_RATIO, axis=0)

    # 종가 기준으로 날짜 정렬 (대만/미국 거래일 차이는 forward-fill) - 하단 % 차이 계산용
    close_df = pd.concat(
        [local["Close"].rename(LOCAL_TICKER), adr["Close"].rename(ADR_TICKER)], axis=1
    ).sort_index()
    close_df = close_df.ffill().dropna()

    # close_df 날짜에 맞춰서도 일별 환율을 별도로 정렬 (해당 날짜의 환율 사용)
    fx_aligned_close = fx_close.sort_index().reindex(close_df.index, method="ffill").bfill()

    # TSM(ADR)를 "해당 날짜의" 환율로 원주 1주 기준 TWD 환산 후, 2330.TW 대비 몇 % 차이 나는지 계산
    # (1 ADR = 원주 5주 이므로 ADR_RATIO로 나눠 원주 1주 환산가를 구함)
    adr_equiv_twd_daily = close_df[ADR_TICKER] * fx_aligned_close / ADR_RATIO
    pct_diff = (adr_equiv_twd_daily / close_df[LOCAL_TICKER] - 1) * 100
    pct_diff.name = "pct_diff"

    return local, tsm_twd, pct_diff, current_usd2twd, dividend_yield


def build_figure(local, tsm_twd, pct_diff, current_usd2twd, dividend_yield, months):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            f"{LOCAL_TICKER} vs {ADR_TICKER}÷{ADR_RATIO} (TWD, 오른쪽) "
            f"+ Yearly Dividend Rate (%, 왼쪽) — Candlestick",
            f"Premium/Discount: {ADR_TICKER}÷{ADR_RATIO}×USD/TWD(해당일 환율 반영) "
            f"vs {LOCAL_TICKER} (%)",
        ),
    )

    # --- 상단 차트: 캔들차트 2개 (오른쪽 y축, TWD), 시가배당율 라인 (왼쪽 y축, %) ---
    fig.add_trace(
        go.Scatter(
            x=dividend_yield.index,
            y=dividend_yield,
            name=f"{LOCAL_TICKER} 시가배당율 (%, 왼쪽)",
            mode="lines",
            line=dict(color="#8c564b", width=2, dash="dot"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Candlestick(
            x=local.index,
            open=local["Open"],
            high=local["High"],
            low=local["Low"],
            close=local["Close"],
            name=f"{LOCAL_TICKER} (TWD)",
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Candlestick(
            x=tsm_twd.index,
            open=tsm_twd["Open"],
            high=tsm_twd["High"],
            low=tsm_twd["Low"],
            close=tsm_twd["Close"],
            name=f"{ADR_TICKER}÷{ADR_RATIO} 환산 (TWD)",
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Yearly Dividend Rate (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price (TWD)", row=1, col=1, secondary_y=True)
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
        title=f"TSMC (2330.TW) vs TSM ADR — 최근 {months}개월",
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

    local, tsm_twd, pct_diff, current_usd2twd, dividend_yield = build_data(args.months)
    fig = build_figure(local, tsm_twd, pct_diff, current_usd2twd, dividend_yield, args.months)

    output_path = "tsmc_adr_comparison.html"
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

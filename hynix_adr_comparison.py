#!/usr/bin/env python3
"""
hynix_adr_comparison.py
------------------------
SK하이닉스(KOSPI: 000660.KS)와 나스닥에 상장된 ADR(SKHY)의 가격을 비교하는 스크립트.

- 상단 차트: 000660.KS 캔들차트 + SKHY(원화 환산) 캔들차트를 단일 y축(KRW)에 표시
  * SKHY는 일별 USD/KRW 환율을 곱하고, ADR 비율(1 ADR = 보통주 10주)을 반영해
    "보통주 환산 원화가"로 변환해 000660.KS와 같은 축(KRW)에서 직접 비교 가능
  * 두 캔들차트 모두 상승일=빨강, 하락일=파랑 (한국 관례)
- 하단 차트: (SKHY * 10 * 현재 USD/KRW환율) 이 000660.KS 대비 몇 % 차이 나는지 (%)
  * 하단 차트 x축에는 range slider(슬라이드 바)가 있어 원하는 구간을 드래그로 선택/확대 가능

사용법:
    python hynix_adr_comparison.py --months 6

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
        help="조회할 개월 수 (오늘로부터 n개월 전까지의 일간 데이터를 사용)",
    )
    return parser.parse_args()


def fetch_ohlc(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """지정한 티커의 일간 OHLC(Open, High, Low, Close)를 DataFrame으로 반환합니다."""
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
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


def build_data(months: int):
    end_date = datetime.today() + timedelta(days=1)  # 오늘자 데이터까지 포함
    start_date = end_date - timedelta(days=int(months * 30.44) + 5)

    print(f"데이터 조회 기간: {start_date.date()} ~ {end_date.date()}")

    print(f"[1/3] {KOSPI_TICKER} (KOSPI) 데이터 가져오는 중...")
    kospi = fetch_ohlc(KOSPI_TICKER, start_date, end_date)

    print(f"[2/3] {ADR_TICKER} (Nasdaq ADR) 데이터 가져오는 중...")
    adr = fetch_ohlc(ADR_TICKER, start_date, end_date)

    print(f"[3/3] {FX_TICKER} (USD/KRW 환율) 데이터 가져오는 중...")
    fx_close = fetch_ohlc(FX_TICKER, start_date, end_date)["Close"]

    # 참고용: 가장 최근 USD/KRW 환율 (차트 제목 표시용)
    current_usd2krw = float(fx_close.iloc[-1])
    print(f"참고 - 최근 USD/KRW 환율(최근 종가 기준): {current_usd2krw:,.2f}")

    # SKHY 거래일에 맞춰 일별 환율을 정렬(직전 영업일 환율로 forward-fill)
    fx_aligned_adr = fx_close.sort_index().reindex(adr.index, method="ffill")
    fx_aligned_adr = fx_aligned_adr.bfill()  # 맨 앞부분에 결측치가 있으면 뒤 값으로 채움

    # SKHY(USD, ADR) -> 보통주 환산 원화가 (KRW) : 일별 환율 반영
    skhy_krw = adr.mul(fx_aligned_adr * ADR_RATIO, axis=0)

    # 종가 기준으로 날짜 정렬 (한국/미국 거래일 차이는 forward-fill) - 하단 % 차이 계산용
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

    return kospi, skhy_krw, pct_diff, current_usd2krw


def build_figure(kospi, skhy_krw, pct_diff, current_usd2krw, months):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.62, 0.38],
        subplot_titles=(
            f"{KOSPI_TICKER} vs {ADR_TICKER}×{ADR_RATIO} (환율 반영, KRW 단일 축) — Candlestick",
            f"Premium/Discount: {ADR_TICKER}×{ADR_RATIO}×USD/KRW(해당일 환율 반영) "
            f"vs {KOSPI_TICKER} (%)",
        ),
    )

    # --- 상단 차트: 캔들차트 2개, 단일 y축(KRW), 한국식 색상(상승=빨강/하락=파랑) ---
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
    )

    fig.update_yaxes(title_text="Price (KRW)", side="right", row=1, col=1)
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

    kospi, skhy_krw, pct_diff, current_usd2krw = build_data(args.months)
    fig = build_figure(kospi, skhy_krw, pct_diff, current_usd2krw, args.months)

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

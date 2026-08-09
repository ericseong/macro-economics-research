#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semi_valuation_chart.py
------------------------
메모리 반도체 주(또는 임의 종목)의 장기 valuation 대비 이익을 한 눈에 보기 위한 차트.

한 차트(1개의 plot area) 안에 아래 시계열을 함께 그린다.
  1) daily 주가 (OHLC candlestick, 좌측 축) - 항상 표시
  2) daily PER   (우측 축, --per 지정 시에만 표시)
  3) daily PBR   (우측 축, --pbr 지정 시에만 표시)
  4) 분기별 영업이익 (막대, 우측 축) - 데이터가 있으면 항상 표시

데이터 소스
  - PER/PBR/OHLC(일봉) : ../quickndirty/msdata/{symbol}_d.csv   (--msdata-dir 로 변경 가능)
  - 분기별 영업이익      : ../quickndirty/fsdata/{symbol}.csv     (--fsdata-dir 로 변경 가능)
  로컬 CSV가 없으면 yfinance 로 주가만이라도 받아오도록 fallback 한다
  (PER/PBR/영업이익은 yfinance 로는 안정적으로 구하기 어려워 그 경우 해당 라인은 생략된다).

사용 예
  python semi_valuation_chart.py --symbol 005930 --years 10 --per --pbr
  python semi_valuation_chart.py --symbol 000660 --years 5 --per --output sk_hynix.html

출력
  --output 로 지정한 경로(기본 {symbol}_valuation.html)에 인터랙티브 plotly 차트를 저장한다.
  차트 하단에 range slider 가 있어 zoom in/out 이 가능하고, 마우스 스크롤로도 확대/축소할 수 있다
  (scrollZoom 옵션 활성화).
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DATE_COL = "날짜"


# --------------------------------------------------------------------------- #
# 데이터 로딩
# --------------------------------------------------------------------------- #
def load_daily_data(symbol: str, msdata_dir: str) -> pd.DataFrame:
    """일별 주가/PER/PBR 데이터를 로드한다. 로컬 CSV 우선, 없으면 yfinance로 주가만 fallback."""
    path = Path(msdata_dir) / f"{symbol}_d.csv"

    if path.exists():
        df = pd.read_csv(path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)

        # 필요한 컬럼만 정리 (없는 컬럼은 NaN 처리)
        for col in ("o", "h", "l", "c", "per", "pbr"):
            if col not in df.columns:
                df[col] = pd.NA

        out = df[[DATE_COL, "o", "h", "l", "c", "per", "pbr"]].rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close"}
        )
        return out

    # ---- fallback: yfinance 로 주가만 가져오기 ----
    print(
        f"[경고] 로컬 파일을 찾을 수 없습니다: {path} "
        f"-> yfinance 로 주가만 가져옵니다 (PER/PBR 은 비어 있게 됩니다).",
        file=sys.stderr,
    )
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance 가 설치되어 있지 않습니다. `pip install yfinance` 후 다시 시도하거나, "
            f"{path} 경로에 데이터를 준비해주세요."
        )

    # 국내 종목코드는 거래소 접미사가 필요 (.KS: 코스피, .KQ: 코스닥). 코스피를 우선 시도.
    ticker_candidates = [f"{symbol}.KS", f"{symbol}.KQ", symbol]
    hist = None
    for tkr in ticker_candidates:
        try:
            h = yf.Ticker(tkr).history(period="max")
            if not h.empty:
                hist = h
                break
        except Exception:
            continue

    if hist is None or hist.empty:
        raise SystemExit(f"yfinance 에서도 '{symbol}' 데이터를 가져오지 못했습니다.")

    hist = hist.reset_index()
    hist[DATE_COL] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    out = pd.DataFrame(
        {
            DATE_COL: hist[DATE_COL],
            "open": hist["Open"],
            "high": hist["High"],
            "low": hist["Low"],
            "close": hist["Close"],
            "per": pd.NA,
            "pbr": pd.NA,
        }
    )
    return out


def load_quarterly_op_income(symbol: str, fsdata_dir: str) -> pd.DataFrame | None:
    """분기별 영업이익 데이터를 로드한다. 없으면 None 을 반환한다."""
    path = Path(fsdata_dir) / f"{symbol}.csv"

    if not path.exists():
        print(
            f"[경고] 분기별 영업이익 파일을 찾을 수 없습니다: {path} "
            f"-> 영업이익 막대그래프 없이 진행합니다.",
            file=sys.stderr,
        )
        return None

    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # 이 데이터 소스는 'op_margin' 컬럼명이지만 실제로는 분기 영업이익(원) 값을 담고 있다.
    if "op_income" in df.columns:
        income_col = "op_income"
    elif "op_margin" in df.columns:
        income_col = "op_margin"
    else:
        # 그 외 이름으로 들어올 경우를 대비해 revenue/날짜가 아닌 첫 숫자 컬럼을 사용
        numeric_cols = [c for c in df.columns if c not in (DATE_COL, "revenue")]
        if not numeric_cols:
            return None
        income_col = numeric_cols[0]

    out = df[[DATE_COL, income_col]].rename(columns={income_col: "op_income"})
    out = out.dropna(subset=["op_income"])
    return out


# --------------------------------------------------------------------------- #
# 차트 생성
# --------------------------------------------------------------------------- #
def build_chart(
    daily: pd.DataFrame,
    quarterly: pd.DataFrame | None,
    symbol: str,
    years: int,
    show_per: bool,
    show_pbr: bool,
) -> go.Figure:
    fig = go.Figure()

    # 1) 주가 캔들차트 (좌측 축, yaxis) - OHLC 사용
    fig.add_trace(
        go.Candlestick(
            x=daily[DATE_COL],
            open=daily["open"],
            high=daily["high"],
            low=daily["low"],
            close=daily["close"],
            name="주가",
            increasing=dict(line=dict(color="#d62728"), fillcolor="#d62728"),  # 국내 관례: 상승 빨강
            decreasing=dict(line=dict(color="#1f77b4"), fillcolor="#1f77b4"),  # 하락 파랑
            yaxis="y",
        )
    )

    # 우측에 붙는 축들 (PER, PBR, 영업이익)을 선택된 것만 순서대로 구성한다.
    right_axes = []  # list of dicts: {trace, title, showgrid}

    if show_per:
        right_axes.append(
            dict(
                trace=go.Scatter(
                    x=daily[DATE_COL],
                    y=daily["per"],
                    name="PER",
                    mode="lines",
                    line=dict(color="#ff7f0e", width=1.2, dash="dot"),
                    connectgaps=False,
                ),
                title="PER (배)",
                showgrid=True,
            )
        )

    if show_pbr:
        right_axes.append(
            dict(
                trace=go.Scatter(
                    x=daily[DATE_COL],
                    y=daily["pbr"],
                    name="PBR",
                    mode="lines",
                    line=dict(color="#2ca02c", width=1.2, dash="dash"),
                    connectgaps=False,
                ),
                title="PBR (배)",
                showgrid=True,
            )
        )

    if quarterly is not None and not quarterly.empty:
        right_axes.append(
            dict(
                trace=go.Bar(
                    x=quarterly[DATE_COL],
                    y=quarterly["op_income"] / 1e8,  # 억원 단위
                    name="분기 영업이익(억원)",
                    marker=dict(color="#9467bd"),
                    opacity=0.35,
                    width=60 * 24 * 3600 * 1000,  # 약 60일 폭 (ms)
                ),
                title="영업이익 (억원)",
                showgrid=False,
            )
        )

    n_right = len(right_axes)
    # 오른쪽에 축이 필요한 개수만큼 plot 영역(domain)을 왼쪽으로 좁힌다.
    domain_right = {0: 1.0, 1: 0.90, 2: 0.85}.get(n_right, 0.80)

    for i, axis_info in enumerate(right_axes):
        axis_key = f"y{i + 2}"  # y2, y3, y4 ...
        axis_info["trace"].update(yaxis=axis_key)
        fig.add_trace(axis_info["trace"])

    fig.update_layout(
        title=f"{symbol} - 장기 Valuation vs 이익 ({years}년)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=70, r=60 + 60 * n_right, t=80, b=60),
        xaxis=dict(
            domain=[0.0, domain_right],
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="all"),
                ]
            ),
            type="date",
        ),
        yaxis=dict(
            title="주가 (원)",
            side="left",
        ),
    )

    for i, axis_info in enumerate(right_axes):
        axis_num = i + 2  # 2, 3, 4 ...
        if i == 0:
            axis_layout = dict(
                title=axis_info["title"],
                overlaying="y",
                side="right",
                anchor="x",
                showgrid=axis_info["showgrid"],
            )
        else:
            position = (
                domain_right
                if n_right == 1
                else domain_right + (1.0 - domain_right) * i / (n_right - 1)
            )
            axis_layout = dict(
                title=axis_info["title"],
                overlaying="y",
                side="right",
                anchor="free",
                position=position,
                showgrid=axis_info["showgrid"],
            )
        fig.update_layout(**{f"yaxis{axis_num}": axis_layout})

    return fig


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="반도체(또는 임의 종목)의 장기 Valuation(PER/PBR) vs 주가/영업이익 차트 생성"
    )
    parser.add_argument(
        "--symbol", default="005930", help="종목코드 (예: 005930, 000660). 기본값 005930"
    )
    parser.add_argument(
        "--years", type=int, default=10, help="관찰 window (년). 기본값 10"
    )
    parser.add_argument(
        "--msdata-dir",
        default="../quickndirty/msdata",
        help="일별 주가/PER/PBR csv 가 있는 디렉토리",
    )
    parser.add_argument(
        "--fsdata-dir",
        default="../quickndirty/fsdata",
        help="분기별 영업이익 csv 가 있는 디렉토리",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 html 경로 (기본값: {symbol}_valuation.html)",
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="PER 라인을 차트에 포함 (지정하지 않으면 PER은 그리지 않음)",
    )
    parser.add_argument(
        "--pbr",
        action="store_true",
        help="PBR 라인을 차트에 포함 (지정하지 않으면 PBR은 그리지 않음)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="저장 후 브라우저로 자동으로 열지 않음 (기본은 자동으로 엶)",
    )
    args = parser.parse_args()

    symbol = args.symbol
    years = args.years
    output = args.output or f"{symbol}_valuation.html"

    daily = load_daily_data(symbol, args.msdata_dir)
    quarterly = load_quarterly_op_income(symbol, args.fsdata_dir)

    # --years window 적용 (데이터 상 최신 날짜 기준)
    cutoff = daily[DATE_COL].max() - pd.DateOffset(years=years)
    daily = daily[daily[DATE_COL] >= cutoff].reset_index(drop=True)
    if quarterly is not None:
        quarterly = quarterly[quarterly[DATE_COL] >= cutoff].reset_index(drop=True)

    if daily.empty:
        raise SystemExit("선택한 기간에 해당하는 데이터가 없습니다.")

    fig = build_chart(daily, quarterly, symbol, years, show_per=args.per, show_pbr=args.pbr)

    fig.write_html(
        output,
        config={"scrollZoom": True, "displaylogo": False},
        include_plotlyjs="cdn",
    )
    print(f"저장 완료: {output}")

    if not args.no_open:
        output_path = Path(output).resolve()
        webbrowser.open(f"file://{output_path}")
        print(f"브라우저에서 열었습니다: {output_path}")


if __name__ == "__main__":
    main()

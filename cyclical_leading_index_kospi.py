"""
한국 선행종합지수 순환변동치 vs KOSPI 지수 시각화 스크립트
=================================================

기능
----
1. 한국은행 ECOS API에서 "선행종합지수 순환변동치"(경기종합지수, 월간)를
   최근 n년치 가져옵니다.
2. yfinance에서 KOSPI 지수(^KS11)의 일별 OHLC(캔들) 데이터를 같은 기간
   가져옵니다.
3. Plotly로 하나의 차트에 그립니다.
   - 왼쪽(y1) 축: 선행지수 순환변동치 (막대그래프)
   - 오른쪽(y2) 축: KOSPI 지수 (일별 캔들차트)
   - 하단에 range slider 표시, 마우스 스크롤로 확대/축소 가능
4. 결과를 `cyclical_components_leading_index_with_kospi_for_{n}_years.html`
   로 저장하고, 로컬 브라우저에서 자동으로 엽니다.

사전 준비
--------
    pip install requests pandas yfinance plotly

ECOS API 키 파일 위치 (기본값, --credential 옵션으로 변경 가능):
    credentials/credential_ecos_api.txt
    -> 파일 내용은 API 키 문자열 한 줄이면 됩니다. (앞뒤 공백/줄바꿈 무시)
    -> 키 발급: https://ecos.bok.or.kr/api/#/ 회원가입 후 [인증키 신청]

실행 예시
--------
    python cyclical_leading_index_kospi.py --years 10
    python cyclical_leading_index_kospi.py --years 5 --credential my_key.txt
"""

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ECOS_BASE_URL = "https://ecos.bok.or.kr/api"
# 경기종합지수(2020=100) 통계표 코드. ECOS 개편 등으로 코드가 바뀔 수 있어
# 아래에서 이름으로 재검색하는 fallback 로직을 함께 둔다.
DEFAULT_STAT_CODE = "901Y067"
KOSPI_TICKER = "^KS11"


def read_api_key(credential_path: str) -> str:
    """credentials 파일에서 ECOS API 키를 읽어온다."""
    path = Path(credential_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ECOS API 키 파일을 찾을 수 없습니다: {path.resolve()}\n"
            "credentials/credential_ecos_api.txt 경로에 API 키를 저장해두거나 "
            "--credential 옵션으로 경로를 지정해주세요."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{path} 파일이 비어 있습니다. API 키를 확인해주세요.")
    return key


def find_leading_cycle_item_code(api_key: str, stat_code: str = DEFAULT_STAT_CODE):
    """
    StatisticItemList API로 통계표 내 세부 항목을 조회해서
    이름에 '선행'과 '순환변동치'가 모두 들어간 항목의 코드를 찾는다.
    ECOS 쪽 통계표/항목 코드가 개편되어도 이름 기준 검색이라 견고하다.
    """
    url = (
        f"{ECOS_BASE_URL}/StatisticItemList/{api_key}/json/kr/1/100/{stat_code}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "StatisticItemList" not in data:
        # 통계표 코드 자체가 잘못됐을 가능성. 에러 메시지를 그대로 노출.
        raise RuntimeError(
            f"ECOS StatisticItemList 응답 오류: {data}\n"
            f"통계표 코드({stat_code})가 유효한지 https://ecos.bok.or.kr/api/#/DevGuide/DevBasicInfo "
            "에서 확인해주세요."
        )

    rows = data["StatisticItemList"]["row"]
    candidates = [
        r for r in rows
        if "선행" in r.get("ITEM_NAME", "") and "순환변동치" in r.get("ITEM_NAME", "")
    ]

    if not candidates:
        available = "\n".join(f"  - {r['ITEM_CODE']}: {r['ITEM_NAME']}" for r in rows)
        raise RuntimeError(
            "'선행' + '순환변동치'가 포함된 항목을 찾지 못했습니다. "
            f"통계표({stat_code}) 내 사용 가능한 항목 목록:\n{available}"
        )

    item = candidates[0]
    return stat_code, item["ITEM_CODE"], item["ITEM_NAME"]


def fetch_leading_cycle_index(api_key: str, years: int) -> pd.DataFrame:
    """ECOS에서 선행종합지수 순환변동치 월간 데이터를 최근 n년치 가져온다."""
    stat_code, item_code, item_name = find_leading_cycle_item_code(api_key)
    print(f"[ECOS] 사용 통계 항목: {item_name} (STAT_CODE={stat_code}, ITEM_CODE={item_code})")

    end_dt = datetime.today()
    start_dt = end_dt.replace(year=end_dt.year - years)
    start_ym = start_dt.strftime("%Y%m")
    end_ym = end_dt.strftime("%Y%m")

    # StatisticSearch: /{키}/json/kr/{시작건수}/{종료건수}/{통계표코드}/{주기}/{시작일}/{종료일}/{항목코드1}
    url = (
        f"{ECOS_BASE_URL}/StatisticSearch/{api_key}/json/kr/1/1000/"
        f"{stat_code}/M/{start_ym}/{end_ym}/{item_code}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "StatisticSearch" not in data:
        raise RuntimeError(f"ECOS StatisticSearch 응답 오류: {data}")

    rows = data["StatisticSearch"]["row"]
    df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
    df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m")
    df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df = df.rename(columns={"TIME": "date", "DATA_VALUE": "leading_cycle_index"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_kospi(years: int) -> pd.DataFrame:
    """yfinance로 KOSPI 지수 일별 OHLC 데이터를 최근 n년치 가져온다."""
    import yfinance as yf

    end_dt = datetime.today()
    start_dt = end_dt.replace(year=end_dt.year - years)

    df = yf.download(
        KOSPI_TICKER,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("yfinance에서 KOSPI(^KS11) 데이터를 가져오지 못했습니다.")

    # yfinance가 MultiIndex 컬럼을 반환하는 경우(최근 버전) 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={"Date": "date"})
    return df[["date", "Open", "High", "Low", "Close"]]


def build_chart(leading_df: pd.DataFrame, kospi_df: pd.DataFrame, years: int) -> go.Figure:
    # 위: KOSPI 캔들차트 / 아래: 선행지수 순환변동치 막대그래프. x축(시간축)은 공유.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=("KOSPI 지수", "선행종합지수 순환변동치"),
    )

    # 위쪽: KOSPI 일별 캔들차트
    fig.add_trace(
        go.Candlestick(
            x=kospi_df["date"],
            open=kospi_df["Open"],
            high=kospi_df["High"],
            low=kospi_df["Low"],
            close=kospi_df["Close"],
            name="KOSPI",
            increasing_line_color="red",   # 국내 관행: 상승=빨강
            decreasing_line_color="blue",  # 하락=파랑
        ),
        row=1,
        col=1,
    )

    # 아래쪽: 선행지수 순환변동치 (막대그래프)
    fig.add_trace(
        go.Bar(
            x=leading_df["date"],
            y=leading_df["leading_cycle_index"],
            name="선행종합지수 순환변동치",
            marker_color="rgba(99, 110, 250, 0.55)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"한국 선행종합지수 순환변동치 vs KOSPI 지수 (최근 {years}년)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )

    # x축은 공유되며, range slider는 맨 아래 subplot에만 표시
    fig.update_xaxes(type="date", rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(type="date", rangeslider=dict(visible=True), row=2, col=1)

    fig.update_yaxes(title_text="KOSPI 지수", row=1, col=1)
    # 순환변동치는 90(최소)~110(최대) 범위로 고정
    fig.update_yaxes(
        title_text="선행종합지수 순환변동치",
        range=[95, 110],
        row=2,
        col=1,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="한국 선행종합지수 순환변동치와 KOSPI 지수를 함께 시각화합니다."
    )
    parser.add_argument(
        "--years", type=int, default=10, help="최근 몇 년치 데이터를 볼지 (기본값: 10)"
    )
    parser.add_argument(
        "--credential",
        type=str,
        default="credentials/credential_ecos_api.txt",
        help="ECOS API 키가 저장된 파일 경로",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="생성 후 브라우저 자동 실행을 하지 않으려면 지정",
    )
    args = parser.parse_args()

    if args.years <= 0:
        print("오류: --years 는 1 이상의 정수여야 합니다.", file=sys.stderr)
        sys.exit(1)

    print(f"1) ECOS API 키 로드 중... ({args.credential})")
    api_key = read_api_key(args.credential)

    print(f"2) 선행종합지수 순환변동치 최근 {args.years}년치 조회 중...")
    leading_df = fetch_leading_cycle_index(api_key, args.years)
    print(f"   -> {len(leading_df)}개월 데이터 수집 완료 "
          f"({leading_df['date'].min():%Y-%m} ~ {leading_df['date'].max():%Y-%m})")

    print(f"3) KOSPI(^KS11) 일별 시세 최근 {args.years}년치 조회 중...")
    kospi_df = fetch_kospi(args.years)
    print(f"   -> {len(kospi_df)}일 데이터 수집 완료 "
          f"({kospi_df['date'].min():%Y-%m-%d} ~ {kospi_df['date'].max():%Y-%m-%d})")

    print("4) 차트 생성 중...")
    fig = build_chart(leading_df, kospi_df, args.years)

    out_path = Path(
        f"cyclical_components_leading_index_with_kospi_for_{args.years}_years.html"
    ).resolve()
    fig.write_html(str(out_path), config={"scrollZoom": True})
    print(f"5) 저장 완료: {out_path}")

    if not args.no_open:
        webbrowser.open(f"file://{out_path}")
        print("6) 브라우저에서 차트를 열었습니다.")


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime, timedelta
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go  # 이중 Y축 구조에 더 적합한 객체 사용
import yfinance as yf


def get_data(years):
    # 날짜 범위 설정
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    # 1. FRED에서 주간 신규 실업수당 청구 건수 (ICSA) 가져오기
    print(f"FRED에서 신규 실업수당 청구 건수(ICSA) 데이터를 가져오는 중... ({start_date.strftime('%Y-%m-%d')} ~)")
    try:
        df_unemployment = web.DataReader("ICSA", "fred", start_date, end_date)
        df_unemployment.index.name = 'Date'
        df_unemployment.columns = ['Unemployment_Claims']
    except Exception as e:
        print(f"FRED 데이터 로드 실패: {e}")
        return None

    # 2. Yahoo Finance에서 S&P 500 ETF (SPY) 데이터 가져오기
    print("Yahoo Finance에서 SPY 데이터를 가져오는 중...")
    df_spy = yf.download("SPY", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    # yfinance 버전 이슈 대응 (auto_adjust=True 반영)
    if isinstance(df_spy.columns, pd.MultiIndex):
        target_col = 'Close' if 'Close' in df_spy.columns.levels[0] else 'Adj Close'
        df_spy = df_spy[target_col]['SPY'].to_frame(name='SPY_Adj_Close')
    else:
        target_col = 'Close' if 'Close' in df_spy.columns else 'Adj Close'
        df_spy = df_spy[[target_col]].rename(columns={target_col: 'SPY_Adj_Close'})

    # 두 데이터 결합
    df_merged = pd.merge(df_unemployment, df_spy, left_index=True, right_index=True, how='outer').sort_index()

    # 주간 데이터와 일일 데이터 간의 공백을 선형 보간 처리
    df_merged = df_merged.interpolate(method='time')

    return df_merged

def plot_correlation(df, years):
    if df is None or df.empty:
        print("시각화할 데이터가 없습니다.")
        return

    # 이중 Y축을 가진 서브플롯 생성
    fig = go.Figure()

    # 왼쪽 Y축: 주간 신규 실업수당 청구 건수
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Unemployment_Claims'],
            name="신규 실업수당 청구건수 (FRED: ICSA)",
            line=dict(color="crimson", width=2),
            yaxis="y1"
        )
    )

    # 오른쪽 Y축: S&P 500 지수 (SPY)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SPY_Adj_Close'],
            name="S&P 500 지수 (Yahoo: SPY)",
            line=dict(color="royalblue", width=2),
            yaxis="y2"
        )
    )

    # 레이아웃 및 이중 Y축 상세 설정 (titlefont -> title_font로 전면 수정)
    fig.update_layout(
        title_text=f"최근 {years}년간 주간 신규실업건수 vs S&P 500 (SPY) 상관관계",
        title_x=0.5,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),

        # X축 및 하단 레인지 슬라이더 설정
        xaxis=dict(
            title="날짜",
            rangeslider=dict(visible=True),
            type="date"
        ),

        # 왼쪽 Y축 설정
        yaxis=dict(
            title="신규 실업수당 청구 건수 (개)",
            title_font=dict(color="crimson"),  # 최신 Plotly 속성으로 수정
            tickfont=dict(color="crimson")
        ),

        # 오른쪽 Y축 설정
        yaxis2=dict(
            title="S&P 500 (SPY) 주가 ($)",
            title_font=dict(color="royalblue"), # 최신 Plotly 속성으로 수정
            tickfont=dict(color="royalblue"),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )

    # 브라우저에 바로 표시
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="신규실업건수와 S&P500 지수 상관관계 시각화 스크립트")
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="최근 몇 년간의 데이터를 볼지 설정 (기본값: 5)",
    )

    args = parser.parse_args()

    data = get_data(args.years)
    plot_correlation(data, args.years)

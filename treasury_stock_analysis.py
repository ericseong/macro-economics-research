import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from datetime import datetime, timedelta

def get_data(symbols, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # 데이터 다운로드
    tickers = ["^TNX"] + symbols
    data = yf.download(tickers, start=start_date, end=end_date)['Close']

    # 결측치 제거 (휴장일 등 처리)
    data = data.ffill().dropna()
    return data

def main():
    # 1. Argument Parser 설정
    parser = argparse.ArgumentParser(description='Market Data Visualization Tool')
    parser.add_argument('--years', type=int, default=10, help='조회할 기간 (년)')
    parser.add_argument('--include', nargs='+', default=['^GSPC', '^IXIC', '^DJI'],
                        help='포함할 지수 심볼')

    args = parser.parse_args()

    # 심볼별 설명 매핑 (Legend 표시용)
    symbol_names = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ',
        '^DJI': 'Dow Jones',
        '^TNX': 'US 10Y Treasury'
    }

    # 2. 데이터 가져오기
    print(f"데이터를 불러오는 중... (최근 {args.years}년)")
    df = get_data(args.include, args.years)

    # 3. Plotly 차트 생성 (이중 축)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 10년물 국채 금리 추가 (검은색, 왼쪽 Y축)
    if '^TNX' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['^TNX'],
                name=f"^TNX ({symbol_names.get('^TNX')})",
                line=dict(color='black', width=2)
            ),
            secondary_y=False,
        )

    # 지수 데이터 정규화 및 추가 (오른쪽 Y축)
    # 각 지수의 첫 번째 유효 데이터를 1로 설정 (수익률 비교)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    for i, symbol in enumerate(args.include):
        if symbol in df.columns:
            # 정규화: (현재 값 / 시작 값)
            normalized_series = df[symbol] / df[symbol].iloc[0]

            display_name = f"{symbol} ({symbol_names.get(symbol, 'Index')})"

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized_series,
                    name=display_name,
                    line=dict(color=colors[i % len(colors)])
                ),
                secondary_y=True,
            )

    # 4. 레이아웃 및 인터랙션 설정
    fig.update_layout(
        title=f'Market Growth Comparison (Normalized) vs US 10Y Yield',
        xaxis_title='Date',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True), # 하단 스크롤바
            type='date'
        )
    )

    # Y축 제목 설정
    fig.update_yaxes(title_text="Treasury Yield (%)", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Normalized Growth (Start = 1.0)", secondary_y=True)

    # 마우스 휠 줌 활성화 설정
    fig.show(config={'scrollZoom': True})

if __name__ == "__main__":
    main()

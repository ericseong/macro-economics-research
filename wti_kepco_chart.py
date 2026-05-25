import argparse
from datetime import datetime, timedelta
import webbrowser
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # 1. Command-line Arguments 설정 (--years)
    parser = argparse.ArgumentParser(description="WTI 유가 및 한국전력 주가 시각화 스크립트")
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='조회할 최근 연도 수 (기본값: 3)'
    )
    args = parser.parse_args()

    # 2. 데이터 수집 기간 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"데이터 수집 기간: {start_str} ~ {end_str} (최근 {args.years}년)")

    # 3. yfinance를 통한 데이터 다운로드
    print("Yahoo Finance로부터 데이터를 다운로드 중입니다...")
    kepco = yf.download("015760.KS", start=start_str, end=end_str)
    wti = yf.download("CL=F", start=start_str, end=end_str)

    if kepco.empty or wti.empty:
        print("데이터를 불러오지 못했습니다. 티커 및 인터넷 연결을 확인하세요.")
        return

    # Multi-index 컬럼 단일화 처리
    if isinstance(kepco.columns, pd.MultiIndex):
        kepco.columns = kepco.columns.get_level_values(0)
    if isinstance(wti.columns, pd.MultiIndex):
        wti.columns = wti.columns.get_level_values(0)

    # 4. Plotly Subplots 생성 (이중 Y축 적용)
    # - Row 1: WTI & 주가 통합 (이중 Y축)
    # - Row 2: 거래량 (단일 Y축)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],  # 상단 통합 차트를 75% 비율로 크게 배분
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]]
    )

    # [Row 1] WTI 유가 라인차트 (왼쪽 Y축: secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=wti.index,
            y=wti['Close'],
            mode='lines',
            name='WTI 유가 ($)',
            line=dict(color='darkorange', width=2)
        ),
        row=1, col=1, secondary_y=False
    )

    # [Row 1] 한국전력 캔들차트 (오른쪽 Y축: secondary_y=True)
    fig.add_trace(
        go.Candlestick(
            x=kepco.index,
            open=kepco['Open'],
            high=kepco['High'],
            low=kepco['Low'],
            close=kepco['Close'],
            name="한국전력 주가",
            increasing_line_color='red',
            decreasing_line_color='blue'
        ),
        row=1, col=1, secondary_y=True
    )

    # [Row 2] 한국전력 거래량 바 차트
    fig.add_trace(
        go.Bar(
            x=kepco.index,
            y=kepco['Volume'],
            name='거래량',
            marker_color='cadetblue',
            opacity=0.7
        ),
        row=2, col=1, secondary_y=False
    )

    # 5. 레이아웃 및 축 타이틀 고도화
    fig.update_layout(
        title=dict(
            text=f"최근 {args.years}개년 한국전력 주가 vs WTI 국제유가 통합 분석",
            x=0.5,
            font=dict(size=20, weight='bold')
        ),
        template="plotly_white",
        hovermode="x unified",
        height=900,
        showlegend=False,
        # 캔들차트에 자동으로 붙는 상단 기본 슬라이더 강제 제거
        xaxis_rangeslider_visible=False
    )

    # 각 Y축별 직관적인 타이틀 및 색상 지정
    fig.update_yaxes(title_text="WTI 유가 (USD/bbl)", title_font=dict(color="darkorange"), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="한국전력 주가 (KRW)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="거래량 (주)", row=2, col=1)

    # X축 하단 통합 Range Slider 설정 (Row 2 거래량 차트 하단에 부착)
    fig.update_layout(
        xaxis2=dict(
            title="날짜 (Date)",
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # 6. HTML 파일로 저장 및 브라우저 강제 팝업 실행
    output_filename = "kepco_wti_advanced.html"

    # scrollZoom 옵션 유지
    fig.write_html(
        output_filename,
        config={'scrollZoom': True}
    )

    full_path = os.path.abspath(output_filename)
    print(f"성공적으로 통합 차트를 생성했습니다: {full_path}")
    webbrowser.open(f"file://{full_path}")

if __name__ == "__main__":
    main()

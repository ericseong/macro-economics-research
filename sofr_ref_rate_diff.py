import os
import argparse
import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go

# --- 1. 환경 설정 및 FRED API Key 파일 로드 ---
API_KEY_PATH = './credentials/credential_fred_api.txt'

def load_api_key(path):
    """지정된 경로에서 FRED API Key를 로드합니다."""
    try:
        with open(path, 'r') as f:
            api_key = f.read().strip()
            if not api_key:
                 raise ValueError("API Key 파일이 비어 있습니다.")
            return api_key
    except FileNotFoundError:
        print(f"⚠️ 오류: API Key 파일 경로를 찾을 수 없습니다: {path}")
        print("파일 경로 및 파일명(`credential_fred_api.txt`)을 확인해 주세요.")
        raise
    except Exception as e:
        print(f"⚠️ 오류: API Key 파일을 읽는 중 오류가 발생했습니다: {e}")
        raise

try:
    API_KEY = load_api_key(API_KEY_PATH)
    fred = Fred(api_key=API_KEY)
except (FileNotFoundError, ValueError, Exception):
    print("스크립트를 실행할 수 없습니다. 위의 오류 메시지를 확인하세요.")
    raise SystemExit

# --- 2. FRED 시리즈 ID 정의 ---
SERIES_IDS = {
    'SOFR': 'SOFR',              # Secured Overnight Financing Rate (일별)
    'Fed_Funds_Upper': 'DFEDTARU', # Federal Funds Target Range - Upper Limit (일별)
    'Fed_Funds_Lower': 'DFEDTARL', # Federal Funds Target Range - Lower Limit (일별)
}

# --- 3. 데이터 로드 및 필터링 함수 ---
def fetch_and_filter_data(series_ids, period_str):
    """
    FRED API에서 데이터를 가져오고, 지정된 기간(예: 365d)으로 필터링합니다.
    SOFR 데이터가 시작된 '2018-04-02' 이후의 전체 데이터를 가져온 후 필터링합니다.
    """
    print("FRED 데이터 다운로드 중...")

    # 전체 기간 데이터를 먼저 로드합니다. (FRED API의 최소 시작일)
    start_date_all = '2018-04-02'
    data = {}
    for name, series_id in series_ids.items():
        try:
            series_data = fred.get_series(series_id, observation_start=start_date_all)
            data[name] = series_data.rename(name)
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")

    # 모든 시리즈를 날짜(인덱스) 기준으로 병합
    df = pd.concat(data.values(), axis=1).sort_index()
    df.index.name = 'Date'

    print("데이터 로드 완료. 기간 필터링 중...")

    # 지정된 기간(예: '365d')만큼 데이터를 필터링합니다.
    try:
        # 현재 시간으로 Timezone-aware datetime 객체를 생성하여 max()를 구하는 것이 가장 안전하지만,
        # FRED 데이터의 인덱스는 일반적으로 naive datetime이므로 그대로 진행합니다.
        end_date = df.index.max()
        if end_date is None:
             # 데이터가 아예 없는 경우 (API 연결 실패 등)
             raise ValueError("로드된 데이터가 없습니다.")

        offset = pd.Timedelta(period_str)
        # 현재 날짜 기준이 아닌, 로드된 데이터의 마지막 날짜 기준으로 역산
        start_date_filter = end_date - offset

        df_filtered = df[df.index >= start_date_filter]

        if df_filtered.empty:
             raise ValueError(f"기간 {period_str} 내에 데이터가 없습니다. 전체 데이터를 사용합니다.")

        # 사용자에게 필터링된 기간을 명확히 안내
        print(f"최근 {period_str}의 데이터 ({df_filtered.index.min().strftime('%Y-%m-%d')} ~ {df_filtered.index.max().strftime('%Y-%m-%d')})로 차트를 생성합니다.")
        return df_filtered

    except Exception as e:
        print(f"⚠️ 경고: 기간 필터링 중 오류 발생 ({e}). 전체 로드된 데이터를 사용합니다.")
        return df

# --- 4. Plotly 시각화 함수 ---
def plot_interactive_rates(df, title, period_str):
    """Plotly를 사용하여 대화형 금리 차트를 생성하고 표시합니다."""
    print("Plotly 차트 생성 중...")

    # SOFR - Fed Funds Midpoint 계산
    df['Fed_Funds_Midpoint'] = (df['Fed_Funds_Upper'] + df['Fed_Funds_Lower']) / 2
    df['SOFR_vs_Fed_Funds_Diff'] = df['SOFR'] - df['Fed_Funds_Midpoint']

    fig = go.Figure()

    # 1. Fed Funds Target Range (기준금리 범위)를 면적(Shaded Area)으로 표시
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Fed_Funds_Lower'], mode='lines', line=dict(width=0),
        showlegend=False, name='Fed Funds Lower Limit'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Fed_Funds_Upper'], mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)',
        name='Fed Funds Target Range (연준 기준금리 범위)'
    ))

    # 2. SOFR 일별 데이터 플롯
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SOFR'], mode='lines', line=dict(color='red', width=2.5),
        name='SOFR (Secured Overnight Financing Rate)'
    ))

    # 3. Target Range 상/하한선 (선으로 강조)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Fed_Funds_Upper'], mode='lines',
        line=dict(color='gray', width=1, dash='dash'), name='Target Range Upper Limit', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Fed_Funds_Lower'], mode='lines',
        line=dict(color='gray', width=1, dash='dot'), name='Target Range Lower Limit', showlegend=False
    ))

    # 4. 추가된 그래프: (SOFR 금리) - (연준 기준금리 중간값) 차이 (Orange)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SOFR_vs_Fed_Funds_Diff'], mode='lines',
        line=dict(color='orange', width=2, dash='solid'),
        name='SOFR - Fed Funds Midpoint (차이)'
    ))

    # --- 차트 레이아웃 설정 ---
    full_title = f"{title} (기간: 최근 {period_str})"
    fig.update_layout(
        title={
            'text': full_title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="날짜 (Date)",
        yaxis_title="금리 (Percent, %)",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
        # 🚨 수정된 부분: dragmode='pan'으로 설정하여 마우스 드래그를 이동으로 설정
        dragmode='pan',
        # Range Slider 및 Selector 활성화
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )

    # Y축 포맷을 소수점 두 자리 숫자로 지정 (FRED 데이터는 이미 % 값)
    fig.update_yaxes(
        tickformat=".2f"
    )

    fig.show()
    print("대화형 차트가 기본 웹 브라우저에서 열렸습니다.")

# --- 5. 메인 실행 블록 ---
if __name__ == '__main__':
    # Argument Parser 설정
    parser = argparse.ArgumentParser(description="FRED API를 사용하여 SOFR 및 연준 기준금리 일별 추이 차트를 생성합니다.")
    parser.add_argument(
        '--days',
        type=str,
        default='1095d',
        help="차트 표시 기간을 설정합니다. 예: '10d', '100d', '365d'. 기본값은 365d입니다."
    )
    args = parser.parse_args()

    # 1. 데이터 로드 및 기간 필터링
    rates_df = fetch_and_filter_data(SERIES_IDS, args.days)

    # 2. Plotly 차트 생성
    plot_title = "일별 SOFR 및 연준 기준금리(Fed Funds Target Range) 추이"
    plot_interactive_rates(rates_df, plot_title, args.days)

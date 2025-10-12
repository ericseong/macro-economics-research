#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S&P500 & VIX (5m) autosell signal visualizer (Compact Axis default)

핵심
- 브라우저 직접 표시 (Plotly)
- 시간은 Asia/Seoul 로 표기
- 미국 정규장(US/Eastern 09:30–16:00) 분봉만 사용 (장외/주말/휴일 제거)
- 기본: '압축 시간축(Compact Axis)'로 비거래 시간대를 축에서 제거 -> 노이즈 0, 마지막 새벽도 확실히 보임
- 옵션: --axis time 으로 '실시간(Time Axis + rangebreaks)' 모드도 가능
- 입력 윈도우:
  * --start/--end 둘 중 하나라도 없으면 기본값: end=현재(KST), start=30일 전(KST)
  * 입력이 'YYYY-MM-DD' 형식이면 start=그날 00:00, end=다음날 00:00(포함일) 해석
  * 입력이 'YYYY-MM-DD HH:MM[:SS]' 형식이면 해당 시각 그대로 경계를 사용
- 급락 검출 규칙:
  * 각 시각 t에서 '직전 5영업일' 참조구간의 VIX 평균 μ(t), 표준편차 σ(t)
  * '직전 60분' 창 W(t) 내에 |VIX - μ(t)| ≥ k·σ(t) 성립하면 t에서 신호 (k 기본 3)

Usage:
  python spx_vix_autosell.py [--start 2025-09-12[ 09:30]] [--end 2025-10-12[ 16:00]] \
                             [--sigma_k 3] [--min_ref_bars 120] [--axis compact|time] [--debug]
"""

import argparse
import re
from datetime import timedelta, time as dtime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pandas.tseries.offsets import BusinessDay

ASIA_SEOUL = "Asia/Seoul"
US_EASTERN = "America/New_York"

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="KST 기준 시작 (YYYY-MM-DD 또는 YYYY-MM-DD HH:MM[:SS])")
    p.add_argument("--end", help="KST 기준 끝 (YYYY-MM-DD 또는 YYYY-MM-DD HH:MM[:SS])")
    p.add_argument("--sigma_k", type=float, default=6.0, help="k in |VIX-μ| ≥ k·σ (default=3.0)")
    p.add_argument("--min_ref_bars", type=int, default=120, help="5영업일 참조 표본 최소 개수 (부족하면 신호 미계산)")
    p.add_argument("--axis", choices=["compact", "time"], default="compact",
                   help="X-axis mode: 'compact' (default, no off-hours) or 'time' (with rangebreaks)")
    p.add_argument("--debug", action="store_true", help="Print detailed diagnostics")
    return p.parse_args()

def _log(debug: bool, msg: str):
    if debug:
        print("[DEBUG]", msg)

# ----------------------------
# Time parsing helpers
# ----------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _parse_kst_maybe_date(s: str | None):
    """
    입력 문자열을 KST tz-aware Timestamp로 파싱.
    - None: 반환 None, None
    - 'YYYY-MM-DD': (kst_ts_at_00:00, True)
    - 'YYYY-MM-DD HH:MM[:SS]': (kst_ts_exact, False)
    """
    if s is None:
        return None, None
    s = s.strip()
    if _DATE_RE.match(s):
        ts = pd.Timestamp(s).tz_localize(ASIA_SEOUL)  # 00:00
        return ts, True
    # 나머지는 datetime으로 시도 (초는 선택)
    try:
        ts = pd.Timestamp(s).tz_localize(ASIA_SEOUL)
    except TypeError:
        ts = pd.Timestamp(s)
        if ts.tz is None:
            ts = ts.tz_localize(ASIA_SEOUL)
        else:
            ts = ts.tz_convert(ASIA_SEOUL)
    return ts, False

def resolve_view_window_kst(start_str: str | None, end_str: str | None, debug=False):
    """
    --start/--end 입력 해석:
      - 둘 중 하나라도 없으면: end=지금(KST), start=end-30일(KST)
      - date-only 인 경우: start는 00:00, end는 다음날 00:00로 해석
      - datetime 인 경우: 해당 시각 그대로
    반환: (view_lo_kst, view_hi_kst)
    """
    now_kst = pd.Timestamp.now(tz=ASIA_SEOUL)

    if start_str is None or end_str is None:
        end_kst = now_kst if end_str is None else _parse_kst_maybe_date(end_str)[0]
        if end_kst is None:
            end_kst = now_kst
        start_kst = (end_kst - pd.Timedelta(days=30)) if start_str is None else _parse_kst_maybe_date(start_str)[0]
        if start_kst is None:
            start_kst = end_kst - pd.Timedelta(days=30)

        _, start_is_date = _parse_kst_maybe_date(start_str) if start_str is not None else (None, False)
        _, end_is_date   = _parse_kst_maybe_date(end_str) if end_str is not None else (None, False)

        view_lo_kst = start_kst if not start_is_date else start_kst.normalize()
        view_hi_kst = end_kst
        if end_is_date:
            view_hi_kst = end_kst.normalize() + pd.Timedelta(days=1)

    else:
        start_kst, start_is_date = _parse_kst_maybe_date(start_str)
        end_kst,   end_is_date   = _parse_kst_maybe_date(end_str)

        view_lo_kst = start_kst if not start_is_date else start_kst.normalize()
        view_hi_kst = end_kst.normalize() + pd.Timedelta(days=1) if end_is_date else end_kst

    if view_lo_kst >= view_hi_kst:
        raise SystemExit(f"Invalid window: start >= end ({view_lo_kst} >= {view_hi_kst})")

    _log(debug, f"INPUT KST window (inclusive): [{view_lo_kst} .. {view_hi_kst})")
    return view_lo_kst, view_hi_kst

# ----------------------------
# Timezone helpers
# ----------------------------
def _ensure_tz(obj: pd.Series | pd.DataFrame, tz: str):
    out = obj.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out.tz_convert(tz)

def _detect_us_dst(seoul_ts: pd.Timestamp) -> bool:
    return "EDT" in seoul_ts.tz_convert(US_EASTERN).tzname()

# ----------------------------
# Data fetching (KST window -> UTC) with reference buffer
# ----------------------------
def fetch_5m_with_view(view_lo_kst: pd.Timestamp, view_hi_kst: pd.Timestamp, debug=False, ref_buffer_days:int=10):
    """
    μ,σ 참조(5영업일) 확보를 위해 조회 시작 이전에 ref_buffer_days 만큼 더 가져온 뒤,
    정규장 필터 → KST 변환 → 최종 뷰 윈도우로 컷팅.
    """
    # 버퍼 포함 조회 구간
    yf_lo_kst = view_lo_kst - pd.Timedelta(days=ref_buffer_days)
    start_utc = yf_lo_kst.tz_convert("UTC")
    end_utc   = view_hi_kst.tz_convert("UTC")
    _log(debug, f"YF UTC window (exclusive end): [{start_utc} .. {end_utc}) [buffer {ref_buffer_days}d]")
    _log(debug, "YF call uses exact datetimes (not dates).")

    tickers = ["^GSPC", "^VIX"]
    df = yf.download(
        tickers=tickers,
        interval="5m",
        start=start_utc.to_pydatetime(),
        end=end_utc.to_pydatetime(),
        auto_adjust=False,
        prepost=False,      # 장전/장후 제외
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if isinstance(df, pd.DataFrame) and len(df) == 0:
        raise SystemExit("No data returned from yfinance. (Check 5m lookback window & symbols)")

    def extract_close(df_all, ticker):
        if isinstance(df_all.columns, pd.MultiIndex):
            s = df_all[(ticker, "Close")].copy()
        else:
            s = df_all["Close"].copy()
        s.name = ticker
        return s

    spx_raw = extract_close(df, "^GSPC").dropna()
    vix_raw = extract_close(df, "^VIX").dropna()

    if debug and len(spx_raw) > 0:
        _log(True, f"RAW (^GSPC) UTC min/max: {spx_raw.index.min()} .. {spx_raw.index.max()} [{spx_raw.index.tz}]")
    if debug and len(vix_raw) > 0:
        _log(True, f"RAW (^VIX ) UTC min/max: {vix_raw.index.min()} .. {vix_raw.index.max()} [{vix_raw.index.tz}]")

    # 정규장 필터 후 KST로
    spx_kst_all = _filter_us_rth(spx_raw, debug=debug, label="SPX")
    vix_kst_all = _filter_us_rth(vix_raw, debug=debug, label="VIX")

    # 최종 컷 (뷰 구간만 남김)
    spx_kst_view = spx_kst_all[(spx_kst_all.index >= view_lo_kst) & (spx_kst_all.index < view_hi_kst)]
    vix_kst_view = vix_kst_all[(vix_kst_all.index >= view_lo_kst) & (vix_kst_all.index < view_hi_kst)]

    if debug:
        if len(spx_kst_view) > 0:
            _log(True, f"SPX KST after final cut: {spx_kst_view.index.min()} .. {spx_kst_view.index.max()} (n={len(spx_kst_view)})")
        if len(vix_kst_view) > 0:
            _log(True, f"VIX KST after final cut: {vix_kst_view.index.min()} .. {vix_kst_view.index.max()} (n={len(vix_kst_view)})")

    # 공통 인덱스로 정합 (뷰 구간)
    dfj = pd.concat([spx_kst_view, vix_kst_view], axis=1, join="inner").sort_index()
    dfj.columns = ["SPX", "VIX"]

    return dfj["SPX"], dfj["VIX"], vix_kst_all  # (뷰 구간 SPX, VIX, 전체기간 VIX for μ·σ)

def _filter_us_rth(s: pd.Series, debug=False, label=""):
    """
    US/Eastern 09:30 <= t < 16:00 만 남김(정규장), 이후 Asia/Seoul로 변환.
    """
    s_eastern = _ensure_tz(s, US_EASTERN)
    t = s_eastern.index.time
    mask = (t >= dtime(9,30)) & (t < dtime(16,0))
    s_rth = s_eastern[mask]
    s_kst = s_rth.tz_convert(ASIA_SEOUL)

    if debug:
        _log(True, f"{label} RTH filter: before={len(s)}, after={len(s_kst)}")
        if len(s) > 0:
            _log(True, f"{label} raw (tz?) min/max: {s.index.min()} .. {s.index.max()} [{s.index.tz}]")
        if len(s_kst) > 0:
            _log(True, f"{label} RTH(KST) min/max: {s_kst.index.min()} .. {s_kst.index.max()} [{s_kst.index.tz}]")
    return s_kst

# ----------------------------
# Compact Axis builder
# ----------------------------
def build_compact_axis_frame(spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    정규장 KST 분봉만 모여있는 상태에서, x축을 '연속 정수 인덱스'로 재매핑.
    - session_date_et: US/Eastern 날짜(세션 기준)
    - bar_idx_in_session: 각 세션 내 5분봉 인덱스 (0..N-1)
    - compact_x: 전체 구간에서 연속 증가 인덱스
    """
    df = pd.concat([spx.rename("SPX"), vix.rename("VIX")], axis=1, join="inner").sort_index()
    if df.empty:
        return df

    # 세션 라벨: US/Eastern 날짜
    idx_et = df.index.tz_convert(US_EASTERN)
    df["session_date_et"] = idx_et.date

    # 세션별 바 인덱스
    df["bar_idx_in_session"] = df.groupby("session_date_et").cumcount()

    # compact_x: 전체 연속 인덱스
    df["compact_x"] = np.arange(len(df), dtype=int)

    # 세션 경계 x
    session_bounds = df.groupby("session_date_et")["compact_x"].agg(["min", "max"]).rename(columns={"min":"x_start","max":"x_end"})
    df = df.join(session_bounds, on="session_date_et")
    return df

# ----------------------------
# Signal detection (5 business-day sigma rule) + DEBUG band logging
# ----------------------------
def detect_sell_signals_sigma(vix_view: pd.Series, vix_all: pd.Series, k: float, min_ref_bars: int, debug: bool=False) -> pd.DatetimeIndex:
    """
    vix_view: 최종 뷰 구간의 VIX (정규장, KST 인덱스)
    vix_all : 참조용으로 view_lo 이전 버퍼 포함 전체 VIX (정규장, KST 인덱스)
    규칙:
      - 시각 t마다 참조구간 R(t) = [t - 5영업일, t) 에서 μ(t), σ(t) 계산 (vix_all 사용)
      - 60분 창 W(t) = [t-60m, t] (vix_all 기준) 내에 |VIX-μ(t)| ≥ k·σ(t) 가 1회라도 있으면 cond(t)=True
      - cond False→True 전환 시 t에서 신호 발생

    추가(요청): DEBUG 모드일 때, 각 t에 대해 'μ(t) + k·σ(t)' 값을 로그로 출력.
    """
    if vix_view.empty or vix_all.empty:
        return vix_view.index[:0]

    idx = vix_view.index
    signals = []
    was_true = False

    for t in idx:
        # 5영업일 참조구간
        ref_start = (t - BusinessDay(5)).to_pydatetime()
        ref_end   = t.to_pydatetime()
        ref = vix_all.loc[(vix_all.index >= ref_start) & (vix_all.index < ref_end)]

        if len(ref) < min_ref_bars:
            if debug:
                print(f"[DEBUG] {t} — ref_bars={len(ref)} < min_ref_bars({min_ref_bars}); skip band calc")
            cond = False
        else:
            mu = ref.mean()
            sigma = ref.std(ddof=1)
            if not np.isfinite(sigma) or sigma == 0:
                if debug:
                    print(f"[DEBUG] {t} — sigma invalid (sigma={sigma}); skip band calc")
                cond = False
            else:
                # 직전 60분 윈도우
                w_start = t - timedelta(minutes=60)
                window = vix_all.loc[(vix_all.index >= w_start) & (vix_all.index <= t)]
                cond = bool((np.abs(window - mu) >= k * sigma).any()) if not window.empty else False

                # === 요청된 디버그 로그: μ + k·σ (상단 밴드) ===
                upper_band = mu + k * sigma
                if debug:
                    curr_vix = vix_all.loc[t] if t in vix_all.index else np.nan
                    print(f"[DEBUG] {t} — mu={mu:.4f}, sigma={sigma:.4f}, k={k:.2f}, mu+k*sigma={upper_band:.4f}, VIX(t)={curr_vix:.4f}")

        if cond and not was_true:
            signals.append(t)
        was_true = cond

    return pd.DatetimeIndex(signals)

# ----------------------------
# rangebreaks (time axis용) + Figure
# ----------------------------
def _build_rangebreaks_for_seoul(index: pd.DatetimeIndex):
    """
    time axis 모드에서만 사용:
    - 주말: ["sat","mon"]
    - 야간: 문자열 "HH:MM:SS" 경계 (상한 1초 낮춤)
      * EDT: 05:00:00 ~ 22:29:59 접기
      * EST: 06:00:00 ~ 23:29:59 접기
    """
    if len(index) == 0:
        return []
    is_edt = _detect_us_dst(index[0])
    hour_bounds = ["05:00:00", "22:29:59"] if is_edt else ["06:00:00", "23:29:59"]
    return [dict(bounds=["sat","mon"]), dict(pattern="hour", bounds=hour_bounds)]

def build_figure_time_axis(spx: pd.Series, vix: pd.Series, signals: pd.DatetimeIndex) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # traces
    fig.add_trace(go.Scatter(x=spx.index, y=spx.values, name="S&P 500 (^GSPC)", mode="lines"), secondary_y=False)
    fig.add_trace(go.Scatter(x=vix.index, y=vix.values, name="VIX (^VIX)", mode="lines"), secondary_y=True)

    if len(signals) > 0:
        spx_at_sig = spx.reindex(signals).dropna()
        fig.add_trace(
            go.Scatter(
                x=spx_at_sig.index, y=spx_at_sig.values, mode="markers",
                name="SELL (|VIX-μ| ≥ k·σ in 60m)",
                marker_symbol="triangle-down", marker_size=12, marker_color="orange"
            ),
            secondary_y=False
        )

    rb = _build_rangebreaks_for_seoul(spx.index)
    x_min = spx.index.min() if len(spx) else None
    x_max = spx.index.max() if len(spx) else None

    fig.update_layout(
        title="S&P 500 & VIX (5m, RTH-only, Asia/Seoul) — Time Axis",
        hovermode="x unified", legend_title_text="Series", dragmode="pan",
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_xaxes(
        title_text=f"Time ({ASIA_SEOUL})",
        range=[x_min, x_max] if x_min is not None else None,
        rangeslider=dict(visible=True),
        rangebreaks=rb,
    )
    fig.update_yaxes(title_text="S&P 500", secondary_y=False)
    fig.update_yaxes(title_text="VIX", secondary_y=True)
    return fig

def build_figure_compact_axis(df_compact: pd.DataFrame, signals: pd.DatetimeIndex) -> go.Figure:
    """
    x = compact_x (연속 정수). 호버/레이블로 실제 KST 시각을 보여줌.
    세션 경계에 vertical line을 그려서 하루 단위 구분을 유지.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # traces
    fig.add_trace(
        go.Scatter(
            x=df_compact["compact_x"], y=df_compact["SPX"],
            text=df_compact.index.strftime("%Y-%m-%d %H:%M:%S %Z"),
            hovertemplate="KST %{text}<br>S&P 500: %{y}<extra></extra>",
            name="S&P 500 (^GSPC)", mode="lines"
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df_compact["compact_x"], y=df_compact["VIX"],
            text=df_compact.index.strftime("%Y-%m-%d %H:%M:%S %Z"),
            hovertemplate="KST %{text}<br>VIX: %{y}<extra></extra>",
            name="VIX (^VIX)", mode="lines"
        ),
        secondary_y=True
    )

    # SELL markers (compact_x 로 매핑)
    if len(signals) > 0:
        sig_idx = df_compact.index.get_indexer(signals, method="nearest")
        sig_x = df_compact["compact_x"].iloc[sig_idx]
        sig_y = df_compact["SPX"].iloc[sig_idx]
        fig.add_trace(
            go.Scatter(
                x=sig_x, y=sig_y,
                mode="markers",
                name="SELL (|VIX-μ| ≥ k·σ in 60m)",
                marker_symbol="triangle-down", marker_size=12, marker_color="orange",
                text=signals.strftime("%Y-%m-%d %H:%M:%S %Z"),
                hovertemplate="Signal @ %{text}<br>S&P 500: %{y}<extra></extra>",
            ),
            secondary_y=False
        )

    # 세션 경계선
    for _, g in df_compact.groupby("session_date_et"):
        x0 = int(g["x_start"].iloc[0])
        fig.add_vline(x=x0, line_width=1, line_dash="dot", line_color="rgba(150,150,150,0.4)")

    fig.update_layout(
        title="S&P 500 & VIX (5m, RTH-only, Asia/Seoul) — Compact Axis",
        hovermode="x unified",
        legend_title_text="Series",
        dragmode="pan",
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(title="Sessions (compact)", rangeslider=dict(visible=True)),
    )
    fig.update_yaxes(title_text="S&P 500", secondary_y=False)
    fig.update_yaxes(title_text="VIX", secondary_y=True)
    return fig

# ----------------------------
# Session-end sanity check (US/Eastern 세션 기준)
# ----------------------------
def assert_session_last_bar(spx_kst: pd.Series):
    """
    검증 기준을 'KST 일자'가 아니라 'US/Eastern 세션일'로 바꿔서
    각 세션의 마지막 봉이 15:55~15:59 ET 근처인지 확인한다.
    """
    if spx_kst.empty:
        return

    df = spx_kst.to_frame("SPX").copy()
    idx_et = df.index.tz_convert(US_EASTERN)
    df["session_date_et"] = idx_et.date

    for d_et, g in df.groupby("session_date_et"):
        last_ts_et = g.index.tz_convert(US_EASTERN).max()
        hhmm = last_ts_et.hour * 60 + last_ts_et.minute
        ok = (15*60 + 53) <= hhmm <= (15*60 + 59)  # 15:53 ~ 15:59 허용
        if not ok:
            print(f"[WARN] Unexpected session last bar on {d_et}: {last_ts_et} (ET clock={last_ts_et.time()}). "
                  f"Expected around 15:55 ET.")

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    pio.renderers.default = "browser"

    # start/end 해석 (기본값 포함)
    view_lo_kst, view_hi_kst = resolve_view_window_kst(args.start, args.end, debug=args.debug)

    # yfinance 5m는 보통 ~30일 제한: 사용자 직접 범위를 넓혔을 때 경고
    try:
        if (view_hi_kst - view_lo_kst) > pd.Timedelta(days=31):
            print("[WARN] yfinance 5m는 보통 ~30일 제한. 구간을 나누거나 더 긴 인터벌(15m/30m/1h)을 고려하세요.")
    except Exception:
        pass

    # 데이터 수집 (뷰+버퍼)
    spx_view, vix_view, vix_all = fetch_5m_with_view(view_lo_kst, view_hi_kst, debug=args.debug, ref_buffer_days=10)
    if spx_view.empty or vix_view.empty:
        raise SystemExit("No data after filtering. Check symbols, dates, and market hours.")

    # 세션 종가 검증(US/Eastern 세션 기준)
    assert_session_last_bar(spx_view)

    # ===== sigma 기반 신호 + 디버그 밴드 출력 =====
    signals = detect_sell_signals_sigma(
        vix_view=vix_view,
        vix_all=vix_all,
        k=args.sigma_k,
        min_ref_bars=args.min_ref_bars,
        debug=args.debug,           # <- 여기서 디버그 출력 활성화
    )

    # 차트
    if args.axis == "time":
        fig = build_figure_time_axis(spx_view, vix_view, signals)
    else:
        dfc = build_compact_axis_frame(spx_view, vix_view)
        fig = build_figure_compact_axis(dfc, signals)

    fig.show()
    print(f"[INFO] KST window: {view_lo_kst} .. {view_hi_kst}")
    print(f"[INFO] Points: SPX={len(spx_view):,}, VIX={len(vix_view):,}, Signals={len(signals):,}")
    print(f"[INFO] Axis mode: {args.axis}")
    print(f"[INFO] Sigma k: {args.sigma_k}, min_ref_bars: {args.min_ref_bars}")

if __name__ == "__main__":
    main()


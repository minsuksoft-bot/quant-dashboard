import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ==========================================
# [4] 스마트폰 호환 & [6] 하이엔드 디자인 (토스/보그 감성)
# ==========================================
st.set_page_config(page_title="QUANT SIGNAL", page_icon="⬛", layout="centered")

st.markdown("""
<style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css");
    html, body, [class*="css"] { font-family: 'Pretendard', sans-serif !important; background-color: #FAFAFA; }
    .block-container { padding-top: 2rem !important; padding-left: 0.8rem !important; padding-right: 0.8rem !important; max-width: 100vw !important; overflow-x: hidden !important; }
    .main-title { font-size: 2.2rem; font-weight: 800; color: #000000; letter-spacing: -0.03em; margin-bottom: 0rem; text-transform: uppercase; }
    .sub-title { font-size: 0.95rem; font-weight: 400; color: #666666; letter-spacing: -0.01em; margin-bottom: 1.5rem; }
    .stButton > button { background-color: #000000 !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 0.6rem 1.5rem !important; width: 100% !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 1px solid #EAEAEA; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: transparent; border-radius: 0px; padding: 5px 0px; font-size: 1rem; font-weight: 600; color: #999999; }
    .stTabs [aria-selected="true"] { color: #000000 !important; border-bottom: 2.5px solid #000000 !important; }
    
    /* [7] 테이블 최적화 (모바일 튕김 방지 및 네이티브 스크롤) */
    .table-container { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 10px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05); background-color: #ffffff; border: 1px solid #F0F0F0; margin-bottom: 2rem; }
    .custom-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: center; }
    .custom-table th { background-color: #F8F9FA; color: #555555; font-weight: 700; padding: 8px 6px; border-bottom: 2px solid #EAEAEA; white-space: nowrap; }
    .custom-table td { padding: 10px 6px; border-bottom: 1px solid #F5F5F5; color: #111111; white-space: nowrap; }
    .custom-table tbody tr:hover { background-color: #FAFAFA; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Quant Signal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Technical Dashboard for Nasdaq</p>', unsafe_allow_html=True)

# ==========================================
# [10] 보안 장치: 관리자 강제 초기화 사이드바
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Admin Control")
    st.caption("비상 시 쿨타임(1시간)을 무시하고 데이터를 즉시 새로 받아옵니다.")
    admin_password = st.text_input("관리자 패스워드", type="password")
    
    if admin_password == "quant1234":
        st.success("관리자 권한 인증 완료")
        if st.button("🚨 쿨타임 무시 강제 초기화", use_container_width=True):
            st.cache_data.clear() # 캐시 강제 삭제
            st.rerun()
    elif admin_password:
        st.error("비밀번호가 틀렸습니다.")

# ==========================================
# [12] 원본 파이썬 로직 100% 동일 (조건 및 지표)
# ==========================================
def is_valid_ticker(ticker):
    return bool(re.match(r'^[A-Za-z0-9.-]+$', str(ticker)))

def is_golden_alignment(ma5, ma10, ma20, ma50, idx):
    try:
        return ma5.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma50.iloc[idx]
    except IndexError:
        return False

def find_signals(data_df, nasdaq_tickers, offset_days=0, is_backtest=False):
    passed_tickers = []
    scanned_count = 0 

    for ticker in nasdaq_tickers:
        try:
            # yf.download(group_by='ticker') 구조 파싱 (원본의 yf.Ticker(t).history 와 동일한 데이터 추출)
            if isinstance(data_df.columns, pd.MultiIndex):
                if ticker not in data_df.columns.levels[0]:
                    continue
                hist = data_df[ticker].dropna(subset=['Close'])
            else:
                if ticker not in data_df:
                    continue
                hist = data_df[ticker].dropna(subset=['Close'])

            if len(hist) > 0:
                scanned_count += 1

            full_high = hist['High'] 

            # [3] 백테스팅: 과거 시점으로 잘라내기
            if offset_days > 0:
                hist = hist.iloc[:-offset_days]

            if len(hist) < 60:
                continue

            close = hist['Close']
            ma5 = close.rolling(window=5).mean()
            ma10 = close.rolling(window=10).mean()
            ma20 = close.rolling(window=20).mean()
            ma50 = close.rolling(window=50).mean()

            x_day = None
            score = 0
            near_ma = None

            for back in range(3, 11):
                i = -back
                price = close.iloc[i]
                for ma, pts in [(ma20, 3), (ma10, 2), (ma5, 1)]:
                    ma_val = ma.iloc[i]
                    if ma_val * 0.995 <= price <= ma_val * 1.005:
                        x_day = i
                        score = pts
                        near_ma = ma
                        break
                if x_day is not None:
                    break

            if x_day is None:
                continue

            cond1 = (is_golden_alignment(ma5, ma10, ma20, ma50, x_day-1) or
                     is_golden_alignment(ma5, ma10, ma20, ma50, x_day-2))

            cond2 = (is_golden_alignment(ma5, ma10, ma20, ma50, x_day+1) or
                     is_golden_alignment(ma5, ma10, ma20, ma50, x_day+2))

            cond3 = ((close.iloc[x_day-1] > near_ma.iloc[x_day-1]) or
                     (close.iloc[x_day-2] > near_ma.iloc[x_day-2]))

            cond4 = ((close.iloc[x_day+1] > near_ma.iloc[x_day+1]) or
                     (close.iloc[x_day+2] > near_ma.iloc[x_day+2]))

            cond5 = (near_ma.iloc[-1] < close.iloc[-1]) and (close.iloc[x_day+1] < close.iloc[-1])

            if cond1 and cond2 and cond3 and cond4 and cond5:
                signal_date = hist.index[x_day].strftime('%Y-%m-%d')
                signal_close = close.iloc[x_day]

                item = {
                    'Ticker': ticker,
                    'Close': round(close.iloc[-1], 2),
                    '눌림목 점수': score,
                    '시그널 발생일자': signal_date
                }

                if is_backtest:
                    signal_timestamp = hist.index[x_day]
                    future_highs = full_high.loc[signal_timestamp:]
                    if len(future_highs) > 0:
                        max_high = future_highs.max()
                        max_return = ((max_high / signal_close) - 1) * 100
                    else:
                        max_high = signal_close
                        max_return = 0.0
                    item['최대 수익률(%)'] = round(max_return, 2)

                passed_tickers.append(item)
        except Exception:
            continue

    return passed_tickers, scanned_count

# ==========================================
# [1] 리소스 최적화 및 서버 캐싱 엔진 (3600초)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def run_dashboard_analysis():
    file = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/pub?output=csv"
    tickers_df = pd.read_csv(file)
    
    # [11] 800개 종목 확장 수신
    nasdaq_tickers = [t for t in tickers_df['Symbol'].tolist()[:800] if is_valid_ticker(t)]
    
    data = yf.download(nasdaq_tickers, period="6mo", group_by="ticker", threads=True, progress=False)
    
    try:
        downloaded_count = len(data.columns.levels[0]) if isinstance(data.columns, pd.MultiIndex) else len(data.columns)
    except:
        downloaded_count = 0
    
    sector_cache = {}

    # --- Live Signals 분석 ---
    current_passed, scanned_live = find_signals(data, nasdaq_tickers, offset_days=0, is_backtest=False)
    results = []
    
    for item in current_passed:
        ticker = item['Ticker']
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'N/A')
            sector_cache[ticker] = sector
            
            psr = info.get('priceToSalesTrailing12Months', 0)
            rev_growth = info.get('revenueGrowth', 0)
            psg = (psr / (rev_growth * 100)) if (psr and rev_growth and rev_growth > 0) else 0
            
            if not psr or not psg:
                try:
                    revenues = stock.financials.loc['Total Revenue']
                    if len(revenues) >= 2 and revenues.iloc[1] != 0 and not pd.isna(revenues.iloc[1]):
                        g_curr = (revenues.iloc[0] - revenues.iloc[1]) / abs(revenues.iloc[1]) * 100
                        m_cap = info.get('marketCap')
                        if m_cap and g_curr != 0: psr, psg = m_cap / revenues.iloc[0], psr / g_curr
                except: pass
            
            results.append({
                '시그널 발생일자': item['시그널 발생일자'], 
                'Ticker': ticker,
                'Sector': sector,
                '눌림목 점수': item['눌림목 점수'],
                'PSR': round(psr if pd.notna(psr) else 0, 2), 
                'PSG': round(psg if pd.notna(psg) else 0, 2)
            })
        except Exception: 
            sector_cache[ticker] = 'N/A'
            continue
                
    curr_df = pd.DataFrame(results)
    if not curr_df.empty:
        # [2] 시그널 정렬: 1순위 시그널 발생일(내림차순)
        curr_df = curr_df.sort_values(by=['시그널 발생일자', '눌림목 점수'], ascending=[False, False])
    
    # --- 1-Month Backtest 분석 ---
    bt_passed, _ = find_signals(data, nasdaq_tickers, offset_days=21, is_backtest=True)
    bt_results = []
    
    for item in bt_passed:
        ticker = item['Ticker']
        # 백테스트 종목 섹터 매칭
        if ticker in sector_cache and sector_cache[ticker] != 'N/A':
            sector = sector_cache[ticker]
        else:
            try:
                sector = yf.Ticker(ticker).info.get('sector', 'N/A')
                sector_cache[ticker] = sector
            except:
                sector = 'N/A'
                
        bt_results.append({
            '시그널 발생일자': item['시그널 발생일자'], 
            'Ticker': ticker, 
            'Sector': sector,
            '눌림목 점수': item['눌림목 점수'], 
            '최대 수익률(%)': item['최대 수익률(%)']
        })

    bt_df = pd.DataFrame(bt_results)
    if not bt_df.empty:
        # [9] 정렬 변경: 1-Month는 최대 수익률 높은 순서
        bt_df = bt_df.sort_values(by=['최대 수익률(%)', '시그널 발생일자'], ascending=[False, False])
        
    return curr_df, bt_df, datetime.now(), downloaded_count, scanned_live

# ==========================================
# 화면 실행 및 디버깅 데이터 표시
# ==========================================
with st.spinner('Scanning Market Data...'):
    curr_df, bt_df, last_update, dl_count, scanned_count = run_dashboard_analysis()

# [11] 디버깅 및 수신 종목 수 표기
st.caption(f"🔄 Last Updated: **{last_update.strftime('%m.%d %H:%M:%S')}** | 📥 API 수신: **{dl_count}개** | 🔍 검토: **{scanned_count}개**")

# ==========================================
# [5] 1시간 쿨타임 일반 새로고침 (누구나 사용 가능)
# ==========================================
if st.button("새로고침 (Refresh)", use_container_width=True):
    time_diff = (datetime.now() - last_update).total_seconds()
    if time_diff >= 3600 or (curr_df.empty and bt_df.empty):
        st.cache_data.clear() # 1시간이 지났으므로 정상적으로 캐시 파기
        st.rerun()
    else:
        remain_min = int((3600 - time_diff) // 60)
        st.warning(f"⚠️ 데이터 새로고침한지 1시간이 지나지 않았습니다. (약 {remain_min}분 후 누구나 가능)")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["LIVE SIGNALS", "1-MONTH TEST"])

# ==========================================
# [8] 칼럼 커스텀 렌더링
# ==========================================
with tab1:
    if curr_df.empty:
        st.info("현재 조건에 부합하는 종목이 없습니다.")
    else:
        display_df = curr_df.copy()
        # Live: 현재가 삭제, 섹터 포함
        display_df = display_df[['시그널 발생일자', 'Ticker', 'Sector', '눌림목 점수', 'PSR', 'PSG']]
        display_df.rename(columns={'시그널 발생일자': '발생일', 'Ticker': '종목', 'Sector': '섹터', '눌림목 점수': '점수'}, inplace=True)
        display_df['PSR'] = display_df['PSR'].apply(lambda x: f"{x:.2f}")
        display_df['PSG'] = display_df['PSG'].apply(lambda x: f"{x:.2f}")
        st.markdown(f'<div class="table-container">{display_df.to_html(index=False, classes="custom-table")}</div>', unsafe_allow_html=True)

with tab2:
    if bt_df.empty:
        st.info("1달 전 조건에 부합했던 종목이 없습니다.")
    else:
        display_bt = bt_df.copy()
        # 1-Month: 현재가/최고가 삭제, 섹터 포함
        display_bt = display_bt[['시그널 발생일자', 'Ticker', 'Sector', '눌림목 점수', '최대 수익률(%)']]
        display_bt.rename(columns={'시그널 발생일자': '발생일', 'Ticker': '종목', 'Sector': '섹터', '눌림목 점수': '점수', '최대 수익률(%)': '수익률'}, inplace=True)
        display_bt['수익률'] = display_bt['수익률'].apply(lambda x: f"{x:,.2f}%")
        st.markdown(f'<div class="table-container">{display_bt.to_html(index=False, classes="custom-table")}</div>', unsafe_allow_html=True)

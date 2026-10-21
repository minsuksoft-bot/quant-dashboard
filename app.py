# -*- coding: utf-8 -*-
"""QUANT SIGNAL + NASDAQ FACTOR LAB | single-file Streamlit app
Deploy as app.py in the existing Streamlit/GitHub repository.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re, time, json, os
from datetime import datetime
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# APP CONFIG / STYLE
# =========================
st.set_page_config(
    page_title="QUANT SIGNAL | Factor Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(r"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css');
:root { color-scheme: light; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: #203247;
}
.stApp {
  background: linear-gradient(145deg, #f4f8fc 0%, #eaf2fa 48%, #f7faff 100%);
  color: #203247;
}
[data-testid="stHeader"] { background: rgba(244,248,252,.96); }
.block-container {
  padding-top: 3.55rem !important; padding-bottom: 2.5rem !important;
  padding-left: clamp(.7rem, 2.7vw, 2rem) !important;
  padding-right: clamp(.7rem, 2.7vw, 2rem) !important;
  max-width: 1500px !important;
}
.app-hero {
  position: relative; overflow: hidden; margin: .45rem 0 1.35rem;
  padding: 1.35rem 1.45rem 1.25rem;
  border: 1px solid #d5e3f1; border-radius: 23px;
  background: linear-gradient(120deg, #ffffff 0%, #e8f3ff 64%, #d9efff 100%);
  box-shadow: 0 10px 30px rgba(42, 83, 125, .09);
}
.app-hero:after { content:""; position:absolute; width:210px; height:210px; right:-75px; top:-100px; border-radius:50%; border:28px solid rgba(48,145,215,.10); pointer-events:none; }
.main-title { margin:0 !important; padding:0 !important; color:#173b60; font-size:clamp(1.8rem,5vw,3rem); line-height:1.22; font-weight:850; letter-spacing:-.045em; white-space:normal; overflow-wrap:normal; word-break:keep-all; }
.sub-title { margin:.45rem 0 0 !important; color:#58728d; font-size:clamp(.78rem,2.5vw,1rem); line-height:1.55; letter-spacing:.015em; }
.eyebrow, .page-kicker { color:#1686c4; font-size:.72rem; font-weight:850; letter-spacing:.13em; margin-bottom:.45rem; }
h1, h2, h3, h4 { color:#203e5c !important; letter-spacing:-.025em; }
p, [data-testid="stMarkdownContainer"] { overflow-wrap:break-word; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#f0f6fc,#e6eff8); border-right:1px solid #d5e3f0; }
[data-testid="stSidebar"] h2 { font-size:1.12rem; color:#234867 !important; }
[data-testid="stRadio"] > label { font-size:.78rem; color:#58728d; }
[data-testid="stRadio"] div[role="radiogroup"] { gap:.35rem; }
[data-testid="stRadio"] div[role="radiogroup"] label { border:1px solid #d7e4f1; border-radius:11px; padding:.48rem .58rem; background:#f9fcff; }
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:linear-gradient(100deg,#d9edff,#c9eaff); border-color:#6bb9e8; }
.stButton > button, .stDownloadButton > button { width:100%; min-height:2.55rem; border-radius:12px; border:1px solid #9bc9e8; background:linear-gradient(115deg,#e3f3ff,#cbeaff); color:#164b73; font-weight:750; transition:filter .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(.98); border-color:#4aa9e3; color:#103d60; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#ffffff,#edf6ff); border:1px solid #d7e6f3; padding:11px 13px; border-radius:15px; box-shadow:0 4px 14px rgba(40,85,125,.04); }
[data-testid="stMetricLabel"] { color:#58728d !important; }
[data-testid="stMetricValue"] { color:#173f63 !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { width:100%; max-width:100%; border:1px solid #d6e3ef; border-radius:12px; overflow:hidden; background:#fff; }
[data-testid="stDataFrame"] iframe { max-width:100% !important; }
[data-testid="stTable"] table { width:100% !important; table-layout:fixed; }
[data-testid="stTable"] th { background:#eaf4fd !important; color:#234867 !important; font-size:.78rem !important; }
[data-testid="stTable"] td { color:#2d4358 !important; font-size:.78rem !important; overflow-wrap:anywhere; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color:#526d86 !important; }
hr { border-color:#d8e5f1 !important; }
.menu-section-title { margin:.2rem 0 .25rem; color:#234867; font-size:1.18rem; font-weight:800; }
.menu-section-note { margin:0 0 1rem; color:#627b94; font-size:.88rem; line-height:1.5; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#d5e3ef !important; border-radius:19px !important; background:linear-gradient(155deg,#ffffff,#f0f7fd) !important; box-shadow:0 5px 18px rgba(41,81,119,.045); }
.menu-card { min-height:9.2rem; padding:.08rem .04rem .25rem; display:flex; flex-direction:column; }
/* 홈 메뉴만 카드 높이를 통일해 2열 그리드가 반듯하게 보이도록 */
[data-testid="stVerticalBlockBorderWrapper"]:has(.menu-card) { min-height: 15.2rem; height:100%; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.menu-card) > div { min-height:15.2rem; height:100%; display:flex; flex-direction:column; justify-content:space-between; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.menu-card) .stButton { margin-top:auto; }
.menu-icon { display:inline-flex; align-items:center; justify-content:center; width:2.6rem; height:2.6rem; border-radius:13px; background:linear-gradient(145deg,#e4f4ff,#d2ebff); border:1px solid #c1e3f8; }
.menu-number { margin-top:.55rem; }
.menu-desc { min-height:3rem; }
.menu-icon { color:#238ec8; font-size:1.55rem; line-height:1.1; margin-bottom:.8rem; }
.menu-number { color:#2189c1; font-size:.68rem; font-weight:850; letter-spacing:.11em; margin-bottom:.32rem; }
.menu-title { color:#203f5b; font-size:clamp(.98rem,2.6vw,1.24rem); line-height:1.35; font-weight:800; margin-bottom:.42rem; }
.menu-desc { color:#617990; font-size:clamp(.76rem,2vw,.9rem); line-height:1.55; min-height:2.65rem; }
.page-title { color:#203f5b; font-size:clamp(1.25rem,3.5vw,1.75rem); font-weight:820; line-height:1.3; margin-bottom:.5rem; }

/* Force the home menu to remain a true two-column grid on phones.
   Streamlit normally stacks columns on narrow screens; only rows containing
   menu cards are overridden, so tables and detail pages remain responsive. */
[data-testid="stHorizontalBlock"]:has(.menu-card) {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
  gap: .72rem !important;
  align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.menu-card) > [data-testid="stColumn"] {
  width: auto !important;
  min-width: 0 !important;
  flex: none !important;
  padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.menu-card) [data-testid="stVerticalBlockBorderWrapper"] {
  width: 100% !important;
  min-width: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.menu-card) .menu-card {
  min-width: 0 !important;
}

@media (max-width:640px) {
  .block-container { padding-top:3.9rem !important; padding-left:.62rem !important; padding-right:.62rem !important; }
  .app-hero { margin-top:.45rem; padding:1.05rem 1rem 1rem; border-radius:19px; }
  .main-title { font-size:clamp(1.75rem,7.2vw,2.2rem); }
  .sub-title { font-size:.78rem; }
  .menu-card { min-height:8.6rem; padding:0; }
  [data-testid="stVerticalBlockBorderWrapper"]:has(.menu-card),
  [data-testid="stVerticalBlockBorderWrapper"]:has(.menu-card) > div { min-height:13.2rem; height:100%; }
  .menu-desc { min-height:3.2rem; font-size:.76rem; }
  .menu-title { font-size:1rem; }
  [data-testid="stMetric"] { padding:8px 9px; }
  [data-testid="stMetricValue"] { font-size:1.25rem; }
  [data-testid="stDataFrame"] { font-size:11px !important; }
  [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom:.35rem; }
  [data-testid="stTable"] th, [data-testid="stTable"] td { font-size:.69rem !important; padding:.35rem .25rem !important; }
}
</style>
""", unsafe_allow_html=True)

TICKER_SOURCE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/pub?output=csv"
BENCHMARK = "QQQ"
DEFAULT_MAX_TICKERS = 800

# =========================
# SHARED HELPERS
# =========================
def is_valid_ticker(ticker):
    return bool(re.fullmatch(r'[A-Za-z0-9.\-]+', str(ticker).strip()))

@st.cache_data(ttl=3600, show_spinner=False)
def load_tickers(max_tickers=800):
    df = pd.read_csv(TICKER_SOURCE_URL)
    if 'Symbol' not in df.columns:
        raise ValueError("종목 목록에 Symbol 컬럼이 없습니다.")
    vals, seen = [], set()
    for x in df['Symbol'].dropna().astype(str):
        t = x.strip().upper()
        if is_valid_ticker(t) and t != BENCHMARK and t not in seen:
            vals.append(t); seen.add(t)
        if len(vals) >= max_tickers: break
    return vals

# =========================
# PULLBACK MODEL (preserves existing signal logic)
# =========================
def is_golden_alignment(ma5, ma10, ma20, ma50, idx):
    try: return ma5.iloc[idx] > ma10.iloc[idx] > ma20.iloc[idx] > ma50.iloc[idx]
    except (IndexError, KeyError): return False

def find_pullback_signals(data_df, tickers, offset_days=0, is_backtest=False):
    passed, scanned = [], 0
    for ticker in tickers:
        try:
            if isinstance(data_df.columns, pd.MultiIndex):
                if ticker not in data_df.columns.get_level_values(0): continue
                hist = data_df[ticker].dropna(subset=['Close']).copy()
            else:
                if 'Close' not in data_df.columns: continue
                hist = data_df.dropna(subset=['Close']).copy()
            if len(hist): scanned += 1
            full_high = hist['High'].copy()
            if offset_days > 0: hist = hist.iloc[:-offset_days]
            if len(hist) < 60: continue
            close = hist['Close']
            ma5, ma10, ma20, ma50 = (close.rolling(n).mean() for n in (5,10,20,50))
            x_day, score, near_ma = None, 0, None
            for back in range(3, 11):
                i = -back; price = close.iloc[i]
                for ma, pts in ((ma20,3),(ma10,2),(ma5,1)):
                    val = ma.iloc[i]
                    if pd.notna(val) and val*.995 <= price <= val*1.005:
                        x_day, score, near_ma = i, pts, ma; break
                if x_day is not None: break
            if x_day is None: continue
            cond1 = is_golden_alignment(ma5,ma10,ma20,ma50,x_day-1) or is_golden_alignment(ma5,ma10,ma20,ma50,x_day-2)
            cond2 = is_golden_alignment(ma5,ma10,ma20,ma50,x_day+1) or is_golden_alignment(ma5,ma10,ma20,ma50,x_day+2)
            cond3 = (close.iloc[x_day-1] > near_ma.iloc[x_day-1]) or (close.iloc[x_day-2] > near_ma.iloc[x_day-2])
            cond4 = (close.iloc[x_day+1] > near_ma.iloc[x_day+1]) or (close.iloc[x_day+2] > near_ma.iloc[x_day+2])
            cond5 = (near_ma.iloc[-1] < close.iloc[-1]) and (close.iloc[x_day+1] < close.iloc[-1])
            if cond1 and cond2 and cond3 and cond4 and cond5:
                stamp = hist.index[x_day]
                item = {'발생일': stamp.strftime('%Y-%m-%d'), '종목': ticker,
                        '현재가': round(float(close.iloc[-1]),2), '눌림목 점수':score}
                if is_backtest:
                    highs = full_high.loc[stamp:]
                    item['이후 최고수익률(%)'] = round((float(highs.max())/float(close.iloc[x_day])-1)*100,2) if len(highs) else 0.0
                passed.append(item)
        except Exception:
            continue
    return pd.DataFrame(passed), scanned

@st.cache_data(ttl=3600, show_spinner=False)
def run_pullback_analysis(max_tickers=800):
    tickers = load_tickers(max_tickers)
    # Avoid one huge 800-symbol request: use modest batches and retry failed batches.
    parts = []
    for start in range(0, len(tickers), 25):
        batch = tickers[start:start+25]
        raw = pd.DataFrame()
        for attempt in range(3):
            try:
                raw = yf.download(tickers=batch, period='6mo', auto_adjust=True,
                                  actions=False, group_by='ticker', threads=False,
                                  progress=False, timeout=35, multi_level_index=True)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                pass
            time.sleep(4 * (attempt + 1))
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                parts.append(raw)
            elif len(batch) == 1:
                raw.columns = pd.MultiIndex.from_product([[batch[0]], raw.columns])
                parts.append(raw)
        time.sleep(1.0)
    if not parts:
        raise RuntimeError('Yahoo Finance에서 유효한 가격 데이터를 받지 못했습니다. 잠시 후 다시 실행하세요.')
    raw = pd.concat(parts, axis=1)
    # Remove duplicate columns if any batch retry returned duplicate symbols.
    raw = raw.loc[:, ~raw.columns.duplicated()]
    current, scanned = find_pullback_signals(raw, tickers, 0, False)
    past, _ = find_pullback_signals(raw, tickers, 21, True)
    if not current.empty: current = current.sort_values(['발생일','눌림목 점수'], ascending=[False,False])
    if not past.empty: past = past.sort_values(['이후 최고수익률(%)','발생일'], ascending=[False,False])
    return current, past, datetime.now(), len(tickers), scanned

# =========================
# FACTOR MODEL
# =========================
FACTOR_GROUPS = {
    'Momentum':['mom_12_1','mom_6_1','mom_3_1','rel_mom_6m','rel_mom_12m','short_reversal','high_52w_distance'],
    'Trend':['sma50_gap','sma200_gap','ema20_gap','rsi14_norm'],
    'Breakout / Technical':['breakout20','bb_squeeze','volume_surge'],
    'Risk':['neg_vol_63','neg_vol_252','neg_downside_vol_63','drawdown_126','beta_126'],
    'Liquidity':['log_dollar_volume_20','dollar_volume_trend']
}
FEATURES = [x for group in FACTOR_GROUPS.values() for x in group]

def _extract_history(raw, ticker, batch):
    try:
        if raw is None or raw.empty: return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0): h = raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(1): h = raw.xs(ticker,axis=1,level=1).copy()
            else: return None
        elif len(batch)==1: h=raw.copy()
        else: return None
        h.columns=[str(c) for c in h.columns]
        h=h.loc[:,~h.columns.duplicated()]
        if isinstance(h.index,pd.DatetimeIndex) and h.index.tz is not None: h.index=h.index.tz_localize(None)
        if 'Close' not in h: return None
        for c in ['High','Low','Volume']:
            if c not in h: h[c]=np.nan
        return h[['Close','High','Low','Volume']].sort_index().dropna(subset=['Close'])
    except Exception: return None

def _robust_z(s):
    x=pd.to_numeric(s,errors='coerce'); out=pd.Series(0.0,index=x.index)
    z=x.dropna()
    if len(z)<5: return out
    lo,hi=z.quantile(.02),z.quantile(.98); z=z.clip(lo,hi)
    med=z.median(); scale=1.4826*(z-med).abs().median()
    if not np.isfinite(scale) or scale<1e-8: scale=z.std(ddof=0)
    if not np.isfinite(scale) or scale<1e-8: scale=1.0
    out.loc[z.index]=((z-med)/scale).clip(-3,3)
    return out

def _features_for_ticker(ticker,h,dates,bmret,bmclose):
    h=h.reindex(dates); c=h.Close.astype(float); hi=h.High.astype(float); vol=h.Volume.astype(float)
    ret=c.pct_change(); sma50=c.rolling(50,min_periods=40).mean(); sma200=c.rolling(200,min_periods=160).mean()
    ema20=c.ewm(span=20,adjust=False,min_periods=15).mean(); ma20=c.rolling(20,min_periods=15).mean()
    bb=4*c.rolling(20,min_periods=15).std()/ma20.replace(0,np.nan)
    adv20=(c*vol).rolling(20,min_periods=15).mean(); adv60=(c*vol).rolling(60,min_periods=45).mean()
    v63=ret.rolling(63,min_periods=45).std()*np.sqrt(252); v252=ret.rolling(252,min_periods=180).std()*np.sqrt(252)
    down=ret.where(ret<0,0.0); dv=down.rolling(63,min_periods=45).std()*np.sqrt(252)
    beta=ret.rolling(126,min_periods=63).cov(bmret)/bmret.rolling(126,min_periods=63).var().replace(0,np.nan)
    priorhi=hi.shift(1).rolling(20,min_periods=15).max()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan)); rsi=rsi.where(loss.ne(0),100)
    bm6=bmclose.shift(21)/bmclose.shift(126)-1; bm12=bmclose.shift(21)/bmclose.shift(252)-1
    f=pd.DataFrame(index=dates)
    f['mom_12_1']=c.shift(21)/c.shift(252)-1; f['mom_6_1']=c.shift(21)/c.shift(126)-1; f['mom_3_1']=c.shift(5)/c.shift(63)-1
    f['short_reversal']=-(c/c.shift(21)-1); f['rel_mom_6m']=(c.shift(21)/c.shift(126)-1)-bm6; f['rel_mom_12m']=(c.shift(21)/c.shift(252)-1)-bm12
    f['high_52w_distance']=c/c.rolling(252,min_periods=180).max()-1
    f['sma50_gap']=c/sma50-1; f['sma200_gap']=c/sma200-1; f['ema20_gap']=c/ema20-1; f['rsi14_norm']=(rsi-50)/50
    f['breakout20']=c/priorhi-1; f['bb_squeeze']=-bb
    f['volume_surge']=vol.rolling(5,min_periods=4).mean()/vol.rolling(20,min_periods=15).mean()-1
    f['neg_vol_63']=-v63; f['neg_vol_252']=-v252; f['neg_downside_vol_63']=-dv
    f['drawdown_126']=c/c.rolling(126,min_periods=80).max()-1; f['beta_126']=beta
    f['log_dollar_volume_20']=np.log1p(adv20.clip(lower=0)); f['dollar_volume_trend']=np.log((adv20+1)/(adv60+1))
    f['Close']=c; f['ADV20']=adv20; f['Obs252']=c.rolling(252,min_periods=1).count(); f['Ticker']=ticker
    return f.replace([np.inf,-np.inf],np.nan)

@st.cache_data(ttl=21600, show_spinner=False)
def run_factor_model(max_tickers=800, history_start='2014-01-01', batch_size=25):
    tickers=load_tickers(max_tickers)
    symbols=list(dict.fromkeys(tickers+[BENCHMARK])); hist={}; failures=[]
    for start in range(0,len(symbols),batch_size):
        batch=symbols[start:start+batch_size]; raw=None
        for attempt in range(3):
            try:
                raw=yf.download(tickers=batch,start=history_start,auto_adjust=True,actions=False,
                                repair=False,group_by='ticker',threads=False,progress=False,
                                timeout=35,multi_level_index=True)
                break
            except Exception as e:
                if attempt==2: failures.extend(batch)
                time.sleep(5*(attempt+1))
        for t in batch:
            h=_extract_history(raw,t,batch)
            if h is not None and len(h)>252: hist[t]=h
        time.sleep(1.5)

    if BENCHMARK not in hist: raise RuntimeError('QQQ 가격 데이터를 받지 못했습니다. 잠시 후 다시 실행하세요.')
    bmclose=hist[BENCHMARK].Close.dropna().sort_index(); dates=bmclose.index; bmret=bmclose.pct_change()
    sample_pos=list(range(252,len(dates)-1,21))
    if len(dates)>252: sample_pos.append(len(dates)-1)
    sample_pos=sorted(set(sample_pos)); sample_dates=dates.take(sample_pos)
    features=[]
    for i,(ticker,h) in enumerate(hist.items()):
        if ticker==BENCHMARK: continue
        f=_features_for_ticker(ticker,h,dates,bmret,bmclose).reindex(sample_dates)
        f.index.name='Date'; features.append(f.reset_index())
    panel=pd.concat(features,ignore_index=True)
    for f in FEATURES: panel[f]=panel.groupby('Date')[f].transform(_robust_z)
    panel[FEATURES]=panel[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)

    # Build future 63-trading-day QQQ-relative excess-return labels.
    rec=[]; date_to_pos={d:i for i,d in enumerate(dates)}
    for pos in sample_pos:
        if pos+63>=len(dates): continue
        d=dates[pos]; cross=panel[panel.Date==d]
        cross=cross[(cross.Close>=5)&(cross.ADV20>=5_000_000)&(cross.Obs252>=245)]
        bmr=float(bmclose.iloc[pos+63]/bmclose.iloc[pos]-1)
        for r in cross.itertuples(index=False):
            h=hist.get(r.Ticker)
            if h is None: continue
            c=h.Close.reindex(dates); p0,p1=c.iloc[pos],c.iloc[pos+63]
            if pd.notna(p0) and pd.notna(p1) and p0>0 and p1>0:
                row={f:getattr(r,f) for f in FEATURES}; row.update({'Ticker':r.Ticker,'Date':d,'Pos':pos,'LabelEnd':pos+63,'Y':float(p1/p0-1-bmr)})
                rec.append(row)
    training=pd.DataFrame(rec)
    if training.empty: raise RuntimeError('학습 표본이 없습니다. 데이터 수집 또는 필터를 확인하세요.')
    training['YTrain']=training.groupby('Date').Y.transform(lambda x:x.clip(x.quantile(.01),x.quantile(.99)) if x.notna().sum()>20 else x)
    latest=dates[-1]; pos=len(dates)-1
    train=training[(training.LabelEnd<=pos)&(training.Pos>=pos-8*252)]
    current=panel[panel.Date==latest].copy()
    current=current[(current.Close>=5)&(current.ADV20>=5_000_000)&(current.Obs252>=245)].copy()
    if len(train)<1500 or train.Date.nunique()<36: raise RuntimeError(f'학습 표본 부족: {len(train)} rows, {train.Date.nunique()} dates')
    if current.empty: raise RuntimeError('현재 추천 후보가 없습니다.')
    imputer=SimpleImputer(strategy='median'); X=imputer.fit_transform(train[FEATURES]); y=train.YTrain.to_numpy()
    scaler=StandardScaler(); Xs=scaler.fit_transform(X); ridge=Ridge(alpha=12).fit(Xs,y)
    tree=HistGradientBoostingRegressor(max_iter=100,learning_rate=.06,max_leaf_nodes=15,max_depth=6,min_samples_leaf=30,l2_regularization=5,random_state=17).fit(X,y)
    Xi=imputer.transform(current[FEATURES]); pred=.5*ridge.predict(scaler.transform(Xi))+.5*tree.predict(Xi)
    current['예상 초과수익률(63일)']=pred; current['Ridge 예측']=ridge.predict(scaler.transform(Xi)); current['트리 예측']=tree.predict(Xi)
    Xstd=scaler.transform(Xi)
    for j,f in enumerate(FEATURES): current[f+'_기여도']=.5*Xstd[:,j]*ridge.coef_[j]
    ranked=current.sort_values('예상 초과수익률(63일)',ascending=False).reset_index(drop=True)
    group_rows=[]
    for r in ranked.head(100).itertuples(index=False):
        for group,fs in FACTOR_GROUPS.items():
            group_rows.append({'Ticker':r.Ticker,'팩터 그룹':group,'Ridge 기여도':sum(getattr(r,f+'_기여도') for f in fs)})
    group_df=pd.DataFrame(group_rows)
    # walk-forward: only matured labels, non-overlapping quarterly evaluation
    bt=[]; hold=[]; eval_positions=sample_pos[::3]
    for p in eval_positions:
        if p+63>=len(dates): continue
        tr=training[(training.LabelEnd<=p)&(training.Pos>=p-8*252)]
        if len(tr)<1500 or tr.Date.nunique()<36: continue
        d=dates[p]; cands=panel[panel.Date==d].copy(); cands=cands[(cands.Close>=5)&(cands.ADV20>=5_000_000)&(cands.Obs252>=245)]
        if len(cands)<20: continue
        try:
            imp=SimpleImputer(strategy='median'); tx=imp.fit_transform(tr[FEATURES]); ty=tr.YTrain.to_numpy(); sc=StandardScaler(); txs=sc.fit_transform(tx)
            rm=Ridge(alpha=12).fit(txs,ty); tm=HistGradientBoostingRegressor(max_iter=100,learning_rate=.06,max_leaf_nodes=15,max_depth=6,min_samples_leaf=30,l2_regularization=5,random_state=17).fit(tx,ty)
            cx=imp.transform(cands[FEATURES]); cands['Prediction']=.5*rm.predict(sc.transform(cx))+.5*tm.predict(cx)
            picks=cands.nlargest(20,'Prediction'); rets=[]
            for r in picks.itertuples(index=False):
                c=hist[r.Ticker].Close.reindex(dates); p0,p1=c.iloc[p],c.iloc[p+63]
                if pd.notna(p0) and pd.notna(p1) and p0>0:
                    rr=float(p1/p0-1); rets.append(rr); hold.append({'SignalDate':d.strftime('%Y-%m-%d'),'Ticker':r.Ticker,'PredictedExcessReturn':r.Prediction,'RealizedReturn':rr,'BenchmarkReturn':float(bmclose.iloc[p+63]/bmclose.iloc[p]-1)})
            if rets:
                bmr=float(bmclose.iloc[p+63]/bmclose.iloc[p]-1); port=float(np.mean(rets))
                bt.append({'SignalDate':d.strftime('%Y-%m-%d'),'PortfolioReturn':port-.001,'BenchmarkReturn':bmr,'ExcessReturn':port-.001-bmr,'Holdings':len(rets)})
        except Exception: continue
    bt_df=pd.DataFrame(bt); hold_df=pd.DataFrame(hold)
    importance=pd.DataFrame({'팩터':FEATURES,'Ridge 절대계수':np.abs(ridge.coef_)}).sort_values('Ridge 절대계수',ascending=False)
    meta={'as_of':str(latest.date()),'requested_tickers':len(tickers),'downloaded_tickers':len(hist)-1,'eligible_candidates':len(current),'training_rows':len(train),'training_dates':int(train.Date.nunique()),'backtest_periods':len(bt_df),'failed_symbols_or_batches':len(failures),'note':'Price-only model; no point-in-time fundamental/revisions data; current universe can have survivorship bias.'}
    return ranked.head(100), group_df, importance, bt_df, hold_df, meta

# =========================
# NAVIGATION / GLOBAL HEADER
# =========================
MENU = [
    '대시보드',
    '팩터 모델 · 추천 종목',
    '팩터 모델 · 팩터 분석',
    '팩터 모델 · 백테스트',
    '팩터 모델 · 개별 종목',
    '눌림목 모델 · 실시간 신호',
    '눌림목 모델 · 1개월 테스트',
    '모델 설정'
]
PAGE_META = {
    '대시보드': ('HOME / OVERVIEW', '리서치 대시보드', '팩터 모델과 눌림목 모델을 한 화면에서 선택하세요.'),
    '팩터 모델 · 추천 종목': ('FACTOR MODEL / RANKING', '팩터 모델 · 추천 종목', '팩터 점수를 기준으로 종목 후보를 정렬합니다.'),
    '팩터 모델 · 팩터 분석': ('FACTOR MODEL / ATTRIBUTION', '팩터 분석', '모델의 변수 중요도와 종목별 팩터 기여도를 확인합니다.'),
    '팩터 모델 · 백테스트': ('FACTOR MODEL / BACKTEST', '팩터 모델 · 백테스트', '과거 워크포워드 성과를 QQQ와 비교합니다.'),
    '팩터 모델 · 개별 종목': ('FACTOR MODEL / STOCK DETAIL', '개별 종목 분석', '선택한 종목의 예측값과 팩터 기여도를 확인합니다.'),
    '눌림목 모델 · 실시간 신호': ('PULLBACK MODEL / LIVE', '눌림목 모델 · 실시간 신호', '현재 가격 데이터에서 눌림목 조건을 검색합니다.'),
    '눌림목 모델 · 1개월 테스트': ('PULLBACK MODEL / TEST', '눌림목 모델 · 1개월 테스트', '최근 한 달 구간의 눌림목 신호와 이후 최고수익률을 확인합니다.'),
    '모델 설정': ('CONFIGURATION / DATA', '모델 설정', '종목 수, 가격 데이터 시작일, 수집 묶음 크기를 조정합니다.'),
}

if 'page' not in st.session_state:
    st.session_state.page = '대시보드'
if 'factor_results' not in st.session_state:
    st.session_state.factor_results = None
if 'pullback_results' not in st.session_state:
    st.session_state.pullback_results = None

# Show the sidebar only within a functional page, not on the home menu.
if st.session_state.page != '대시보드':
    with st.sidebar:
        st.markdown('## QUANT SIGNAL')
        st.caption('FACTOR LAB  /  PULLBACK')
        if st.button('⌂  홈 대시보드', key='nav_home', use_container_width=True):
            st.session_state.page = '대시보드'
            st.session_state.pop('page_radio', None)
            st.rerun()
        st.divider()
        selected_page = st.radio('MENU', MENU[1:], index=(MENU[1:].index(st.session_state.page) if st.session_state.page in MENU[1:] else 0), key='page_radio')
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
        st.divider()
        st.caption('데이터: Yahoo Finance / Google Sheets')
page = st.session_state.page

# Header lives in normal document flow with generous top padding so it is not
# hidden behind the fixed Streamlit toolbar when the page first opens.
st.markdown(
    '<div class="app-hero">'
    '<div class="eyebrow">QUANTITATIVE RESEARCH WORKSPACE</div>'
    '<div class="main-title">QUANT SIGNAL</div>'
    '<div class="sub-title">FACTOR LAB  ·  PULLBACK SIGNALS  ·  800 STOCK UNIVERSE</div>'
    '</div>',
    unsafe_allow_html=True,
)

def render_page_heading(page_name):
    kicker, title, note = PAGE_META[page_name]
    st.markdown(f'<div class="page-kicker">{kicker}</div><div class="page-title">{title}</div><div class="menu-section-note">{note}</div>', unsafe_allow_html=True)

def show_stable_table(data, height=360, column_width='small'):
    """Fixed-height, compact tables reduce page jumps and horizontal overflow."""
    if data is None:
        st.info('표시할 데이터가 없습니다.')
        return
    view = data.copy()
    config = {}
    for col in view.columns:
        dtype = view[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            config[col] = st.column_config.NumberColumn(str(col), width=column_width)
        else:
            config[col] = st.column_config.TextColumn(str(col), width=column_width)
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        height=height,
        row_height=32,
        column_config=config,
    )

# =========================
# PAGES
# =========================
if page == '대시보드':
    st.markdown('<div class="menu-section-title">분석 메뉴</div><div class="menu-section-note">원하는 기능을 선택하세요. 각 카드는 동일한 높이의 2열 레이아웃으로 배치했습니다.</div>', unsafe_allow_html=True)
    f = st.session_state.factor_results
    p = st.session_state.pullback_results
    menu_cards = [
        ('01', '⌂', '대시보드', '모델 실행 상태와 최근 분석 결과를 확인합니다.', '대시보드'),
        ('02', '↗', '추천 종목', '팩터 점수 상위 종목과 예상 초과수익률을 확인합니다.', '팩터 모델 · 추천 종목'),
        ('03', '◈', '팩터 분석', '팩터 중요도와 종목별 기여도를 분석합니다.', '팩터 모델 · 팩터 분석'),
        ('04', '▥', '백테스트', 'QQQ 비교, 누적수익률, 낙폭과 테스트 기록을 봅니다.', '팩터 모델 · 백테스트'),
        ('05', '⌕', '개별 종목 분석', '선택한 종목의 예측값과 팩터별 근거를 확인합니다.', '팩터 모델 · 개별 종목'),
        ('06', '⌁', '눌림목 실시간 신호', '이동평균 정배열 기반의 현재 신호를 검색합니다.', '눌림목 모델 · 실시간 신호'),
        ('07', '◷', '눌림목 1개월 테스트', '최근 한 달 신호와 이후 최고수익률을 확인합니다.', '눌림목 모델 · 1개월 테스트'),
        ('08', '⚙', '모델 설정', '분석 종목 수와 가격 데이터 수집 범위를 조정합니다.', '모델 설정'),
    ]
    for row_start in range(0, len(menu_cards), 2):
        cols = st.columns(2, gap='medium')
        for col, item in zip(cols, menu_cards[row_start:row_start+2]):
            num, icon, title, desc, target = item
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="menu-card"><div class="menu-icon">{icon}</div>'
                        f'<div class="menu-number">{num} / QUANT SIGNAL</div>'
                        f'<div class="menu-title">{title}</div><div class="menu-desc">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if target == '대시보드':
                        open_clicked = st.button('현재 상태 보기  ↗', key='home_open_'+num, use_container_width=True)
                    else:
                        open_clicked = st.button('메뉴 열기  ↗', key='home_open_'+num, use_container_width=True)
                    if open_clicked:
                        st.session_state.page = target
                        st.session_state.pop('page_radio', None)
                        st.rerun()
    # Compact operational status instead of a separate quick-launch sidebar.
    st.markdown('### 최근 실행 상태')
    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric('팩터 모델', '실행 완료' if f else '대기 중')
        if f:
            st.caption(f"기준일 {f['meta'].get('as_of', 'N/A')} · 적격 후보 {f['meta'].get('eligible_candidates', 0)}개")
    with status_cols[1]:
        st.metric('눌림목 모델', '실행 완료' if p else '대기 중')
        if p:
            st.caption(f'최근 신호 {len(p[0])}개')
    st.caption('참고: 모델 예측은 투자 권유나 수익 보장이 아닙니다. 팩터 모델은 가격 데이터 기반이며 현재 종목 유니버스 사용에 따른 생존편향 가능성이 있습니다.')

if page=='모델 설정':
    render_page_heading('모델 설정')
    st.markdown('### 분석 범위')
    st.write('팩터 모델과 눌림목 모델의 종목 수는 기본 800개입니다. 팩터 모델은 장기 가격 데이터를 필요로 하므로 첫 실행에 시간이 걸릴 수 있습니다.')
    max_tickers=st.select_slider('검토 종목 수',options=[100,200,400,600,800],value=800)
    history_start=st.selectbox('팩터 모델 가격 데이터 시작일',['2014-01-01','2016-01-01','2018-01-01'],index=0)
    batch_size=st.selectbox('Yahoo Finance 묶음 크기',[10,20,25,40],index=2)
    st.caption('묶음 크기가 작을수록 요청을 나누지만 총 요청 수는 늘 수 있습니다. Yahoo Finance의 비공식 인터페이스에는 요청 제한이 있을 수 있습니다.')
    st.session_state['settings']={'max_tickers':max_tickers,'history_start':history_start,'batch_size':batch_size}
    if st.button('캐시 초기화',key='clear_cache'):
        st.cache_data.clear(); st.success('앱 캐시를 초기화했습니다. 다음 실행 시 데이터를 다시 수집합니다.')
    st.markdown('### 관리자 보안')
    st.write('비밀번호를 코드에 직접 저장하지 마세요. 필요한 경우 Streamlit Secrets의 `ADMIN_PASSWORD`를 설정하세요.')
    try: admin_expected=st.secrets.get('ADMIN_PASSWORD','')
    except Exception: admin_expected=''
    if admin_expected:
        admin_input=st.text_input('관리자 비밀번호',type='password')
        if admin_input==admin_expected:
            if st.button('관리자 캐시 강제 초기화'): st.cache_data.clear(); st.success('캐시 초기화 완료')
    else: st.info('현재 ADMIN_PASSWORD Secret이 설정되어 있지 않습니다. 관리자 전용 기능은 비활성화되어 있습니다.')

# Run queued model from dashboard
if st.session_state.pop('run_factor',False):
    settings=st.session_state.get('settings',{'max_tickers':800,'history_start':'2014-01-01','batch_size':25})
    try:
        with st.spinner('팩터 모델 실행 중: 가격 수집·팩터 계산·학습·백테스트를 진행합니다. 첫 실행은 오래 걸릴 수 있습니다.'):
            res=run_factor_model(settings['max_tickers'],settings['history_start'],settings['batch_size'])
        st.session_state.factor_results={'ranked':res[0],'groups':res[1],'importance':res[2],'backtest':res[3],'holdings':res[4],'meta':res[5]}
        st.success('팩터 모델 분석이 완료됐습니다. 팩터 모델 메뉴에서 결과를 확인하세요.')
    except Exception as e: st.error(f'팩터 모델 실행 실패: {e}')
if st.session_state.pop('run_pullback',False):
    settings=st.session_state.get('settings',{'max_tickers':800})
    try:
        with st.spinner('눌림목 모델 실행 중: 6개월 가격 데이터를 수집하고 신호를 분석합니다.'):
            st.session_state.pullback_results=run_pullback_analysis(settings['max_tickers'])
        st.success('눌림목 분석이 완료됐습니다.')
    except Exception as e: st.error(f'눌림목 모델 실행 실패: {e}')

if page=='팩터 모델 · 추천 종목':
    render_page_heading('팩터 모델 · 추천 종목')
    if st.button('팩터 모델 실행 / 갱신',key='factor_run_recs'):
        settings=st.session_state.get('settings',{'max_tickers':800,'history_start':'2014-01-01','batch_size':25})
        try:
            with st.spinner('분석 중입니다. 첫 실행에는 시간이 걸립니다.'):
                res=run_factor_model(settings['max_tickers'],settings['history_start'],settings['batch_size'])
            st.session_state.factor_results={'ranked':res[0],'groups':res[1],'importance':res[2],'backtest':res[3],'holdings':res[4],'meta':res[5]}
            st.success('완료')
        except Exception as e: st.error(f'실행 실패: {e}')
    f=st.session_state.factor_results
    if f:
        st.caption(f"기준일 {f['meta']['as_of']} · 수집 종목 {f['meta']['downloaded_tickers']} · 적격 후보 {f['meta']['eligible_candidates']}")
        df=f['ranked'].copy(); df.insert(0,'순위',range(1,len(df)+1))
        show=df[['순위','Ticker','Close','ADV20','예상 초과수익률(63일)','Ridge 예측','트리 예측']].copy()
        for c in ['예상 초과수익률(63일)','Ridge 예측','트리 예측']: show[c]=show[c].map(lambda x:f'{x:.2%}')
        show=show.rename(columns={'Ticker':'종목','Close':'현재가','ADV20':'평균 거래대금(USD)'})
        display_count = st.selectbox('표시 종목 수', [20, 30, 50, 100], index=0)
        # Keep the default table narrow enough for mobile: detailed model outputs remain in CSV.
        compact = show[['순위', '종목', '현재가', '예상 초과수익률(63일)', '평균 거래대금(USD)']].head(display_count).copy()
        compact['평균거래대금($M)'] = (pd.to_numeric(compact.pop('평균 거래대금(USD)'), errors='coerce') / 1_000_000).map(lambda x: f'{x:,.0f}' if pd.notna(x) else '—')
        compact = compact.rename(columns={'예상 초과수익률(63일)': '예상 초과(63일)'})
        compact['현재가'] = pd.to_numeric(compact['현재가'], errors='coerce').map(lambda x: f'${x:,.2f}' if pd.notna(x) else '—')
        # Static table: avoids the embedded grid capturing vertical swipe gestures on mobile.
        st.table(compact)
        st.download_button('추천 종목 CSV 다운로드',df.to_csv(index=False).encode('utf-8-sig'),'factor_recommendations.csv','text/csv')
    else: st.info('먼저 대시보드 또는 이 페이지에서 팩터 모델을 실행하세요.')

if page=='팩터 모델 · 팩터 분석':
    render_page_heading('팩터 모델 · 팩터 분석')
    f=st.session_state.factor_results
    if not f: st.info('먼저 팩터 모델을 실행하세요.')
    else:
        st.markdown('### 전체 변수 중요도')
        show_stable_table(f['importance'].rename(columns={'Ridge 절대계수': 'Ridge 계수'}), height=360)
        st.bar_chart(f['importance'].head(12).set_index('팩터'))
        st.markdown('### 종목별 팩터 그룹 기여도')
        tickers=f['ranked'].Ticker.tolist()
        chosen=st.selectbox('종목 선택',tickers)
        gd=f['groups'][f['groups'].Ticker==chosen].set_index('팩터 그룹')
        st.bar_chart(gd[['Ridge 기여도']])
        show_stable_table(gd.drop(columns=['Ticker'], errors='ignore').reset_index(), height=300)
        st.caption('기여도는 Ridge 선형 구성요소만 반영합니다. 트리 모델의 비선형 효과를 완전히 분해한 값은 아닙니다. 글로벌 중요도는 모델 전체의 인과적 효과를 뜻하지 않습니다.')

if page=='팩터 모델 · 백테스트':
    render_page_heading('팩터 모델 · 백테스트')
    f=st.session_state.factor_results
    if not f: st.info('먼저 팩터 모델을 실행하세요.')
    else:
        bt=f['backtest']
        if bt.empty: st.warning('충분한 과거 학습 표본을 확보한 백테스트 기간이 없습니다.')
        else:
            st.metric('백테스트 기간 수',len(bt))
            st.metric('기간 평균 초과수익률',f"{bt.ExcessReturn.mean():.2%}")
            st.metric('초과수익 기간 비율',f"{(bt.ExcessReturn>0).mean():.1%}")
            plot=bt.copy(); plot['SignalDate']=pd.to_datetime(plot.SignalDate); plot=plot.set_index('SignalDate')
            plot['Portfolio cumulative']=(1+plot.PortfolioReturn).cumprod()-1
            plot['QQQ cumulative']=(1+plot.BenchmarkReturn).cumprod()-1
            st.line_chart(plot[['Portfolio cumulative','QQQ cumulative']])
            bt_view = bt.rename(columns={'SignalDate': '기준일', 'PortfolioReturn': '전략수익률', 'BenchmarkReturn': 'QQQ 수익률', 'ExcessReturn': '초과수익률', 'Holdings': '보유수'})
            show_stable_table(bt_view, height=360)
            st.download_button('백테스트 CSV 다운로드',bt.to_csv(index=False).encode('utf-8-sig'),'factor_backtest.csv','text/csv')
            st.caption('주의: 현재 유니버스 기반 과거 재구성은 생존편향이 남을 수 있습니다. 비용은 단순 10bp 가정이며, 배당/체결/시장충격 등을 완전히 반영하지 않습니다.')

if page=='팩터 모델 · 개별 종목':
    render_page_heading('팩터 모델 · 개별 종목')
    f=st.session_state.factor_results
    if not f: st.info('먼저 팩터 모델을 실행하세요.')
    else:
        alltick=f['ranked'].Ticker.tolist(); ticker=st.selectbox('분석할 종목',alltick)
        row=f['ranked'][f['ranked'].Ticker==ticker].iloc[0]
        a,b,c=st.columns(3); a.metric('현재가',f"${row.Close:.2f}"); b.metric('예상 초과수익률',f"{row['예상 초과수익률(63일)']:.2%}"); c.metric('랭킹',f"#{alltick.index(ticker)+1}")
        gd=f['groups'][f['groups'].Ticker==ticker].sort_values('Ridge 기여도',ascending=False)
        st.markdown('### 팩터 그룹 기여도'); st.bar_chart(gd.set_index('팩터 그룹')[['Ridge 기여도']]); show_stable_table(gd.drop(columns=['Ticker'], errors='ignore'), height=340)
        st.caption('현재 가격 팩터 모델의 상대적 예측 결과이며 목표주가나 수익 보장이 아닙니다. 재무제표·애널리스트 전망 수정 팩터는 포함되지 않았습니다.')

if page in ('눌림목 모델 · 실시간 신호','눌림목 모델 · 1개월 테스트'):
    render_page_heading(page)
    if st.button('눌림목 분석 실행 / 갱신',key='pullback_run_'+page):
        settings=st.session_state.get('settings',{'max_tickers':800})
        try:
            with st.spinner('최대 800개 종목의 6개월 가격 데이터를 분석합니다.'):
                st.session_state.pullback_results=run_pullback_analysis(settings['max_tickers'])
        except Exception as e: st.error(f'실행 실패: {e}')
    p=st.session_state.pullback_results
    if p:
        live,past,updated,nreq,nscan=p
        st.caption(f"마지막 분석 {updated:%Y-%m-%d %H:%M:%S} · 요청 종목 {nreq} · 스캔 종목 {nscan}")
        data=live if page.endswith('실시간 신호') else past
        if data.empty: st.info('현재 조건에 부합하는 종목이 없습니다.')
        else:
            data_view = data.rename(columns={'발생일': '발생일', '종목': '종목', '현재가': '현재가', '눌림목 점수': '점수', '이후 최고수익률(%)': '이후 최고(%)'})
            show_stable_table(data_view, height=360)
            st.download_button('현재 표 CSV 다운로드',data.to_csv(index=False).encode('utf-8-sig'),'pullback_signals.csv','text/csv')
        if page.endswith('1개월 테스트'):
            st.warning('이 테스트의 최고수익률은 신호 발생 후 현재까지의 최고가를 사용하므로, 실제 체결 가능한 수익률과 다릅니다. 향후 고가를 이용한 성과는 매도 시점을 사전에 알 수 없다는 한계가 있습니다.')
    else: st.info('눌림목 분석을 실행하면 결과가 여기에 표시됩니다.')

# footer
st.divider()
st.caption('QUANT SIGNAL  ·  Research tool only  ·  Predictions are uncertain and not investment advice.')

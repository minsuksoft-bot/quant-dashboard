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
st.set_page_config(page_title="QUANT SIGNAL | Factor Lab", page_icon="⬛", layout="centered")
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.8/dist/web/static/pretendard.css');
html, body, [class*="css"] {font-family:'Pretendard',sans-serif!important;}
.block-container {padding-top:1.2rem!important;padding-left:.8rem!important;padding-right:.8rem!important;max-width:100vw!important;}
.main-title {font-size:2rem;font-weight:850;letter-spacing:-.04em;margin-bottom:0;}
.sub-title {font-size:.9rem;color:#777;margin-bottom:1rem;}
.stButton>button {border-radius:9px;font-weight:650;width:100%;}
div[data-testid="stMetric"] {border:1px solid #eee;padding:12px;border-radius:12px;}
[data-testid="stDataFrame"] {width:100%;}
</style>
""", unsafe_allow_html=True)

TICKER_SOURCE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTgA3hs0AXyyXwkg3U6j902doYiv9U8BrwejaodapPou48w7j2jX56Zlwh4RKJFmV6wBV4TJ21U3_cF/pub?output=csv"
BENCHMARK = "QQQ"
DEFAULT_MAX_TICKERS = 800

st.markdown('<p class="main-title">QUANT SIGNAL</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">NASDAQ FACTOR LAB · Technical Signals · Mobile-first</p>', unsafe_allow_html=True)

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
# NAVIGATION
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
with st.sidebar:
    st.markdown('## QUANT SIGNAL')
    st.caption('Factor Model + Pullback Model')
    page=st.radio('페이지 메뉴',MENU,index=0)
    st.divider()
    st.caption('데이터 소스: Yahoo Finance / Google Sheets')

if 'factor_results' not in st.session_state: st.session_state.factor_results=None
if 'pullback_results' not in st.session_state: st.session_state.pullback_results=None

# =========================
# PAGES
# =========================
if page=='대시보드':
    st.subheader('Market Research Dashboard')
    st.write('팩터 모델과 기존 눌림목 모델을 분리해 운영합니다. 각 분석은 해당 메뉴에서 실행할 수 있습니다.')
    f=st.session_state.factor_results; p=st.session_state.pullback_results
    c1,c2=st.columns(2)
    with c1:
        st.markdown('### Factor Model')
        if f:
            st.metric('추천 후보',f['meta']['eligible_candidates']); st.caption(f"마지막 분석: {f['meta']['as_of']}")
        else: st.info('아직 실행된 팩터 분석이 없습니다.')
        if st.button('팩터 모델 실행',key='dash_factor'): st.session_state['run_factor']=True; st.rerun()
    with c2:
        st.markdown('### Pullback Model')
        if p:
            st.metric('현재 신호',len(p[0])); st.caption('최근 실행 결과')
        else: st.info('아직 실행된 눌림목 분석이 없습니다.')
        if st.button('눌림목 모델 실행',key='dash_pullback'): st.session_state['run_pullback']=True; st.rerun()
    st.warning('모델 예측은 투자 권유나 수익 보장이 아닙니다. 팩터 모델은 가격 데이터 기반이며 생존편향이 남을 수 있습니다.')

if page=='모델 설정':
    st.subheader('모델 설정')
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
    st.subheader('Factor Model · Recommendations')
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
        st.dataframe(show.head(st.selectbox('표시 종목 수',[20,30,50,100],index=0)),use_container_width=True,hide_index=True)
        st.download_button('추천 종목 CSV 다운로드',df.to_csv(index=False).encode('utf-8-sig'),'factor_recommendations.csv','text/csv')
    else: st.info('먼저 대시보드 또는 이 페이지에서 팩터 모델을 실행하세요.')

if page=='팩터 모델 · 팩터 분석':
    st.subheader('Factor Model · Attribution')
    f=st.session_state.factor_results
    if not f: st.info('먼저 팩터 모델을 실행하세요.')
    else:
        st.markdown('### 전체 변수 중요도')
        st.dataframe(f['importance'],use_container_width=True,hide_index=True)
        st.bar_chart(f['importance'].head(12).set_index('팩터'))
        st.markdown('### 종목별 팩터 그룹 기여도')
        tickers=f['ranked'].Ticker.tolist()
        chosen=st.selectbox('종목 선택',tickers)
        gd=f['groups'][f['groups'].Ticker==chosen].set_index('팩터 그룹')
        st.bar_chart(gd[['Ridge 기여도']])
        st.dataframe(gd,use_container_width=True)
        st.caption('기여도는 Ridge 선형 구성요소만 반영합니다. 트리 모델의 비선형 효과를 완전히 분해한 값은 아닙니다. 글로벌 중요도는 모델 전체의 인과적 효과를 뜻하지 않습니다.')

if page=='팩터 모델 · 백테스트':
    st.subheader('Factor Model · Backtest')
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
            st.dataframe(bt,use_container_width=True,hide_index=True)
            st.download_button('백테스트 CSV 다운로드',bt.to_csv(index=False).encode('utf-8-sig'),'factor_backtest.csv','text/csv')
            st.caption('주의: 현재 유니버스 기반 과거 재구성은 생존편향이 남을 수 있습니다. 비용은 단순 10bp 가정이며, 배당/체결/시장충격 등을 완전히 반영하지 않습니다.')

if page=='팩터 모델 · 개별 종목':
    st.subheader('Factor Model · Stock Detail')
    f=st.session_state.factor_results
    if not f: st.info('먼저 팩터 모델을 실행하세요.')
    else:
        alltick=f['ranked'].Ticker.tolist(); ticker=st.selectbox('분석할 종목',alltick)
        row=f['ranked'][f['ranked'].Ticker==ticker].iloc[0]
        a,b,c=st.columns(3); a.metric('현재가',f"${row.Close:.2f}"); b.metric('예상 초과수익률',f"{row['예상 초과수익률(63일)']:.2%}"); c.metric('랭킹',f"#{alltick.index(ticker)+1}")
        gd=f['groups'][f['groups'].Ticker==ticker].sort_values('Ridge 기여도',ascending=False)
        st.markdown('### 팩터 그룹 기여도'); st.bar_chart(gd.set_index('팩터 그룹')[['Ridge 기여도']]); st.dataframe(gd,use_container_width=True,hide_index=True)
        st.caption('현재 가격 팩터 모델의 상대적 예측 결과이며 목표주가나 수익 보장이 아닙니다. 재무제표·애널리스트 전망 수정 팩터는 포함되지 않았습니다.')

if page in ('눌림목 모델 · 실시간 신호','눌림목 모델 · 1개월 테스트'):
    st.subheader('Pullback Model · '+('Live Signals' if page.endswith('실시간 신호') else '1-Month Test'))
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
            st.dataframe(data,use_container_width=True,hide_index=True)
            st.download_button('현재 표 CSV 다운로드',data.to_csv(index=False).encode('utf-8-sig'),'pullback_signals.csv','text/csv')
        if page.endswith('1개월 테스트'):
            st.warning('이 테스트의 최고수익률은 신호 발생 후 현재까지의 최고가를 사용하므로, 실제 체결 가능한 수익률과 다릅니다. 향후 고가를 이용한 성과는 매도 시점을 사전에 알 수 없다는 한계가 있습니다.')
    else: st.info('눌림목 분석을 실행하면 결과가 여기에 표시됩니다.')

# footer
st.divider()
st.caption('QUANT SIGNAL · Research tool only · Predictions are uncertain and not investment advice.')

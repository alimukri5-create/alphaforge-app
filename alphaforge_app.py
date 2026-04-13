import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import argrelextrema
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ==========================================
# HELPERS FOR YFINANCE DATA (FIXES ERRORS)
# ==========================================

def safe_close(df, ticker=None):
    """Safely extract close prices from yfinance"""
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker and ticker in df.columns.get_level_values(1):
                return df['Close'][ticker]
            return df['Close'].iloc[:, 0]
        return df['Close']
    except:
        return None

def safe_high_low_vol(df, ticker=None):
    """Safely extract high, low, volume"""
    if df is None or df.empty:
        return None, None, None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker and ticker in df.columns.get_level_values(1):
                return df['High'][ticker], df['Low'][ticker], df['Volume'][ticker]
            return df['High'].iloc[:, 0], df['Low'].iloc[:, 0], df['Volume'].iloc[:, 0]
        return df['High'], df['Low'], df['Volume']
    except:
        return None, None, None

# ==========================================
# OPTIONS DATA LAYER (Reads options, trades spot)
# ==========================================

def get_options_implied_data(ticker):
    """Get IV, expected move, and skew from options - for spot trading decisions"""
    try:
        stock = yf.Ticker(ticker)
        current_price = safe_close(stock.history(period="1d"), ticker)
        if current_price is None or current_price.empty:
            return None
            
        current_price = float(current_price.iloc[-1])
        expirations = stock.options
        
        if len(expirations) == 0:
            return {'error': 'No options available'}
        
        # Get nearest expiration (30-45 days out ideally)
        target_date = datetime.now() + timedelta(days=30)
        closest_exp = min(expirations, 
                         key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))
        
        opt = stock.option_chain(closest_exp)
        
        # ATM IV
        calls = opt.calls
        puts = opt.puts
        
        atm_call_idx = (calls['strike'] - current_price).abs().argsort()[:1]
        atm_put_idx = (puts['strike'] - current_price).abs().argsort()[:1]
        
        call_iv = float(calls.iloc[atm_call_idx]['impliedVolatility'].iloc[0])
        put_iv = float(puts.iloc[atm_put_idx]['impliedVolatility'].iloc[0])
        atm_iv = (call_iv + put_iv) / 2
        
        # Expected move from straddle
        call_price = float(calls.iloc[atm_call_idx]['lastPrice'].iloc[0])
        put_price = float(puts.iloc[atm_put_idx]['lastPrice'].iloc[0])
        straddle = call_price + put_price
        expected_move_pct = (straddle / current_price) * 100
        
        # Simple skew: 5% OTM put vs 5% OTM call
        otm_put_strike = current_price * 0.95
        otm_call_strike = current_price * 1.05
        
        put_5 = puts[puts['strike'] <= otm_put_strike]
        call_5 = calls[calls['strike'] >= otm_call_strike]
        
        if not put_5.empty and not call_5.empty:
            put_5_iv = float(put_5.iloc[-1]['impliedVolatility'])
            call_5_iv = float(call_5.iloc[0]['impliedVolatility'])
            skew = put_5_iv - call_5_iv  # Positive = fear of downside
        else:
            skew = 0
        
        # IV interpretation
        if atm_iv < 0.30:
            iv_status = "CHEAP"
            iv_signal = "buy"  # Options underpricing = good entry timing
        elif atm_iv > 0.50:
            iv_status = "EXPENSIVE"
            iv_signal = "avoid"  # High premium, wait for vol crush
        else:
            iv_status = "FAIR"
            iv_signal = "neutral"
        
        return {
            'atm_iv': atm_iv,
            'iv_status': iv_status,
            'iv_signal': iv_signal,
            'expected_move_pct': expected_move_pct,
            'expected_move_dollars': current_price * (expected_move_pct/100),
            'skew': skew,
            'skew_signal': 'fear' if skew > 0.03 else 'greed' if skew < -0.01 else 'neutral',
            'expiration': closest_exp
        }
    except Exception as e:
        return {'error': str(e)}

# ==========================================
# SETUP SCANNER CLASS
# ==========================================

class LongOnlyScanner:
    def __init__(self):
        self.spy_data = None
        self.market_ok = False
        
    def check_market_regime(self):
        """Global gate: Is market safe for long entries?"""
        try:
            self.spy_data = yf.download("SPY", period="1y", progress=False)
            spy_close = safe_close(self.spy_data, "SPY")
            
            if spy_close is None or spy_close.empty:
                return False
                
            weekly_spy = spy_close.resample('W-Fri').last()
            weekly_ema20 = weekly_spy.ewm(span=20, adjust=False).mean()
            weekly_trend = bool(weekly_spy.iloc[-1] > weekly_ema20.iloc[-1])
            
            sma50 = spy_close.rolling(50).mean()
            daily_trend = bool(spy_close.iloc[-1] > sma50.iloc[-1])
            
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix_close = safe_close(vix_data, "^VIX")
            
            if vix_close is not None and not vix_close.empty:
                vix_current = float(vix_close.iloc[-1])
                vix_low = vix_current < 25
            else:
                vix_low = True
                
            self.market_ok = weekly_trend and daily_trend and vix_low
            return self.market_ok
            
        except Exception as e:
            st.error(f"Market check failed: {e}")
            return False
    
    def squeeze_breakout_signal(self, ticker):
        """Bollinger Squeeze breakout detection"""
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < 50:
                return False, 0, "No data", None
            
            close = safe_close(df, ticker)
            high, low, volume = safe_high_low_vol(df, ticker)
            
            if close is None or volume is None:
                return False, 0, "Data error", None
            
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = sma20 + (2 * std20)
            lower = sma20 - (2 * std20)
            bandwidth = (upper - lower) / sma20
            squeeze = bandwidth <= bandwidth.rolling(120).min()
            vol_avg = volume.rolling(20).mean()
            
            curr_close = float(close.iloc[-1])
            prev_upper = float(upper.iloc[-2])
            curr_vol = float(volume.iloc[-1])
            prev_vol_avg = float(vol_avg.iloc[-2])
            was_squeezed = bool(squeeze.iloc[-2])
            
            breakout = curr_close > prev_upper
            volume_spike = curr_vol > (prev_vol_avg * 1.3)
            is_setup = breakout and volume_spike and was_squeezed
            
            score = 0
            if is_setup:
                score = 50
                if squeeze.iloc[-20:].sum() >= 5:
                    score += 20
                if curr_vol > (prev_vol_avg * 2):
                    score += 20
                if curr_close > float(sma20.iloc[-1]):
                    score += 10
            
            reason = ""
            if is_setup:
                reason = f"Squeeze breakout"
            elif not was_squeezed:
                reason = "No squeeze"
            elif not breakout:
                reason = "No breakout"
            else:
                reason = "Low volume"
            
            return is_setup, score, reason, curr_close
            
        except Exception as e:
            return False, 0, str(e), None
    
    def relative_strength_ok(self, ticker):
        """Is stock keeping up with SPY?"""
        try:
            stock_df = yf.download(ticker, period="3mo", progress=False)
            stock_close = safe_close(stock_df, ticker)
            spy_close = safe_close(self.spy_data, "SPY")
            
            if spy_close is None:
                spy_df = yf.download("SPY", period="3mo", progress=False)
                spy_close = safe_close(spy_df, "SPY")
            
            if stock_close is None or spy_close is None:
                return False
            
            stock_ret = (float(stock_close.iloc[-1]) / float(stock_close.iloc[-20]) - 1)
            spy_ret = (float(spy_close.iloc[-1]) / float(spy_close.iloc[-20]) - 1)
            
            return stock_ret >= (spy_ret - 0.03)
        except:
            return False
    
    def check_earnings_risk(self, ticker):
        """Avoid setups right before earnings"""
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                next_earn = earnings.index[0]
                days_to = (next_earn - datetime.now()).days
                if 0 <= days_to <= 3:
                    return False, f"Earnings in {days_to} days"
            return True, "OK"
        except:
            return True, "Unknown"

def scan_watchlist(watchlist, use_options_filter=True):
    """Main scanning function"""
    scanner = LongOnlyScanner()
    
    if not scanner.check_market_regime():
        return [], "🛑 MARKET NOT OK - Check SPY trend and VIX", []
    
    results = []
    rejected = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist))
        
        # Technical checks
        is_setup, score, reason, price = scanner.squeeze_breakout_signal(ticker)
        if not is_setup:
            rejected.append(f"{ticker}: {reason}")
            continue
        
        if not scanner.relative_strength_ok(ticker):
            rejected.append(f"{ticker}: Weak vs SPY")
            continue
        
        earn_ok, earn_msg = scanner.check_earnings_risk(ticker)
        if not earn_ok:
            rejected.append(f"{ticker}: {earn_msg}")
            continue
        
        # Options overlay (optional)
        options_data = None
        if use_options_filter:
            options_data = get_options_implied_data(ticker)
            if options_data and 'iv_signal' in options_data:
                if options_data['iv_signal'] == 'avoid':
                    rejected.append(f"{ticker}: IV too expensive ({options_data['atm_iv']:.1%})")
                    continue
        
        # PASSED ALL FILTERS
        result = {
            'Ticker': ticker,
            'Price': f"${price:.2f}",
            'Score': score,
            'Setup': 'Squeeze',
            'Quality': 'HIGH' if score > 80 else 'MEDIUM'
        }
        
        if options_data and 'error' not in options_data:
            result['IV'] = f"{options_data['atm_iv']:.1%}"
            result['Exp_Move'] = f"{options_data['expected_move_pct']:.1f}%"
            result['IV_Status'] = options_data['iv_status']
        
        results.append(result)
    
    progress_bar.empty()
    return results, "✅ Market OK", rejected

# ==========================================
# STREAMLIT APP
# ==========================================

st.set_page_config(page_title="AlphaForge Alpha", page_icon="🎯", layout="wide")

st.title("🎯 AlphaForge Alpha Generation")

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price & Zones", 
    "🎲 Monte Carlo", 
    "🧠 Models", 
    "⚠️ Risk",
    "🎯 Setup Scanner"
])

# ==========================================
# TAB 5: SETUP SCANNER (Standalone)
# ==========================================

with tab5:
    st.header("Long-Only Setup Scanner")
    st.caption("Squeeze breakouts + Options intelligence (for spot trading)")
    
    # Market check
    scanner = LongOnlyScanner()
    market_ok = scanner.check_market_regime()
    
    if market_ok:
        st.success("✅ MARKET OK FOR LONGS")
    else:
        st.error("🛑 MARKET REGIME BLOCKING NEW LONGS")
        st.info("SPY must be above weekly 20 EMA + daily 50 SMA, and VIX < 25")
    
    # Options filter toggle
    use_options = st.checkbox("Use Options Data Filters (IV < 50% only)", value=True)
    if use_options:
        st.info("This reads options data to avoid buying when implied vol is expensive, even though you're trading spot equity")
    
    st.divider()
    
    # Watchlist input
    default_list = "AAPL, MSFT, NVDA, TSLA, AMD, NFLX, CRM, META, AMZN, GOOGL, COIN, HOOD, RBLX, U"
    watchlist_input = st.text_area("Enter tickers (comma-separated):", value=default_list, height=100)
    
    if market_ok and st.button("🔍 SCAN FOR SETUPS", type="primary"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        
        with st.spinner(f"Scanning {len(tickers)} tickers... This takes 30-60 seconds"):
            results, market_msg, rejected = scan_watchlist(tickers, use_options_filter=use_options)
        
        if not results:
            st.warning("No setups found matching criteria")
            with st.expander("See why stocks were filtered out"):
                for r in rejected[:20]:
                    st.write(f"• {r}")
        else:
            st.success(f"Found {len(results)} qualified setups out of {len(tickers)} scanned")
            
            # Results table
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Individual analysis
            st.divider()
            for result in results:
                with st.expander(f"{result['Ticker']} - {result['Quality']} ({result['Score']}/100)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Price:** {result['Price']}")
                        st.write(f"**Setup:** Squeeze breakout")
                    with col2:
                        if 'IV' in result:
                            st.write(f"**IV:** {result['IV']} ({result['IV_Status']})")
                            st.write(f"**Expected Move:** ±{result['Exp_Move']}")
                        else:
                            st.write("No options data")
                    with col3:
                        # Calculate stops
                        try:
                            price_str = result['Price'].replace('$', '')
                            price = float(price_str)
                            ticker = result['Ticker']
                            
                            # ATR for stop
                            df = yf.download(ticker, period="3mo", progress=False)
                            if not df.empty:
                                close = safe_close(df, ticker)
                                high, low, _ = safe_high_low_vol(df, ticker)
                                if close is not None:
                                    tr1 = high - low
                                    tr2 = abs(high - close.shift())
                                    tr3 = abs(low - close.shift())
                                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                                    atr = tr.rolling(14).mean().iloc[-1]
                                    
                                    stop = price - (2 * atr)
                                    target = price + (3 * atr * 2)  # 3R
                                    
                                    st.write(f"**Stop:** ${stop:.2f}")
                                    st.write(f"**Target:** ${target:.2f}")
                        except:
                            pass
    
    elif not market_ok:
        st.button("🔍 SCAN FOR SETUPS", type="primary", disabled=True)
        st.warning("Scanning disabled - wait for market regime to improve")

# ==========================================
# SIDEBAR & OTHER TABS
# ==========================================

st.sidebar.header("📊 Single Ticker Analysis")
analysis_ticker = st.sidebar.text_input("Ticker", value="MSFT", key="analysis_ticker").upper()
benchmark = st.sidebar.text_input("Benchmark", value="SPY")
timeframe = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Models:**")
mc_on = st.sidebar.checkbox("Monte Carlo", value=True)
regime_on = st.sidebar.checkbox("Regime Detection", value=True)
risk_on = st.sidebar.checkbox("Risk Metrics", value=True)

svj_on = st.sidebar.checkbox("Jump-Trend", value=True)
meanrev_on = st.sidebar.checkbox("Mean Reversion", value=True)
support_on = st.sidebar.checkbox("Support Level", value=True)
momentum_on = st.sidebar.checkbox("Momentum", value=True)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner(f"Analyzing {analysis_ticker}..."):
        
        try:
            # Load data
            stock = yf.Ticker(analysis_ticker)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error("No data found")
                st.stop()
            
            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'][analysis_ticker] if analysis_ticker in data.columns.get_level_values(1) else data['Close'].iloc[:, 0]
            else:
                prices = data['Close']
                
            current_price = float(prices.iloc[-1])
            returns = prices.pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            
            # Simple models
            signals = []
            
            # Jump-trend (SVJ style)
            if svj_on:
                jumps = np.abs(returns) > (3 * returns.std())
                jump_count = jumps.sum()
                if returns.mean() > 0 and jump_count < len(returns) * 0.05:
                    signals.append(("Jump-Trend", "BULLISH", 70))
                else:
                    signals.append(("Jump-Trend", "NEUTRAL", 50))
            
            # Mean reversion (Kalman style)
            if meanrev_on:
                ema20 = prices.ewm(span=20).mean().iloc[-1]
                dev = (current_price - ema20) / ema20
                if dev < -0.05:
                    signals.append(("Mean Reversion", "BULLISH", 75))
                elif dev > 0.05:
                    signals.append(("Mean Reversion", "BEARISH", 60))
                else:
                    signals.append(("Mean Reversion", "NEUTRAL", 50))
            
            # Support (Topological style)
            if support_on:
                recent_low = float(prices.tail(60).min())
                if current_price < recent_low * 1.02:
                    signals.append(("Support", "BULLISH", 80))
                else:
                    signals.append(("Support", "NEUTRAL", 50))
            
            # Momentum (LSTM style)
            if momentum_on:
                rsi = 50
                try:
                    delta = prices.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = (100 - (100 / (1 + rs))).iloc[-1]
                except:
                    pass
                
                if rsi < 35:
                    signals.append(("Momentum", "BULLISH", 75))
                elif rsi > 65:
                    signals.append(("Momentum", "BEARISH", 75))
                else:
                    signals.append(("Momentum", "NEUTRAL", 50))
            
            # Display tabs
            with tab1:
                st.subheader(f"{analysis_ticker} Price & Zones")
                
                # Zones
                base_mult = 1.0
                if vol > 0.30:
                    base_mult = 1.3
                
                z1 = current_price * (1 - 0.02 * base_mult)
                z2 = current_price * (1 - 0.05 * base_mult)
                z3 = current_price * (1 - 0.12 * base_mult)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Price'))
                fig.add_hrect(y0=z1, y1=current_price, fillcolor="green", opacity=0.2, annotation_text="Zone 1")
                fig.add_hrect(y0=z2, y1=z1, fillcolor="blue", opacity=0.2, annotation_text="Zone 2")
                fig.add_hrect(y0=z3, y1=z2, fillcolor="purple", opacity=0.2, annotation_text="Zone 3")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"**Current:** ${current_price:.2f}")
                st.write(f"**Volatility:** {vol:.1%}")
            
            with tab2:
                if mc_on:
                    st.subheader("Monte Carlo (30 days)")
                    
                    n_sims = 1000
                    mu = returns.mean()
                    sigma = returns.std()
                    
                    np.random.seed(42)
                    paths = np.zeros((n_sims, 30))
                    paths[:, 0] = current_price
                    
                    for i in range(1, 30):
                        paths[:, i] = paths[:, i-1] * (1 + np.random.normal(mu, sigma, n_sims))
                    
                    fig_mc = go.Figure()
                    for i in range(0, 100, 5):
                        fig_mc.add_trace(go.Scatter(x=list(range(30)), y=paths[i], mode='lines', 
                                                   line=dict(color='gray', width=0.5), opacity=0.3, showlegend=False))
                    
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), y=np.percentile(paths, 95, axis=0), 
                                               mode='lines', line=dict(color='green'), name='95%'))
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), y=np.percentile(paths, 50, axis=0), 
                                               mode='lines', line=dict(color='blue'), name='Median'))
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), y=np.percentile(paths, 5, axis=0), 
                                               mode='lines', line=dict(color='red'), name='5%'))
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    final = paths[:, -1]
                    prob_up = np.mean(final > current_price)
                    st.metric("Probability Higher", f"{prob_up:.1%}")
            
            with tab3:
                st.subheader("Model Signals")
                for name, signal, strength in signals:
                    emoji = "🟢" if signal == "BULLISH" else "🔴" if signal == "BEARISH" else "🟡"
                    st.write(f"{emoji} **{name}:** {signal} ({strength}%)")
                
                # Options overlay for single ticker
                st.divider()
                st.subheader("Options Intelligence")
                opt_data = get_options_implied_data(analysis_ticker)
                
                if opt_data and 'error' not in opt_data:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("IV", f"{opt_data['atm_iv']:.1%}", opt_data['iv_status'])
                    col2.metric("Expected Move", f"±{opt_data['expected_move_pct']:.1f}%")
                    col3.metric("Skew", f"{opt_data['skew']:.2f}", opt_data['skew_signal'])
                    
                    if opt_data['iv_signal'] == 'avoid':
                        st.error("⚠️ IV too expensive - Consider waiting for vol crush")
                    elif opt_data['iv_signal'] == 'buy':
                        st.success("✅ IV cheap - Good timing for entry")
                else:
                    st.info("No options data available for this ticker")
            
            with tab4:
                st.subheader("Risk Metrics")
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                cum_ret = (1 + returns).cumprod()
                running_max = cum_ret.expanding().max()
                max_dd = ((cum_ret - running_max) / running_max).min()
                
                st.metric("Sharpe", f"{sharpe:.2f}")
                st.metric("Max Drawdown", f"{max_dd:.1%}")
                st.metric("Volatility", f"{vol:.1%}")
                
                if regime_on:
                    vol_20 = returns.rolling(20).std() * np.sqrt(252)
                    v_mean = vol_20.mean()
                    v_std = vol_20.std()
                    current_v = vol_20.iloc[-1]
                    
                    if current_v > v_mean + v_std:
                        st.error("HIGH VOL REGIME - Wider stops needed")
                    elif current_v < v_mean - v_std:
                        st.success("LOW VOL REGIME - Trend following favorable")
                    else:
                        st.info("TRANSITION REGIME - Caution")
            
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("AlphaForge Alpha - Educational purposes only")

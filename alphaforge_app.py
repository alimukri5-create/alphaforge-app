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
# SETUP SCANNER CODE (Integrated)
# ==========================================

class LongOnlyScanner:
    def __init__(self):
        self.spy_data = None
        self.vix_data = None
        self.weekly_trend = None
        self.daily_trend = None
        self.vix_low = None
        self.market_ok = False
        
    def check_market_regime(self):
        """Global gate: Is market safe for long entries?"""
        try:
            self.spy_data = yf.download("SPY", period="1y", progress=False)
            if self.spy_data.empty:
                return False
                
            weekly_spy = self.spy_data.resample('W-Fri').last()
            weekly_ema20 = weekly_spy['Close'].ewm(span=20, adjust=False).mean()
            self.weekly_trend = weekly_spy['Close'].iloc[-1] > weekly_ema20.iloc[-1]
            
            sma50 = self.spy_data['Close'].rolling(50).mean()
            self.daily_trend = self.spy_data['Close'].iloc[-1] > sma50.iloc[-1]
            
            self.vix_data = yf.download("^VIX", period="5d", progress=False)
            if not self.vix_data.empty:
                vix_current = self.vix_data['Close'].iloc[-1]
                self.vix_low = vix_current < 25
            else:
                self.vix_low = True
                
            self.market_ok = self.weekly_trend and self.daily_trend and self.vix_low
            return self.market_ok
            
        except Exception as e:
            st.error(f"Market regime check failed: {e}")
            return False
    
    def get_market_status_text(self):
        if not self.market_ok:
            return "🛑 NO NEW LONGS"
        status = "✅ MARKET OK FOR LONGS\n\n"
        status += f"Weekly Trend (SPY): {'UP' if self.weekly_trend else 'DOWN'}\n"
        status += f"Daily Trend (SPY): {'UP' if self.daily_trend else 'DOWN'}\n"
        status += f"VIX < 25: {'YES' if self.vix_low else 'NO'}"
        return status
    
    def squeeze_breakout_signal(self, ticker):
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < 50:
                return False, 0, "No data"
            
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['STD20'] = df['Close'].rolling(20).std()
            df['Upper'] = df['SMA20'] + (2 * df['STD20'])
            df['Lower'] = df['SMA20'] - (2 * df['STD20'])
            df['Bandwidth'] = (df['Upper'] - df['Lower']) / df['SMA20']
            df['Squeeze'] = df['Bandwidth'] <= df['Bandwidth'].rolling(120).min()
            df['VolAvg'] = df['Volume'].rolling(20).mean()
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            price_breakout = curr['Close'] > prev['Upper']
            volume_confirmed = curr['Volume'] > (prev['VolAvg'] * 1.3)
            was_in_squeeze = prev['Squeeze']
            squeeze_duration = df['Squeeze'].iloc[-20:].sum()
            
            is_setup = price_breakout and volume_confirmed and was_in_squeeze
            
            score = 0
            if is_setup:
                score = 50
                if squeeze_duration >= 5:
                    score += 20
                if curr['Volume'] > (prev['VolAvg'] * 2):
                    score += 20
                if curr['Close'] > curr['SMA20']:
                    score += 10
            
            reason = ""
            if is_setup:
                reason = f"Squeeze breakout ({int(squeeze_duration)} days compression)"
            elif not was_in_squeeze:
                reason = "No squeeze detected"
            elif not price_breakout:
                reason = "No price breakout"
            elif not volume_confirmed:
                reason = "Low volume"
            
            return is_setup, score, reason
            
        except Exception as e:
            return False, 0, f"Error: {str(e)}"
    
    def relative_strength_check(self, ticker, lookback=20):
        try:
            stock = yf.download(ticker, period="3mo", progress=False)['Close']
            spy = self.spy_data['Close'] if self.spy_data is not None else yf.download("SPY", period="3mo", progress=False)['Close']
            
            if stock.empty or spy.empty or len(stock) < lookback:
                return False, 0
            
            stock_ret = (stock.iloc[-1] / stock.iloc[-lookback] - 1)
            spy_ret = (spy.iloc[-1] / spy.iloc[-lookback] - 1)
            
            rs_ok = stock_ret >= (spy_ret - 0.03)
            rs_score = min(100, max(0, (stock_ret - spy_ret + 0.05) * 500))
            
            return rs_ok, rs_score
        except:
            return False, 0
    
    def check_events(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            if earnings is not None and not earnings.empty:
                next_earnings = earnings.index[0]
                days_to_earnings = (next_earnings - datetime.now()).days
                if 0 <= days_to_earnings <= 3:
                    return False, f"Earnings in {days_to_earnings} days"
            return True, "No events"
        except:
            return True, "Event check unavailable"

def scan_watchlist(watchlist):
    scanner = LongOnlyScanner()
    market_ok = scanner.check_market_regime()
    if not market_ok:
        return [], scanner.get_market_status_text()
    
    results = []
    progress_bar = st.progress(0)
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist))
        
        is_squeeze, squeeze_score, reason = scanner.squeeze_breakout_signal(ticker)
        rs_ok, rs_score = scanner.relative_strength_check(ticker)
        event_ok, event_msg = scanner.check_events(ticker)
        
        if is_squeeze and rs_ok and event_ok:
            total_score = (squeeze_score * 0.6) + (rs_score * 0.4)
            results.append({
                'Ticker': ticker,
                'Setup': 'Squeeze Breakout',
                'Score': int(total_score),
                'Squeeze_Reason': reason,
                'Event_Status': event_msg,
                'Entry_Quality': 'HIGH' if total_score > 80 else 'MEDIUM'
            })
    
    progress_bar.empty()
    results = sorted(results, key=lambda x: x['Score'], reverse=True)
    return results, scanner.get_market_status_text()

def get_exit_levels(ticker, entry_price):
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty:
            return None
            
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        stop_loss = entry_price - (2 * atr)
        risk = entry_price - stop_loss
        target = entry_price + (3 * risk)
        
        return {
            'stop_loss': stop_loss,
            'target': target,
            'atr': atr,
            'risk_pct': (risk / entry_price) * 100
        }
    except:
        return None

def setup_scanner_tab():
    st.header("🎯 Long-Only Setup Scanner")
    st.caption("Squeeze breakout detection + Market regime filter")
    
    scanner = LongOnlyScanner()
    market_ok = scanner.check_market_regime()
    
    if market_ok:
        st.success(scanner.get_market_status_text())
    else:
        st.error("🛑 MARKET REGIME: NO NEW LONGS")
        st.warning("SPY below weekly/daily trends OR VIX > 25. Avoid new entries.")
        return
    
    st.divider()
    st.subheader("Scan Watchlist")
    
    default_list = "AAPL, MSFT, NVDA, TSLA, AMD, NFLX, CRM, META, AMZN, GOOGL, COIN, HOOD, RBLX, U"
    watchlist_input = st.text_area("Enter tickers (comma-separated):", value=default_list)
    
    if st.button("🔍 SCAN FOR SETUPS", type="primary"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        
        with st.spinner(f"Scanning {len(tickers)} tickers..."):
            results, market_msg = scan_watchlist(tickers)
        
        if not results:
            st.info("No squeeze breakout setups found.")
            st.caption("Look for: Price breaking upper Bollinger Band after 5+ days compression + volume spike.")
        else:
            st.success(f"Found {len(results)} setup(s)")
            
            df = pd.DataFrame(results)
            st.dataframe(df[['Ticker', 'Score', 'Entry_Quality', 'Squeeze_Reason', 'Event_Status']], 
                        use_container_width=True, hide_index=True)
            
            for result in results:
                with st.expander(f"{result['Ticker']} (Score: {result['Score']}/100)"):
                    try:
                        current_price = yf.Ticker(result['Ticker']).history(period="1d")['Close'].iloc[-1]
                        exits = get_exit_levels(result['Ticker'], current_price)
                        
                        if exits:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Current:** ${current_price:.2f}")
                                st.write(f"**Setup:** {result['Squeeze_Reason']}")
                            with col2:
                                st.write(f"**Stop:** ${exits['stop_loss']:.2f} ({exits['risk_pct']:.1f}%)")
                                st.write(f"**Target (3R):** ${exits['target']:.2f}")
                    except:
                        st.write("Price data unavailable")
    
    st.divider()
    st.subheader("Position Monitor")
    col1, col2 = st.columns(2)
    with col1:
        pos_ticker = st.text_input("Ticker", key="pos_ticker").upper()
        entry_price = st.number_input("Entry Price", min_value=0.0, format="%.2f", key="entry_price")
    
    if pos_ticker and entry_price > 0:
        try:
            exits = get_exit_levels(pos_ticker, entry_price)
            if exits:
                current = yf.Ticker(pos_ticker).history(period="1d")['Close'].iloc[-1]
                pnl_pct = ((current - entry_price) / entry_price) * 100
                
                st.write(f"**Current:** ${current:.2f} ({pnl_pct:+.1f}%)")
                st.write(f"**Stop Level:** ${exits['stop_loss']:.2f}")
                
                if current < exits['stop_loss']:
                    st.error("🚨 STOP HIT - EXIT NOW")
                elif current > exits['target']:
                    st.success("🎯 TARGET HIT - Take 1/2 profits")
                else:
                    st.info("⏱️ HOLD - Within risk range")
        except:
            st.error("Could not fetch position data")

# ==========================================
# ORIGINAL ALPHAFORGE APP CODE
# ==========================================

st.set_page_config(page_title="AlphaForge Alpha", page_icon="🎯", layout="wide")

st.title("🎯 AlphaForge Alpha Generation")
st.subheader("Probabilistic Forecasting & Statistical Edge")

st.sidebar.header("📊 Analysis Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="MSFT").upper()
benchmark = st.sidebar.text_input("Benchmark", value="SPY").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Alpha Models:**")
mc_on = st.sidebar.checkbox("Monte Carlo Forward Paths", value=True)
regime_on = st.sidebar.checkbox("Regime Detection", value=True)
coint_on = st.sidebar.checkbox("Cointegration vs Benchmark", value=True)
risk_on = st.sidebar.checkbox("Advanced Risk Metrics", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Base Models:**")
svj_on = st.sidebar.checkbox("SVJ", value=True)
kalman_on = st.sidebar.checkbox("Kalman", value=True)
topo_on = st.sidebar.checkbox("Topological", value=True)
lstm_on = st.sidebar.checkbox("LSTM", value=True)
evt_on = st.sidebar.checkbox("EVT", value=True)
tech_on = st.sidebar.checkbox("Technical", value=True)

if st.sidebar.button("🚀 Run Alpha Analysis", type="primary"):
    with st.spinner(f"Generating alpha analysis for {ticker}..."):
        
        try:
            # Load data
            stock = yf.Ticker(ticker)
            bench = yf.Ticker(benchmark)
            data = stock.history(period=timeframe)
            bench_data = bench.history(period=timeframe)
            
            if data.empty:
                st.error("No data found")
                st.stop()
            
            prices = data['Close']
            current_price = float(prices.iloc[-1])
            returns = prices.pct_change().dropna()
            
            # Load benchmark if available
            if not bench_data.empty and coint_on:
                bench_prices = bench_data['Close']
                bench_returns = bench_prices.pct_change().dropna()
                # Align lengths
                min_len = min(len(returns), len(bench_returns))
                aligned_returns = returns.iloc[-min_len:]
                aligned_bench = bench_returns.iloc[-min_len:]
            else:
                aligned_bench = None
            
            # ==========================================
            # BASE MODELS (Your existing 6)
            # ==========================================
            
            # SVJ
            if svj_on:
                vol = returns.std() * np.sqrt(252)
                jump_threshold = 3 * returns.std()
                jumps = np.abs(returns) > jump_threshold
                jump_intensity = jumps.sum() / len(returns) * 252
                trend = returns.mean() * 252
                
                if trend > 0.05 and vol < 0.4:
                    svj_sig = ("BULLISH", 85, f"Trend {trend:.1%}, {jump_intensity:.1f} jumps/yr")
                elif trend < -0.05:
                    svj_sig = ("BEARISH", 80, f"Negative trend {trend:.1%}")
                else:
                    svj_sig = ("NEUTRAL", 50, "Consolidation")
            else:
                svj_sig = ("OFF", 0, "Disabled")
                vol = returns.std() * np.sqrt(252)
                jump_intensity = 0
            
            # Kalman
            if kalman_on:
                kf_trend = prices.ewm(span=20).mean().iloc[-1]
                deviation = (current_price - kf_trend) / kf_trend
                
                if deviation < -0.05:
                    kalman_sig = ("BULLISH", 75, f"{deviation:.1%} below trend")
                elif deviation > 0.05:
                    kalman_sig = ("BEARISH", 75, f"{deviation:.1%} above trend")
                else:
                    kalman_sig = ("NEUTRAL", 50, "Aligned")
            else:
                kalman_sig = ("OFF", 0, "Disabled")
            
            # Topological
            if topo_on:
                local_min = argrelextrema(prices.values, np.less, order=10)[0]
                
                if len(local_min) > 0:
                    recent_support = prices.iloc[local_min[-1]]
                    persistence = len(prices) - local_min[-1]
                else:
                    recent_support = prices.tail(60).min()
                    persistence = 0
                
                if current_price < recent_support * 1.02:
                    topo_sig = ("BULLISH", 85, f"At support ${recent_support:.2f} ({persistence}d)")
                else:
                    topo_sig = ("NEUTRAL", 50, f"Support ${recent_support:.2f}")
            else:
                topo_sig = ("OFF", 0, "Disabled")
                recent_support = prices.tail(60).min()
            
            # LSTM
            if lstm_on:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                
                if rsi < 35 and vol > 0.25:
                    lstm_sig = ("BULLISH", 80, f"RSI {rsi:.1f}, oversold")
                elif rsi > 65:
                    lstm_sig = ("BEARISH", 75, f"RSI {rsi:.1f}, overbought")
                else:
                    lstm_sig = ("NEUTRAL", 50, f"RSI {rsi:.1f}")
            else:
                lstm_sig = ("OFF", 0, "Disabled")
                rsi = 50
            
            # EVT
            if evt_on:
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                tail_risk = abs(var_99) * current_price
                
                if abs(var_99) > 0.04:
                    evt_sig = ("NEUTRAL", 60, f"⚠️ Tail risk {var_99:.2%}")
                else:
                    evt_sig = ("NEUTRAL", 70, f"VaR 95%: {var_95:.2%}")
            else:
                evt_sig = ("OFF", 0, "Disabled")
                var_95 = -0.02
            
            # Technical
            if tech_on:
                sma_50 = prices.rolling(50).mean().iloc[-1]
                sma_200 = prices.rolling(200).mean().iloc[-1]
                
                if current_price > sma_50 > sma_200:
                    tech_sig = ("BULLISH", 75, "Golden cross")
                elif current_price < sma_50 < sma_200:
                    tech_sig = ("BEARISH", 75, "Death cross")
                else:
                    tech_sig = ("NEUTRAL", 50, "Mixed")
            else:
                tech_sig = ("OFF", 0, "Disabled")
            
            # ==========================================
            # ALPHA GENERATION MODELS (The New Stuff)
            # ==========================================
            
            # 1. MONTE CARLO FORWARD SIMULATION
            if mc_on:
                n_sims = 1000
                n_days = 30
                mu = returns.mean()
                sigma = returns.std()
                
                np.random.seed(42)
                mc_paths = np.zeros((n_sims, n_days))
                mc_paths[:, 0] = current_price
                
                for i in range(1, n_days):
                    shocks = np.random.normal(mu, sigma, n_sims)
                    mc_paths[:, i] = mc_paths[:, i-1] * (1 + shocks)
                
                final_prices = mc_paths[:, -1]
                mc_stats = {
                    'prob_up': np.mean(final_prices > current_price),
                    'median': np.median(final_prices),
                    'mean': np.mean(final_prices),
                    'p5': np.percentile(final_prices, 5),
                    'p25': np.percentile(final_prices, 25),
                    'p75': np.percentile(final_prices, 75),
                    'p95': np.percentile(final_prices, 95),
                    'paths': mc_paths
                }
            else:
                mc_stats = None
            
            # 2. REGIME DETECTION (Hidden Markov Style)
            if regime_on:
                vol_20 = returns.rolling(20).std() * np.sqrt(252)
                vol_mean = vol_20.mean()
                vol_std = vol_20.std()
                current_vol = vol_20.iloc[-1]
                
                # Regime classification
                if current_vol > vol_mean + vol_std:
                    regime = "HIGH_VOL"
                    regime_conf = 0.8
                    regime_desc = "High volatility - reduce size, widen stops"
                    regime_color = "red"
                elif current_vol < vol_mean - vol_std:
                    regime = "LOW_VOL"
                    regime_conf = 0.8
                    regime_desc = "Low volatility - trend following effective"
                    regime_color = "green"
                else:
                    regime = "TRANSITION"
                    regime_conf = 0.6
                    regime_desc = "Regime uncertainty - caution"
                    regime_color = "yellow"
                
                regime_stats = {
                    'regime': regime,
                    'confidence': regime_conf,
                    'description': regime_desc,
                    'color': regime_color,
                    'current_vol': current_vol,
                    'mean_vol': vol_mean
                }
            else:
                regime_stats = None
            
            # 3. COINTEGRATION / STATISTICAL ARBITRAGE
            if coint_on and aligned_bench is not None:
                correlation = aligned_returns.corr(aligned_bench)
                
                # Price ratio z-score
                price_ratio = prices / bench_prices.reindex(prices.index, method='ffill')
                ratio_mean = price_ratio.mean()
                ratio_std = price_ratio.std()
                current_z = (price_ratio.iloc[-1] - ratio_mean) / ratio_std
                
                if abs(current_z) > 2:
                    coint_signal = "MEAN_REVERSION"
                    coint_strength = "STRONG"
                    if current_z > 2:
                        coint_trade = f"SHORT {ticker} / LONG {benchmark}"
                    else:
                        coint_trade = f"LONG {ticker} / SHORT {benchmark}"
                elif abs(current_z) > 1:
                    coint_signal = "MEAN_REVERSION"
                    coint_strength = "MODERATE"
                    coint_trade = "Watch for entry"
                else:
                    coint_signal = "EQUILIBRIUM"
                    coint_strength = "NONE"
                    coint_trade = "No arbitrage"
                
                coint_stats = {
                    'correlation': correlation,
                    'z_score': current_z,
                    'signal': coint_signal,
                    'strength': coint_strength,
                    'trade': coint_trade
                }
            else:
                coint_stats = None
            
            # 4. ADVANCED RISK METRICS
            if risk_on and aligned_bench is not None:
                # Sharpe
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                
                # Information Ratio (alpha vs benchmark)
                excess = aligned_returns - aligned_bench
                tracking_err = excess.std() * np.sqrt(252)
                info_ratio = (excess.mean() * 252) / tracking_err if tracking_err > 0 else 0
                
                # Max Drawdown
                cum_ret = (1 + returns).cumprod()
                running_max = cum_ret.expanding().max()
                drawdown = (cum_ret - running_max) / running_max
                max_dd = drawdown.min()
                
                # Sortino
                downside = returns[returns < 0]
                downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.01
                sortino = (returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
                
                risk_stats = {
                    'sharpe': sharpe,
                    'info_ratio': info_ratio,
                    'max_dd': max_dd,
                    'sortino': sortino,
                    'volatility': vol
                }
            else:
                risk_stats = None
            
            # ==========================================
            # ENSEMBLE & ZONES
            # ==========================================
            
            all_sigs = [svj_sig, kalman_sig, topo_sig, lstm_sig, evt_sig, tech_sig]
            active = [s for s in all_sigs if s[0] != "OFF"]
            
            bullish = sum([s[1] for s in active if s[0] == "BULLISH"])
            bearish = sum([s[1] for s in active if s[0] == "BEARISH"])
            total = sum([s[1] for s in active])
            
            if total == 0:
                composite, confidence = "NEUTRAL", 50
            else:
                net = (bullish - bearish) / total
                if net > 0.3:
                    composite, confidence = "STRONG_BUY", int(75 + net * 25)
                elif net > 0.1:
                    composite, confidence = "BUY", int(65 + net * 35)
                elif net < -0.3:
                    composite, confidence = "STRONG_SELL", int(75 - net * 25)
                elif net < -0.1:
                    composite, confidence = "SELL", int(65 - net * 35)
                else:
                    composite, confidence = "HOLD", 50
            
            # Dynamic zones with regime adjustment
            base_mult = 1.0
            if regime_stats:
                if regime_stats['regime'] == "HIGH_VOL":
                    base_mult = 1.3  # Wider zones
                elif regime_stats['regime'] == "LOW_VOL":
                    base_mult = 0.9  # Tighter zones
            
            z1 = current_price * (1 - 0.02 * base_mult)
            z2 = current_price * (1 - 0.05 * base_mult)
            z3 = current_price * (1 - 0.12 * base_mult)
            z2 = max(z2, recent_support * 0.98)
            
            # ==========================================
            # DISPLAY - 5 TABS (Added Setup Scanner)
            # ==========================================
            
            st.success(f"✅ Alpha analysis complete for {ticker}")
            
            # Top Metrics
            cols = st.columns([1, 1, 1, 1, 1.5])
            cols[0].metric("Price", f"${current_price:.2f}")
            cols[1].metric("Signal", composite, f"{confidence}%")
            cols[2].metric("Volatility", f"{vol:.1%}")
            if risk_stats:
                cols[3].metric("Sharpe", f"{risk_stats['sharpe']:.2f}")
                cols[4].metric("Max DD", f"{risk_stats['max_dd']:.1%}")
            else:
                cols[3].metric("Jumps/Yr", f"{jump_intensity:.1f}")
            
            st.divider()
            
            # Main tabs - NOW 5 TABS INSTEAD OF 4
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Price & Zones", 
                "🎲 Monte Carlo", 
                "🧠 Model Consensus", 
                "⚠️ Risk & Regime",
                "🎯 Setup Scanner"  # NEW TAB
            ])
            
            with tab1:
                # Price chart with zones
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', 
                                        name='Price', line=dict(color='black', width=2)))
                
                # Zone bands
                fig.add_hrect(y0=z1, y1=current_price, fillcolor="green", opacity=0.2, 
                             annotation_text="Zone 1")
                fig.add_hrect(y0=z2, y1=z1, fillcolor="blue", opacity=0.2, 
                             annotation_text="Zone 2")
                fig.add_hrect(y0=z3, y1=z2, fillcolor="purple", opacity=0.2, 
                             annotation_text="Zone 3")
                
                fig.update_layout(height=500, title=f"{ticker} Price Action")
                st.plotly_chart(fig, use_container_width=True)
                
                # Zone table
                zone_df = pd.DataFrame({
                    'Zone': ['Zone 1 (Immediate)', 'Zone 2 (Support)', 'Zone 3 (Deep Value)'],
                    'Price Range': [f"${z1:.2f} - ${current_price:.2f}",
                                   f"${z2:.2f} - ${z1:.2f}",
                                   f"${z3:.2f} - ${z2:.2f}"],
                    'Fill Prob': ['70%', '40%', '15%'],
                    'R/R': ['2.5:1', '3.5:1', '5:1']
                })
                st.dataframe(zone_df, hide_index=True, use_container_width=True)
            
            with tab2:
                if mc_stats:
                    st.subheader("Monte Carlo Forward Simulation (30 Days)")
                    
                    # Plot paths
                    fig_mc = go.Figure()
                    
                    # Sample paths
                    for i in range(0, min(100, len(mc_stats['paths'])), 5):
                        fig_mc.add_trace(go.Scatter(
                            x=list(range(30)), y=mc_stats['paths'][i],
                            mode='lines', line=dict(color='gray', width=0.5),
                            opacity=0.2, showlegend=False
                        ))
                    
                    # Percentiles
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), 
                                               y=np.percentile(mc_stats['paths'], 95, axis=0),
                                               mode='lines', line=dict(color='green', width=2),
                                               name='95th %ile'))
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), 
                                               y=np.percentile(mc_stats['paths'], 50, axis=0),
                                               mode='lines', line=dict(color='blue', width=3),
                                               name='Median'))
                    fig_mc.add_trace(go.Scatter(x=list(range(30)), 
                                               y=np.percentile(mc_stats['paths'], 5, axis=0),
                                               mode='lines', line=dict(color='red', width=2),
                                               name='5th %ile'))
                    
                    fig_mc.add_hline(y=current_price, line_dash="dash", line_color="black")
                    fig_mc.update_layout(height=400, xaxis_title="Days", yaxis_title="Price ($)")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # MC stats
                    mc_cols = st.columns(4)
                    mc_cols[0].metric("Prob Up", f"{mc_stats['prob_up']:.1%}")
                    mc_cols[1].metric("Expected", f"${mc_stats['median']:.2f}", 
                                     f"{(mc_stats['median']/current_price-1)*100:+.1f}%")
                    mc_cols[2].metric("Worst 5%", f"${mc_stats['p5']:.2f}")
                    mc_cols[3].metric("Best 5%", f"${mc_stats['p95']:.2f}")
                else:
                    st.info("Enable Monte Carlo in sidebar")
            
            with tab3:
                st.subheader("Model Consensus")
                
                model_names = ["SVJ", "Kalman", "Topological", "LSTM", "EVT", "Technical"]
                for name, sig in zip(model_names, all_sigs):
                    if sig[0] == "OFF":
                        continue
                    emoji = "🟢" if sig[0] == "BULLISH" else "🔴" if sig[0] == "BEARISH" else "🟡"
                    with st.expander(f"{emoji} {name}: {sig[0]} ({sig[1]}%)"):
                        st.write(sig[2])
                
                # Net score
                st.info(f"**Ensemble Net Score:** {net:.2f} (Bullish: {bullish}, Bearish: {bearish})")
            
            with tab4:
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.subheader("Risk Metrics")
                    if risk_stats:
                        st.metric("Sharpe Ratio", f"{risk_stats['sharpe']:.2f}")
                        st.metric("Information Ratio", f"{risk_stats['info_ratio']:.2f}")
                        st.metric("Sortino Ratio", f"{risk_stats['sortino']:.2f}")
                        st.metric("Max Drawdown", f"{risk_stats['max_dd']:.1%}")
                    else:
                        st.info("Enable Risk Metrics in sidebar")
                
                with col_r2:
                    st.subheader("Regime Detection")
                    if regime_stats:
                        color = regime_stats['color']
                        if color == "red":
                            st.error(f"**{regime_stats['regime']}** ({regime_stats['confidence']:.0%})")
                        elif color == "green":
                            st.success(f"**{regime_stats['regime']}** ({regime_stats['confidence']:.0%})")
                        else:
                            st.warning(f"**{regime_stats['regime']}** ({regime_stats['confidence']:.0%})")
                        
                        st.write(regime_stats['description'])
                        st.write(f"Current: {regime_stats['current_vol']:.1%} vs Mean: {regime_stats['mean_vol']:.1%}")
                    else:
                        st.info("Enable Regime Detection in sidebar")
                
                if coint_stats:
                    st.subheader(f"Statistical Arbitrage: {ticker} vs {benchmark}")
                    st.metric("Correlation", f"{coint_stats['correlation']:.2f}")
                    st.metric("Z-Score", f"{coint_stats['z_score']:.2f}σ")
                    
                    if coint_stats['strength'] == "STRONG":
                        st.error(f"🚨 **{coint_stats['signal']}** - {coint_stats['trade']}")
                    elif coint_stats['strength'] == "MODERATE":
                        st.warning(f"⚠️ **{coint_stats['signal']}** - {coint_stats['trade']}")
                    else:
                        st.success(f"✅ **{coint_stats['signal']}** - {coint_stats['trade']}")
            
            # NEW TAB 5 - Setup Scanner
            with tab5:
                setup_scanner_tab()
            
            st.divider()
            
            # Final recommendation
            st.subheader("🎯 Alpha-Optimized Strategy")
            
            reasons = []
            if composite in ["BUY", "STRONG_BUY"]:
                reasons.append(f"Ensemble signal: {composite}")
            if mc_stats and mc_stats['prob_up'] > 0.6:
                reasons.append(f"Monte Carlo {mc_stats['prob_up']:.0%} prob of gain")
            if regime_stats and regime_stats['regime'] == "LOW_VOL":
                reasons.append("Low vol regime (favorable)")
            if coint_stats and coint_stats['strength'] == "STRONG":
                reasons.append(f"Stat arb: {coint_stats['z_score']:.1f}σ deviation")
            
            if reasons:
                st.write("**Positive Factors:**")
                for r in reasons:
                    st.write(f"• {r}")
            
            # Risk warning
            if regime_stats and regime_stats['regime'] == "HIGH_VOL":
                st.warning("⚠️ High volatility regime - Consider smaller size, wider stops")
            
            if risk_stats and risk_stats['max_dd'] < -0.25:
                st.error(f"🚨 High historical drawdown ({risk_stats['max_dd']:.1%}) - High risk ticker")
            
            st.caption("AlphaForge AlphaGen | Mathematical edge through probabilistic forecasting")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Configure models and click 'Run Alpha Analysis'")
    
    st.subheader("📚 Alpha Generation Features")
    st.write("**Monte Carlo:** 1,000 forward paths for probabilistic price forecasting")
    st.write("**Regime Detection:** Volatility state identification (High/Low/Transition)")
    st.write("**Cointegration:** Statistical arbitrage opportunities vs benchmark")
    st.write("**Risk Metrics:** Sharpe, Sortino, Information Ratio, Max Drawdown")
    st.write("**Base Models:** SVJ, Kalman, Topological, LSTM, EVT, Technical")

st.divider()
st.caption("For sophisticated quantitative analysis | Educational purposes")

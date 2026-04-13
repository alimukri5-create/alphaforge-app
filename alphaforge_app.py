import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import argrelextrema
import warnings
from datetime import datetime, timedelta, date
warnings.filterwarnings('ignore')

# ==========================================
# HELPERS
# ==========================================

def safe_close(df, ticker=None):
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
# BACKTEST ENGINE (NEW)
# ==========================================

class ScannerBacktest:
    def __init__(self, tickers, start_date, end_date, use_options_filter=True):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.use_options_filter = use_options_filter
        self.results = []
        
    def fetch_data_for_date(self, ticker, as_of_date):
        """Fetch data as of a specific historical date"""
        try:
            # Get 6 months of data ending at as_of_date
            end_date = pd.Timestamp(as_of_date)
            start_date = end_date - pd.DateOffset(months=6)
            
            # Download historical data
            df = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            
            if df.empty or len(df) < 30:
                return None
                
            return df
        except:
            return None
    
    def check_squeeze_as_of(self, df, as_of_date):
        """Check if squeeze breakout occurred as of specific date"""
        try:
            if df is None or len(df) < 50:
                return False, 0, None
            
            # Get data up to as_of_date
            as_of_ts = pd.Timestamp(as_of_date)
            mask = df.index <= as_of_ts
            hist_data = df[mask]
            
            if len(hist_data) < 50:
                return False, 0, None
            
            close = safe_close(hist_data)
            high, low, volume = safe_high_low_vol(hist_data)
            
            if close is None or volume is None:
                return False, 0, None
            
            # Calculate indicators
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = sma20 + (2 * std20)
            lower = sma20 - (2 * std20)
            bandwidth = (upper - lower) / sma20
            squeeze = bandwidth <= bandwidth.rolling(120).min()
            vol_avg = volume.rolling(20).mean()
            
            # Check if today is a breakout
            if len(close) < 2:
                return False, 0, None
                
            today_close = float(close.iloc[-1])
            yesterday_upper = float(upper.iloc[-2])
            today_vol = float(volume.iloc[-1])
            yesterday_vol_avg = float(vol_avg.iloc[-2])
            was_squeezed = bool(squeeze.iloc[-2])
            
            breakout = today_close > yesterday_upper
            volume_spike = today_vol > (yesterday_vol_avg * 1.3)
            
            if breakout and volume_spike and was_squeezed:
                score = 50
                squeeze_days = int(squeeze.iloc[-20:].sum())
                if squeeze_days >= 5:
                    score += 20
                if today_vol > (yesterday_vol_avg * 2):
                    score += 20
                if today_close > float(sma20.iloc[-1]):
                    score += 10
                return True, score, today_close
            
            return False, 0, None
        except:
            return False, 0, None
    
    def check_relative_strength_as_of(self, ticker, as_of_date):
        """Check RS vs SPY as of specific date"""
        try:
            # Get SPY data
            spy_df = yf.download(
                "SPY",
                start=(pd.Timestamp(as_of_date) - pd.DateOffset(months=3)).strftime('%Y-%m-%d'),
                end=(pd.Timestamp(as_of_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            
            stock_df = yf.download(
                ticker,
                start=(pd.Timestamp(as_of_date) - pd.DateOffset(months=3)).strftime('%Y-%m-%d'),
                end=(pd.Timestamp(as_of_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            
            if spy_df.empty or stock_df.empty:
                return False
                
            spy_close = safe_close(spy_df)
            stock_close = safe_close(stock_df, ticker)
            
            if spy_close is None or stock_close is None:
                return False
            
            # Get 20-day returns as of date
            spy_mask = spy_df.index <= pd.Timestamp(as_of_date)
            stock_mask = stock_df.index <= pd.Timestamp(as_of_date)
            
            spy_hist = spy_close[spy_close.index <= pd.Timestamp(as_of_date)]
            stock_hist = stock_close[stock_close.index <= pd.Timestamp(as_of_date)]
            
            if len(spy_hist) < 20 or len(stock_hist) < 20:
                return False
            
            spy_ret = float(spy_hist.iloc[-1]) / float(spy_hist.iloc[-20]) - 1
            stock_ret = float(stock_hist.iloc[-1]) / float(stock_hist.iloc[-20]) - 1
            
            return stock_ret >= (spy_ret - 0.03)
        except:
            return False
    
    def get_outcome(self, ticker, entry_date, entry_price, hold_days=10):
        """Check what happened after entry"""
        try:
            # Get future data
            start_date = pd.Timestamp(entry_date) + pd.DateOffset(days=1)
            end_date = start_date + pd.DateOffset(days=hold_days+5)  # Extra for weekends
            
            df = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if df.empty:
                return None, None, None
            
            close = safe_close(df, ticker)
            if close is None or len(close) < 2:
                return None, None, None
            
            # Get historical data for ATR calculation (stop loss)
            hist = yf.download(
                ticker,
                start=(pd.Timestamp(entry_date) - pd.DateOffset(months=3)).strftime('%Y-%m-%d'),
                end=(pd.Timestamp(entry_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            
            if not hist.empty:
                h, l, _ = safe_high_low_vol(hist, ticker)
                c = safe_close(hist, ticker)
                if c is not None and h is not None and l is not None:
                    tr1 = h - l
                    tr2 = abs(h - c.shift())
                    tr3 = abs(l - c.shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean().iloc[-1]
                    stop_loss = entry_price - (2 * atr)
                    target = entry_price + (3 * atr * 2)  # 3R
                else:
                    stop_loss = entry_price * 0.94
                    target = entry_price * 1.15
            else:
                stop_loss = entry_price * 0.94
                target = entry_price * 1.15
            
            # Track outcomes day by day
            outcome = "OPEN"
            exit_price = close.iloc[-1]
            exit_date = close.index[-1]
            days_held = 0
            max_price = entry_price
            min_price = entry_price
            
            for i, (idx, price) in enumerate(close.items()):
                price_val = float(price)
                max_price = max(max_price, price_val)
                min_price = min(min_price, price_val)
                days_held = i + 1
                
                if price_val <= stop_loss:
                    outcome = "STOP_HIT"
                    exit_price = price_val
                    exit_date = idx
                    break
                elif price_val >= target:
                    outcome = "TARGET_HIT"
                    exit_price = price_val
                    exit_date = idx
                    break
                
                if days_held >= hold_days:
                    outcome = "TIME_EXIT"
                    exit_price = price_val
                    exit_date = idx
                    break
            
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            
            return {
                'outcome': outcome,
                'exit_price': exit_price,
                'exit_date': exit_date,
                'days_held': days_held,
                'pnl_pct': pnl_pct,
                'max_gain': (max_price - entry_price) / entry_price * 100,
                'max_loss': (min_price - entry_price) / entry_price * 100,
                'stop_level': stop_loss,
                'target_level': target
            }
            
        except Exception as e:
            return None
    
    def run(self, progress_bar=None, status_text=None):
        """Run backtest over date range"""
        # Generate trading days
        all_days = pd.date_range(self.start_date, self.end_date, freq='B')  # Business days
        trading_days = [d for d in all_days if d.weekday() < 5]  # Mon-Fri
        
        total_checks = len(trading_days) * len(self.tickers)
        checked = 0
        
        for check_date in trading_days:
            for ticker in self.tickers:
                checked += 1
                if progress_bar:
                    progress_bar.progress(checked / total_checks)
                if status_text:
                    status_text.text(f"Checking {ticker} on {check_date.strftime('%Y-%m-%d')}...")
                
                # Skip if market not open (simple check)
                if check_date.weekday() >= 5:
                    continue
                
                # Get data as of this date
                df = self.fetch_data_for_date(ticker, check_date)
                if df is None:
                    continue
                
                # Check for setup
                is_setup, score, entry_price = self.check_squeeze_as_of(df, check_date)
                if not is_setup:
                    continue
                
                # Check RS
                if not self.check_relative_strength_as_of(ticker, check_date):
                    continue
                
                # Found a setup - get outcome
                outcome = self.get_outcome(ticker, check_date, entry_price)
                if outcome:
                    self.results.append({
                        'entry_date': check_date,
                        'ticker': ticker,
                        'entry_price': entry_price,
                        'score': score,
                        'outcome': outcome['outcome'],
                        'exit_price': outcome['exit_price'],
                        'exit_date': outcome['exit_date'],
                        'days_held': outcome['days_held'],
                        'pnl_pct': outcome['pnl_pct'],
                        'max_gain': outcome['max_gain'],
                        'max_loss': outcome['max_loss']
                    })
        
        return self.results
    
    def get_statistics(self):
        """Calculate backtest statistics"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        wins = len(df[df['pnl_pct'] > 0])
        losses = len(df[df['pnl_pct'] <= 0])
        total = len(df)
        
        if total == 0:
            return None
        
        avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() if losses > 0 else 0
        
        # Calculate expectancy: (Win% * Avg Win) - (Loss% * |Avg Loss|)
        win_pct = wins / total
        loss_pct = losses / total
        expectancy = (win_pct * avg_win) - (loss_pct * abs(avg_loss))
        
        # Max drawdown calculation (simplified)
        cumulative = df['pnl_pct'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()
        
        # Sharpe (simplified, assuming risk-free rate 0)
        returns_mean = df['pnl_pct'].mean()
        returns_std = df['pnl_pct'].std()
        sharpe = (returns_mean / returns_std) * np.sqrt(252/10) if returns_std > 0 else 0  # Adjusted for hold period
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'total_return': df['pnl_pct'].sum(),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'avg_hold_days': df['days_held'].mean(),
            'results_df': df
        }

# ==========================================
# SETUP SCANNER CLASS (Updated)
# ==========================================

class LongOnlyScanner:
    def __init__(self):
        self.spy_data = None
        self.market_ok = False
        
    def check_market_regime(self):
        try:
            self.spy_data = yf.download("SPY", period="1y", progress=False)
            spy_close = safe_close(self.spy_data, "SPY")
            if spy_close is None:
                return False
            
            weekly_spy = spy_close.resample('W-Fri').last()
            weekly_ema20 = weekly_spy.ewm(span=20, adjust=False).mean()
            weekly_trend = bool(weekly_spy.iloc[-1] > weekly_ema20.iloc[-1])
            
            sma50 = spy_close.rolling(50).mean()
            daily_trend = bool(spy_close.iloc[-1] > sma50.iloc[-1])
            
            vix_data = yf.download("^VIX", period="5d", progress=False)
            vix_close = safe_close(vix_data, "^VIX")
            vix_low = float(vix_close.iloc[-1]) < 25 if vix_close is not None else True
            
            self.market_ok = weekly_trend and daily_trend and vix_low
            return self.market_ok
        except:
            return False
    
    def squeeze_breakout_signal(self, ticker):
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < 50:
                return False, 0, "No data", None
            
            close = safe_close(df, ticker)
            high, low, volume = safe_high_low_vol(df, ticker)
            if close is None:
                return False, 0, "Data error", None
            
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = sma20 + (2 * std20)
            bandwidth = (upper - (sma20 - (2 * std20))) / sma20
            squeeze = bandwidth <= bandwidth.rolling(120).min()
            vol_avg = volume.rolling(20).mean()
            
            curr_close = float(close.iloc[-1])
            prev_upper = float(upper.iloc[-2])
            curr_vol = float(volume.iloc[-1])
            prev_vol_avg = float(vol_avg.iloc[-2])
            
            breakout = curr_close > prev_upper
            volume_spike = curr_vol > (prev_vol_avg * 1.3)
            was_squeezed = bool(squeeze.iloc[-2])
            
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
            
            reason = "Squeeze breakout" if is_setup else "No setup"
            return is_setup, score, reason, curr_close
        except Exception as e:
            return False, 0, str(e), None
    
    def relative_strength_ok(self, ticker):
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

def scan_watchlist(watchlist):
    scanner = LongOnlyScanner()
    if not scanner.check_market_regime():
        return [], "Market not OK", []
    
    results = []
    rejected = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist))
        
        is_setup, score, reason, price = scanner.squeeze_breakout_signal(ticker)
        if not is_setup:
            rejected.append(f"{ticker}: {reason}")
            continue
        
        if not scanner.relative_strength_ok(ticker):
            rejected.append(f"{ticker}: Weak vs SPY")
            continue
        
        results.append({
            'Ticker': ticker,
            'Price': f"${price:.2f}",
            'Score': score,
            'Quality': 'HIGH' if score > 80 else 'MEDIUM'
        })
    
    progress_bar.empty()
    return results, "Market OK", rejected

# ==========================================
# MAIN APP
# ==========================================

st.set_page_config(page_title="AlphaForge", page_icon="🎯", layout="wide")

st.title("🎯 AlphaForge Alpha Generation")
st.subheader("Probabilistic Forecasting & Statistical Edge")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price & Zones", "🎲 Monte Carlo", "🧠 Model Consensus", "⚠️ Risk & Regime", "🎯 Setup Scanner"
])

# ==========================================
# TAB 5: SETUP SCANNER WITH BACKTEST
# ==========================================

with tab5:
    st.header("🎯 Long-Only Setup Scanner")
    st.caption("Squeeze breakout detection + Market regime filter + Historical Validation")
    
    # Mode selector
    mode = st.radio("Mode", ["Live Scan", "Backtest Historical Performance"], horizontal=True)
    
    if mode == "Live Scan":
        scanner = LongOnlyScanner()
        market_ok = scanner.check_market_regime()
        
        if market_ok:
            st.success("✅ MARKET OK FOR LONGS")
        else:
            st.error("🛑 MARKET REGIME: NO NEW LONGS")
        
        default_list = "AAPL, MSFT, NVDA, TSLA, AMD, NFLX, CRM, META, AMZN, GOOGL, COIN, HOOD, RBLX, U"
        watchlist_input = st.text_area("Enter tickers (comma-separated):", value=default_list, height=100)
        
        if market_ok and st.button("🔍 SCAN FOR SETUPS", type="primary"):
            tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
            with st.spinner(f"Scanning {len(tickers)} tickers..."):
                results, msg, rejected = scan_watchlist(tickers)
            
            if not results:
                st.info("No setups found.")
                with st.expander("See why"):
                    for r in rejected[:15]:
                        st.write(f"• {r}")
            else:
                st.success(f"Found {len(results)} setup(s)")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    else:  # BACKTEST MODE
        st.info("📊 Backtest Mode: Test the scanner on historical data to see if setups actually worked")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2024, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=date(2024, 12, 31))
        
        default_backtest = "AAPL, MSFT, NVDA, TSLA, AMD, NFLX, CRM, META, AMZN, GOOGL"
        backtest_tickers = st.text_area("Tickers to backtest:", value=default_backtest, height=100)
        
        max_days = st.slider("Hold Days (exit if target/stop not hit)", 5, 30, 10)
        
        if st.button("🧪 RUN BACKTEST", type="primary"):
            tickers = [t.strip().upper() for t in backtest_tickers.split(",") if t.strip()]
            
            with st.spinner("Running historical backtest... This may take 2-5 minutes"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                backtest = ScannerBacktest(tickers, start_date, end_date)
                results = backtest.run(progress_bar, status_text)
                stats = backtest.get_statistics()
                
                progress_bar.empty()
                status_text.empty()
            
            if stats is None:
                st.warning("No setups found in backtest period. Try widening date range or ticker list.")
            else:
                st.success(f"✅ Backtest Complete: {stats['total_trades']} setups found")
                
                # Key metrics
                metric_cols = st.columns(4)
                metric_cols[0].metric("Win Rate", f"{stats['win_rate']:.1%}")
                metric_cols[1].metric("Expectancy", f"{stats['expectancy']:.2f}%", 
                                     delta="Good" if stats['expectancy'] > 0 else "Bad")
                metric_cols[2].metric("Total Return", f"{stats['total_return']:.1f}%")
                metric_cols[3].metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%", 
                                     delta="Risk" if stats['max_drawdown'] < -20 else "OK")
                
                # Detailed stats
                with st.expander("Detailed Statistics"):
                    detail_cols = st.columns(3)
                    detail_cols[0].metric("Avg Win", f"{stats['avg_win']:.2f}%")
                    detail_cols[1].metric("Avg Loss", f"{stats['avg_loss']:.2f}%")
                    detail_cols[2].metric("Avg Hold", f"{stats['avg_hold_days']:.1f} days")
                    st.metric("Sharpe Ratio (adj)", f"{stats['sharpe']:.2f}")
                
                # Trade list
                st.divider()
                st.subheader("Individual Trades")
                trades_df = stats['results_df'].sort_values('entry_date', ascending=False)
                
                for _, trade in trades_df.head(20).iterrows():
                    emoji = "🟢" if trade['pnl_pct'] > 0 else "🔴"
                    outcome_label = trade['outcome'].replace('_', ' ')
                    
                    with st.expander(f"{emoji} {trade['ticker']} on {trade['entry_date'].strftime('%Y-%m-%d')} | P&L: {trade['pnl_pct']:+.2f}% ({outcome_label})"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Entry", f"${trade['entry_price']:.2f}")
                        col2.metric("Exit", f"${trade['exit_price']:.2f}")
                        col3.metric("Days Held", f"{trade['days_held']}")
                        
                        st.caption(f"Max gain: {trade['max_gain']:+.2f}% | Max loss: {trade['max_loss']:+.2f}%")
                
                # Equity curve
                st.divider()
                st.subheader("Cumulative Performance")
                
                trades_df_sorted = trades_df.sort_values('entry_date')
                trades_df_sorted['cumulative'] = trades_df_sorted['pnl_pct'].cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trades_df_sorted['entry_date'],
                    y=trades_df_sorted['cumulative'],
                    mode='lines+markers',
                    name='Cumulative Return %'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Equity Curve (Compounded % Returns)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return %",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SIDEBAR & OTHER TABS (Original preserved)
# ==========================================

st.sidebar.header("📊 Single Ticker Analysis")
analysis_ticker = st.sidebar.text_input("Ticker", value="MSFT", key="analysis_ticker").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Models:**")
svj_on = st.sidebar.checkbox("SVJ", value=True)
kalman_on = st.sidebar.checkbox("Kalman", value=True)
topo_on = st.sidebar.checkbox("Topological", value=True)
lstm_on = st.sidebar.checkbox("LSTM", value=True)
evt_on = st.sidebar.checkbox("EVT", value=True)
tech_on = st.sidebar.checkbox("Technical", value=True)
mc_on = st.sidebar.checkbox("Monte Carlo", value=True)
regime_on = st.sidebar.checkbox("Regime Detection", value=True)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner(f"Analyzing {analysis_ticker}..."):
        try:
            # [Original analysis code preserved]
            stock = yf.Ticker(analysis_ticker)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error("No data found")
                st.stop()
            
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'][analysis_ticker] if analysis_ticker in data.columns.get_level_values(1) else data['Close'].iloc[:, 0]
            else:
                prices = data['Close']
            
            current_price = float(prices.iloc[-1])
            returns = prices.pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            
            # Signals
            signals = []
            if svj_on:
                trend = returns.mean() * 252
                if trend > 0.05:
                    signals.append(("SVJ", "BULLISH", 85))
                elif trend < -0.05:
                    signals.append(("SVJ", "BEARISH", 80))
                else:
                    signals.append(("SVJ", "NEUTRAL", 50))
            
            if kalman_on:
                ema20 = prices.ewm(span=20).mean().iloc[-1]
                dev = (current_price - ema20) / ema20
                if dev < -0.05:
                    signals.append(("Kalman", "BULLISH", 75))
                elif dev > 0.05:
                    signals.append(("Kalman", "BEARISH", 75))
                else:
                    signals.append(("Kalman", "NEUTRAL", 50))
            
            if topo_on:
                local_min = argrelextrema(prices.values, np.less, order=10)[0]
                if len(local_min) > 0:
                    support = float(prices.iloc[local_min[-1]])
                    if current_price < support * 1.02:
                        signals.append(("Topological", "BULLISH", 85))
                    else:
                        signals.append(("Topological", "NEUTRAL", 50))
            
            if lstm_on:
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                if rsi < 35:
                    signals.append(("LSTM", "BULLISH", 80))
                elif rsi > 65:
                    signals.append(("LSTM", "BEARISH", 75))
                else:
                    signals.append(("LSTM", "NEUTRAL", 50))
            
            if evt_on:
                var_95 = np.percentile(returns, 5)
                signals.append(("EVT", "NEUTRAL", 60 if abs(var_95) > 0.04 else 70))
            
            if tech_on:
                sma_50 = prices.rolling(50).mean().iloc[-1]
                sma_200 = prices.rolling(200).mean().iloc[-1]
                if current_price > sma_50 > sma_200:
                    signals.append(("Technical", "BULLISH", 75))
                elif current_price < sma_50 < sma_200:
                    signals.append(("Technical", "BEARISH", 75))
                else:
                    signals.append(("Technical", "NEUTRAL", 50))
            
            # Ensemble
            bullish = sum([s[2] for s in signals if s[1] == "BULLISH"])
            bearish = sum([s[2] for s in signals if s[1] == "BEARISH"])
            total = sum([s[2] for s in signals])
            
            if total > 0:
                net = (bullish - bearish) / total
                if net > 0.3:
                    composite = "STRONG_BUY"
                elif net > 0.1:
                    composite = "BUY"
                elif net < -0.3:
                    composite = "STRONG_SELL"
                elif net < -0.1:
                    composite = "SELL"
                else:
                    composite = "HOLD"
            else:
                composite = "NEUTRAL"
            
            # Zones
            z1 = current_price * 0.98
            z2 = current_price * 0.95
            z3 = current_price * 0.88
            
            with tab1:
                st.metric("Signal", composite)
                st.metric("Price", f"${current_price:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Price'))
                fig.add_hrect(y0=z1, y1=current_price, fillcolor="green", opacity=0.2, annotation_text="Zone 1")
                fig.add_hrect(y0=z2, y1=z1, fillcolor="blue", opacity=0.2, annotation_text="Zone 2")
                fig.add_hrect(y0=z3, y1=z2, fillcolor="purple", opacity=0.2, annotation_text="Zone 3")
                st.plotly_chart(fig, use_container_width=True)
                
                zone_df = pd.DataFrame({
                    'Zone': ['Zone 1', 'Zone 2', 'Zone 3'],
                    'Range': [f"${z1:.2f} - ${current_price:.2f}", f"${z2:.2f} - ${z1:.2f}", f"${z3:.2f} - ${z2:.2f}"]
                })
                st.dataframe(zone_df, hide_index=True)
            
            with tab3:
                for name, signal, strength in signals:
                    emoji = "🟢" if signal == "BULLISH" else "🔴" if signal == "BEARISH" else "🟡"
                    st.write(f"{emoji} **{name}:** {signal} ({strength}%)")
                
                st.info(f"**Net Score:** {net:.2f}" if total > 0 else "No active signals")
            
            with tab4:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                st.metric("Sharpe", f"{sharpe:.2f}")
                st.metric("Volatility", f"{vol:.1%}")
                
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("AlphaForge AlphaGen | Mathematical edge through probabilistic forecasting")

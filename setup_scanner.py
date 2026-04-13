import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
            # Get SPY data
            self.spy_data = yf.download("SPY", period="1y", progress=False)
            if self.spy_data.empty:
                return False
                
            # Weekly trend (20 EMA)
            weekly_spy = self.spy_data.resample('W-Fri').last()
            weekly_ema20 = weekly_spy['Close'].ewm(span=20, adjust=False).mean()
            self.weekly_trend = weekly_spy['Close'].iloc[-1] > weekly_ema20.iloc[-1]
            
            # Daily trend (50 SMA)
            sma50 = self.spy_data['Close'].rolling(50).mean()
            self.daily_trend = self.spy_data['Close'].iloc[-1] > sma50.iloc[-1]
            
            # VIX check
            self.vix_data = yf.download("^VIX", period="5d", progress=False)
            if not self.vix_data.empty:
                vix_current = self.vix_data['Close'].iloc[-1]
                self.vix_low = vix_current < 25
            else:
                self.vix_low = True  # Assume OK if data missing
                
            self.market_ok = self.weekly_trend and self.daily_trend and self.vix_low
            return self.market_ok
            
        except Exception as e:
            st.error(f"Market regime check failed: {e}")
            return False
    
    def get_market_status_text(self):
        """Returns formatted market status for display"""
        if not self.market_ok:
            return "🛑 NO NEW LONGS"
        
        status = "✅ MARKET OK FOR LONGS\n\n"
        status += f"Weekly Trend (SPY): {'UP' if self.weekly_trend else 'DOWN'}\n"
        status += f"Daily Trend (SPY): {'UP' if self.daily_trend else 'DOWN'}\n"
        status += f"VIX < 25: {'YES' if self.vix_low else 'NO'}"
        return status
    
    def squeeze_breakout_signal(self, ticker):
        """Volatility compression breakout detection"""
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < 50:
                return False, 0, "No data"
            
            # Calculate Bollinger Bands
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['STD20'] = df['Close'].rolling(20).std()
            df['Upper'] = df['SMA20'] + (2 * df['STD20'])
            df['Lower'] = df['SMA20'] - (2 * df['STD20'])
            df['Bandwidth'] = (df['Upper'] - df['Lower']) / df['SMA20']
            
            # Squeeze detection (narrowest in 120 days)
            df['Squeeze'] = df['Bandwidth'] <= df['Bandwidth'].rolling(120).min()
            
            # Volume average
            df['VolAvg'] = df['Volume'].rolling(20).mean()
            
            # Current and previous bar
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Breakout conditions
            price_breakout = curr['Close'] > prev['Upper']
            volume_confirmed = curr['Volume'] > (prev['VolAvg'] * 1.3)
            was_in_squeeze = prev['Squeeze']
            
            # Calculate squeeze duration (how long it was compressed)
            squeeze_duration = df['Squeeze'].iloc[-20:].sum()
            
            is_setup = price_breakout and volume_confirmed and was_in_squeeze
            
            # Quality score (0-100)
            score = 0
            if is_setup:
                score = 50  # Base for breakout
                if squeeze_duration >= 5:  # Sustained compression
                    score += 20
                if curr['Volume'] > (prev['VolAvg'] * 2):  # Heavy volume
                    score += 20
                if curr['Close'] > curr['SMA20']:  # Above 20dma
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
        """Check if stock is keeping up with SPY"""
        try:
            stock = yf.download(ticker, period="3mo", progress=False)['Close']
            spy = self.spy_data['Close'] if self.spy_data is not None else yf.download("SPY", period="3mo", progress=False)['Close']
            
            if stock.empty or spy.empty or len(stock) < lookback:
                return False, 0
            
            stock_ret = (stock.iloc[-1] / stock.iloc[-lookback] - 1)
            spy_ret = (spy.iloc[-1] / spy.iloc[-lookback] - 1)
            
            # Stock must be within -3% of SPY or beating it
            rs_ok = stock_ret >= (spy_ret - 0.03)
            rs_score = min(100, max(0, (stock_ret - spy_ret + 0.05) * 500))  # Normalize to score
            
            return rs_ok, rs_score
        except:
            return False, 0
    
    def check_events(self, ticker):
        """Basic event risk check"""
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get earnings date
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
    """Main scanning function"""
    scanner = LongOnlyScanner()
    
    # First check market regime
    market_ok = scanner.check_market_regime()
    if not market_ok:
        return [], scanner.get_market_status_text()
    
    results = []
    
    progress_bar = st.progress(0)
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist))
        
        # Squeeze signal
        is_squeeze, squeeze_score, reason = scanner.squeeze_breakout_signal(ticker)
        
        # RS check
        rs_ok, rs_score = scanner.relative_strength_check(ticker)
        
        # Event check
        event_ok, event_msg = scanner.check_events(ticker)
        
        # Combined score
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
    
    # Sort by score descending
    results = sorted(results, key=lambda x: x['Score'], reverse=True)
    return results, scanner.get_market_status_text()

def get_exit_levels(ticker, entry_price):
    """Calculate exit levels for a position"""
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if df.empty:
            return None
            
        # ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Stop loss (2x ATR)
        stop_loss = entry_price - (2 * atr)
        
        # Target (3R = 3x risk)
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

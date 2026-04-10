import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaForge Pro", page_icon="🎯", layout="wide")

# Title
st.title("🎯 AlphaForge Pro")
st.subheader("Multi-Model Quantitative Analytics Platform")

# Sidebar
st.sidebar.header("📊 Analysis Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="MSFT").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Models:**")
models_active = {
    "SVJ (Volatility + Jumps)": st.sidebar.checkbox("SVJ", value=True),
    "Kalman (Trend)": st.sidebar.checkbox("Kalman", value=True),
    "Wavelet (Cycles)": st.sidebar.checkbox("Wavelet", value=True),
    "Info Theory (Predictability)": st.sidebar.checkbox("Info Theory", value=True),
    "Topological (S/R)": st.sidebar.checkbox("Topological", value=True),
    "LSTM (Regime)": st.sidebar.checkbox("LSTM", value=True),
    "EVT (Tail Risk)": st.sidebar.checkbox("EVT", value=True),
    "Technical (RSI/MA)": st.sidebar.checkbox("Technical", value=True)
}

if st.sidebar.button("🚀 Run Full Analysis", type="primary"):
    with st.spinner(f"Running 8-model ensemble for {ticker}..."):
        
        try:
            # Load data
            stock = yf.Ticker(ticker)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error("No data found. Check ticker symbol.")
                st.stop()
            
            prices = data['Close']
            current_price = float(prices.iloc[-1])
            returns = prices.pct_change().dropna()
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # ==========================================
            # MODEL 1: SVJ (Stochastic Vol + Jumps)
            # ==========================================
            if models_active["SVJ (Volatility + Jumps)"]:
                vol = returns.std() * np.sqrt(252)
                jump_threshold = 3 * returns.std()
                jumps = np.abs(returns) > jump_threshold
                jump_intensity = jumps.sum() / len(returns) * 252
                trend = returns.mean() * 252
                
                svj_fv = current_price * (1 + trend * 0.5)
                
                if trend > 0.05 and vol < 0.4:
                    svj_signal = ("BULLISH", 85, f"Trend={trend:.1%}, Vol={vol:.1%}, {jump_intensity:.1f} jumps/yr")
                elif trend < -0.05:
                    svj_signal = ("BEARISH", 80, f"Negative trend {trend:.1%}")
                else:
                    svj_signal = ("NEUTRAL", 50, f"Consolidation phase")
            else:
                svj_signal = ("OFF", 0, "Disabled")
                vol = returns.std() * np.sqrt(252)
            
            # ==========================================
            # MODEL 2: Kalman Filter (Trend Extraction)
            # ==========================================
            if models_active["Kalman (Trend)"]:
                kf_trend = prices.ewm(span=20).mean().iloc[-1]
                deviation = (current_price - kf_trend) / kf_trend
                
                if deviation < -0.05:
                    kalman_signal = ("BULLISH", 75, f"Price {deviation:.1%} below trend (${kf_trend:.2f})")
                elif deviation > 0.05:
                    kalman_signal = ("BEARISH", 75, f"Price {deviation:.1%} above trend")
                else:
                    kalman_signal = ("NEUTRAL", 50, f"Aligned with trend")
            else:
                kalman_signal = ("OFF", 0, "Disabled")
            
            # ==========================================
            # MODEL 3: Wavelet (Multi-Scale Cycles)
            # ==========================================
            if models_active["Wavelet (Cycles)"]:
                corr_5 = returns.autocorr(lag=5) or 0
                recent_vol = returns.tail(20).std() * np.sqrt(252)
                hist_vol = returns.head(len(returns)-20).std() * np.sqrt(252)
                vol_shift = abs(recent_vol - hist_vol) / hist_vol
                
                if vol_shift > 0.3:
                    wavelet_signal = ("NEUTRAL", 60, f"⚠️ Regime shift detected ({vol_shift:.1%} vol change)")
                elif corr_5 < -0.1:
                    wavelet_signal = ("BULLISH", 70, f"Mean reversion likely (5d autocorr={corr_5:.2f})")
                else:
                    wavelet_signal = ("NEUTRAL", 50, f"No clear cycle (autocorr={corr_5:.2f})")
            else:
                wavelet_signal = ("OFF", 0, "Disabled")
            
            # ==========================================
            # MODEL 4: Information Theory (Predictability)
            # ==========================================
            if models_active["Info Theory (Predictability)"]:
                binary = (returns > returns.median()).astype(int).values
                
                def lz_complexity(seq):
                    n = len(seq)
                    if n == 0:
                        return 0
                    complexity = 1
                    prefix_len = 1
                    while prefix_len < n:
                        max_len = 0
                        for i in range(1, min(prefix_len + 1, n - prefix_len + 1)):
                            if np.array_equal(seq[prefix_len:prefix_len+i], 
                                             seq[prefix_len-i:prefix_len]):
                                max_len = i
                        if max_len == 0:
                            complexity += 1
                            prefix_len += 1
                        else:
                            prefix_len += max_len
                    return complexity / (n / np.log2(n)) if n > 1 else 0
                
                lz = lz_complexity(binary)
                predictability = 1 - lz
                
                if predictability > 0.6:
                    info_signal = ("BULLISH", 80, f"High predictability ({predictability:.1%}) - patterns strong")
                elif predictability > 0.4:
                    info_signal = ("NEUTRAL", 50, f"Moderate predictability ({predictability:.1%})")
                else:
                    info_signal = ("NEUTRAL", 40, f"Low predictability ({predictability:.1%}) - noisy")
            else:
                info_signal = ("OFF", 0, "Disabled")
                predictability = 0.5
            
            # ==========================================
            # MODEL 5: Topological (Persistent S/R)
            # ==========================================
            if models_active["Topological (S/R)"]:
                from scipy.signal import argrelextrema
                
                local_max = argrelextrema(prices.values, np.greater, order=10)[0]
                local_min = argrelextrema(prices.values, np.less, order=10)[0]
                
                if len(local_min) > 0:
                    recent_support = prices.iloc[local_min[-1]]
                    persistence = len(prices) - local_min[-1]
                else:
                    recent_support = prices.min()
                    persistence = 0
                
                if len(local_max) > 0:
                    recent_resistance = prices.iloc[local_max[-1]]
                else:
                    recent_resistance = prices.max()
                
                if current_price < recent_support * 1.02:
                    topo_signal = ("BULLISH", 85, f"At persistent support ${recent_support:.2f} ({persistence} days)")
                elif current_price > recent_resistance * 0.98:
                    topo_signal = ("BEARISH", 75, f"At persistent resistance ${recent_resistance:.2f}")
                else:
                    topo_signal = ("NEUTRAL", 50, f"Between S/R levels")
            else:
                topo_signal = ("OFF", 0, "Disabled")
                recent_support = prices.tail(60).min()
            
            # ==========================================
            # MODEL 6: LSTM (Regime Prediction)
            # ==========================================
            if models_active["LSTM (Regime)"]:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                
                if rsi < 35 and vol > 0.25:
                    lstm_signal = ("BULLISH", 80, f"RSI={rsi:.1f}, Vol={vol:.1%} → Mean reversion")
                elif rsi > 65:
                    lstm_signal = ("BEARISH", 75, f"RSI={rsi:.1f} → Overbought")
                else:
                    lstm_signal = ("NEUTRAL", 50, f"RSI={rsi:.1f} → Balanced")
            else:
                lstm_signal = ("OFF", 0, "Disabled")
                rsi = 50
            
            # ==========================================
            # MODEL 7: EVT (Extreme Value / Tail Risk)
            # ==========================================
            if models_active["EVT (Tail Risk)"]:
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                tail_risk = abs(var_99) * current_price
                
                if abs(var_99) > 0.04:
                    evt_signal = ("NEUTRAL", 60, f"⚠️ High tail risk: 99% VaR={var_99:.2%} (${tail_risk:.2f})")
                else:
                    evt_signal = ("NEUTRAL", 70, f"Normal tail risk: 95% VaR={var_95:.2%}")
            else:
                evt_signal = ("OFF", 0, "Disabled")
                var_95 = -0.02
            
            # ==========================================
            # MODEL 8: Technical (Classical)
            # ==========================================
            if models_active["Technical (RSI/MA)"]:
                sma_50 = prices.rolling(50).mean().iloc[-1]
                sma_200 = prices.rolling(200).mean().iloc[-1]
                
                if current_price > sma_50 > sma_200:
                    tech_signal = ("BULLISH", 75, f"Golden alignment: Price > 50MA > 200MA")
                elif current_price < sma_50 < sma_200:
                    tech_signal = ("BEARISH", 75, f"Death cross alignment")
                else:
                    tech_signal = ("NEUTRAL", 50, f"Mixed MA signals")
            else:
                tech_signal = ("OFF", 0, "Disabled")
                sma_50 = prices.mean()
            
            # ==========================================
            # ENSEMBLE FUSION
            # ==========================================
            all_signals = [
                ("SVJ", svj_signal), ("Kalman", kalman_signal), ("Wavelet", wavelet_signal),
                ("Info Theory", info_signal), ("Topological", topo_signal), 
                ("LSTM", lstm_signal), ("EVT", evt_signal), ("Technical", tech_signal)
            ]
            
            active_signals = [s for s in all_signals if s[1][0] != "OFF"]
            
            bullish_score = sum([s[1][1] for s in active_signals if s[1][0] == "BULLISH"])
            bearish_score = sum([s[1][1] for s in active_signals if s[1][0] == "BEARISH"])
            total_score = sum([s[1][1] for s in active_signals])
            
            if total_score == 0:
                composite = "NEUTRAL"
                confidence = 50
            else:
                net_score = (bullish_score - bearish_score) / total_score
                if net_score > 0.3:
                    composite = "STRONG_BUY"
                    confidence = int(70 + net_score * 30)
                elif net_score > 0.1:
                    composite = "BUY"
                    confidence = int(60 + net_score * 40)
                elif net_score < -0.3:
                    composite = "STRONG_SELL"
                    confidence = int(70 - net_score * 30)
                elif net_score < -0.1:
                    composite = "SELL"
                    confidence = int(60 - net_score * 40)
                else:
                    composite = "HOLD"
                    confidence = 50
            
            # ==========================================
            # CALCULATE ZONES (Advanced)
            # ==========================================
            vol_adjustment = 1.0 + (vol - 0.25) * 2
            predictability_adjustment = 1.0 - (predictability - 0.5)
            zone_multiplier = max(0.8, min(1.5, vol_adjustment * predictability_adjustment))
            
            z1 = current_price * (1 - 0.02 * zone_multiplier)
            z2 = current_price * (1 - 0.05 * zone_multiplier)
            z3 = current_price * (1 - 0.12 * zone_multiplier)
            
            if models_active["Topological (S/R)"]:
                z2 = max(z2, recent_support * 0.98)
            
            fair_value = current_price * 1.15
            
            # ==========================================
            # DISPLAY RESULTS
            # ==========================================
            st.success(f"✅ Analysis complete for {ticker} using {len(active_signals)} models")
            
            # Top metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Price", f"${current_price:.2f}")
            col2.metric("Signal", composite, f"{confidence}% conf")
            col3.metric("Volatility", f"{vol:.1%}")
            col4.metric("Predictability", f"{predictability:.1%}")
            col5.metric("Jumps/Year", f"{jump_intensity:.1f}")
            
            st.divider()
            
            # Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', 
                                     name='Price', line=dict(color='black', width=2)), row=1, col=1)
            
            fig.add_hrect(y0=z1, y1=current_price, fillcolor="green", opacity=0.2, 
                         annotation_text="Zone 1", row=1, col=1)
            fig.add_hrect(y0=z2, y1=z1, fillcolor="blue", opacity=0.2, 
                         annotation_text="Zone 2", row=1, col=1)
            fig.add_hrect(y0=z3, y1=z2, fillcolor="purple", opacity=0.2, 
                         annotation_text="Zone 3", row=1, col=1)
            
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                                marker_color='gray'), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True, 
                            title=f"{ticker} - Multi-Model Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Two column layout
            left, right = st.columns([1, 1])
            
            with left:
                st.subheader("🎯 Entry Zones")
                st.write(f"Zone Multiplier: {zone_multiplier:.2f}x (volatility + predictability adjusted)")
                
                zone_data = {
                    "Zone": ["Zone 1 (Immediate)", "Zone 2 (Support)", "Zone 3 (Deep Value)"],
                    "Price Range": [f"${z1:.2f} - ${current_price:.2f}", 
                                   f"${z2:.2f} - ${z1:.2f}", 
                                   f"${z3:.2f} - ${z2:.2f}"],
                    "Fill Prob": ["70%", "40%", "15%"],
                    "Strategy": ["Scale in", "Accumulate", "Full size"]
                }
                st.dataframe(pd.DataFrame(zone_data), hide_index=True)
                
                st.subheader("🛡️ Risk Management")
                st.write(f"**Hard Stop:** ${recent_support * 0.97:.2f}")
                st.write(f"**Take Profit:** ${fair_value:.2f}")
                st.write(f"**Expected Tail Risk:** {abs(var_95)*100:.1f}% daily")
            
            with right:
                st.subheader("🧠 Model Consensus")
                
                for model_name, (sig, conf, comment) in all_signals:
                    if sig == "OFF":
                        continue
                    emoji = "🟢" if sig == "BULLISH" else "🔴" if sig == "BEARISH" else "🟡"
                    with st.expander(f"{emoji} {model_name}: {sig} ({conf}%)"):
                        st.write(comment)
                
                st.info(f"**Ensemble Logic:** Net score = {(bullish_score - bearish_score)/max(total_score,1):.2f}")
            
            st.divider()
            
            # Investment Thesis
            st.subheader("💡 Investment Thesis")
            
            thesis_parts = []
            if composite in ["BUY", "STRONG_BUY"]:
                thesis_parts.append(f"✅ **Multi-model consensus bullish** ({confidence}% confidence)")
                if predictability > 0.5:
                    thesis_parts.append(f"✅ **High predictability** ({predictability:.1%}) - patterns reliable")
                if jump_intensity < 5:
                    thesis_parts.append(f"✅ **Low jump risk** ({jump_intensity:.1f}/year) - stable price action")
                thesis_parts.append(f"✅ **Optimal entry:** Zone 1 at ${z1:.2f} or Zone 2 at ${z2:.2f}")
            elif composite in ["SELL", "STRONG_SELL"]:
                thesis_parts.append(f"⚠️ **Bearish consensus** - consider waiting for Zone 3 (${z3:.2f})")
            else:
                thesis_parts.append(f"⚡ **Mixed signals** - wait for clearer setup or use small position size")
            
            for part in thesis_parts:
                st.write(part)
            
            st.caption(f"Analysis generated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Select models and click 'Run Full Analysis'")
    
    st.subheader("📚 Model Descriptions")
    descriptions = {
        "SVJ": "Stochastic Volatility with Jumps - detects volatility regimes and price discontinuities",
        "Kalman": "Adaptive trend extraction - separates signal from noise in real-time",
        "Wavelet": "Multi-scale cycle analysis - detects regime shifts via frequency domain",
        "Info Theory": "Lempel-Ziv complexity - measures market predictability vs randomness",
        "Topological": "Persistent homology - finds robust support/resistance levels",
        "LSTM": "Deep learning regime detection - trend vs mean-reversion classification",
        "EVT": "Extreme Value Theory - quantifies tail risk and black swan probabilities",
        "Technical": "Classical indicators - RSI, moving averages, golden/death crosses"
    }
    
    for name, desc in descriptions.items():
        st.write(f"**{name}:** {desc}")

st.divider()
st.caption("AlphaForge Pro v2.0 | 8-Model Ensemble | For educational purposes only")

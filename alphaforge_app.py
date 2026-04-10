import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AlphaForge", page_icon="🎯", layout="wide")

st.title("🎯 AlphaForge")
st.subheader("Stock Analysis Dashboard")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"], index=1)

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner(f"Loading {ticker}..."):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error("No data found. Check ticker symbol.")
                st.stop()
            
            current_price = float(data['Close'].iloc[-1])
            returns = data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # MAs
            sma_50 = float(data['Close'].rolling(50).mean().iloc[-1])
            sma_200 = float(data['Close'].rolling(200).mean().iloc[-1])
            support = float(data['Low'].tail(60).min())
            
            # Signal
            if rsi < 30 and current_price > sma_50:
                composite = "BUY"
                confidence = 75
            elif rsi > 70:
                composite = "SELL"
                confidence = 75
            else:
                composite = "HOLD"
                confidence = 50
            
            # Zones
            if volatility < 0.25:
                z1, z2, z3 = current_price * 0.98, current_price * 0.95, current_price * 0.88
            elif volatility < 0.45:
                z1, z2, z3 = current_price * 0.95, current_price * 0.88, current_price * 0.75
            else:
                z1, z2, z3 = current_price * 0.93, current_price * 0.85, current_price * 0.70
            
            # Display
            st.success(f"Analysis complete for {ticker}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${current_price:.2f}")
            c2.metric("Signal", composite)
            c3.metric("Confidence", f"{confidence}%")
            c4.metric("Volatility", f"{volatility:.1%}")
            
            st.divider()
            
            # Simple chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
            fig.add_hrect(y0=z1, y1=current_price, fillcolor="green", opacity=0.2)
            fig.add_hrect(y0=z2, y1=z1, fillcolor="blue", opacity=0.2)
            fig.add_hrect(y0=z3, y1=z2, fillcolor="purple", opacity=0.2)
            fig.update_layout(title=f"{ticker} with Entry Zones", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Entry Zones")
            z1_col, z2_col, z3_col = st.columns(3)
            z1_col.metric("Zone 1 (Immediate)", f"${z1:.2f}")
            z2_col.metric("Zone 2 (Support)", f"${z2:.2f}")
            z3_col.metric("Zone 3 (Deep Value)", f"${z3:.2f}")
            
            st.subheader("Risk Management")
            st.write(f"**Stop Loss:** ${support * 0.98:.2f}")
            st.write(f"**Take Profit:** ${current_price * 1.15:.2f}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Try: AAPL, MSFT, NVDA, TSLA, AMZN")
else:
    st.info("👈 Enter a ticker and click 'Run Analysis'")
    st.write("**Examples:** AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META")

st.divider()
st.caption("AlphaForge v2.0 | Educational purposes only")

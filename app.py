import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# App title
st.title("📈 Cash Flow Forecast Dashboard")

# Load forecast data
@st.cache_data
def load_forecast():
    if os.path.exists("forecast.csv"):
        df = pd.read_csv("forecast.csv")
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    return pd.DataFrame()

forecast_df = load_forecast()

st.subheader("📊 Forecasted Daily Net Cash Flow")

if not forecast_df.empty:
    # --- Line Chart with Confidence Interval ---
    st.markdown("### 🔮 Forecast with Confidence Intervals")
    st.markdown("This chart shows the forecasted daily net cash flow along with upper and lower confidence intervals predicted by Facebook Prophet.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(
        title="📉 Daily Net Cash Flow Forecast",
        xaxis_title="Date",
        yaxis_title="Net Cash Flow",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Weekly Aggregation Chart ---
    st.markdown("### 📆 Weekly Aggregated Forecast")
    st.markdown("This chart aggregates daily forecasts into weekly values to help identify broader trends.")
    weekly_df = forecast_df[['ds', 'yhat']].copy()
    weekly_df['week'] = weekly_df['ds'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_summary = weekly_df.groupby('week')['yhat'].sum().reset_index()
    fig2 = px.bar(weekly_summary, x='week', y='yhat', title='🗓️ Weekly Forecasted Net Cash Flow', labels={'yhat': 'Total Net Cash Flow'})
    st.plotly_chart(fig2, use_container_width=True)

    # --- Cumulative Cash Flow Chart ---
    st.markdown("### 📈 Cumulative Forecast")
    st.markdown("This chart shows the cumulative sum of forecasted cash flow, useful for estimating total liquidity projection.")
    forecast_df['cumulative_cashflow'] = forecast_df['yhat'].cumsum()
    fig3 = px.line(forecast_df, x='ds', y='cumulative_cashflow', title='💰 Cumulative Net Cash Flow Forecast', labels={'ds': 'Date', 'cumulative_cashflow': 'Cumulative Cash Flow'})
    st.plotly_chart(fig3, use_container_width=True)

    # --- Moving Average Forecast ---
    st.markdown("### 📊 7-Day Moving Average Forecast")
    st.markdown("This smoothed view of the forecasted cash flow helps identify longer-term patterns, filtering out daily volatility.")
    forecast_df['moving_avg'] = forecast_df['yhat'].rolling(window=7).mean()
    fig4 = px.line(forecast_df, x='ds', y='moving_avg', title='📉 7-Day Moving Average of Forecasted Cash Flow', labels={'ds': 'Date', 'moving_avg': '7-Day Moving Avg'})
    st.plotly_chart(fig4, use_container_width=True)

    # --- Daily Change Chart ---
    st.markdown("### 📈 Daily Change in Forecasted Cash Flow")
    st.markdown("This chart shows the daily difference in forecasted net cash flow to highlight volatility and trend shifts.")
    forecast_df['daily_change'] = forecast_df['yhat'].diff()
    fig5 = px.bar(forecast_df, x='ds', y='daily_change', title='📊 Daily Change in Forecasted Cash Flow', labels={'ds': 'Date', 'daily_change': 'Change'})
    st.plotly_chart(fig5, use_container_width=True)

else:
    st.warning("No forecast.csv file found. Please generate it first.")
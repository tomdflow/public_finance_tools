# Secondary packages
import numpy as np
import pandas as pd

# Data packages
import yfinance as yf

import streamlit as st
from finance import Finance_Tools

# Package settings
pd.options.plotting.backend = "plotly" # Set pandas plotting package to plotly


st.set_page_config(
   page_title="Finance Tools view",
   layout="wide",
)#initial_sidebar_state="expanded", page_icon="🧊"

st.title('Finance functions display')

col1, col2 = st.columns(2)
with col1:
   asset = st.text_input(label="Stock", value='AAPL')
with col2:
   start_time = st.date_input(label='Start date', value=pd.to_datetime('2010-01-01'))

data = yf.download(asset, start_time)#['Close'] , interval='1h'
 
st.dataframe(data)  # Display the data in a table

ft = Finance_Tools(data, asset)

st.header(asset)

#st.write(lol.data_analytics())

#st.write('Candlestick chart')
st.plotly_chart(ft.candlestick(show=False))
st.write('252 day rolling volatility')
st.plotly_chart(ft.volatility().plot())

col3, col4 = st.columns(2)

with col3:
   st.write('Returns artihmetic')
   st.plotly_chart(ft.returns(return_pd=False).ret_hist(show=False))

   st.write('Returns log')
   st.plotly_chart(ft.log_returns(return_pd=False).ret_hist(show=False))


   intraday_ret_sig, overnight_ret_sig, weekend_ret_sig = ft.test_all_returns()

   st.write(f'Intraday Returns: significant={intraday_ret_sig}')
   st.plotly_chart(ft.intraday_returns(return_pd=False).ret_hist(show=False))

   st.write(f'Overnight Returns: significant={overnight_ret_sig}')
   st.plotly_chart(ft.overnight_returns(return_pd=False).ret_hist(show=False))

   st.write(f'Weekend Returns: significant={weekend_ret_sig}')
   st.plotly_chart(ft.simple_weekend_returns(return_pd=False).ret_hist(show=False))

   st.write('Significance of different weekday returns vs the overall returns')
   weekday_fig, weekday_df = ft.weekday_returns()
   st.plotly_chart(weekday_fig)
   st.write(weekday_df)

   st.write('Significance of diffferent months')
   seas_months = ft.month_returns()
   st.write(seas_months)

   st.write('High low range')
   st.plotly_chart(ft.high_low_range().plot())

with col4:
   st.write('Comparison of return distribution with benchmark')
   st.plotly_chart(ft.ret_hist_benchmark(show=False, benchmark='^SPX'))

   st.write("Performance Comparison to other ticker")
   comp_asset = st.text_input(label="Stock", value='MSFT') # get the ticker from input
   bench_data = yf.download(comp_asset, start_time)['Close'] # download ticker data
   st.plotly_chart(ft.returns(return_pd=False).ret_hist_comparison(bench_data, other_name=comp_asset, show=False)) # add histogram comparison
   st.plotly_chart(ft.cumulative_comparison(bench_data, other_name=comp_asset).plot()) # cumulative performance comparison chart

   # Historic correl chart
   st.write("Rolling 252-day correlation")
   st.plotly_chart(ft.correlation(bench_data).plot())





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
#data.columns = data.columns.str.lower()
lol = Finance_Tools(data, asset)


st.header(asset)

#st.write(lol.data_analytics())

st.write('Candlestick chart')
st.plotly_chart(lol.candlestick(show=False))

col3, col4 = st.columns(2)

with col3:
   st.write('Returns artihmetic')
   st.plotly_chart(lol.returns(return_pd=False).ret_hist(show=False))

   st.write('Returns log')
   st.plotly_chart(lol.log_returns(return_pd=False).ret_hist(show=False))

   st.write('Intraday Returns')
   st.plotly_chart(lol.intraday_returns(return_pd=False).ret_hist(show=False))

   st.write('Overnight Returns')
   st.plotly_chart(lol.overnight_returns(return_pd=False).ret_hist(show=False))

with col4:
   st.write('Weekend Returns')
   #st.plotly_chart(lol.weekend_return()['weekend_return'].plot(kind='hist'))
   st.plotly_chart(lol.simple_weekend_returns().plot(kind='hist'))
   st.write(lol.weekday_returns())

   st.write('High low range')
   st.plotly_chart(lol.high_low_range().plot())

   st.write('Returns significance')
   #st.write(lol.test_all_returns())


from pprint import pprint

# Secondary packages
import numpy as np
import pandas as pd
import scipy.stats as stats

# Plotting packages
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

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

asset = st.text_input(label="Stock", value='AAPL')

data = yf.download(asset, '2010-01-01')#['Close'] , interval='1h'
#data.columns = data.columns.str.lower()
lol = Finance_Tools(data, asset)


st.header(asset)

#st.write(lol.data_analytics())

st.write('Returns artihmetic')
st.plotly_chart(lol.returns().plot(kind='hist'))

st.write('Returns log')
st.plotly_chart(lol.log_returns().plot(kind='hist'))

st.write('Intraday Returns')
st.plotly_chart(lol.intraday_returns().plot(kind='hist'))

st.write('Overnight Returns')
st.plotly_chart(lol.overnight_returns().plot(kind='hist'))

st.write('Weekend Returns')
#st.plotly_chart(lol.weekend_return()['weekend_return'].plot(kind='hist'))
st.plotly_chart(lol.simple_weekend_returns().plot(kind='hist'))

st.write('High low range')
st.plotly_chart(lol.high_low_range().plot())

st.write('Returns significance')
#st.write(lol.test_all_returns())


st.write('Candlestick chart')
st.plotly_chart(lol.candlestick(show=False))
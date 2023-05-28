import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import datetime

st.set_page_config(page_title = "Support and resistance levels",
                   page_icon = ':📈:',
                   layout = 'wide')

st.title('📈 Technical analysis 📉')
st.header('Find support and resistance levels for :blue[price action] analysis!')
st.markdown('''<span style="font-size:18px; font-weight:500;">
This demo includes an implemented <em>Agglomerative Clustering</em>
algorithm that can assist you in automatically detecting 
potential support and resistance levels in financial markets.
</span>''', unsafe_allow_html = True)
st.markdown('##')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url__money = "https://assets1.lottiefiles.com/packages/lf20_06a6pf9i.json"
lottie_money = load_lottieurl(lottie_url__money)

st.sidebar.header('Please choose parameters: ')

ticker = st.text_input('''Select stock to analyse: 
(Make sure the ticker you search for is supported 
by _Yahoo! Finance_).''', 'BNB-USD')

interval = st.sidebar.selectbox(
    'Select the time interval',
    ('1d', '5d', '1wk', '1mo', '3mo'))

timedelta = {'1d': 1, '5d': 5, '1wk' : 7, '1mo' : 30, '3mo' : 90}

start = st.sidebar.date_input(
    "Select the beginning date",
    datetime.date(2022, 1, 1))

end = st.sidebar.date_input(
    "Select the ending date",
    datetime.date(2023, 1, 1))

if end < start + datetime.timedelta(timedelta[interval]):
    raise ValueError('End date cannot be earlier than start date.')

df = yf.download(ticker, start = start, end = end, interval = interval)
df.index = pd.to_datetime(df.index).strftime("%d-%m-%Y")
df = df.drop(columns = ["Adj Close"])

num_clusters = st.sidebar.slider(
    'Select the number of clusters (affects number of levels you will get)',
    1, 7, 3)

rolling_wave_length = st.sidebar.slider(
    '''Select the length of rolling wave 
    (select more the more long-term biased you are)''',
    1, len(df)//5, 1)

left_column, right_column = st.columns(2)

left_column.markdown('<span style="font-size:20px; font-weight:600; letter-spacing:2px;">Preview data:</span>',
            unsafe_allow_html = True)
left_column.dataframe(df, height = 400, use_container_width=True)

with right_column:
    st_lottie(lottie_money, key="money", quality = 'high', height = 400)

#creating function
def calculate_support_resistance(df, rolling_wave_length, num_clusters):
    date = df.index
    df.reset_index(inplace=True)
    
    max_waves_temp = df.High.rolling(rolling_wave_length).max().rename('waves')
    min_waves_temp = df.Low.rolling(rolling_wave_length).min().rename('waves')
   
    max_waves = pd.concat([max_waves_temp, pd.Series(np.zeros(len(max_waves_temp)) + 1)], axis=1)
    min_waves = pd.concat([min_waves_temp, pd.Series(np.zeros(len(min_waves_temp)) + -1)], axis=1)
    max_waves.drop_duplicates('waves', inplace=True)
    min_waves.drop_duplicates('waves', inplace=True)
    
    waves = pd.concat([max_waves, min_waves]).sort_index()
    waves = waves[waves[0] != waves[0].shift()].dropna()
    
    x = np.concatenate((waves.waves.values.reshape(-1, 1),
                        (np.zeros(len(waves)) + 1).reshape(-1, 1)), axis=1)
    
    cluster = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    cluster.fit_predict(x)
    waves['clusters'] = cluster.labels_
    waves2 = waves.loc[waves.groupby('clusters')['waves'].idxmax()]
    df.index = date
    waves2.waves.drop_duplicates(keep='first', inplace=True)
    
    return waves2.reset_index().waves
support_resistance_levels = calculate_support_resistance(df, rolling_wave_length, num_clusters)

#creating a plot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.06, subplot_titles=('OHLC', 'Volume'), 
               row_width=[0.3, 0.7])

fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = "Market data"), row = 1, col = 1)

i = 0
for level in support_resistance_levels.to_list():
    fig.add_hline(y=level, line_width=1, 
                  line_dash="dash", row=1, col=1,
                  line_color="snow")
    i += 1

fig.update_xaxes(
    rangeslider_visible = False)

colors = []

for i in range(len(df.Close)):
    if i != 0:
        if df.Close[i] > df.Close[i-1]:
            colors.append('lightgreen')
        else:
            colors.append('lightcoral')
    else:
        colors.append('lightcoral')

fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, 
                     marker=dict(color=colors)), row=2, col=1)

fig.update_traces(name= 'Volume', selector=dict(type='bar'))

text = f'{ticker} Chart'

fig.update_layout(
    title=go.layout.Title(
        text=text,
        xref="paper",
        x=0))

#show chart
st.plotly_chart(fig, use_container_width=True)

st.markdown("""<span style="font-size:13px; font-weight:400;">
Disclaimer: It's important to note that while this demonstration provides a useful approach to 
identifying support and resistance levels in financial markets, 
it is not intended to be taken as financial advice. 
Trading decisions should be made based on careful analysis of multiple factors, 
including market conditions, 
risk tolerance, 
and individual financial goals.
</span>""", unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown('''
            <div style="position: relative; bottom: 0px; width: 100%;">
                <span class="e1_33">
                    <p style="text-align:center">
                        Designed with ❤️ by 
                        <a href="https://www.linkedin.com/in/amelia-doli%C5%84ska-55613a270/">
                        <em>
                        Amelia Dolińska
                        </em>
                        </a> 
                    </p>
                </span>
            </div>
            ''',
            unsafe_allow_html=True)
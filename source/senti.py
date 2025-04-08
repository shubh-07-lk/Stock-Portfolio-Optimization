import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import yfinance as yf  # type: ignore
from keras.models import load_model  # type: ignore
import streamlit as st  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import requests  # For fetching news articles

# Constants
START_DATE = '2010-01-01'
END_DATE = '2025-04-08'
NEWS_API_KEY = '8d61da2584bf42b8a4b6a9f58afb8660'  # Replace with your actual News API key

# Streamlit title
st.title('Stock Portfolio Optimization')

# Sidebar CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        font-size: 18px;
    }
    .sidebar .sidebar-content .stText, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header('User Input')
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
train_size = st.sidebar.slider('Training Data Size (%)', 50, 90, 70)

# Downloading the stock data
df = yf.download(user_input, start=START_DATE, end=END_DATE)

# Displaying data statistics
st.subheader('Data from 2010 - 2025')
st.write(df.describe())

# Fetching current stock data
current_stock = yf.Ticker(user_input)
today_data = current_stock.history(period='1d')

# Checking if data is available
if not today_data.empty:
    current_price = today_data['Close'][0]
    opening_price = today_data['Open'][0]
    price_change_percentage = ((current_price - opening_price) / opening_price) * 100

    # Displaying current price, opening price, and gain/loss percentage
    st.sidebar.subheader('Current Stock Information')
    st.sidebar.write(f"**Current Price:** ${current_price:.2f}")
    st.sidebar.write(f"**Today's Opening Price:** ${opening_price:.2f}")
    st.sidebar.write(f"**Today's Change:** {price_change_percentage:.2f}% {'⬆️' if price_change_percentage > 0 else '⬇️'}")
else:
    st.sidebar.write("No current data available for this stock.")

# Visualization function with black background
def plot_with_background(x, y, title, xlabel, ylabel, color='blue'):
    fig = plt.figure(figsize=(12, 6), facecolor='black')
    plt.plot(x, y, label='Closing Price', color=color)
    plt.title(title, color='white')
    plt.xlabel(xlabel, color='white')
    plt.ylabel(ylabel, color='white')
    plt.legend(facecolor='black', edgecolor='white', fontsize=10)
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('black')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(fig)

# Plotting closing price
plot_with_background(df.index, df['Close'], f'{user_input} Closing Price', 'Date', 'Price')

# Moving Average Calculations and Plotting
ma100 = df['Close'].rolling(100).mean()
plot_with_background(df.index, ma100, f'{user_input} 100-Day Moving Average', 'Date', 'Price', color='red')

ma200 = df['Close'].rolling(200).mean()
plot_with_background(df.index, ma200, f'{user_input} 200-Day Moving Average', 'Date', 'Price', color='green')

# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * (train_size / 100))])
data_testing = pd.DataFrame(df['Close'][int(len(df) * (train_size / 100)):])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the model
model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Rescale the predictions
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plotting predicted vs original prices
st.subheader('Predicted vs Original Prices')
fig2 = plt.figure(figsize=(12, 6), facecolor='black')
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.title('Predicted vs Original Prices', color='white')
plt.xlabel('Time', color='white')
plt.ylabel('Price', color='white')
plt.legend(facecolor='black', edgecolor='white', fontsize=10)
plt.gca().set_facecolor('black')
plt.xticks(color='white')
plt.yticks(color='white')
st.pyplot(fig2)

# News Sentiment Analysis
def fetch_news_sentiment(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for article in articles:
        description = article['description']
        if description:
            score = analyzer.polarity_scores(description)
            sentiment_scores.append(score['compound'])
    
    return np.mean(sentiment_scores) if sentiment_scores else 0.0

# Sidebar for news sentiment analysis
if st.sidebar.button('Analyze News Sentiment'):
    try:
        sentiment_score = fetch_news_sentiment(user_input)
        st.sidebar.subheader('Sentiment Score from News')
        st.sidebar.write(f'The average sentiment score for {user_input} based on news is: {sentiment_score:.2f}')

        # Analysis interpretation
        st.sidebar.subheader('Sentiment Analysis Interpretation')
        if sentiment_score > 0.5:
            st.sidebar.write("The sentiment is strongly positive, indicating favorable news coverage. This could imply optimism around the stock, which might positively influence buying decisions.")
        elif 0 < sentiment_score <= 0.5:
            st.sidebar.write("The sentiment is moderately positive, suggesting some favorable coverage. While there's optimism, some neutral or mixed opinions may also exist.")
        elif -0.5 < sentiment_score <= 0:
            st.sidebar.write("The sentiment is moderately negative, with slightly unfavorable or mixed news. This may reflect mild concerns that could affect investor confidence.")
        else:
            st.sidebar.write("The sentiment is strongly negative, indicating predominantly unfavorable news. This could indicate negative sentiment, potentially leading to caution among investors.")

    except Exception as e:
        st.sidebar.write('Error fetching news or analyzing sentiment:', e)

# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.markdown('<span style="color:black;">This application allows you to predict stock trends based on historical data and analyze sentiment from news sources. You can adjust the training data size for analysis.</span>', unsafe_allow_html=True)

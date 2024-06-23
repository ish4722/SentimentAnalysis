# SentimentAnalysis

Stock Sentiment Analysis Using Machine Learning Techniques  
Stock sentiment analysis using machine learning techniques involves analyzing textual data  to predict stock price movements based on the sentiments expressed in the text. 
Key Steps in Stock Sentiment Analysis: 
1. Data Collection: Gather textual data from various sources such as news articles,  social media posts, financial reports, and forums.We gathered the information of the  company from new articles of Business Insider website.And the information about  the stocks values from taken from Yahoo Finance using its API Yfinance. 
2. Data Preprocessing:Clean and preprocess the text data to remove noise (e.g., HTML  tags, special characters, stop words).Tokenize the text into words or phrases.We used  the lemmatization technique to tokenise the news. 
3. Sentiment Labeling:Label the textual data with sentiment scores(e.g., positive,  negative, neutral).Use techniques like rule-based approaches, lexicon-based methods,  or pre-trained sentiment analysis models.In our project we used  
SentimentIntensityAnalyzer. 
4. Feature Extraction:Convert text data into numerical features using methods like:Bag  of Words (BoW),Term Frequency-Inverse Document Frequency (TF-IDF).We used  the TfidfVectorizer for this step. 
5. Labeling of data:Data was labelled as 0 or 1 according to the closing prices initially. 
6. Model Training:Train machine learning models on the sentiment labelled columns  and on extracted features to predict 0 or 1 on closing data.We used pre defined model  dataintensityAnalyser,SVM,Random Forest then find the max accuracy and used its  prediction for further use. 
7. Trading Strategy Development:Develop trading strategies based on the sentiment  analysis results.Buy stocks with positive label and sell stocks with zero label. 
Project by:ISHAN:23112043 1
Ishan Tandon 23112043 2nd Year(newly sophomore) 
8. Backtesting and Evaluation:Backtest the trading strategy on historical data to  evaluate its performance.Metrics for evaluation include Sharpe ratio, maximum  drawdowns, number of trades executed, and win ratio. 
9. Deployment:To find the final portfolio if initial portfolio was fixed at $100000 using  the trading strategy followed by us.
    
 DETAILED EXPLAINATION OF EACH STEP


 1.The script starts by scraping news articles from the Business Insider website, collecting  headlines and dates of the articles.Here, the script requests webpages, parses the HTML  content, and extracts news headlines and dates. The data is stored in a DataFrame. 
 
2.Next, the collected dates are converted from relative terms (like "h" for hours) to integers.  Dates older than 1500 days are removed.The date column is cleaned by replacing 'h' with '0',  removing 'd' and ',' characters, and converting the dates to integers. Dates greater than 1500  are filtered out. 

3.Relative days are converted to actual dates.A function to convert relative days to dd-mm-yy  format. 

4.The news articles with the same date are merged into a single entry.

5.Historical stock data for Google (GOOGL),Amazon(AMZN),Meta(META) is fetched using  the yfinance library.This script fetches five years of historical stock data and converts the  index to dates. 

6.The news data is merged with the stock price data on the date.For the dates on which stock  values were not available i.e. on non trading days :the news of these days were added to the  previous trading days to avoid the problem caused due to NaN. 

7.A binary target variable is created based on the next day's closing price i.e. we added a  column named price_up which has 1 if next day closing price is higher than today and 0 if  lower.

8.Sentiment analysis is performed using TextBlob and VADER sentiment analyzers.It  calculates sentiment scores (compound,negative,positive,neutral)for each news article and  stores them in the DataFrame. 

9.The news articles are preprocessed, and TF-IDF features are extracted.The news articles are  tokenized, stop words are removed, and words are lemmatized. TF-IDF features are then  extracted and added to the DataFrame. 


• TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is a widely used  technique in natural language processing (NLP) for converting text data into numerical  features.

• TF-IDF assigns higher weights to words that are frequent in a particular document but not  across all documents. This helps in emphasizing the most relevant words for each  document, which can be more informative than simply counting word frequencies (as in  Bag-of-Words). 

• It reduces noise by providing lesser weight to common but less informative words like  “is”,”the” which improves the performance of machine learning. 

• It is relatively simpler to implement and computationally efficient.They are  numerical ,making them compatible for different machine algorithm like linear models,  decision trees.

10.Data is splitted into 70:30 ratio to achieve a max accuracy in prediction. 

11.Two models are trained: Linear Discriminant Analysis (LDA) and Random Forest. The  best model is selected based on accuracy.As every model accuracy was ranging between 46  to 53 so we decided to choose the one with the best accuracy. 

RANDOM FOREST 
• Random Forest can handle high-dimensional data, which is useful given the large number  of features generated from TF-IDF. 
• It is less likely to overfit compared to other models because it aggregates the predictions of  multiple decision trees. 
• Random Forest is known for its high accuracy and ability to handle both numerical and  categorical data effectively. 

12.The best model is used to predict the binary target variable on the last part of dataset. 

13.The trading strategy is implemented and trades are simulated based on the signals.The  script simulates buying and selling shares based on the trading signals, updating the portfolio  value at each step.If the signal turns to be 1 we buy the share and sell if its zero. 

Initial Portfolio : $100000 

Stocks in portfolio: 
• META(meta) 
• AMAZON(amzn) 
• GOOGLE(Goog) 

Currency:USD 

CONSTRAINTS: 
The accuracy and reliability of sentiment analysis may be affected by factors such as  ambiguity in language, sarcasm, and context-dependent interpretations, requiring  robust NLP techniques and validation procedures. The availability and quality of  textual data may vary across different stocks and time periods. The performance of  the sentiment analysis model may be influenced by changes in market conditions,  investor behavior, and external events, requiring regular updates and recalibration  of the model. 

THANK YOU 



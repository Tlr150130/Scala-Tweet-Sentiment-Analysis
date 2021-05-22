# Scala-Tweet-Sentiment-Analysis
Use of Scala to analyze positive, negative, and neutral sentiments using a Logistic Regression. Regularization was implemented to reduce overfitting. The L1 and L2 regularization was split with an alpha of 0.9. The optimal lambdda value was determined using 10-K cross validation. All opterations were done through the use of a pipeline. Hosted on DataBricks.

## Data
The airport sentiment data used was from Kaggle.
https://www.kaggle.com/crowdflower/twitter-airline-sentiment

## DataBricks Link
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2359164240640367/2096290446018463/1498264311209917/latest.html

## Usefulness
The application of sentiment analysis on tweet data is prevailent in algorithmic trading. However, sentiment analysis offers a significant competitive edge regarding asset trading is still up for debate. It would be advised to include other features to supplement the trading decision. 

## Improvements
The order of the words may prove to have significance on the sentiment of the phrase/tweet. It would be beneficial to test different machine learning models such as a recurrent neural network or LSTM neural network. 

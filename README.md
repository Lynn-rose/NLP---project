# Tweet Sentiment Analysis

![TWEET](https://github.com/Lynn-rose/NLP---project/blob/main/images/WhatsApp%20Image%202024-08-05%20at%2010.22.14%20AM.jpeg)
## BUSINESS UNDERSTANDING
### Introduction
Apple and Google have been constantly innovating and changing their products, services, and customer experiences. This has led to a surge in customer feedback and a need for companies to analyze and understand the sentiments expressed by their users. Tweets provide a unique platform for companies to gather feedback and respond to their customers, which can be a valuable resource for understanding consumer behavior and making data-driven decisions. This project aims to analyze the sentiments expressed in tweets about Apple and Google products.


## Problem Statement
Accurately classifying the sentiments expressed in tweets about topics or brands into specific classes- positive, negative or neutral is a huge challenge for companies like Apple and Google. Given the diverse nature of informal data, with its use of slang, abbreviations, coming up with a reliable sentiment analysis model that can effectively interpret and classify the tweets can be a complex task. Getting this task right provides a wide variety of novel information for a company like Apple by providing insights and creating better understanding overall of how consumers interact with products/brands.


## Data Understanding
Contributors evaluated tweets about multiple brands and products. The crowd was asked if the tweet expressed positive, negative, or no emotion towards a brand and/or product. If some emotion was expressed they were also asked to say which brand or product was the target of that emotion.

The dataset contains the following columns:

1. 'tweet_text' column: Contains the text of the tweet.
2. 'emotion_in_tweet_is_directed_at' column: Contains the person or entity or brand that the tweet is directed at.
3. 'is_there_an_emotion_directed_at_a_brand_or_product' column: Indicates the kind of emotion in the tweet directed at the brand or product

The dataset has a total of 9093 data points.


## Objective
To build a model that can rate the sentiment of a tweet based on its content.


## Data Preprocessing
The following steps were taken in the preparation of the data ready for modelling.

1. Data Cleaning: Selected relevant columns, handled missing values by replacing the nan values with a placeholder.

2. Tokenisation: It is done to break down the text into smaller units like sentences or words, depending on the task.

3. Removing Stop Words: Stop words are often removed during pre-processing because they can add noise to the data

4. Lemmatization: Lemmatization is generally preferred for tasks where preserving the word's meaning and part of speech is crucial

5. Train-Test Split: Split the dataset into training (80%) and testing (20%) sets.



## Modeling
The following models were used during the developing of a machine learning model that can predict sentiments expressed in tweets about topics or brands into specific classes- positive, negative or neutral is a huge challenge for companies like Apple and Google. Naive Bayes,Logistic Regression, Decision Tree, and Deep Learning models

1. Logistic Regression
2. Decision Tree Classifier
3. Naive Bayes model
4. Support Vector Classifier
5. Deep Learning models:

    * CNN (Convolutional Neural Network)
    * GRU (Gated Recurrent Unit)
    * LSTM (Long Short-Term Memory)


## Evaluation
In this project, multiple evaluation metrics were used to assess the performance of different models in predicting emotions for different tweets. These metrics provided insights into the accuracy, precision, recall, and overall predictive power of the models.

1. Accuracy Definition: The proportion of true results (both true positives and true negatives) among the total number of cases examined.

2. Precision Definition: The proportion of positive identifications that were actually correct.

3. Recall (Sensitivity) Definition: The proportion of actual positives that were correctly identified.

4. F1-Score Definition: The harmonic mean of precision and recall, providing a balance between the two metrics.

5. Confusion Matrix Definition: A table used to describe the performance of a classification model, showing the actual vs. predicted classifications.


## Conlusion
Based on the evaluation metrics (accuracy, classification reports), the Support Vector Classifier outperformed other models with the highest test accuracy. This model was selected as the final model for predicting emotions.The following are the results of the metrics used during the modeling:

Support Vector Classifier

Train Accuracy: 0.88

Test Accuracy: 0.72

This model can help target entities identify emotions whether positive negative or neutral.


 ## For More Information  
Please review my full analysis in my [Jupyter Notebook](https://github.com/Lynn-rose/NLP---project/blob/main/index.ipynb) or my [Presentation](https://github.com/Lynn-rose/phase-3-project/blob/main/Predicting%20H1N1%20Vaccine%20Uptake.pdf)

For any additional questions, please contact Lynn Rose Achieng, lynn90952@gmail.com

```python

```

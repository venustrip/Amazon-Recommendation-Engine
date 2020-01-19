## Introduction

This capstone project is part of the Data Science career track program at Springboard. An intelligent recommendation engine has been built for beauty products that are sold on Amazon.com. By providing personalized recommendations, the ecomm giant will be able to give it's customers the ability to take their brand experiences into their hands and make informed decisions. This level of personalization will support higher customer retention and reiniforce brand loyalty. A hybrid recommendation engine is developed powered by the Amazon dataset, offering a combination of popularityibased recommendation, personalized content-based recommendation and personalized collaborative recommendation at users' choice.

## Data Collection

The dataset has been downloaded from the website: http://jmcauley.ucsd.edu/data/amazon/links.html
(Citation:
Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering
R. He, J. McAuley
WWW, 2016)

There are 2 individual json files containing the beauty product reviews and metadata from Amazon. The reviews file has ~2 million reviews spanning May 1996 - July 2014. The metadata file contains the details of ~260K beauty products sold at Amazon.

## Data Wrangling

After loading the json files into pandas dataframe, the raw data was cleaned, structured and enriched into a cleaner format.
The metadata file contains the details of ~260K beauty products sold at Amazon. There are 9 columns in the file including asin, description, title, brand, categories, related, price, sales rank and image url. No duplicate values were found in the asin column.

The code for data wrangling of metadata file can be found at: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/clean_meta.ipynb

The reviews file has ~2 million reviews collected between May-1996 and July-2014. There are 9 columns in the file including: reviewerID, reviewerName, asin, helpful, reviewText, overall, summary, unixReviewTime and reviewTime.

The code for data wrangling of reviews file can be found at: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/clean_reviews.ipynb

## EDA

EDA (Exploratory Data Analysis) is the initial basic exploration of the dataset in a systematic manner using visual methods. This step included identification and elimination of outliers as well as checking for correlations between the independent variables. 

Key observations during the EDA were:

1. Most of the reviews (62%) had an overall rating of 5 which means positive reviews. 
2. Most of the polarity scores were above 0, which means most reviews had positive sentiments in the data.
3. Price and Polarity had a strong positive correlation. 
4. Overall and Polarity also had a positive correlation, although not too strong, which is odd.
5. There were quite a few reviews where the polarity (sentiment analysis) was inconsistent with the overall rating.

## Recommender Systems

Different types of recommender systems were built using Machine learning algorithms.

### 1. Popularity-based

This is the simplest recommendation system. As is obvious from the name, it simply recommends the popular and highly rated items to all the users. The bigest drawback of this system is that it is non-personalized and recommends the same items to everyone. Since users' have different tastes and preferences, this is not a very useful filtering method.

### 2. Content-based

This method recommends an item based on its features and how similar they are to features of other items in the data set. It is based on similarity of item attributes, which can be determined using cosine similarity or nearest neighbor algorithms. The brand_title, main_cat and description features of the items were combined into a single feature and converted into vector form, in order to effectively determine similarities between different items.

### 3. Collaborative

This is a personalized method for recommending items to users based on historical user and item interactions and user ratings for items. Two approaches to handle this are: Memory-based and Model-based. Memory based collaborative filtering can be further classified into User-User similarity and Item-Item similarity. Model based collaborative filtering utilizes the concept of Matrix Factorization. Latent features are identified that determine how a user rated an item and the utility matrix is decomposed into it's constituents.

### 4. Hybrid

Hybrid method combines 2 or more of the above methods to improve the performance of the recommender. Shortcomings such as cold start problem can be overcome using the hybrid technique.  

## Performance Evaluation

RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) were used to evaluate the performance of each recommender system and the scores were compared to determine the best performance. Strong recommender systems can have positive effects on user experience. They can result into higher customer satisfaction and retention, and in turn boost revenues.

## Further Readings
Link to the Full Report: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/final_report.ipynb

Link to the Slide Deck: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/slide_deck.ipynb


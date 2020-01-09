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

After loading the json files into pandas dataframe, the raw data was cleaned, structured and enriched into a cleaner format.\n",
    "\n",
    "The metadata file contains the details of ~260K beauty products sold at Amazon. There are 9 columns in the file including asin, description, title, brand, categories, related, price, sales rank and image url. No duplicate values were found in the asin column. \n",
    "\n",
    "The code for data wrangling of metadata file can be found at: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/clean_meta.ipynb\n",
    "\n",
    "The reviews file has ~2 million reviews collected between May-1996 and July-2014. There are 9 columns in the file including: reviewerID, reviewerName, asin, helpful, reviewText, overall, summary, unixReviewTime and reviewTime.\n",
    "\n",
    "The code for data wrangling of reviews file can be found at: https://github.com/venustrip/Amazon-Recommendation-Engine/blob/master/clean_reviews.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## EDA\n",
    "\n",
    "EDA (Exploratory Data Analysis) is the initial basic exploration of the dataset in a systematic manner using visual methods. This step will include identification and elimination of outliers as well as checking for correlations between the independent variables. This step was also completed in the first milestone."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Machine Learning\n",
    "\n",
    "Different types of recommender systems will be built using Machine learning algorithms. \n",
    "The algorithms can be classified into two categories: Content-based filtering and Collaborative filtering. Modern recommenders use a hybrid approach which is a combination of these two.\n",
    "Recommender systems are applied where many users interact with many items. We have a rich dataset with item attributes and historical user reviews for these items, that will be used to train and test the models. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performance Evaluation\n",
    "\n",
    "RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) were used to evaluate the performance of each recommender system and the scores were compared to determine the best performance. Strong recommender systems can have positive effects on user experience. They can result into higher customer satisfaction and retention, and in turn boost revenues."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Machine Learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are 4 common ways to build a recommender engine. \n",
    "1. Popularity based\n",
    "2. Content based\n",
    "3. Collaborative\n",
    "4. Hybrid \n",
    "\n",
    "\n",
    "Each will be explored more in detail as we build them one by one."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 Popularity-based Recommender\n",
    "\n",
    "This is the simplest recommendation system. As is obvious from the name, it simply recommends the popular and highly rated items to all the users. The bigest drawback of this system is that it is non-personalized and recommends the same items to everyone. Since users' have different tastes and preferences, this is not a very useful filtering method."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One way of measuring popularity is by counting the number of times an item was rated. Higher the rating count, more popular the item is."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Another way to measure the popularity of an item is by checking it's related_count. This count was extracted by adding the number of times an item was clicked or viewed or bought after viewing or together with another item. The higher the related_count, the more popular the item should be."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We also have sales ranking for the items either under beauty or under health_personal_care categories. Another way of determining popular items is by looking for their sales rank under either of these columns."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So, the caveat to Popularity based recommenders is that they do not filter items based on personal preferences and recommend the same top-N items to every single user. The method is pretty simple, but not very effective."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Content-based Filtering\n",
    "\n",
    "This method recommends an item based on its features and how similar they are to features of other items in the data set. It is based on similarity of item attributes, which can be determined using cosine similarity or nearest neighbor algorithms.\n",
    "\n",
    "The past interactions of a given user with items is taken into account, ignoring all other users. Items are recommended based on comparison between contents of the items and a user profile. The content of each item could be represented as descriptor or terms. \n",
    "\n",
    "We will combine the brand_title, main_cat and description features of the items into a single feature and convert it into vector form so it can be effectively used for determining similarities between different items."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For a fair analysis, we will include only those items from the dataset that have been reviewed more than 10 times and less then 7000 times. Less than 10 ratings means not many users have reviewed the item. More than 7000 ratings means the item is already quite popular and would show up as a recommendation, regardless."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So we are now left with 33,878 items that we will use for filtering using their content.\n",
    "\n",
    "We will define a function to get the pairwise similarity scores of all items compared to a given item. The results will be sorted and top-5 similar items will be returned."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the first approach, we will use TfidfVectorizer function. TF-IDF is heavily used in Natural Language Processing and is used in information retrieval using feature extraction processes. It is a measure used to evaluate how important a word is to a document in document corpus. The three text columns will be combined into one and their text will be used to fit the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we will compute the similarity matrix using linear_kernel of sklearn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's pick a random item in index location 2 and display it's categories and brand_title. Our assumption here is that a user has reviewed and liked this item and therefore we are identifying the top-N items that are most similar to it and recommending them to that user."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So we see that it is a hair care product/shampoo. \n",
    "\n",
    "Execute the recommend function on this item and find the top-5 items similar to it."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The other approach is to use the CountVectorizer, which counts the number of times a token shows up in the document and uses this value as its weight. It is simpler than TfidfVectorizer. We will then compute the similarity matrix using cosine_similarity of sklearn."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above two approaches returned only 1 common item - nexxus promend shampoo 338oz.\n",
    "\n",
    "#### Pros and Cons of Content-based Filtering\n",
    "\n",
    "##### Pros:\n",
    "The recommendations are specific to one user, therefore the model does not need any data about other users. This makes it easier to scale to a large number of users. It is very effective in recommending niche items, since it can capture the specific interests of a user. With sufficient description, the cold start problem can be eliminated.\n",
    "\n",
    "##### Cons:\n",
    "The feature representation of the items must be very rich because the model solely depends on that. The model can make recommendations based only on existing interests of the user. So it tends to over-specialize and will recommend items only similar to those already used and rated."

# Building a Recommender System
## SUMMARY :
The goal of this project is to build a recommendation engine for the users. In this context, there are plenty of different ways to build this engine. It might depend on the user habits, contents or genres.

In order to build that engine, I benefit from the movie data set from the webpage Kaggle. I used tmdb 5000 movies dataset. there are two datasets movies and credits. I provided dataset in repository. dataset Link https://www.kaggle.com/tmdb/tmdb-movie-metadata

I builded simple content based recommendation system using Sklearn packages TfidfVectorizer and sigmoid_kernel :

## TfidfVectorizer :
Transform a count matrix to a normalized tf or tf-idf representation

Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification. Read More https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

## sigmoid_kernel :
SVM algorithms use a set of mathematical functions that are defined as the kernel. The function of kernel is to take data as input and transform it into the required form. Different SVM algorithms use different types of kernel functions. These functions can be different types. For example linear, nonlinear, polynomial, radial basis function (RBF), and sigmoid. Read More https://data-flair.training/blogs/svm-kernel-functions/

## Content based recommender :
This recommendation is built using the movie descriptions. From each description, I used TF-IDF to check the words creating bins. At the end, I was able to recommend movies depending on the descriptions.

When we think about this approach, it is a more detailed recommendation on just suggesting from a variety of genres. When we assume that the descriptions are summaries of each movie without giving away the key concepts, we can actually get a better sense of the similarities utilizing this information.

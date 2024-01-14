Types of Recommendation System
    Content Based
    Collaborative Fitting
    Hybrid

This one is on Content Based

DATA --> PreProcessing --> Model --> Website --> Deploy

Dataset: TMDB 5000 Dataset, Kaggle

Convert text to vector: Bag of words
    Add all tags
    calculate frequency of top 5000 words = 'word' (any number "5000") // no stop words (and or etc) // use skitlearn
    Ask 'word' in each tag of one perticular movie ==> Vector
    repeat above step to all 5000 movies

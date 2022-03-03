# Hands-On Guide for Sentiment Analysis of Conference Calls
Walks you through how to extract sentiment from quarterly conference calls, comparing three different approaches: Finbert vs Loughran Mcdonald vs Naive Bayes.
Provides examples and practical considerations at every level of the process -- from data-collection to sentiment modeling to quantitative analysis.

**These notebooks were part of a larger presentation titled "Hands-On Data Science in Investment Management," presented at the Columbus CFA Society.**

#### [You can download the full presentation here.](https://github.com/talsan/calltone/raw/master/notebooks/CFA%20of%20Columbus%20-%20Hands%20on%20Data-Science.pdf)

## Process Overview
1. [Data Collection](https://github.com/talsan/calltone/blob/master/notebooks/Data%20Collection.ipynb) - text from conference calls, universe, sectors, returns, growth/value indicies
2. [Sentiment Modeling](https://github.com/talsan/calltone/blob/master/notebooks/sentiment_models.ipynb) - Finbert + Loughran & Mcdonald + Naive Bayes (via Textblob)
3. [Quantitative Analysis](https://github.com/talsan/calltone/blob/master/notebooks/sentiment_analysis.ipynb) - Risk and Return Characteristics

## Notebooks
### 1. Data Collection Notebook
[`Data_Collection.ipynb`](https://github.com/talsan/calltone/blob/master/notebooks/Data%20Collection.ipynb) -- steps required to build the corpus and other relevant data for this project. Data includes text from conference calls ([detailed in a seperate repo](https://github.com/talsan/foolcalls)), universe constituents, sector constituents, returns, growth/value indices.
#### Architecture Overview
![Architecture Overview](https://github.com/talsan/calltone/blob/master/img/architecture_overview.png)


### 2. Sentiment Modelling Notebook
[`sentiment_models.ipynb`](https://github.com/talsan/calltone/blob/master/notebooks/sentiment_models.ipynb) -- how to build 3 sentiment models (finbert, Loughran & Mcdonald, Naive Bayes). Includes pre-processing steps like tokenization and lemmatization.
#### Top Calls from 2019 - 2020
![Sentiment Model Example](https://github.com/talsan/calltone/blob/master/img/example_output_sentiment_model.png)


### 3. Quantitative Analysis Notebook
[`sentiment_analysis.ipynb`](https://github.com/talsan/calltone/blob/master/notebooks/sentiment_analysis.ipynb) -- how to analyse and contextualize results with respect to returns, sectors, growth/value, etc. Connects sentiment models to market/economic data.
#### Aggregate Market Sentiment
![Aggregate Market Sentiment](https://github.com/talsan/calltone/blob/master/img/example_output_sentiment_analysis.png)




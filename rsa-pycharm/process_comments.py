from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

pd.set_option('display.max_columns', None)

# Get data
df = pd.read_csv('reddit-sentiment-data.csv')
print(df.head(10))

# Define a function to tokenize headlines and remove stopwords
sia = SIA()
results = []
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')


def process_text(text):
    tokens = []
    for line in text:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)
    return tokens


# Tokenize and remove stopwords from comments and titles
comment_tok = list(df['comment'])
comment_tok = process_text(comment_tok)
title_tok = list(df['title'].drop_duplicates())
title_tok = process_text(title_tok)


# Get Freq Distributions
comment_freq = nltk.FreqDist(comment_tok)
print(comment_freq.most_common(10))
title_freq = nltk.FreqDist(title_tok)
print(title_freq.most_common(10))


# Use Vader Sentiment Analyzer to rank text as positive, negative, or neutral

# df['Comments'] = df['Comments'].apply(list)
# df['neg'] = [sia.polarity_scores(x)['neg'] for x in df['Comments']]
# df['neu'] = [sia.polarity_scores(x)['neu'] for x in df['Comments']]
# df['pos'] = [sia.polarity_scores(x)['pos'] for x in df['Comments']]

# df['pol_score'] = 0
# df['pol_score'] = df['pol_score'].apply(sia.polarity_scores(df['Comments']))

# For each line in the headlines list, calculate and define the polarity score,
# define pol_score under 'headline', and append the results to the results list.
# for line in headlines:
#    pol_score = sia.polarity_scores(line)
#    pol_score['headline'] = line
#    results.append(pol_score)

# Assign Pos and Neg Values
# df['label'] = 0
# df.loc[df['compound'] > 0.2, 'label'] = 1
# df.loc[df['compound'] < -0.2, 'label'] = -1
# print(df.loc[df['label'] != 0].head(5))

# list_text = list(s['Comments'])
# final = process_text(list_text)

# Add results to dataframe
# df = title_ser.to_frame()
# df = pd.concat([df, comment_tok], axis=1)
#
# print(df.head)



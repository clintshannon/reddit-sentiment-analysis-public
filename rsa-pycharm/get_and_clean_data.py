from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import praw

# Create a reddit client
reddit = praw.Reddit(client_id='TtaGbxrcEqrFPqjHoFyZig',
                     client_secret='ANwuNmen1-zv_7rZ-JJp7n7nx3bIGg',
                     user_agent='sentiment-analysis')

# Define a set for headlines to avoid duplicates
headlines = set()

# Iterate through the datascience sub titles
for submission in reddit.subreddit('datascience').new(limit=None):
    headlines.add(submission.title)
    display.clear_output()

# print(len(headlines))

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# Use NLTK’s built-in Vader Sentiment Analyzer to rank a piece of text as positive,
# negative or neutral using a lexicon of positive and negative words.

sia = SIA()
results = []

# For each line in the headlines list, calculate and define the polarity score,
# define pol_score under 'headline', and append the results to the results list.
for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

# pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head(5)

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
print(df.loc[df['label'] != 0].head(5))

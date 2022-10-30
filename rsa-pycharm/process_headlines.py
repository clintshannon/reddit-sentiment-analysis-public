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
# Get info here https://www.reddit.com/prefs/apps/
reddit = praw.Reddit(client_id='TtaGbxrcEqrFPqjHoFyZig',
                     client_secret='ANwuNmen1-zv_7rZ-JJp7n7nx3bIGg',
                     user_agent='sentiment-analysis')

# Define a set for headlines to avoid duplicates
headlines = set()

# Iterate through the datascience sub titles
for submission in reddit.subreddit('datascience').new(limit=None):
    headlines.add(submission.title)
    display.clear_output()

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

df = pd.DataFrame.from_records(results)
df.head(5)

# Assign Pos and Neg Values
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
print(df.loc[df['label'] != 0].head(5))

# Positive and Negative Stats
print(df.label.value_counts())
print(df.label.value_counts(normalize=True) * 100)

# Plot Pos and Neg Stats
fig, ax = plt.subplots(figsize=(8, 8))
counts = df.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

# Processing Text
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# Define a function to tokenize headlines and remove stopwords.
def process_text(text):
    tokens = []
    for line in text:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)
    return tokens


# Positive words
pos_lines = list(df[df.label == 1].headline)
pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)
print(pos_freq.most_common(10))

# Negative words
neg_lines = list(df[df.label == -1].headline)
neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)
print(neg_freq.most_common(10))

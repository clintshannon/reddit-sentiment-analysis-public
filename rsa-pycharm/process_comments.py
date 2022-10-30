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

subreddit = reddit.subreddit("learnprogramming").top(limit=5, time_filter='week')
data = {}
titles = {}

for submission in subreddit:
    post_comments = set()
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        post_comments.add(comment.body)
    post_id = submission.id
    data[post_id] = post_comments
    titles[post_id] = submission.title
    display.clear_output()

# Convert comments and titles to series
s_comments = pd.Series(data, name='Comments')
s_comments.index.name = 'Post ID'
s_comments.reset_index()

s_titles = pd.Series(titles, name='Titles')
s_titles.index.name = 'Post ID'
s_titles.reset_index()

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


# Tokenize and remove stopwords
t_comments = s_comments.apply(process_text)
# t_titles = s_titles.apply(process_text)

df = s_titles.to_frame()
df = pd.concat([df, t_comments], axis=1)

# Use NLTK’s built-in Vader Sentiment Analyzer to rank a piece of text as positive,
# negative or neutral using a lexicon of positive and negative words.
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
results = []

df['Comments'] = df['Comments'].apply(list)
df['neg'] = [sia.polarity_scores(x)['neg'] for x in df['Comments']]
df['neu'] = [sia.polarity_scores(x)['neu'] for x in df['Comments']]
df['pos'] = [sia.polarity_scores(x)['pos'] for x in df['Comments']]

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
print(df.head)



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
# post_id = set()
data = {}

for submission in subreddit:
    post_comments = set()
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        post_comments.add(comment.body)
    post_id = submission.id
    data[post_id] = post_comments
    display.clear_output()

s = pd.Series(data, name='Comments')
s.index.name = 'Post ID'
s.reset_index()

print(s.head(5))


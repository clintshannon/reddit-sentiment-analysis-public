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
post_id = set()

for submission in subreddit:
    post_id.add(submission.id)
    display.clear_output()
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        print(comment.body)

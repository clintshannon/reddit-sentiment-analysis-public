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

print(len(headlines))

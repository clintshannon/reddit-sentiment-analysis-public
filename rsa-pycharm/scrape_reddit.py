import pandas as pd
import praw
import os

pd.set_option('display.max_columns', None)

# Create reddit client, define subreddit, and add other tools
reddit = praw.Reddit(client_id='TtaGbxrcEqrFPqjHoFyZig',
                     client_secret='ANwuNmen1-zv_7rZ-JJp7n7nx3bIGg',
                     user_agent='sentiment-analysis')
subreddit = reddit.subreddit("learnprogramming").top(limit=5, time_filter='week')

# Create a dictionary to store all data
sub_dict = {"posts": {"id": [],
                      "title": [],
                      "score": []
                      },
            "comments": {"id": [],
                         "c_id": [],
                         "comment": [],
                         "c_score": []
                         }
            }

# Add data to the dictionary
for submission in subreddit:
    sub_dict["posts"]["id"].append(submission.id)
    sub_dict["posts"]["title"].append(submission.title)
    sub_dict["posts"]["score"].append(submission.score)
    submission.comment_sort = "top"
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        sub_dict["comments"]["id"].append(comment.link_id)
        sub_dict["comments"]["c_id"].append(comment.id)
        sub_dict["comments"]["comment"].append(comment.body)
        sub_dict["comments"]["c_score"].append(comment.score)

# Create dataframes from the dictionary and clean up columns
post_df = pd.DataFrame(sub_dict["posts"])
comment_df = pd.DataFrame(sub_dict["comments"])
comment_df["id"] = comment_df["id"].str[3:]
# comment_df = comment_df.nlargest(25, "c_score")

# Merge dataframes
all_df = pd.merge(post_df, comment_df, on="id")
# path = r'C:\Users\clint\my-files\projects\reddit-sentiment-analysis\rsa-pycharm\'
# all_df.to_csv(os.path.join(path, r'reddit-data-for-sentiment.csv'))
all_df.to_csv(r'reddit-sentiment-data.csv', index=False)

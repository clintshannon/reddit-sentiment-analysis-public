import pandas as pd
import praw
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
pd.set_option('display.max_columns', None)

# Create reddit client, define subreddit, and add NLP tools
reddit = praw.Reddit(client_id='<reddit-client-id>',
                     client_secret='reddit-client-secret',
                     user_agent='app-name')
subreddit = reddit.subreddit("Fantasy").top(limit=3, time_filter='week')
sia = SIA()
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')


def process_text(text):
    tokens = []
    for line in text:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)
    return tokens


# Create a dictionary to store all data
sub_dict = {"posts": {"id": [],
                      "title": [],
                      "score": []
                      },
            "comments": {"id": [],
                         "c_id": [],
                         "comment": [],
                         "c_score": [],
                         "comment_set": set(),
                         "sia_results": []
                         }
            }

# Add reddit data to the dictionary
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
        sub_dict["comments"]["comment_set"].add(comment.body)

# For each comment, calculate the polarity score, and append to results list.
for line in sub_dict["comments"]["comment_set"]:
    pol_score = sia.polarity_scores(line)
    pol_score['comment'] = line
    sub_dict["comments"]["sia_results"].append(pol_score)

# Assign Pos and Neg Values
sia_df = pd.DataFrame(sub_dict["comments"]["sia_results"])
sia_df['label'] = 0
sia_df.loc[sia_df['compound'] > 0.2, 'label'] = 1
sia_df.loc[sia_df['compound'] < -0.2, 'label'] = -1
print("Sentiment Sample:\n")
print(sia_df.loc[sia_df['label'] != 0].head(5))

# Positive, Negative, and Neutral Ratio
print("Sentiment Proportions:\n")
print(sia_df.label.value_counts(normalize=True) * 100)

# Plot Pos and Neg Stats
fig, ax = plt.subplots(figsize=(8, 8))
counts = sia_df.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

plt.savefig('sentiment_analysis.jpg', bbox_inches='tight')

# Tokenize and remove stopwords from comments and titles
# comment_tok = list(sia_df['comment'])
# comment_tok = process_text(comment_tok)
# df['sentiment'] = df['text'].apply(lambda tweet: TextBlob(tweet).sentiment)
freq = []
blob = TextBlob(str(sub_dict['posts']['title']))
for noun in blob.noun_phrases:
    freq.append(noun)
title_tok = list(sub_dict['posts']['title'])
title_tok = str(process_text(title_tok))
blob2 = TextBlob(title_tok)
# for more in blob2.noun_phrases:
#     freq.append(more)

# Get Freq Distributions
# comment_freq = nltk.FreqDist(comment_tok)
# print(comment_freq.most_common(10))
# title_freq = nltk.FreqDist(title_tok)
freq = nltk.FreqDist(freq)
print("Common post themes:\n")
print(freq.most_common(10))

# print(sia_df.head(5))

# Create dataframes from the dictionary and clean up columns
# post_df = pd.DataFrame(sub_dict["posts"])
# comment_df = pd.DataFrame(sub_dict["comments"])
# comment_df["id"] = comment_df["id"].str[3:]

# Merge dataframes and export csv
# all_df = pd.merge(post_df, comment_df, on="id")
# all_df.to_csv(r'reddit-sentiment-data.csv', index=False)

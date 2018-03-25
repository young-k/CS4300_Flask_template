import praw

reddit = praw.Reddit(client_id='wQaHdUWqKKgaLQ',
                     client_secret='mIDBYtZGLe_Pve7RiGcw2VWc-y4',
                     user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0')

subs = []
for submission in reddit.subreddit('changemyview').top(limit=1000):
  subs.append(submission)

print(subs)

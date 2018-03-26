import praw

f = open("sample.txt", "w+")

reddit = praw.Reddit(client_id='wQaHdUWqKKgaLQ',
                     client_secret='mIDBYtZGLe_Pve7RiGcw2VWc-y4',
                     user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0')

subs = []
for submission in reddit.subreddit('changemyview').top(limit=100):
  subs.append(submission)

for submission in subs:
  print(submission.title.encode('utf8'))
  f.write(submission.title.encode('utf8'))
  f.write("\n")
  f.write("=======================================")
  f.write("\n\n")

  submission.comment_sort = 'top'
  top_comments = submission.comments.list()
  real_comments = [comment for comment in top_comments if isinstance(comment, praw.models.Comment)]

  real_comments.sort(key=lambda c: c.score, reverse=True)

  for comment in real_comments[:20]:
    f.write("\tScore: " + str(comment.score))
    f.write("\n")
    f.write("\t\t" + comment.body.encode('utf8').replace("\n", " "))
    f.write("\n\n")

  f.write("\n\n")

f.close()

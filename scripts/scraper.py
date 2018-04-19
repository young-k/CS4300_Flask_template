import json
import lxml
import praw
import requests
from bs4 import BeautifulSoup

def write_sample_to_file(num_posts, num_comments):
  f = open("testing.txt", "w+")

  reddit = praw.Reddit(client_id='wQaHdUWqKKgaLQ',
                       client_secret='mIDBYtZGLe_Pve7RiGcw2VWc-y4',
                       user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0')

  subs = []
  for submission in reddit.subreddit('changemyview').top(limit=num_posts):
    subs.append(submission)

  ret = []
  for submission in subs:
    delta_detection = submission.author is not None

    curr = {}
    curr["body"] = ""
    soup = BeautifulSoup(submission.selftext_html, 'lxml')
    for paragraph in soup.find_all("p"):
      curr["body"] += paragraph.text

    if delta_detection: 
      curr["author"] = submission.author.name
    else:
      curr["author"] = "[deleted]"

    curr["title"] = submission.title.encode('utf8')
    curr["url"] = submission.url
    print(curr["title"])

    submission.comment_sort = 'top'
    top_comments = submission.comments.list()

    real_comments = [comment for comment in top_comments \
        if isinstance(comment, praw.models.Comment)]

    real_comments = [comment for comment in real_comments \
        if comment.author is not None]

    real_comments.sort(key=lambda c: c.score, reverse=True)

    curr["top_comments"] = []
    curr["delta_comments"] = []
    for comment in real_comments[:num_comments]:
      curr_comment = {}
      curr_comment["score"] = comment.score
      curr_comment["author"] = comment.author.name
      curr_comment["body"] = comment.body.encode('utf8')
      curr["top_comments"].append(curr_comment)

      if delta_detection:
        for reply in comment._replies:
          if isinstance(reply, praw.models.Comment) and \
              reply.author is not None:
            if reply.author.name == curr["author"]:
              delta_given = "!delta" in reply.body or u'\u2206' in reply.body
              if delta_given:
                curr["delta_comments"].append(curr_comment)

    ret.append(curr)

  json.dump(ret, f)
  f.close()

if __name__ == "__main__":
  write_sample_to_file(1000, 40)

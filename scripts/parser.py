import json
import re

with open('data.txt', 'r') as f:
    data = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

posts = []
post = None

for i, line in enumerate(data):
    nextline = data[i + 1] if i < (len(data) - 1) else ''
    prevline = data[i - 1] if i > 0 else ''

    if nextline == '=' * 39:
        if post is not None:
            post['comments'] = comments
            posts.append(post)
        post = {'title': line}
        comments = []

    if prevline.startswith('Score:'):
        score = int(re.findall(r'\d+', prevline)[0])
        comments.append({'score': score, 'comment': line})

post['comments'] = comments
posts.append(post)

with open('../data/data.json', 'w') as f:
    json.dump(posts, f)

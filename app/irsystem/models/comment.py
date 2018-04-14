from . import *

class Comment(Base):
  __tablename__ = 'comments'

  id = db.Column(db.Integer, primary_key=True)
  body = db.Column(db.Text, nullable=False)
  score = db.Column(db.Integer, nullable=False)

  submission_id = db.Column(
      db.Integer,
      db.ForeignKey('submissions.id', ondelete='CASCADE')
  )
  submission = db.relationship('Submission', backref='comments')

  user_id= db.Column(
      db.Integer,
      db.ForeignKey('users.id', ondelete='CASCADE')
  )
  user = db.relationship('User', backref='comments')

  def __init__(self, **kwargs):
    self.body = kwargs.get('body')
    self.score = kwargs.get('score')
    self.submission_id = kwargs.get('submission_id')

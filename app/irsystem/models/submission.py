from . import *

class Submission(Base):
  __tablename__ = 'submissions'

  id = db.Column(db.Integer, primary_key=True)
  score = db.Column(db.Integer, nullable=False)
  title = db.Column(db.String(255), nullable=False)

  def __init__(self, **kwargs):
    self.score = kwargs.get('score')
    self.title = kwargs.get('title')

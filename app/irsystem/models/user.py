from . import *

class User(Base):
  __tablename__ = 'users'

  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(255), nullable=False, unique=True)
  delta = db.Column(db.Integer, nullable=True)

  def __init__(self, **kwargs):
    self.username = kwargs.get('username')

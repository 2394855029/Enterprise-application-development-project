from sqlalchemy.orm import Session

from . import models, schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_recent_papers(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Paper).order_by(models.Paper.created_at.desc()).offset(skip).limit(limit).all()


def create_paper(db: Session, paper: schemas.PaperCreate):
    db_paper = models.Paper(title=paper.title, abstract=paper.abstract, authors=paper.authors, keywords=paper.keywords, content=paper.content, analysis_result=paper.analysis_result, created_at=paper.created_at, updated_at=paper.updated_at, md5_hash=paper.md5_hash, path=paper.path)
    db.add(db_paper)
    db.commit()
    db.refresh(db_paper)
    return db_paper


def get_paper(db: Session, paper_id: int):
    return db.query(models.Paper).filter(models.Paper.id == paper_id).first()


def get_papers(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Paper).offset(skip).limit(limit).all()


def update_paper(db: Session, paper_id: int, paper: schemas.PaperCreate):
    db_paper = db.query(models.Paper).filter(models.Paper.id == paper_id).first()
    db_paper.title = paper.title
    db_paper.abstract = paper.abstract
    db_paper.authors = paper.authors
    db_paper.keywords = paper.keywords
    db_paper.content = paper.content
    db_paper.analysis_result = paper.analysis_result
    db_paper.updated_at = paper.updated_at
    db.commit()
    db.refresh(db_paper)
    return db_paper

def delete_paper(db: Session, paper_id: int):
    db_paper = db.query(models.Paper).filter(models.Paper.id == paper_id).first()
    db.delete(db_paper)
    db.commit()
    return db_paper


def search_papers(db: Session, keyword: str):
    return db.query(models.Paper).filter(models.Paper.result.like('%'+keyword+'%')).all()


def get_paper_by_md5_hash(db: Session, md5_hash: str):
    return db.query(models.Paper).filter(models.Paper.md5_hash == md5_hash).first()


def create_chat_history(db: Session, chat_history: schemas.ChatHistoryCreate):
    db_chat_history = models.ChatHistory(prompt=chat_history.prompt, response=chat_history.response, created_at=chat_history.created_at)
    db.add(db_chat_history)
    db.commit()
    db.refresh(db_chat_history)
    return db_chat_history
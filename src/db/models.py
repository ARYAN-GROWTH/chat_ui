from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from src.db.connection import Base

class User(Base):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'public'}
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(255), unique=True, nullable=True)
    display_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    __table_args__ = {'schema': 'public'}
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class QueryLog(Base):
    __tablename__ = 'query_logs'
    __table_args__ = {'schema': 'public'}
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=True, index=True)
    user_query = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=True)
    validation_status = Column(String(50), nullable=True)
    execution_status = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    row_count = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserMemory(Base):
    __tablename__ = 'user_memory'
    __table_args__ = {'schema': 'public'}
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('public.users.id', ondelete='CASCADE'), unique=True)
    memory_summary = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

class SessionSummary(Base):
    __tablename__ = 'session_summaries'
    __table_args__ = (
        UniqueConstraint('session_id', 'user_id', name='uq_session_user'),
        {'schema': 'public'}
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey('public.users.id', ondelete='CASCADE'))
    summary = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

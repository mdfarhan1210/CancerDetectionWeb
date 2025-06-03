from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, timedelta, timezone

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    profile_picture = Column(String(255), nullable=True, default=None)
    
    images = relationship("Image", back_populates="owner")

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)                          
    filepath = Column(String(255), nullable=False)
    owner_id = Column(Integer, ForeignKey('users.id'))
    prediction = Column(String(255), nullable=False) 
    
    owner = relationship("User", back_populates="images")

class OTPEntry(Base):
    __tablename__ = 'otp_entries'

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(255), unique=True, index=True)
    otp = Column(String(20), index=True)
    expiry = Column(DateTime, default=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))
from pydantic import BaseModel, EmailStr, constr
from typing import List, Optional

class ImageBase(BaseModel):
    filename: str
    prediction: str

class ImageCreate(ImageBase):
    pass

class Image(ImageBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    username: str
    email: str
    phone: str
    profile_picture: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    images: List[Image] = []

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    profile_picture: Optional[str] = None 

    class Config:
        from_attributes = True

class ChatInput(BaseModel):
    message: str
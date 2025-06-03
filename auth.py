from fastapi import Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
import models, schemas, database

# Secret key for JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(request: Request, db: Session = Depends(database.get_db)):
    token = request.cookies.get("access_token")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    
    return user

def create_user(user: schemas.UserCreate, db: Session):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, email=user.email, phone=user.phone, hashed_password=hashed_password, profile_picture=user.profile_picture)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def verify_otp(phone: str, otp: str, db: Session) -> bool:
    otp_entry = db.query(models.OTPEntry).filter(models.OTPEntry.phone_number == phone).first()
    if otp_entry and otp_entry.otp == otp:
        current_utc_time = datetime.now(timezone.utc).replace(tzinfo=timezone.utc)
        
        if otp_entry.expiry.replace(tzinfo=timezone.utc) > current_utc_time:
            db.delete(otp_entry)
            db.commit()
            return True
    return False
import os
# Suppress TensorFlow informational and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, Depends, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from datetime import timedelta
import models, schemas, database, auth
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from database import engine
from twilio.rest import Client
from datetime import datetime, timezone
from pyotp import TOTP, random_base32
from gradcam import GradCAM
import cv2

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load the .env file
load_dotenv()

# Serve static files (e.g., CSS, JavaScript)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template directory setup
templates = Jinja2Templates(directory="templates")

# Load pre-trained models
model_paths = {
    "CustomSequentialModel": "models/benign_malignant_classifier.h5",
    "ResNet50": "models/benign_malignant_classifier_resnet50.h5",
    "VGG16": "models/benign_malignant_classifier_vgg16.h5",
    "VGG19": "models/benign_malignant_classifier_vgg19.h5"
}
models_dict = {name: load_model(path) for name, path in model_paths.items()}

ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')  
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')    
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER') 

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Image Preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]  # Remove alpha channel if present
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Image Prediction function
def predict_image(image: Image.Image, model_name: str) -> dict:
    processed_image = preprocess_image(image)
    model = models_dict[model_name] 
    prediction = model.predict(processed_image)
    predicted_class, predicted_class_idx = ('benign', 0) if prediction[0] < 0.5 else ('malignant', 1)
    confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
    return {
        "model": model_name,
        "predicted_class": predicted_class,
        "predicted_class_idx": predicted_class_idx,
        "confidence": confidence
    }

def get_max_accuracy_model():
    accuracy_dict = dict()
    for name, model in models_dict.items():
        if name == 'CustomSequentialModel':
            optimizer = Adam(learning_rate=0.001)
        elif name == 'ResNet50':
            optimizer = 'adam'
        elif name == 'VGG16':
            optimizer = Adam(learning_rate=1e-4)
        elif name == 'VGG19':
            optimizer = Adam(learning_rate=1e-5)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    
        validation_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = validation_datagen.flow_from_directory(
            'data/test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        accuracy_dict[name] = model.evaluate(validation_generator)[1]

    max_accuracy_model_name = max(accuracy_dict, key=accuracy_dict.get)
    
    return max_accuracy_model_name
 
# Using GradCAM to generate heatmap
def generate_heatmap(image: Image.Image, grad_cam):
    processed_image = preprocess_image(image)
    
    # Compute heatmap
    heatmap = grad_cam.compute_heatmap(processed_image)
    
    return heatmap

def generate_heatmap_colored(heatmap, original_image, grad_cam):
    # Convert original PIL image to NumPy array for visualization
    original_image = np.array(original_image)

    # Assuming original_image is in RGB, but OpenCV uses BGR for display purposes
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Overlay the heatmap on the original image
    heatmap_colored, output_image = grad_cam.overlay_heatmap(heatmap, original_image)

    # Convert the image back to RGB for display with Matplotlib
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    is_success_h, heatmap_colored_encoded = cv2.imencode('.png', heatmap_colored)
    is_success_o, output_image_encoded = cv2.imencode('.png', output_image)
    
    if is_success_h and is_success_o:
        return heatmap_colored_encoded, output_image_encoded
    
    raise RuntimeError(f"Failed to encode heatmap iamge") 

# OTP sending function
def send_otp(phone_number, db: Session):
    totp = TOTP(random_base32())
    otp = totp.now()
    message = client.messages.create(
        body=f"Your OTP is: {otp} which is valid for 5 minutes",
        from_=TWILIO_NUMBER,
        to=phone_number
    )
    
    # Check if there's already an OTP for this number and update or create new
    otp_entry = db.query(models.OTPEntry).filter(models.OTPEntry.phone_number == phone_number).first()
    if otp_entry:
        otp_entry.otp = str(otp)
        otp_entry.expiry = datetime.now(timezone.utc) + timedelta(minutes=5)  # Reset expiry time
    else:
        otp_entry = models.OTPEntry(phone_number=phone_number, otp=str(otp))
        db.add(otp_entry)
    
    db.commit()
    db.refresh(otp_entry)

# GET method to display home section
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    token = request.cookies.get("access_token")
    user_is_authenticated = token is not None
        
    return templates.TemplateResponse("home.html", {"request": request, "user_is_authenticated": user_is_authenticated})
    
# GET method to display the signup form
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

# POST method to send OTP 
@app.post("/request-otp/", response_class=HTMLResponse)
async def request_otp(phone: str = Form(...), db: Session = Depends(database.get_db)):
    send_otp(phone, db)
    return HTMLResponse(content="OTP sent successfully.")

# POST method to verify OTP 
@app.post("/verify-otp/", response_class=HTMLResponse)
async def verify_otp(phone: str = Form(...), otp: str = Form(...), db: Session = Depends(database.get_db)):
    if auth.verify_otp(phone, otp, db):
        return HTMLResponse(content="OTP verified successfully.")
    else:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
# POST method to handle signup form submission
@app.post("/signup", response_class=HTMLResponse)
async def signup(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    phone: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db)
):    
    # Validate file type
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .png, .jpg, or .jpeg file.")
    
    # Save the uploaded image
    filename = f"{username}_profile_picture.jpg"
    filepath = f"static/uploads/{filename}"
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    
    user_data = schemas.UserCreate(
        username=username, email=email, phone=phone, profile_picture=filepath, password=password
    )
    auth.create_user(user_data, db)
    
    # Redirect to the login page after successful signup
    return RedirectResponse(url="/login", status_code=303)

# GET method to display the login form
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# POST method to handle login form submission
@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db)
):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    # Set the token as an HTTP-only cookie
    response = RedirectResponse(url="/", status_code=303)
    # response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    response.set_cookie(key="access_token", value=f"{access_token}", httponly=True)
    return response

# GET method to display the upload page
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request, current_user: schemas.User = Depends(auth.get_current_user)):
    return templates.TemplateResponse("upload.html", {"request": request})

# POST method to handle image upload and prediction
@app.post("/predict/", response_class=HTMLResponse)
async def upload_image(
    request: Request,
    file: UploadFile = File(None),
    image_data: str = Form(None),
    current_user: schemas.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    model = get_max_accuracy_model()
    if file:
        # Validate file type
        if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .png, .jpg, or .jpeg file.")
        
        # Save the uploaded image
        filename = f"{current_user.username}_{file.filename}"
        filepath = f"static/uploads/{filename}"
            # Ensure the upload directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as buffer:
            buffer.write(file.file.read())

        # Open and preprocess the image for prediction
        image = Image.open(filepath)
        
    elif image_data:
        # Decode the base64 image
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
    

    predictions = predict_image(image, model)

    #############################################################################################
    # GradCam
    
    model = models_dict[model] 
    
    predicted_class_idx = predictions.get("predicted_class_idx")
    gradcam = GradCAM(model=model, classIdx=predicted_class_idx)

    heatmap = generate_heatmap(image, gradcam)
    heatmap_colored, output_image = generate_heatmap_colored(heatmap, image, gradcam)
    
    heatmap_filename = f"{current_user.username}_heatmap_{file.filename}"
    heatmap_filepath = f"static/uploads/{heatmap_filename}"
    with open(heatmap_filepath, "wb") as buffer:
        buffer.write(heatmap_colored)

    heatmap_with_original_image_filename = f"{current_user.username}_output_image_{file.filename}"
    heatmap_with_original_image_filepath = f"static/uploads/{heatmap_with_original_image_filename}"
    with open(heatmap_with_original_image_filepath, "wb") as buffer:
        buffer.write(output_image)
    
    # Save image record in the database
    image_record = models.Image(filename=filename, filepath=filepath, prediction=predictions.get("predicted_class"), owner_id=current_user.id)
    db.add(image_record)
    db.commit()
    db.refresh(image_record)
    
    # Return the result page with predictions
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "predictions": predictions,
            "image_url": f"/static/uploads/{filename}",
            "heatmap_image_url": f"/static/uploads/{heatmap_filename}",
            "heatmap_with_original_image_url": f"/static/uploads/{heatmap_with_original_image_filename}"
        }
    )

@app.get("/logout", response_class=RedirectResponse)
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")
    return response

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    token = request.cookies.get("access_token")
    user_is_authenticated = token is not None
    
    return templates.TemplateResponse("about.html", {"request": request, "user_is_authenticated": user_is_authenticated})

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    user_images = db.query(models.Image).filter(models.Image.owner_id == current_user.id).all()
    return templates.TemplateResponse("profile.html", {"request": request, "user": current_user, "images": user_images})

@app.get("/edit_profile", response_class=HTMLResponse)
async def about(request: Request, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    token = request.cookies.get("access_token")
    user_is_authenticated = token is not None
    
    return templates.TemplateResponse("edit_profile.html", {"request": request, "user": current_user, "user_is_authenticated": user_is_authenticated})

@app.post("/profile/update")
async def update_user_profile(
    request: Request,
    username: str = Form(None),
    email: str = Form(None),
    password: str = Form(None),
    file: UploadFile = File(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    try:
        user = db.query(models.User).filter(models.User.id == current_user.id).first()
        if not user:
            return HTTPException(status_code=404, detail="User not found")
        
        shouldCommitTheDb = False
        
        if username:
            user.username = username
            shouldCommitTheDb = True
        if email:
            user.email = email
            shouldCommitTheDb = True
        if password:
            user.hashed_password = auth.get_password_hash(password)
            shouldCommitTheDb = True
        if file:
            # Save the uploaded image
            filename = f"{current_user.username}_profile_picture.jpg"
            filepath = f"static/uploads/{filename}"
            
            with open(filepath, "wb") as buffer:
                buffer.write(file.file.read())  
            user.profile_picture = filepath
            shouldCommitTheDb = True
        
        if not File:
            pass
        if shouldCommitTheDb:
            db.commit()
            db.refresh(user)

        user_images = db.query(models.Image).filter(models.Image.owner_id == current_user.id).all()
        return templates.TemplateResponse("profile.html", {"request": request, "user": current_user, "images": user_images})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"cant update -> {str(e)}")
    
@app.post("/chat")
async def chatbotChat(
    request: Request,
    input: schemas.ChatInput
    ):
    user_messasge = input.message

    response = f"Through the sever it is delivered... typed message {user_messasge}"
    return {"response": response}
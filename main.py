# from urllib import request

# from fastapi import FastAPI,Request,Form
# import jinja2
# from pydantic import BaseModel,Field
# import pandas as pd
# import numpy as np
# import joblib
# from typing import Annotated
# import sys
# import os 
# from fastapi.templating import Jinja2Templates
# from sklearn._loss import _loss as sklearn_loss


# app=FastAPI()
# sys.modules.setdefault('_loss',sklearn_loss)


# model_data=joblib.load('heart.joblib')
# model=model_data["model"]
# feature_names = model_data["feature"]


# template=Jinja2Templates(directory="templates")

# class heartData(BaseModel):
#   age:int = Field(ge=0,le=120,description="Age of patient")
#   sex:int = Field(...,description="Sex")
#   cp:int =Field(le=3,ge=0,description="Chest Pain type")
#   trestbps:int = Field(...,description="Testing blood pressure")
#   chol:int = Field(...,description="Serum cholesterol")
#   fbs:int =Field(...,description="Fasting blood sugar")
#   restecg:int=Field(...,description="Resting electrocardiographic")
#   thalach:int = Field(...,description="Maximum heart rate achieve")
#   exang:int=Field(...,description="Exercise-induced agina")
#   oldpeak:int=Field(...,description="ST depression induced")
#   slope:int=Field(...,description="Slope of peak excercise")
#   ca:int=Field(...,description="Number of mejore vessel")
#   thal:int=Field(...,description="Thalassemia")

# # @app.get("/")
# # async def home():
# #   return template.TemplateResponse(request=request,name="index.html")

# @app.get("/")
# async def home(request: Request):
#     return template.TemplateResponse(
#         request=request,
#         name="index.html"
        
#     )

# @app.post("/predict")
# def predict(request:Request,data:Annotated[heartData,Form()]):
#   input_data=pd.DataFrame([data.model_dump()])
#   input_data=input_data[feature_names]
#   prediction=model.predict(input_data)
#   if prediction[0]==1:
#     result="The patient is likely to have heart disease"
#   else:
#     result="The patient is unlikely to have heart disease"
#   return template.TemplateResponse(request=request,template_name="result.html",context={"request":request,"result":result})

from fastapi import FastAPI,Request,Form    
import jinja2
from pydantic import BaseModel , Field
import pandas as pd
import numpy as np
import joblib
from typing import Annotated
import sys
import os
from urllib import request 
from fastapi.templating import Jinja2Templates
from sklearn._loss import _loss as sklearn_loss



app = FastAPI()
sys.modules.setdefault("_loss", sklearn_loss)
model_data = joblib.load('heart.joblib')
model = model_data["model"]
feature_names = model_data["feature"]


template = Jinja2Templates(directory = "templates")


class heartData(BaseModel):
    age: int = Field(...,ge = 0,le =120, description="Age of the patient")
    sex: int = Field(..., description="Sex")
    cp: int = Field(le = 3 ,ge = 0, description="Chest pain type")
    trestbps: int = Field(..., description="Resting blood pressure")
    chol: int = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar")
    restecg: int = Field(..., description="Resting electrocardiographic results")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of the peak exercise ST segment")
    ca: int = Field(..., description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., description="Thalassemia")

@app.get("/")
async def home(request: Request):
    return template.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )
    
@app.post("/predict")
def predict(request: Request, data: Annotated[heartData, Form()]):
    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data[feature_names]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "The patient is likely to have heart disease."
    else:
        result = "The patient is unlikely to have heart disease."
        
    return template.TemplateResponse(
    request=request,
    name="index.html",
    context={"request": request, "result": result}
)
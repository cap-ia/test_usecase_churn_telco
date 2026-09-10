from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}
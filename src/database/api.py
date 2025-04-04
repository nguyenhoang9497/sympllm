from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Chat(BaseModel):
    message: str
    model: Optional[str] = None
    
db = None
ollamaModel = None

@app.post("/chat")
async def chat(chat: Chat):
    response = ollamaModel.queryDatabase(chat.message, db, chat.model)
    return {"response": response}



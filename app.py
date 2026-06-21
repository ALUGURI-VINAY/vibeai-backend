from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="AI Vibe Developer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    persona: str
    empathy: int
    humor: int
    professionalism: int


@app.get("/")
def root():
    return {"message": "AI Vibe Backend Running with Groq"}


@app.post("/chat")
def chat(data: ChatRequest):
    prompt = f"""
You are {data.persona}.

Empathy Level: {data.empathy}/100
Humor Level: {data.humor}/100
Professionalism Level: {data.professionalism}/100

User Message:
{data.message}

Respond according to the selected personality and vibe settings.
Keep the response helpful, natural, and emotionally intelligent.
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_completion_tokens=500
    )

    return {
        "reply": completion.choices[0].message.content
    }
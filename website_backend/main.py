from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chatbot_call import CulturalChatbot  
import nest_asyncio
from pyngrok import ngrok
from typing import List
import uvicorn
import os

ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
public_url = ngrok.connect(8000)

chatbot = CulturalChatbot()

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

class OnboardingData(BaseModel):
    answers: List[str]

# -------------------------------------------------------
# THESE METHODS ARE FOR THE CULTURAL CHATBOT

@app.post("/chat")
def chat_endpoint(request: PromptRequest):
    tagged_vector = chatbot.tag_user(request.prompt)
    chatbot.update_user_profile(tagged_vector)
    tagged_prompt = chatbot.tag_prompt(request.prompt)
    response = chatbot.generate_response(tagged_prompt)
    return {"response": response}

@app.post("/onboarding")
async def receive_onboarding(data: OnboardingData):
    answers = data.answers
    print("Received onboarding answers:", answers)
    tone = chatbot.normalize_letter(answers[0])
    style = chatbot.normalize_letter(answers[1])
    
    if tone == 'informal' and chatbot.cultural_vector[0] > 0.40:
        chatbot.cultural_vector[0] = 0.40
    elif tone == 'formal' and chatbot.cultural_vector[0] < 0.60:
        chatbot.cultural_vector[0] = 0.60
    
    if style == 'exploratory' and chatbot.cultural_vector[3] > 0.40:
        chatbot.cultural_vector[3] = 0.40
    elif style == 'guided' and chatbot.cultural_vector[3] < 0.60:
        chatbot.cultural_vector[3] = 0.60
    
    print(f'cultural vector: {chatbot.cultural_vector}')     
    return {"message": "Onboarding answers received successfully"}

# -------------------------------------------------------


# @app.post("/chat")
# def chat_endpoint(request: PromptRequest):
#     tagged_vector = chatbot.tag_user(request.prompt)
#     chatbot.update_user_profile(tagged_vector)
#     tagged_prompt = chatbot.tag_prompt(request.prompt)
#     response = chatbot.generate_response(tagged_prompt)
#     return {"response": response}

# -------------------------------------------------------

# added script to expose the chatbot class for testing on Colab to an ngrok server
nest_asyncio.apply()

public_url = ngrok.connect(8000)
print(f"Your public URL is: {public_url}")

uvicorn.run(app, host="0.0.0.0", port=8000)
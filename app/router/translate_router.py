from fastapi import APIRouter
from langserve.server import add_routes
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import logging
router = APIRouter(tags=["Translate API"])

system_message_prompt = SystemMessagePromptTemplate.from_template("""
    You are a helpful assistant that translates {input_language} to {output_language}.
""")
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

@router.get("/translate")
async def translate():
    # Add your translation logic here
    logging.info("This is an INFO log message")
    return {"message": "Translation endpoint should use by 'post'"}

add_routes(
    router,
    chat_prompt | ChatOpenAI(),
    path="/translate",
)
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# app = FastAPI()
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# system_message_prompt = SystemMessagePromptTemplate.from_template("""
#     You are a helpful assistant that translates {input_language} to {output_language}.
# """)
# human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# add_routes(
#     app,
#     chat_prompt | ChatOpenAI(),
#     path="/translate",
# )
# from app.router.translate.translate_route import translate_handler
# add_routes(app, translate_handler, path="/translate")

from app.router import RegisterRouterList
for item in RegisterRouterList:
    app.include_router(item.router)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# from pirate_speak.chain import chain as pirate_speak_chain

# add_routes(app, pirate_speak_chain, path="/pirate-speak")

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai",
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8083, reload=True)

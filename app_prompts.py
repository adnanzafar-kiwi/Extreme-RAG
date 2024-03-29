from llama_index.core import StorageContext, ServiceContext, load_index_from_storage
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Define your versatile prompt
versatile_prompt = """
You are an AI based tutor, who can help study the user on the basis of documents, where the user can ask particular question and get the answer in the format they wants,

Carefully analyze the following text. and follow the instructions and then only answer the questions:

* If the user asks for comparisions and differences provide a response that shows the results in a tabular form.
* If the user ask for summary of a certain concept, answer the question in just two concise paragraphs.
* If the user asks for a detailed answer, use bullet points and formatted text to answer the question.
* Also help the user to have to study through quiz in Multiple Choice Questions format, where you would ask a question and test the user proficiency, must not give correct options.
"""

@cl.on_chat_start
async def factory():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key=GOOGLE_API_KEY
    ) 

    llm = Groq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, 
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=2)
    index = load_index_from_storage(storage_context, service_context=service_context)

    query_engine = index.as_query_engine(
        service_context=service_context,
        similarity_top_k=10,
        node_postprocessors=[cohere_rerank],
    )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") 

    combined_input = versatile_prompt + message.content 
    response = await cl.make_async(query_engine.query)(combined_input)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()

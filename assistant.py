from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

import os


# Security Best Practice: Use environment variables for sensitive information
db_user = os.environ.get("DB_USER", "retail-admin")  # Default to 'retail-admin' if not set
db_password = os.environ.get("DB_PASSWORD")  # No default; it MUST be set
db_ip_address = "35.236.0.111" 
db_name = "retail"

if not db_password:
    raise ValueError("DB_PASSWORD environment variable must be set!")

db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_ip_address}:5432/{db_name}"

print(f"Connecting to: {db_url.replace(db_password, '***')}") 

def get_auto_rag_assistant(
    # llm_model: str = "llama3-70b-8192",
    llm_model: str = "llama-3.1-70b-versatile",
    embeddings_model: str = "text-embedding-3-small",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq Auto RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    embeddings_table = (
        "auto_rag_documents_groq_ollama" if embeddings_model == "nomic-embed-text" else "auto_rag_documents_groq_openai"
    )

    return Assistant(
        name="auto_rag_assistant_groq",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="auto_rag_assistant_groq", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 3 references are added to the prompt
            num_documents=3,
        ),
        description="You are Project Blue Bot, an AI assistant specialized in searching and analyzing US government documents to answer questions about UFOs. Your goal is to provide accurate and insightful information on UFO-related topics based on official records.",
        instructions=[
            "Given a user query, first ALWAYS search your knowledge base using the `search_knowledge_base` tool to see if you have relevant information.",
            "If you dont find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the users question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search for information
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        tools=[DuckDuckGo()],
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # Adds chat history to messages
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

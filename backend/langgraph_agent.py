from fastapi.responses import StreamingResponse
import json
import os
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated,Optional
from langgraph.graph.message import  AnyMessage,add_messages
from IPython.display import Image,display
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from pydantic import BaseModel
from fastapi import FastAPI
from langchain_core.messages import HumanMessage,ToolMessage
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
import secrets

load_dotenv()


NEXUS_API_KEY = os.environ.get("NEXUS_API_KEY")
print(f"\n✓ NEXUS API KEY loaded from .env\n")
if not NEXUS_API_KEY:
    raise RuntimeError("NEXUS_API_KEY is not set")

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:8501"],allow_methods=["*"],allow_headers=["*"])

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Check X-Api-Key header on every protected request."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Pass it as header: X-Api-Key: <your-key>"
        )
    if x_api_key != NEXUS_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )

#Tool for Finding Paper
api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv,description="Query arxiv paper")

#Tool for Recent Data search
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=500)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

#Fetching Tavilyapikey
api_key=os.environ.get("TAVILY_API_KEY")
tavily_search=TavilySearchResults(api_key=api_key)
tools=[arxiv,wiki,tavily_search]

llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7
)
llm_with_tools=llm.bind_tools(tools)

#Creating State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

#Building graph and adding nodes
builder=StateGraph(State)
builder.add_node("Tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

#Building edges between nodes
builder.add_edge(START,"Tool_calling_llm")
builder.add_conditional_edges("Tool_calling_llm",tools_condition)
builder.add_edge("tools","Tool_calling_llm")
graph=builder.compile()

#Viewing graph
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    """Public endpoint — no auth needed. Frontend pings this to check if server is up."""
    return {"status": "online"}

@app.post("/verify-key")
def verify_key(x_api_key: Optional[str] = Header(None)):
    """
    Frontend calls this to validate the key before allowing access.
    Returns 200 if valid, 401/403 if not.
    """
    verify_api_key(x_api_key)
    return {"valid": True}

@app.post("/ask_stream")
def ask_stream(req: dict,x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)
    def event_generator():

        for event in graph.stream({
            "messages": [HumanMessage(content=req["query"])]
        }):

            # event contains node updates
            for node, value in event.items():

                msg = value.get("messages", [])[-1]

                # tool call
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    yield json.dumps({
                        "type": "tool",
                        "data": msg.tool_calls
                    }) + "\n"

                # final AI output
                elif hasattr(msg, "content"):
                    yield json.dumps({
                        "type": "answer",
                        "data": msg.content
                    }) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

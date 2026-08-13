from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

SYSTEM_PROMPT = """You are TailorTalk, an AI fashion assistant specializing in sarees.
Your primary role is to help users find sarees that are visually similar to an image they provide.

Rules:
1. When a user asks to find similar sarees, you MUST use the `search_similar_sarees` tool.
2. DO NOT invent or generate fake similarity scores or fake saree names. Only return what the tool gives you.
3. If the user hasn't uploaded an image, ask them to upload one.
4. Present the tool's results clearly and elegantly. Mention the similarity score, name, and price.
5. If the tool returns an error, inform the user politely.
"""


class AgentWrapper:
    """
    Thin wrapper around the LangGraph react agent that exposes an
    .invoke({"input": ...}) interface compatible with app.py.
    """

    def __init__(self, graph):
        self._graph = graph

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        result = self._graph.invoke({"messages": [("human", user_input)]})
        # result["messages"] is a list of BaseMessage objects; last one is the AI reply
        last_msg = result["messages"][-1]
        output = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {"output": output}


def create_tailortalk_agent(tools):
    """
    Creates the TailorTalk agent using the modern LangGraph create_react_agent API.
    """
    # Use Groq for fast, free inference
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    graph = create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)

    return AgentWrapper(graph)

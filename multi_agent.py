from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START,END, add_messages
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Literal
import langchain_openai
import asyncio, json, os
from functools import partial
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
import sys

class ResearchState(TypedDict):
    query          : str
    retrieved_docs : List[str]
    sources        : List[str]  
    summary        : str
    critique       : Dict[str, Any]
    final_answer   : str
    iteration      : int
    answer_type    : str  
 
 
research_llm = init_chat_model(
    model       = 'gpt-4.1-nano',
    temperature = 0.1
)

def router_node(state: ResearchState, research_llm: BaseChatModel) -> Literal["retriever", "direct_answer"]:
    """ 
    Decides whether the query requires retrieval from external knowledge base
 
    Return:
    "retriever" : query needs external documents 
    "direct_answer" : query can be satisfied with common knowledge
    """
 
    prompt = f"""
        You are a query router for a research assistant agent.
 
        You decide if the given query needs document retrieval or can be answered from general knowledge
 
        Query : {state["query"]}
 
        Query -> if asks about specific facts, events, details, niche topic, scientific nitpicks : "RETRIEVE"
        Query -> if asks general information that is conceptual, definitional, or widely known : "DIRECT"
 
        Respond with exactly one word: RETRIEVE or DIRECT
    """
    decision = research_llm.invoke(prompt)
 
    return "retriever" if decision.content.strip() == "RETRIEVE" else "direct_answer"



async def retriever_node(state: ResearchState, search_tool) -> dict:
    """ Fetched the top-k relevant documents pertaining the query.
    
    Reads : state["query"]
    writes : state["retreived_docs"]
    """
 
    query = state["query"]
 
    raw     = await search_tool.ainvoke({"query": query, "k": 5})

    chunks  = json.loads(raw[0]["text"])
    clean_docs = [chunk["content"] for chunk in chunks]

    import os
    sources    = list(set([os.path.basename(chunk["source"]) for chunk in chunks]))

 
    return {
        "retrieved_docs": clean_docs,
        "sources": sources,
        "answer_type": "retrieved"
    }



def summarizer_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """ Condenses retreived documents into structured key points
    
    Reads : state["retrieved_docs"]
    Write:  state["summary]
    
    """
 
    docs    = state["retrieved_docs"]
    context = "\n\n---\n\n".join(docs)
 
    prompt = f"""
        You are a research summarizer
 
        Summarixe the following source documents clearly, into structured key points that directly address the user's query.
 
        Source documents:
        {context}
 
        Instructions:
        1) Extract only most relevant information
        2) Remove redundancy across documents
        3) Organize by topic or importance
        4) Be concise but complete
        """
    
    summary = research_llm.invoke(prompt).content
 
    return{
        "summary" : summary
    }



class CritiqueOutput(BaseModel):
    score         : float     = Field(ge=0.0, le=1.0)
    issues        : List[str]
    needs_revision: bool
 
 
def critic_node(state: ResearchState, research_llm: BaseChatModel) -> dict:
    score         : float     = Field(ge=0.0, le=1.0, description="Qualit score from 0 to 1")
    issues        : List[str] = Field(description="List of specific problems with the summary")
    needs_revision: bool      = Field(description="True if summary should be revised, else False")
    """ 
    Evaluate the summary and return a structured critic
 
    Reads:  state["query"], state["summary"], state["iteration"]
    Writes: state["critique"], state["iteration"]
    
    """
    query   = state["query"]
    summary = state["summary"]
    prompt  = f"""
        You are a ciritical evaluator for a research assistant.
        Evaluate the following summary against the user's original query
        Query: {query}
 
    Summary : {summary}
 
Assess the summary on :
1) Relevance : does it answer the query?
2) Completeness: are key points missing?
3) Accuracy: are any claims unsupported or vague?
4) Clarity: is it well-structured and easy to read?
 
Be specific, List concrete issues, not generic complaints
 
 """
    
    structured_llm = research_llm.with_structured_output(CritiqueOutput)
    critic : critic_node = structured_llm.invoke(prompt)
 
    return {
        "critique" : critic.model_dump(),
        "iteration": state.get("iteration", 0) + 1
    }
 
 
MAX_ITERATIONS = 3
 
def should_refine(state: ResearchState) -> Literal["refine", "synthesizer"]:
    """
    Decides whether to loop back through refine_node or proceed to synthesizer.
 
    Returns:
        "refine"      → critic flagged issues and we haven't hit the iteration cap
        "synthesizer" → quality is good enough, or we've run out of attempts
    """
    critique  = state["critique"]
    iteration = state["iteration"]
 
    if critique["needs_revision"] and iteration < MAX_ITERATIONS:
        return "refine"
 
    return "synthesizer"



def refine_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Rewrites the summary using the critic's specific feedback.
 
    Reads:  state["summary"], state["critique"]
    Writes: state["summary"]  (overwrites with improved version)
    """
    summary  = state["summary"]
    issues   = state["critique"]["issues"]
    score    = state["critique"]["score"]
 
    formatted_issues = "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))
 
    prompt = f"""
    You are a research editor improving a summary based on reviewer feedback.
 
    Current quality score: {score:.2f} / 1.0
 
    Issues identified by the critic:
    {formatted_issues}
 
    Original summary:
    {summary}
 
    Instructions:
    - Fix each issue listed above
    - Do not remove content that was already correct
    - Keep the structure clean and well-organised
    - Do not add unsupported claims
    """
 
    improved_summary = llm.invoke(prompt).content
 
    return {
        "summary": improved_summary
    }


def synthesizer_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Produces the final structured answer from the polished summary.
 
    Reads:  state["query"], state["summary"]
    Writes: state["final_answer"]
    """
    query   = state["query"]
    summary = state["summary"]
 
    prompt = f"""
    You are a research assistant writing a final answer for a user.
 
    User's query: {query}
 
    Research summary:
    {summary}
 
    Instructions:
    - Answer the query directly and completely
    - Use the summary as your source of truth
    - Use clear headings if the answer has multiple parts
    - Cite the key points from the summary where relevant
    - Be concise — no padding or repetition
    """
 
    final_answer = llm.invoke(prompt).content
 
    return {
        "final_answer": final_answer
    }


def direct_answer_node(state: ResearchState, llm: BaseChatModel) -> dict:
    """
    Answers the query directly from LLM knowledge, skipping retrieval.
    Writes to state["summary"] so synthesizer_node handles final formatting.
 
    Reads:  state["query"]
    Writes: state["summary"]
    """
    query = state["query"]
 
    prompt = f"""
    You are a knowledgeable research assistant.
 
    Answer the following query from your general knowledge.
    Be accurate, structured, and thorough.
 
    Query: {query}
    """
 
    answer = llm.invoke(prompt).content
 
    return {
        "summary": answer,
        "sources": [],  
        "answer_type": "direct"

    }



async def build_and_run(query: str) -> str:
 
    client =  MultiServerMCPClient({
        "research_rag": {
            "command"  : "python",
            "args"     : ["mcp-rag-server.py"],  
            "transport": "stdio"
        }
    }) 
    tools       = await client.get_tools()
    search_tool = next(t for t in tools if t.name == "search_papers")

    builder = StateGraph(ResearchState)

    builder.add_node("retriever",     partial(retriever_node,     search_tool=search_tool))
    builder.add_node("direct_answer", partial(direct_answer_node, llm=research_llm))        
    builder.add_node("summarizer",    partial(summarizer_node,    llm=research_llm))        
    builder.add_node("critic",        partial(critic_node,        research_llm=research_llm))  
    builder.add_node("refine",        partial(refine_node,        llm=research_llm))        
    builder.add_node("synthesizer",   partial(synthesizer_node,   llm=research_llm))        

    builder.add_conditional_edges(START, partial(router_node, research_llm=research_llm))   

    builder.add_edge("retriever",     "summarizer")    
    builder.add_edge("summarizer",    "critic")        
    builder.add_conditional_edges("critic", should_refine)  
    builder.add_edge("refine",        "critic")        
    builder.add_edge("direct_answer", "synthesizer")   
    builder.add_edge("synthesizer",   END)             

    graph = builder.compile()

    result = await graph.ainvoke({"query": query})
    return {
    "final_answer": result["final_answer"],
    "sources"     : result.get("sources", []),
    "answer_type" : result.get("answer_type", "unknown"),
    "iterations"  : result.get("iteration", 0)
}


 
if __name__ == "__main__":
    query  = "what are the resolution levels a tumor cell microscopy is captured with?"
    answer = asyncio.run(build_and_run(query))
    print(answer)
 
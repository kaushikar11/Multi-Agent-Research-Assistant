from dotenv import load_dotenv
load_dotenv()
 
import os, json, asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import sys
 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import FAISS_INDEX_PATH, EMBEDDING_MODEL
 
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_db  = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


app = Server("research-rag-server")
 
 
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_papers",
            description=(
                "Search the research paper knowledge base and return "
                "the most relevant text chunks for a given query. "
                "Use this whenever the query needs specific facts, "
                "details, or findings from the ingested papers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
 
        Tool(
            name="search_with_threshold",
            description=(
                "Search papers and only return results above a relevance "
                "score threshold. Returns empty if no good matches found. "
                "Use this when you want quality-controlled retrieval and "
                "would rather get nothing than a bad result."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum relevance score 0.0-1.0 (default 0.7)",
                        "default": 0.7
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max chunks to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]
 
 
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_papers":
        query = arguments["query"]
        k     = arguments.get("k", 5)
        print(f"[search_papers] query='{query}' k={k}", file=sys.stderr, flush=True)
        docs  = vector_db.similarity_search(query, k=k)
        print(f"[search_papers] returned {len(docs)} chunks", file=sys.stderr, flush=True)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source" : doc.metadata.get("source", "unknown"),
                "page"   : doc.metadata.get("page", "?")
            })
 
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2)
        )]
    elif name == "search_with_threshold":
            query     = arguments["query"]
            threshold = arguments.get("threshold", 0.7)
            k         = arguments.get("k", 5)
            print(f"[search_with_threshold] query='{query}' threshold={threshold} k={k}", file=sys.stderr, flush=True)
            results_with_scores = vector_db.similarity_search_with_relevance_scores(query, k=k)
            filtered = [(doc, score) for doc, score in results_with_scores if score >= threshold]
            print(f"[search_with_threshold] {len(filtered)}/{len(results_with_scores)} chunks passed threshold", file=sys.stderr, flush=True)
        
    
            results_with_scores = vector_db.similarity_search_with_relevance_scores(
                query, k=k
            )
    
            filtered = [
                (doc, score)
                for doc, score in results_with_scores
                if score >= threshold
            ]
    
            if not filtered:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "found"  : False,
                        "message": f"No results above threshold {threshold}",
                        "results": []
                    })
                )]
    
            results = []
            for doc, score in filtered:
                results.append({
                    "content": doc.page_content,
                    "source" : doc.metadata.get("source", "unknown"),
                    "page"   : doc.metadata.get("page", "?"),
                    "score"  : round(float(score), 4)
                })
    
            return [TextContent(
                type="text",
                text=json.dumps({"found": True, "results": results}, indent=2)
            )]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
 
 
asyncio.run(main())
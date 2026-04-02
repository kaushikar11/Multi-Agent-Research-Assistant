# Research Assistant

A multi-agent RAG system for querying research papers. Built with LangGraph, FAISS, and MCP.

---

## Architecture

```
papers/                  Raw PDF documents
ingest.py                Loads PDFs, chunks text, builds FAISS index
mcp_rag_server.py        Serves FAISS index as MCP tools over stdio
multi_agent.py           LangGraph agent graph — connects to MCP server
app.py                   Streamlit web interface
```

**Agent pipeline**

```
query
  router          decides: retrieve from papers or answer directly
  retriever       calls MCP tool, fetches top-k chunks from FAISS
  summarizer      condenses retrieved chunks into key points
  critic          scores the summary, identifies issues
  refine          rewrites summary based on critic feedback (max 3x)
  synthesizer     produces the final structured answer
```

**MCP tools exposed by the server**

| Tool | Description |
|---|---|
| `search_papers` | Returns top-k relevant chunks for a query |
| `search_with_threshold` | Returns chunks above a relevance score threshold |

---

## Setup

**1. Install dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Set environment variables**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

**3. Ingest documents**

Place PDF files in the `papers/` folder, then run:

```bash
python ingest.py
```

This creates the `faiss_index/` directory. Run once, or re-run whenever you add new papers.

---

## Running

**Streamlit UI**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Command line**

```bash
python multi_agent.py
```

Runs with the hardcoded query at the bottom of the file. Change it to test different queries.

---

## Claude Desktop Integration

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-rag": {
      "command": "/absolute/path/to/venv/bin/python3",
      "args": ["/absolute/path/to/mcp_rag_server.py"]
    }
  }
}
```

Use absolute paths. Restart Claude Desktop after saving. The tools `search_papers` and `search_with_threshold` will appear in the tools menu and Claude can query your papers directly in conversation.

---

## Requirements

```
python-dotenv
langchain
langchain-community
langchain-core
langchain-openai
langchain-text-splitters
langgraph
langchain-mcp-adapters
mcp
pydantic
faiss-cpu
pypdf
pymupdf
streamlit
ipykernel
```

---

## Notes

- The MCP server must be started from the project root directory, or FAISS index paths must be absolute. The current implementation uses absolute paths derived from the file location so it works from any directory.
- All logs from the MCP server are written to stderr and will not interfere with the MCP JSON-RPC channel on stdout.
- The Pydantic V1 warnings visible on Python 3.14 are from upstream dependencies and do not affect functionality.
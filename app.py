import streamlit as st
import asyncio
from multi_agent import build_and_run

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Research Assistant")
st.caption("Powered by LangGraph · FAISS · MCP")
st.divider()

query = st.text_area(
    label="Ask a question about your papers",
    placeholder="e.g. What resolution levels are tumor cell microscopy images captured at?",
    height=100
)

run = st.button("Ask", use_container_width=True, type="primary")

if run:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking through your papers..."):
            result = asyncio.run(build_and_run(query))

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if result["answer_type"] == "retrieved":
                st.success("📚 From Papers")
            else:
                st.info("💡 General Knowledge")

        with col2:
            itr = result["iterations"]
            if itr == 0:
                st.metric("Revisions", "None")
            else:
                st.metric("Revisions", itr)

        with col3:
            src_count = len(result["sources"])
            st.metric("Sources", src_count if src_count > 0 else "None")

        if result["sources"]:
            st.divider()
            st.markdown("**Sources**")
            for s in result["sources"]:
                st.markdown(f"📄 `{s}`")
        else:
            st.divider()
            st.markdown("**Sources**")
            st.caption("No papers retrieved — answered from general knowledge.")

        st.divider()
        st.markdown("### Answer")
        st.markdown(result["final_answer"])

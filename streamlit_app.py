import requests
import streamlit as st

st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="📚",
)

API_URL = "http://127.0.0.1:8000/chat"

# PyPDFLoader numbers pages from 0. Set to 0 if you want the raw value.
PAGE_OFFSET = 1

st.title("Agentic AI RAG Chatbot")
st.caption("Ask questions about the Agentic AI eBook")

with st.form("chat_form"):
    query = st.text_input(
        "Question",
        placeholder="What is Agentic AI according to the eBook?",
    )

    submitted = st.form_submit_button(
        "Ask",
        type="primary",
    )

if submitted:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json={"query": query.strip()},
                timeout=120,
            )

        if response.status_code != 200:
            st.error(f"FastAPI returned HTTP {response.status_code}")
            st.code(response.text)
            st.stop()

        data = response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI is not running.\n\n"
            "Start it with:\n"
            "`uv run uvicorn app:app --reload`"
        )
        st.stop()

    except requests.exceptions.Timeout:
        st.error("FastAPI request timed out.")
        st.stop()

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    st.subheader("Answer")
    st.write(data.get("final_answer", ""))

    confidence = float(data.get("confidence_score", 0.0) or 0.0)
    support = data.get("answer_support")

    if support is None:
        st.metric(
            "Retrieval Confidence",
            f"{confidence:.2f}",
            help="How closely the retrieved passages match the question. "
            "It does not measure whether the answer is correct.",
        )
    else:
        left, right = st.columns(2)

        left.metric(
            "Retrieval Confidence",
            f"{confidence:.2f}",
            help="How closely the retrieved passages match the question. "
            "It does not measure whether the answer is correct.",
        )

        right.metric(
            "Answer Support",
            f"{float(support):.2f}",
            help="Share of the answer's key terms found in the retrieved "
            "passages. A low value suggests the answer added details.",
        )

    st.subheader("Retrieved Context")

    chunks = data.get("retrieved_context_chunks", [])

    if not chunks:
        st.info("No context was retrieved.")
    else:
        for i, chunk in enumerate(chunks, 1):
            if isinstance(chunk, str):
                text = chunk
                page = "-"
                section = ""
                score = None
            else:
                text = chunk.get("text", "")
                section = chunk.get("section", "")
                score = chunk.get("semantic_score", chunk.get("final_score"))

                raw_page = chunk.get("page")
                page = (
                    raw_page + PAGE_OFFSET
                    if isinstance(raw_page, int)
                    else "-"
                )

            label = f"Chunk {i} · Page {page}"

            if score is not None:
                label += f" · Similarity {float(score):.2f}"

            with st.expander(label):
                if section:
                    st.caption(f"Section: {section}")

                st.write(text)
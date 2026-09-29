import streamlit as st

from src.graph import rag_graph


st.set_page_config(
    page_title="Agentic AI RAG",
    page_icon="📚",
)

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
            result = rag_graph.invoke(
                {"question": query.strip()}
            )

    except Exception as exc:
        st.error("Failed to process the question.")
        st.exception(exc)
        st.stop()

    st.subheader("Answer")
    st.write(result.get("answer", ""))

    confidence = float(
        result.get("score", 0.0) or 0.0
    )

    support = result.get("answer_support")

    if support is None:
        st.metric(
            "Retrieval Confidence",
            f"{confidence:.2f}",
            help=(
                "How closely the retrieved passages match "
                "the question. It does not measure whether "
                "the answer is correct."
            ),
        )
    else:
        left, right = st.columns(2)

        left.metric(
            "Retrieval Confidence",
            f"{confidence:.2f}",
            help=(
                "How closely the retrieved passages match "
                "the question. It does not measure whether "
                "the answer is correct."
            ),
        )

        right.metric(
            "Answer Support",
            f"{float(support):.2f}",
            help=(
                "Share of the answer's key terms found in "
                "the retrieved passages."
            ),
        )

    st.subheader("Retrieved Context")

    chunks = result.get(
        "retrieved_context_chunks",
        [],
    )

    if not chunks:
        st.info("No context was retrieved.")
    else:
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            section = chunk.get("section", "")
            score = chunk.get(
                "semantic_score",
                chunk.get("final_score"),
            )

            raw_page = chunk.get("page")

            page = (
                raw_page + PAGE_OFFSET
                if isinstance(raw_page, int)
                else "-"
            )

            label = f"Chunk {i} · Page {page}"

            if score is not None:
                label += (
                    f" · Similarity {float(score):.2f}"
                )

            with st.expander(label):
                if section:
                    st.caption(
                        f"Section: {section}"
                    )

                st.write(text)
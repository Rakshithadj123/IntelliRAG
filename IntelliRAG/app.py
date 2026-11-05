
import streamlit as st
import ollama

from ollama_engine import read_pdf, build_vector_store
from agents import run_intellirag


st.set_page_config(
    page_title="IntelliRAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IntelliRAG")
st.subheader(
    "Multi-Agent Research Assistant Using Agentic RAG"
)

st.write(
    "Upload a PDF and ask questions. "
    "AI responses are generated locally through Ollama."
)


def get_installed_models():
    """List chat models installed in Ollama."""
    response = ollama.list()
    names = []

    for item in response.models:
        name = getattr(item, "model", None)

        if not name and isinstance(item, dict):
            name = item.get("model") or item.get("name")

        if name:
            names.append(name)

    return sorted(set(names))


# Session state stores data temporarily.
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None


with st.sidebar:
    st.header("Settings")

    if st.button("Refresh installed models"):
        st.rerun()

    try:
        installed_models = get_installed_models()

        if installed_models:
            model_name = st.selectbox(
                "Select an installed Ollama model",
                installed_models
            )
        else:
            model_name = None
            st.warning(
                "No installed models found. "
                "Download a chat model using Ollama first."
            )

    except Exception as error:
        model_name = None

        st.error(
            "Cannot connect to Ollama. "
            "Check that Ollama is installed and running."
        )

        st.caption(str(error))

    st.divider()

    st.caption(
        "PDF text and chat history are stored temporarily "
        "in application memory."
    )


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=False,
    help="Choose a PDF from your computer."
)


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # Identify a newly selected file.
    import hashlib

    current_file_id = hashlib.sha256(file_bytes).hexdigest()

    if current_file_id != st.session_state.processed_file_id:
        st.info(f"Selected file: {uploaded_file.name}")

        if st.button("Process PDF", type="primary"):
            try:
                with st.spinner("Reading and processing PDF..."):
                    pdf_text = read_pdf(file_bytes)
                    chunks = build_vector_store(pdf_text)

                    st.session_state.document_chunks = chunks
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.processed_file_id = current_file_id
                    st.session_state.chat_history = []

                st.success(
                    f"PDF processed successfully! "
                    f"Created {len(chunks)} text chunks."
                )

            except Exception as error:
                st.error(f"PDF processing failed: {error}")

    else:
        st.success(
            f"Processed document: {st.session_state.document_name}"
        )


if st.session_state.document_chunks:
    st.divider()

    st.subheader("Ask your document")

    with st.form("question_form"):
        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: Explain heaps and heap sort"
            )
        )

        submitted = st.form_submit_button(
            "Get Answer",
            type="primary"
        )

    if submitted:
        if not model_name:
            st.warning(
                "Select an installed Ollama model first."
            )

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    f"Researching with {model_name}..."
                ):
                    answer = run_intellirag(
                        question.strip(),
                        st.session_state.document_chunks,
                        model_name
                    )

                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "answer": answer,
                    "model": model_name
                })

            except Exception as error:
                st.error(f"Could not generate an answer: {error}")


if st.session_state.chat_history:
    st.divider()
    st.header("Research Results")

    for item in reversed(st.session_state.chat_history):
        with st.expander(
            item["question"],
            expanded=True
        ):
            st.caption(f"Model: {item['model']}")
            st.markdown(item["answer"])

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

else:
    if not st.session_state.document_chunks:
        st.info(
            "Upload a PDF and click Process PDF to get started."
        )

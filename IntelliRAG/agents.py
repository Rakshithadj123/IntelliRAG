
import ollama

from ollama_engine import retrieve_context


def call_model(model_name, prompt):
    """Generate a response using the selected Ollama model."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response["message"]["content"]


def planner_agent(question, model_name):
    """Plan how to answer the question."""
    prompt = f"""
You are the planning agent in IntelliRAG.

Create a short plan for answering the question.
Identify the important concepts to look for.

Question:
{question}

Return a short numbered plan.
"""
    return call_model(model_name, prompt)


def retrieval_agent(question, document_chunks):
    """Retrieve passages from the uploaded PDF."""
    return retrieve_context(question, document_chunks)


def grader_agent(question, context, model_name):
    """Evaluate whether retrieved text is relevant."""
    prompt = f"""
You are the relevance-checking agent.

Question:
{question}

Retrieved document passages:
{context}

Does the context contain useful information for answering
the question?

Explain briefly. Do not generate the final answer.
"""
    return call_model(model_name, prompt)


def generator_agent(question, context, plan, model_name):
    """Generate an answer grounded in the retrieved text."""
    prompt = f"""
You are the answer-generation agent in IntelliRAG.

Answer the question using the document passages below.
Follow the research plan where useful.

Rules:
- Use simple, clear language.
- Explain concepts step by step when appropriate.
- Use headings and bullet points when helpful.
- Do not claim that the document says something it does not.
- If the answer is missing from the passages, state this clearly.

Research plan:
{plan}

Question:
{question}

Document passages:
{context}

Write the answer.
"""
    return call_model(model_name, prompt)


def verifier_agent(question, context, draft, model_name):
    """Check the draft against the retrieved passages."""
    prompt = f"""
You are the answer-verification agent.

Check the draft against the document passages.
Correct unsupported claims and remove invented facts.

If the passages do not provide enough information, explain
that limitation. Do not add outside facts as if they came
from the uploaded PDF.

Question:
{question}

Document passages:
{context}

Draft answer:
{draft}

Return the improved final answer only.
"""
    return call_model(model_name, prompt)


def run_intellirag(question, document_chunks, model_name):
    """Run the complete IntelliRAG workflow."""
    if not document_chunks:
        return "Please upload and process a PDF first."

    plan = planner_agent(question, model_name)

    context = retrieval_agent(question, document_chunks)

    if not context.strip():
        return (
            "No readable passages were found. "
            "Please process the PDF again."
        )

    # The grader evaluates relevance, but does not block
    # the answer automatically.
    relevance_review = grader_agent(
        question, context, model_name
    )

    draft = generator_agent(
        question, context, plan, model_name
    )

    final_answer = verifier_agent(
        question, context, draft, model_name
    )

    return (
        f"{final_answer}\n\n"
        "---\n"
        "**Research workflow completed**\n\n"
        f"**Relevance review:** {relevance_review}"
    )

"""Named prompt-template versions.

Each version is a function (query, context) -> prompt string. Keeping
them here, named and versioned, lets us:
  - A/B compare them directly (app/main.py's /ask/compare)
  - reference a specific version in eval baselines (Phase 2) so
    "did quality change" questions are answerable as "compared to
    which prompt version's baseline?"

When build_prompt() in app/rag.py changes, add a new version here rather
than editing in place - keeps old versions runnable for comparison.
"""

PROMPT_VERSIONS = {
    "v1_verbose": (
        "You are a helpful assistant answering questions based on the "
        "user's personal notes. Use ONLY the context below to answer the "
        "question. If the context doesn't contain the answer, say so "
        "clearly rather than guessing.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
    "v2_concise_2sentence": (
        "You are a helpful assistant answering questions based on the "
        "user's personal notes. Use ONLY the context below to answer the "
        "question. If the context doesn't contain the answer, say so "
        "clearly rather than guessing.\n\n"
        "Answer in AT MOST 2 sentences. Be direct - state the answer "
        'first, with no preamble like "Based on the context" or '
        '"According to the notes".\n\n'
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
}

# Which version build_prompt() in app/rag.py currently uses - kept in
# sync manually for now. Phase 2's eval harness reads this to label
# results with the active prompt version.
CURRENT_VERSION = "v1_verbose" # "v1_verbose" or "v2_concise_2sentence"


def render_prompt(version: str, query: str, context: str) -> str:
    template = PROMPT_VERSIONS[version]
    return template.format(query=query, context=context)

"""
Ollama LLM wrapper.
"""

import ollama


class OllamaLLM:
    """Simple wrapper around a local Ollama model."""

    def __init__(self, model: str = "llama3.2") -> None:
        self.model = model

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are a Retrieval-Augmented Generation assistant.

Answer ONLY from the supplied context.

If the answer is not present, reply:
"I could not find the answer in the indexed documents."

Context:
{context}

Question:
{question}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return str(response["message"]["content"])
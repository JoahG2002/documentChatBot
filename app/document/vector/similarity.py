from numpy import float32
from numpy._typing import NDArray

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def process_questions(questions: list[str]) -> tuple[list[str], list[str]]:
    """
    Strips all command-line question off any whitespace, and returns the clean original question with the rephrased ones as a tuple.
    """
    rephrased_questions: list[str] = [""] * (len(questions) - 1)

    for i in range(len(questions) - 1):
        rephrased_questions[i] = questions[i].strip()

    return [questions[-1].strip()], rephrased_questions


def calculate_question_average_similarity(original_question: list[str], rephrased_questions: list[str]) -> float:
    """
    Calculates the average similarity between the original question and a list of related questions. The original question is the reference to compare against alternatives. The average similarity between the original question and all questions in the list is returned.
    """
    sentence_transformer_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    original_embedding: NDArray[float32] = sentence_transformer_model.encode(original_question)[0]
    sentence_embeddings: NDArray[float32] = sentence_transformer_model.encode(rephrased_questions)

    return cosine_similarity([original_embedding], sentence_embeddings)[0].mean()

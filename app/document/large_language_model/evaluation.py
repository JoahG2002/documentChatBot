

class Evaluation:
    __slots__: tuple[str, ...] = ("timestamp",)

    def __init__(self, timestamp: int = 0) -> None:
        self.timestamp: int = timestamp


class QueryResponseEvaluation:
    __slots__: tuple[str, ...] = (
        "timestamp", "large_language_model", "model_name", "large_language_model_token_limit", "sentence_embedding_model",
        "dimension_count_vectors", "top_k_similar_documents", "pdf_page_size", "instruction_text", "query", "llm_response",
        "evaluation", "pdf_document_name"
    )

    def __init__(self,
                 timestamp: int = 0,
                 pdf_page_size: int = 0,
                 dimension_count_vectors: int = 0,
                 top_k_similar_documents: int = 0,
                 large_language_model: str = "",
                 model_name: str = "",
                 pdf_document_name: str = "",
                 sentence_embedding_model: str = "",
                 instruction_text: str = "",
                 query: str = "",
                 llm_response: str = "",
                 large_language_model_token_limit: int = 0) -> None:
        self.timestamp: int = timestamp
        self.pdf_document_name: str = pdf_document_name
        self.large_language_model: str = large_language_model
        self.model_name: str = model_name
        self.large_language_model_token_limit: int = large_language_model_token_limit
        self.sentence_embedding_model: str = sentence_embedding_model
        self.dimension_count_vectors: int = dimension_count_vectors
        self.top_k_similar_documents: int = top_k_similar_documents
        self.pdf_page_size: int = pdf_page_size
        self.instruction_text: str = instruction_text
        self.query: str = query
        self.llm_response: str = llm_response
        self.evaluation: Evaluation | None = None

    def set_evaluation(self) -> None:
        """
        Sets/constructs the evaluation of the query response.
        """
        self.evaluation = None



class ExperimentRound:
    __slots__: tuple[str, ...] = ("index_query", "page_size", "top_k")

    def __init__(self, index_query: int, page_size: int, top_k: int) -> None:
        self.index_query: int = index_query
        self.page_size: int = page_size
        self.top_k: int = top_k


def generate_experiment_rounds(number_of_questions: int = 13,
                               pages_sizes: tuple[int, ...] = (512, 1000),
                               top_k_s: tuple[int, ...] = (3, 5, 10)) -> list[ExperimentRound]:
    """
    Generates all possible combinations required for the LLM experiment.
    """
    i: int = 0
    experiment_rounds: list[ExperimentRound | None] = [None] * (number_of_questions * len(pages_sizes) * len(top_k_s))

    for index_query in range(1, (number_of_questions + 1)):
        for page_size in pages_sizes:
            for top_k in top_k_s:
                experiment_rounds[i] = ExperimentRound(index_query=index_query, page_size=page_size, top_k=top_k)

                i += 1

    return experiment_rounds

import os
import sys
import fitz

import polars as pl

from typing import Optional

from ..constant.constant import limits, directories
from ..document.vector.meta import VectorMetaData, split_text_into_chunks
from ..document.large_language_model.evaluation import QueryResponseEvaluation


def pdf_to_text_pages(pdf_filepath: str) -> Optional[list[VectorMetaData]]:
    """
    Receives the file path of a PDF file, and returns the text of its pages as a list of strings.
    """
    if (not os.path.exists(pdf_filepath)):
        sys.stderr.write(f"[ERROR] PDF file does not exists.\n")

        return None

    pdf_document_pages: list[VectorMetaData] = []

    with fitz.open(pdf_filepath) as pdf_document:
        pdf_filename: str = os.path.basename(pdf_filepath)

        for page_number, page in enumerate(pdf_document, start=1):
            for chunk in split_text_into_chunks(
                text=page.get_text(),
                chunk_size=limits.PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT
            ):
                pdf_document_pages.append(
                    VectorMetaData(content=chunk, page_number=page_number, filename=pdf_filename)
                )

    return pdf_document_pages


def read_query_response_evaluations(large_language_model: str, model_name: str) -> Optional[tuple[QueryResponseEvaluation, ...]]:
    """
    Reads a query response evaluations of certain large language model's model's (ChatGPT-4o, for example) with their metadata.
    """
    return pl.read_csv(f"{directories.LARGE_LANGUAGE_MODEL_RESPONSES}/{large_language_model}{model_name}")


# def read_experiment_rounds() -> tuple[ExperimentRound, ...]:
#     """
#     Reads the data of the current experiment round.
#     """
#     with open("combinatiesExperiment.txt", 'r') as experiment_rounds_file:
#         current_experiment_rounds: list[str] = experiment_rounds_file.readlines()
#
#     experiment_rounds: list[ExperimentRound | None] = [None] * (limits.INDEX_LAST_TASK + 1)
#
#     for i, current_experiment_round in enumerate(current_experiment_rounds):
#         round_elements: list[str] = current_experiment_round.split(',')
#
#         experiment_rounds[i] = ExperimentRound(
#             question=large_language_models.QUESTIONS[int(round_elements[0])],
#             large_language_model=round_elements[1],
#             page_size=int(round_elements[2]),
#             top_k=int(round_elements[3])
#         )
#
#     return tuple(experiment_rounds)

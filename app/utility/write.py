import os
import sys
import time
import json
import random

from typing import Optional, Any

from ..constant.constant import directories, default_texts, limits


def clear_terminal() -> None:
    """
    Clear all output of the terminal in which the current program is run.
    """
    os.system("clear" if (os.name == "posix") else "cls")


def format_vector_database_path(pdf_file_path: str, page_size: Optional[int] = None) -> str:
    """
    Returns a reconstructable Pickle database file path based on a certain PDF file's name.
    """
    base_database_file_path: str = f"{directories.PDF_VECTOR_DATABASES}/{os.path.basename(pdf_file_path).lower()[:-4]}"

    return f"{base_database_file_path}_chunk_{limits.PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT if (not page_size) else page_size}.pkl"


def generate_file_id() -> str:
    """
    Generates a unique file ID (a random integer, cast to a string).
    """
    return str(random.randint(0, 15_000_000))


def write_llm_query_response_to_csv(pdf_document_name: str,
                                    large_language_model: str,
                                    model_name: str,
                                    sentence_embedding_model: str,
                                    dimension_count_vectors: int,
                                    top_k_similar_documents: int,
                                    pdf_page_size: int,
                                    prompt_length: int,
                                    instruction_text: str,
                                    top_k_documents: str,
                                    query: str,
                                    llm_response: str,
                                    token_limit: int) -> None:
    """
    Stores large language model's response to a certain query, with its metadata.
    """
    filepath: str = f"{directories.LARGE_LANGUAGE_MODEL_RESPONSES}/chatResults.csv"

    write_mode: str = ('w' if (not os.path.exists(filepath)) else 'a')

    with open(filepath, write_mode, encoding="utf-8") as model_performance_csv:
        if (write_mode == 'w'):
            model_performance_csv.write(default_texts.MODEL_PERFORMANCE_CSV_COLUMNS_ROW)

        model_performance_csv.write(
            f"\n{int(time.time())},{pdf_document_name},{large_language_model},{model_name},{sentence_embedding_model},"
            f"{dimension_count_vectors},{top_k_similar_documents},{pdf_page_size},\"{instruction_text.strip()}\","
            f"\"{query}\",\"{top_k_documents}\",\"{llm_response}\",{token_limit},{prompt_length},{len(instruction_text)},"
            f"{len(query)},null,null,null,null,null,{len(llm_response)},null\n"  # calculated later
        )

    sys.stdout.write(f"\n[SAVE] LLM response successfully written to: {filepath}\n\n")


def store_experiment_round(
        large_language_model: str,
        model_name: str,
        prompt: str,
        sentence_embedding_model: str,
        instruction_context: str,
        dimension_count_vectors: int,
        page_size: int,
        top_k: int,
        token_limit: int,
        evaluation_response_json: str,
        query: str,
        top_k_documents: str,
        llm_response: str,
        pdf_document: str) -> None:
    """
    Writes experiment details, documents, LLM responses, and evaluation to a JSON text file.
    """
    now: int = int(time.time())
    target_file_path: str = f"./data/experimentsEvaluations/{large_language_model}_{page_size}_{top_k}_{now}.json"

    try:
        json_body_evaluation: dict[str, Any] = json.loads(evaluation_response_json)

    except Exception as e:
        _ = e
        json_body_evaluation = {
            "sourced_cited_correctly": 0,
            "faithfulness_score": 0.0,
            "context_relevance": 0.0,
            "practicality_score": 0.0
        }

    experiment_data: dict[str, Any] = {
        "timestamp": now,
        "dimension_count_vectors": dimension_count_vectors,
        "pdf_document": pdf_document,
        "instruction_context": instruction_context,
        "large_language_model": large_language_model,
        "sentence_embedding_model": sentence_embedding_model,
        "model_name": model_name,
        "page_size": page_size,
        "top_k": top_k,
        "token_limit": token_limit,
        "prompt_length": len(prompt),
        "instruction_length": len(instruction_context),
        "query_length": len(query),
        "evaluation_response": json_body_evaluation,
        "query": query,
        "top_k_documents": top_k_documents.strip(),
        "llm_response": llm_response,
        "llm_response_length": len(llm_response),
        "rephrased_questions": None
    }

    with open(target_file_path, 'w', encoding="utf-8") as experiment_data_file:
        json.dump(experiment_data, experiment_data_file, indent=4, ensure_ascii=False)

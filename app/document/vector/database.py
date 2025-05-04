import os
import sys
import pickle

import numpy as np
from numpy import float32, int64
from numpy.typing import NDArray
from typing import Optional, Literal
from sentence_transformers import SentenceTransformer

from app.utility.read import pdf_to_text_pages
from app.utility.write import format_vector_database_path
from app.document.vector.meta import VectorMetaData, SimilarVector
from app.constant.constant import return_codes, large_language_models, limits


class PDFVectorDatabase:
    __slots__: tuple[str, ...] = (
        "__database_path", "__pdf_file_path", "__model", "__expected_maximum_vector_index", "__current_vector_index",
        "__vectors", "__metadata_vectors", "__top_k", "__page_size"
    )

    def __init__(self,
                 vector_count: int,
                 vector_dimensions: int,
                 pdf_file_path: str,
                 model_name: str = large_language_models.DEFAULT_SENTENCE_EMBEDDING_MODEL,
                 page_size: int | None = None,
                 top_k: int | None = None) -> None:
        self.__database_path: str = format_vector_database_path(pdf_file_path, page_size=page_size)
        self.__pdf_file_path: str = pdf_file_path

        self.__model: SentenceTransformer = SentenceTransformer(model_name)

        self.__expected_maximum_vector_index: int = (vector_count - 1)
        self.__current_vector_index: int = 0
        self.__vectors: NDArray[float32] = np.zeros(shape=(vector_count, vector_dimensions), dtype=float32)
        self.__metadata_vectors: list[VectorMetaData | None] = [None] * vector_count

        self.__top_k: int | None = limits.TOP_K_RELEVANT_DOCUMENT_SEGMENTS if (not top_k) else top_k
        self.__page_size: int | None = limits.PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT if (not page_size) else page_size

        self._load_database()

    def embed_text(self, text: str) -> NDArray[float32]:
        """
        Generates an embedding vector for a given text.
        """
        return self.__model.encode(text)

    def _insert_is_possible(self, vector_shape: tuple[int, ...]) -> None:
        """
        Raises an error based if a vector insert is not possible — based on the allocated memory for the vectors and the newly added vectors dimensions.
        """
        if (self.__current_vector_index >= self.__expected_maximum_vector_index):
            raise IndexError("Expected vector count exceeded! Increase memory allocation / expected_vector_count.")

        if (vector_shape != self.__vectors[self.__current_vector_index].shape):
            raise ValueError(f"Vector shape mismatch. Expected {self.__vectors.shape[1]}, got {vector_shape}")

    def add_text_as_vector(self, text_metadata: VectorMetaData) -> None:
        """
        Adds a document chunk from text as a vector to the vector database.
        """
        text_as_vector: NDArray[float32] = self.embed_text(text_metadata.content)

        self._insert_is_possible(text_as_vector.shape)

        self.__vectors[self.__current_vector_index] = text_as_vector
        self.__metadata_vectors[self.__current_vector_index] = text_metadata

        self.__current_vector_index += 1

    def add_vector(self, vector: NDArray[float32], metadata: VectorMetaData) -> None:
        """
        Adds a document chunk as a vector to the vector database.
        """
        self._insert_is_possible(vector.shape)

        self.__vectors[self.__current_vector_index] = vector
        self.__metadata_vectors[self.__current_vector_index] = metadata

        self.__current_vector_index += 1

    def save_database(self) -> None:
        """
        Writes the current vectors of the database and their metadata to a Pickle file.
        """
        with open(self.__database_path, "wb") as database_file:
            pickle.dump(
                obj={
                    "vectors": self.__vectors if ((self.__current_vector_index + 1) == self.__vectors.__len__()) else self.__vectors[:self.__current_vector_index],
                    "metadata": [metadata.to_dict() for metadata in self.__metadata_vectors if (metadata)]
                },
                file=database_file
            )

        sys.stdout.write(
            f"\n[SAVE] Database written to {self.__database_path} (vector_count={(self.__current_vector_index + 1)},"
            f" (dimensions={self.__vectors[0].shape}).\n"
        )

    def _embed_pdf_document(self, save_afterwards: bool = True) -> Literal[0, 1]:
        """
        Embeds a PDF document from scratch.
        """
        sys.stdout.write(f"\n\nEmbedding PDF document pages ...\n")

        for text_metadata in pdf_to_text_pages(self.__pdf_file_path):
            self.add_text_as_vector(text_metadata)

        if (save_afterwards):
            self.save_database()

        return return_codes.SUCCESS

    def _load_database(self, reloading: bool = False) -> Literal[0, 1]:
        """
        Reads the vector database from disk if one exists.
        """
        if (not os.path.exists(self.__database_path)):
            self._embed_pdf_document()

        with open(self.__database_path, "rb") as database_file:
            data: dict[str, NDArray[float32] | dict[str, str | int]] = pickle.load(database_file)

        self.__vectors = data["vectors"]

        if (reloading):
            self.__metadata_vectors = [None] * len(self.__vectors)

        for i, vector_metadata in enumerate(data["metadata"]):
            self.__metadata_vectors[i] = VectorMetaData(
                content=vector_metadata["content"],
                page_number=vector_metadata["page_number"],
                filename=vector_metadata["filename"]
            )

        self.__current_vector_index = len(self.__vectors)

        sys.stdout.write(
            f"[READ] Successfully loaded {(self.__current_vector_index + 1)} vectors (dimensions={self.__vectors[0].shape}) "
            f"from {self.__database_path}.\n"
        )

        return return_codes.SUCCESS

    def set_top_k(self, top_k: int) -> None:
        """
        Sets the top-k for relevant PDF document searches.
        """
        self.__top_k = top_k

    def set_page_size(self, page_size: int) -> None:
        """
        Sets the page size for relevant PDF document searches, and loads a different chunk-size vector database if the new page size differs from the current.
        """
        if (page_size == self.__page_size):
            sys.stdout.write("\nPage size unchanged.\n")

            return

        self.__page_size = page_size
        self.__database_path = format_vector_database_path(self.__pdf_file_path, page_size=self.__page_size)

        self._load_database(reloading=True)

    def get_relevant_page_vectors(self, query_vector: NDArray[float32]) -> Optional[list[SimilarVector]]:
        """
        Searches the database for the vectors containing the relevant information in regard to the user's query.
        """
        if (self.__current_vector_index == 0):
            return None

        similarities: NDArray[float32] = (
            np.dot(self.__vectors, query_vector) / (np.linalg.norm(self.__vectors, axis=1) * np.linalg.norm(query_vector))
        )

        top_k = min(self.__top_k, self.__current_vector_index)
        k_most_similar_vectors_indeces: NDArray[int64] = np.argsort(similarities)[::-1][:top_k]

        most_similar_vectors: list[SimilarVector | None] = [None] * top_k

        for i, most_similar_index in enumerate(k_most_similar_vectors_indeces):
            most_similar_vectors[i] = SimilarVector(
                similarity=float(similarities[most_similar_index]),
                metadata=self.__metadata_vectors[most_similar_index]
            )

        return most_similar_vectors

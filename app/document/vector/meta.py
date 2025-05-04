

class VectorMetaData:
    __slots__: tuple[str, ...] = ("content", "page_number", "source")

    def __init__(self, content: str, page_number: int, filename: str) -> None:
        self.content: str = content
        self.page_number: int = page_number
        self.source: str = filename

    def to_dict(self) -> dict[str, str | int]:
        """
        Returns the document page instance as a dictionary.
        """
        return {"content": self.content, "page_number": self.page_number, "filename": self.source}


class SimilarVector:
    __slots__: tuple[str, ...] = ("similarity", "metadata")

    def __init__(self, similarity: float, metadata: VectorMetaData) -> None:
        self.similarity: float = similarity
        self.metadata: VectorMetaData = metadata


def split_text_into_chunks(text: str, chunk_size: int) -> list[str]:
    """
    Returns a text body as a list of substrings of a certain chunk size (character count).
    """
    if (text.__len__() <= chunk_size):
        return [text]

    character_count: int = len(text)
    number_of_chunks: int = (character_count // chunk_size)
    count_left_over_characters: int = (character_count % number_of_chunks)

    text_chunks: list[str] = [""] * number_of_chunks
    i: int = 0
    j: int = 0

    while ((i < character_count) and (j < number_of_chunks)):
        text_chunks[j] = text[i:(i + chunk_size)]

        i += chunk_size
        j += 1

    if (count_left_over_characters > 0):
        text_chunks[-1] += text[i:]

    return text_chunks

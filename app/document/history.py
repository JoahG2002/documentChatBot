

class Chat:
    __slots__: tuple[str, ...] = ("response", "query", "timestamp")

    def __init__(self, response: str, query: str, timestamp: int) -> None:
        self.response: str = response
        self.query: str = query
        self.timestamp: int = timestamp

    def get_preview_text(self) -> str:
        """
        Returns the preview text (the first N characters) of the query of the chat.
        """
        return self.query[:7]


class ChatHistory:
    __slots__: tuple[str, ...] = ("remember_until_index", "__chat_history", "__current_index")

    def __init__(self, remember_until: int) -> None:
        self.remember_until_index: int = (remember_until - 1)
        self.__current_index: int = 0
        self.__chat_history: list[Chat | None] = [None] * remember_until

    def add_chat(self, chat: Chat) -> None:
        """
        Adds a chat (a query and a LLM response) to the front of the chat history. The oldest chat will be removed if the ``remember_until`` is exceeded at the time of the addition of the new chat.
        """
        if (self.__current_index < self.remember_until_index):
            self.__chat_history[self.__current_index] = chat

            self.__current_index += 1

            return

        current_chat_history_length: int = len(self.__chat_history)
        renewed_chat_history: list[Chat | None] = [None] * current_chat_history_length
        renewed_chat_history[0] = chat
        i: int = 1

        while (i < current_chat_history_length):
            renewed_chat_history[i] = self.__chat_history[i]

            i += 1

        self.__chat_history = renewed_chat_history

    def get_chat_previews(self) -> list[str]:
        """
        Returns the previews of all available chats — most- to least-recent.
        """
        if (self.__current_index == 0):
            return []

        chat_previews: list[str] = [""] * (self.__current_index + 1)
        index_last_chat: int = self.__current_index

        while (index_last_chat > 0):
            chat_previews[index_last_chat] = self.__chat_history[index_last_chat].get_preview_text()
            index_last_chat -= 1

        return chat_previews

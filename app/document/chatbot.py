import os
import time

from kivy.app import App
from kivy.metrics import dp
from plyer import filechooser
from kivy.uix.label import Label
from typing import Self, Literal
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Rectangle, Color

from ..ui.button import RoundButton
from ..constant.color import colors
from .vector.meta import SimilarVector
from .large_language_model.chatgpt import query_chatgpt
from ..document.vector.database import PDFVectorDatabase
from ..utility.write import write_llm_query_response_to_csv
from ..constant.constant import font_sizes, return_codes, limits, default_texts, large_language_models


class ChatBot(App):
    def __init__(self, **kwargs) -> None:
        super(ChatBot, self).__init__(**kwargs)

        self.title = "DocumentChatBot"

        self.timestamp_last_request: int = 0

        self.pdf_file_path: str = ""
        self.pdf_vector_database: PDFVectorDatabase | None = None

        self.rectangle: Rectangle | None = None

        self.app_layout: BoxLayout = BoxLayout(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(10),
        )

        self.output_text_label: Label = Label(
            text="Hi, ask any question about a complex PDF document.",
            font_size=font_sizes.HEADER,
            size_hint_y=0.7,
            halign="center",
            valign="top",
            text_size=(None, None)
        )

        self.output_scroll: ScrollView = ScrollView(
            size_hint=(1, 0.7),
            do_scroll_x=False,
            do_scroll_y=True
        )

        self.query_input_field: TextInput = TextInput(
            hint_text="Ask a question.",
            foreground_color=colors.WHITE,
            background_color=colors.DARK_GREY,
            cursor_color=colors.SECONDARY_COLOR,
            selection_color=colors.SECONDARY_COLOR_TRANSPARENT,
            size_hint_y=None,
            font_size=font_sizes.REGULAR,
            height=dp(100),
            multiline=True,
            padding=[10, 10, 10, 10]
        )

        self.feedback_label: Label = Label(
            font_size=font_sizes.ERROR,
            size_hint_y=0.7,
            halign="center",
            valign="top",
            color=colors.ERROR_RED,
            text_size=(None, None)
        )

        self.search_button: RoundButton = RoundButton(
            text="search",
            font_size=font_sizes.BUTTON,
            main_color=colors.SECONDARY_COLOR,
            color_after_pressed=colors.DARK_GREY,
            on_press=self._query_answer,
            color=colors.WHITE,
            background_color=colors.SECONDARY_COLOR,
            background_normal="",
            background_disabled_normal="",
            size_hint=(None, None),
            size=(dp(150), dp(50)),
            pos_hint={"center_x": 0.5}
        )

        self.pick_pdf_button: RoundButton = RoundButton(
            text='+',
            font_size=font_sizes.ICONIC_SYMBOLS,
            main_color=colors.MAIN_THEME,
            color_after_pressed=colors.DARK_GREY,
            on_press=self._pick_pdf_file,
            color=colors.WHITE,
            background_color=colors.MAIN_THEME,
            background_normal="",
            background_disabled_normal="",
            size_hint=(None, None),
            size=(dp(60), dp(60)),
            pos_hint={"right": 1}
        )

    def build(self) -> BoxLayout:
        """
        Builds the app's layout on start-up.
        """
        self.app_layout.bind(size=self._update_rect, pos=self._update_rect)

        with self.app_layout.canvas.before:
            Color(*colors.MAIN_THEME)
            self.rectangle = Rectangle(size=self.app_layout.size, pos=self.app_layout.pos)

        app_title_layout: BoxLayout = BoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(70)
        )

        app_title: Label = Label(
            text="documentChatBot",
            font_size=font_sizes.HEADER,
            size_hint_x=0.8,
            valign="top",
        )

        app_title_layout.add_widget(app_title)
        app_title_layout.add_widget(self.pick_pdf_button)

        self.app_layout.add_widget(app_title_layout)

        self.output_text_label.bind(size=self._update_text_size)
        self.output_scroll.add_widget(self.output_text_label)
        self.app_layout.add_widget(self.output_scroll)

        input_area: BoxLayout = BoxLayout(
            orientation="vertical",
            size_hint_y=0.3,
            spacing=dp(10)
        )
        input_area.add_widget(self.query_input_field)
        input_area.add_widget(self.feedback_label)

        button_container: BoxLayout = BoxLayout(
            size_hint_y=None,
            height=dp(70)
        )

        button_container.add_widget(Widget(size_hint_x=0.25))
        button_container.add_widget(self.search_button)
        button_container.add_widget(Widget(size_hint_x=0.25))

        input_area.add_widget(button_container)

        self.app_layout.add_widget(input_area)

        return self.app_layout

    def _update_rect(self, instance: Self, _value: float) -> None:
        """
        Updates the rectangle size when window resizes.
        """
        self.rectangle.pos = instance.pos
        self.rectangle.size = instance.size

    def _update_text_size(self, instance: Self, _value: float) -> None:
        """
        Update text wrapping when size changes.
        """
        self.output_text_label.text_size = (instance.width, None)

    def _pick_pdf_file(self, *_args) -> Literal[0, 1]:
        """
        Lets the user pick a PDF file with a file chooser, and validates the PDF file by checking whether the file path exists, and whether its extensions is really .pdf.
        """
        pdf_file_path: list[str] = filechooser.open_file(title="Select a PDF file", filters=[["*.pdf"]])

        if (not pdf_file_path):
            self.feedback_label.text = "No PDF file provided."

            return return_codes.FAILURE

        self.pdf_file_path: str = pdf_file_path[0]

        if (not self.pdf_file_path.endswith(".pdf")):
            self.pdf_file_path = ""

            self.feedback_label.text = "Please pick an actual PDF file."

            return return_codes.FAILURE

        self.feedback_label.text = ""
        self.output_text_label.text = "Processing PDF file ...\nThis may take a while."

        self.pdf_vector_database = PDFVectorDatabase(
            vector_count=large_language_models.EXPECT_VECTOR_COUNT,
            vector_dimensions=large_language_models.VECTOR_DIMENSION_PER_MODEL[large_language_models.DEFAULT_SENTENCE_EMBEDDING_MODEL],
            pdf_file_path=self.pdf_file_path,
            model_name=large_language_models.DEFAULT_SENTENCE_EMBEDDING_MODEL
        )

        self.output_text_label.text = "PDF document processed; feel free to ask any question."

        return return_codes.SUCCESS

    def _check_input(self, *_args) -> Literal[0, 1]:
        """
        Checks whether the users input is valid, before sending the search request. If the input is invalid, the search procedure is aborted, and errors are display on screen.
        """
        query_input: str = str(self.query_input_field.text)

        if (not query_input):
            self.feedback_label.text = "Input field empty."

            return return_codes.FAILURE

        if (len(query_input) > limits.QUERY_CHARACTER_COUNT):
            self.feedback_label.text = "Character limit exceeded."

            return return_codes.FAILURE

        if (not self.pdf_file_path):
            self.feedback_label.text = "No PDF file provided."

            return return_codes.FAILURE

        self.feedback_label.text = ""

        return return_codes.SUCCESS

    def _handle_query_request(self, query: str) -> None:
        """
        Requests an answer to query from the LLM.
        """
        relevant_page_vectors: list[SimilarVector] | None = self.pdf_vector_database.get_relevant_page_vectors(
            query_vector=self.pdf_vector_database.embed_text(query)
        )

        if (not relevant_page_vectors):
            self.feedback_label.text = "Could not find any relevant document pages."

            return

        self.feedback_label.text = ""

        prompt: str = default_texts.LLM_CHAT_INSTRUCTIONS_ANSWER

        relevant_document_segments: str = "\n\n".join(
            f"[Page {relevant_page.metadata.page_number}]: {relevant_page.metadata.content}"
            for relevant_page in relevant_page_vectors
        )
        prompt += f"Document sections:\n{relevant_document_segments}\n"

        question: str = f"Question: {query}"

        prompt += question

        self.timestamp_last_request = int(time.time())

        response_chatgpt: str | None = query_chatgpt(prompt, model=large_language_models.MODEL_NAME)

        if (not response_chatgpt):
            self.feedback_label.text = "Connection error."

            return

        self.feedback_label.text = ""
        
        self.output_text_label.text = f"""
        {(self.output_text_label.text)}
        ---------------------------------------------------
        
        Question: {query}

        {response_chatgpt}
        """

        self.query_input_field.text = ""
        self.search_button.disabled = False

        write_llm_query_response_to_csv(
            pdf_document_name=os.path.basename(self.pdf_file_path),
            large_language_model=large_language_models.NAME,
            model_name=large_language_models.MODEL_NAME,
            sentence_embedding_model=large_language_models.DEFAULT_SENTENCE_EMBEDDING_MODEL,
            dimension_count_vectors=large_language_models.VECTOR_DIMENSION_PER_MODEL[large_language_models.DEFAULT_SENTENCE_EMBEDDING_MODEL],
            top_k_similar_documents=limits.TOP_K_RELEVANT_DOCUMENT_SEGMENTS,
            pdf_page_size=limits.PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT,
            instruction_text=default_texts.LLM_CHAT_INSTRUCTIONS_ANSWER,
            query=query,
            llm_response=response_chatgpt,
            token_limit=limits.LLM_TOKEN_LIMIT,
            prompt_length=len(prompt),
            top_k_documents=relevant_document_segments
        )

    def _query_answer(self, *_args) -> None:
        """
        Queries the answer to the user's question about the PDF file from the large language model (LLM).
        """
        if (self._check_input() == return_codes.FAILURE):
            return

        if ((int(time.time()) - self.timestamp_last_request) < limits.MINIMUM_SECONDS_SINCE_LAST_API_REQUEST):
            self.feedback_label.text = "Too fast consecutive requests."

            return

        self.feedback_label.text = ""

        self.output_text_label.font_size = font_sizes.REGULAR
        self.output_text_label.halign = "left"
        self.output_text_label.text = "thinking ..."

        self.search_button.disabled = True

        self._handle_query_request(str(self.query_input_field.text))

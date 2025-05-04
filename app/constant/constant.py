import os

from typing import Final, Literal

from app.constant.env import read_env_variables


ENVIRONMENT_VARIABLES: dict[str, str] = read_env_variables()
CURRENT_WORKING_DIRECTORY: str = os.getcwd().replace("\\", '/')


class DefaultText:
    __slots__: tuple[str, ...] = (
        "LLM_CHAT_INSTRUCTIONS_ANSWER", "LLM_RESPONSE_EVALUATION_INSTRUCTIONS", "SEPARATION_LINE",
        "MODEL_PERFORMANCE_CSV_COLUMNS_ROW",
    )

    def __init__(self) -> None:
        self.LLM_CHAT_INSTRUCTIONS_ANSWER: Final[str] = """
        You are a technical assistant for engineers and programmers. Your task is to extract only the most relevant, actionable information from complex legal, regulatory, and technical documents. Ignore irrelevant sections, legal jargon, or unnecessary details. Your response should be concise, practical, and focused on implementation.
        
        Response Guidelines:
        - summarize only the requirements relevant to engineers or developers;
        - avoid legalese—translate laws into code or implementation steps;
        - provide examples where applicable (e.g., API rules, safety limits, compliance steps);
        - avoid unnecessary context or lengthy explanations—answer only what is asked.
        - Provide all required steps with enumerated bullet points.
        
        For example:
        query: I am working on two-factor-authentication (2FA) for my users' e-mail addresses in a mailing list, what regulations should I keep in mind?
        correct answer: 
        1. Make sure all e-mails are sent with their data encrypted [Page X], and do not store the e-mail addresses longer than 5 years [Page Y].
        2. Then, according to this law, ... Just: do X [page N].
        
        IMPORTANT: Always include page references in your answer using the format (Page X) when citing information.
        Each document section below already includes its page number in the format [Page X].
                    
        If no relevant information is found, respond with: "No applicable technical data available."""

        self.LLM_RESPONSE_EVALUATION_INSTRUCTIONS: Final[str] = """
        You evaluating the response to a certain question of a large-language-model chatbot designed to answer programmers and researches
        questions about complex regulatory and technical PDF documents. You'll be given the query, the top-k most relevant extracted pages and the LLMs response.
        
        Your task is to extract evaluate whether only the most relevant, actionable information from the document was extracted. 
        Also whether, irrelevant sections, legal jargon, or unnecessary details were ignored, and whether pages were cited every each important statement block (practicality score).
        
        The chatbot should have very clear responses: 
        - summarize only the requirements relevant to engineers or developers;
        - avoid legalese—translate laws into code or implementation steps;
        - provide examples where applicable (e.g., API rules, safety limits, compliance steps);
        - avoid unnecessary context or lengthy explanations—answer only what is asked.
        - Provide all required steps with enumerated bullet points.
        
        For example:
        query: I am working on two-factor-authentication (2FA) for my users' e-mail addresses in a mailing list, what regulations should I keep in mind?
        correct answer: 
        1. Make sure all e-mails are sent with their data encrypted [Page X], and do not store the e-mail addresses longer than 5 years [Page Y].
        2. Then, according to this law, ... Just: do X [page N].
                        
        If no relevant information is found, this is not a problem, if it is clearly stated by the bot.
 
        - Sources cited correctly: either 0 or 1
        - Faithfulness score: decompose the answer into individual statements -> verify whether each statement is consistent with the context. -> consistent_statement_count / total_statement_count.
        - Context relevance: count_directly_relevant_sentences_in_pages / total_number_of_sentences_pages
        - Practicality score: 0.0–10.0 (does the programmer/researcher exactly know what to do now?)
    
        Answer strictly like, nothing more:
        {
            "sourced_cited_correctly": 1  # or 0,
            "faithfulness_score": a_float,
            "context_relevance": a_float,
            "practicality_score": a_float
        }
        """
        # - Answer relevance: generate a single Python string of potential other questions, with the questions seperated by '_' (I will then embed to and calculate their similarity to the original question, not headache).

        self.MODEL_PERFORMANCE_CSV_COLUMNS_ROW: Final[str] = (
            "timestamp,pdfDocument,largeLanguageModel,modelName,sentenceEmbeddingModel,dimensionCountVectors,"
            "topKSimilarDocuments,pdfPageSize,instructionContextText,query,topKDocuments,lmmResponse,tokenLimit,"
            "promptLength,instructionLength,queryLength,answerFaithfulnessScore,answerRelevanceScore,contextRelevanceScore,"
            "sourcedCited,practicalityScore,llmResponseLength,rephrasedQuestions\n"
        )


class FontSize:
    __slots__: tuple[str, ...] = ("REGULAR", "HEADER", "INFORMATION", "ERROR", "BUTTON", "ICONIC_SYMBOLS")

    def __init__(self) -> None:
        self.REGULAR: Final[str] = "19sp"
        self.INFORMATION: Final[str] = "19sp"
        self.HEADER: Final[str] = "32sp"
        self.ERROR: Final[str] = "18sp"
        self.BUTTON: Final[str] = "22sp"
        self.ICONIC_SYMBOLS: Final[str] = "30sp"


class ReturnCode:
    __slots__: tuple[str, ...] = ("SUCCESS", "FAILURE")

    def __init__(self) -> None:
        self.SUCCESS: Final[Literal[0, 1]] = 0
        self.FAILURE: Final[Literal[0, 1]] = 1


class Limit:
    __slots__: tuple[str, ...] = (
        "QUERY_CHARACTER_COUNT", "CHAT_HISTORY", "TOP_K_RELEVANT_DOCUMENT_SEGMENTS", "LLM_TOKEN_LIMIT",
        "PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT", "MINIMUM_SECONDS_SINCE_LAST_API_REQUEST", "INDEX_LAST_TASK"
    )

    def __init__(self) -> None:
        self.QUERY_CHARACTER_COUNT: Final[int] = 500
        self.CHAT_HISTORY: Final[int] = 10
        self.TOP_K_RELEVANT_DOCUMENT_SEGMENTS: Final[int] = 3
        self.LLM_TOKEN_LIMIT: Final[int] = 1_000
        self.PDF_DOCUMENT_PAGE_CHUNK_CHARACTER_LIMIT: Final[int] = 1_000  # 512
        self.MINIMUM_SECONDS_SINCE_LAST_API_REQUEST: Final[int] = 3
        self.INDEX_LAST_TASK: Final[int] = 141


class Directory:
    __slots__: tuple[str, ...] = ("PDF_VECTOR_DATABASES", "LARGE_LANGUAGE_MODEL_RESPONSES", "ANALYSIS_PLOTS")

    def __init__(self) -> None:
        self.PDF_VECTOR_DATABASES: Final[str] = f"{CURRENT_WORKING_DIRECTORY}/data/pdfVectors"
        self.LARGE_LANGUAGE_MODEL_RESPONSES: Final[str] = f"{CURRENT_WORKING_DIRECTORY}/data/llmResponses/csvFiles"
        self.ANALYSIS_PLOTS: Final[str] = f"./data/plots"


class LargeLanguageModel:
    __slots__: tuple[str, ...] = (
        "NAME", "MODEL_NAME", "DEFAULT_SENTENCE_EMBEDDING_MODEL", "OPEN_AI_API_KEY", "VECTOR_DIMENSION_PER_MODEL",
        "EXPECT_VECTOR_COUNT", "QUESTIONS", "GEMINI_API_KEY", "MAIN_PDF_DOCUMENT_FILE_PATH"
    )

    def __init__(self) -> None:
        self.NAME: Final[str] = "ChatGPT"
        self.MODEL_NAME: Final[str] = "gpt-4o"

        self.DEFAULT_SENTENCE_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
        self.OPEN_AI_API_KEY: Final[str] = ENVIRONMENT_VARIABLES["OPEN_AI_API_KEY"]
        self.GEMINI_API_KEY: Final[str] = ENVIRONMENT_VARIABLES["GEMINI_API_KEY"]

        self.VECTOR_DIMENSION_PER_MODEL: dict[str, int] = {
            self.DEFAULT_SENTENCE_EMBEDDING_MODEL: 384
        }

        self.EXPECT_VECTOR_COUNT: Final[int] = 5_000

        self.QUESTIONS: Final[dict[int, str]] = {
            1: "How can we optimize code or firmware to reduce CPU/GPU load, thereby lowering energy consumption?",
            2: "Can we design or modify hardware systems (like cooling or power delivery) to minimize environmental footprint and improve energy efficiency (e.g., PUE – Power Usage Effectiveness)?",
            3: "How do we assess the environmental impact of our systems throughout their full lifecycle (from production to decommissioning)?",
            4: "How can we document and communicate our system’s environmental performance in a way that aligns with ISO 14001 requirements for transparency and reporting?",
            5: "What logging and reporting features should we build into software systems to support environmental audits and reviews?",
            6: "What software or control systems can we implement to detect and mitigate environmental risks like overheating, power surges, or hazardous material leaks?",
            7: "Can we establish accurate energy baselines and track deviations using analytics or machine learning models?",
            8: "Which Energy Performance Indicators (EnPIs) should we define and monitor in software to reflect meaningful efficiency metrics (e.g., PUE, DCiE, cooling system efficiency)?",
            9: "How can our system provide actionable insights into energy usage trends over time or under different workloads?",
            10: "How can our hardware designs (e.g., for servers, cooling systems, or UPS) be more energy efficient through component selection and layout?",
            11: "Are there feedback loops or alerts built in that notify stakeholders about unusual energy patterns or inefficiencies?",
            12: "Where can I find our organization’s environmental policy, and how do I make sure my system design doesn’t contradict it?",
            13: "Is there a current energy baseline or performance target I should be aware of when designing or coding?",
        }

        self.MAIN_PDF_DOCUMENT_FILE_PATH: Final[str] = "./data/pdfDocuments/cellar_bb8539b7-b1b5-11ec-9d96-01aa75ed71a1.0001.02_DOC_1.pdf"


class Csv:
    __slots__: tuple[str, ...] = (
        "EXPECTED_COUNT_ROWS", "COLUMN_DATATYPES", "ENCODING", "EXPERIMENT_DATA_FILE", "NUMERIC_COLUMNS_QUESTION1",
        "ALPHA"
    )

    def __init__(self) -> None:
        self.EXPECTED_COUNT_ROWS: Final[int] = 800
        self.ENCODING: Final[str] = "utf-8"
        self.EXPERIMENT_DATA_FILE: Final[str] = "./data/centralResults/experimentResults.csv"

        self.COLUMN_DATATYPES: Final[dict[str, str]] = {
            "timestamp": "int32",
            "pdfDocument": "string",
            "largeLanguageModel": "string",
            "modelName": "string",
            "sentenceEmbeddingModel": "string",
            "dimensionCountVectors": "int16",
            "topKSimilarDocuments": "int8",
            "pdfPageSize": "int16",
            "instructionContextText": "string",
            "query": "string",
            "topKDocuments": "string",
            "lmmResponse": "string",
            "tokenLimit": "int16",
            "promptLength": "int16",
            "instructionLength": "int16",
            "queryLength": "int16",
            "answerFaithfulnessScore": "float64",
            "answerRelevanceScore": "float64",
            "contextRelevanceScore": "float64",
            "sourcedCited": "float64",
            "practicalityScore": "float64",
            "llmResponseLength": "float64",
            "rephrasedQuestions": "string"
        }

        self.NUMERIC_COLUMNS_QUESTION1: set[str] = {
            "answerRelevanceScore", "answerFaithfulnessScore", "contextRelevanceScore", "practicalityScore"
        }

        self.ALPHA: Final[float] = 0.05


font_sizes: FontSize = FontSize()
return_codes: ReturnCode = ReturnCode()
limits: Limit = Limit()
default_texts: DefaultText = DefaultText()
directories: Directory = Directory()
large_language_models: LargeLanguageModel = LargeLanguageModel()
csv_: Csv = Csv()

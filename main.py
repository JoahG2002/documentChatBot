import sys

from app.document.chatbot import ChatBot
from app.utility.write import clear_terminal
from app.constant.constant import return_codes
from app.document.vector.similarity import calculate_question_average_similarity, process_questions


def main(argc: int, argv: list[str]) -> None:
    clear_terminal()

    if (argc == 2):
        questions: list[str] = argv[1].split('_')

        if (len(questions) < 2):
            sys.stderr.write(
                "\nError: not enough questions provided. Format should be: \"question1_question2_..._original_question\"\n\n"
            )

            exit(return_codes.FAILURE)

        original_question, rephrased_questions = process_questions(questions)

        average_similarity: float = calculate_question_average_similarity(
            original_question=original_question,
            rephrased_questions=rephrased_questions
        )

        sys.stdout.write(
            f"\nAverage similarity alternative questions: {average_similarity}\n\n"
        )

        return

    clear_terminal()

    ChatBot().run()

    exit(return_codes.SUCCESS)


if (__name__ == "__main__"):
    main(len(sys.argv), sys.argv)

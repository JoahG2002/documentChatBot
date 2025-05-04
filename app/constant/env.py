import os
import sys

from typing import Optional

from . import constant


def read_env_variables() -> Optional[dict[str, str]]:
    """
    Leest de namen en waarden van de omgevingvariablen uit het .env-bestand.
    """
    path_env_file: str = f"{os.getcwd()}/data/.env"

    if (not os.path.exists(path_env_file)):
        sys.stderr.write("\n.env file not found; unzip data.zip.\n")

        exit(constant.return_codes.FAILURE)

    env_variables: dict[str, str] = {}

    with open(path_env_file, 'r') as env_file:
        env_file_lines: list[str] = env_file.read().splitlines()

    for line in env_file_lines:
        variable, value = line.replace('"', '').split(sep='=', maxsplit=1)

        env_variables[variable] = value

    return env_variables

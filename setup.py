from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Ml proj1"
VERSION = "0.0.1"
DESCRIPTION = "second one"
AUTHOR_NAME = "Tarun"
AUTHOR_EMAIL = "dummy@gmail.com"

REQ_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirement_list() -> List[str]:
    with open(REQ_FILE_NAME) as req_file:
        req_list = [req.strip() for req in req_file.readlines()]

    if HYPHEN_E_DOT in req_list:
        req_list.remove(HYPHEN_E_DOT)

    return req_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirement_list()
)

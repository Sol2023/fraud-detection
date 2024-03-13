from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = " -e ."

def get_requirements(file_path: str = "requirements.txt"):

    '''
    This function reads the requirements.txt file and returns a list of required packages
    '''

    requirements: List[str] = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.strip() for r in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="fraud_detection",
    version="0.0.1",
    author="Adam Huang",
    author_email="shuang497@gatech.edu",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
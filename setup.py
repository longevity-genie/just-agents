from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.2.3'
DESCRIPTION = 'Just Agents'
LONG_DESCRIPTION = 'LLM Agents that are implemented without unnecessary complexity'

# Setting up
setup(
    name="just-agents",
    version=VERSION,
    author="Alex Karmazin, Anton Kulaga",
    author_email="antonkulaga@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=["litellm>=1.50.1", "numpydoc", "requests", "typer>=0.12.5"],
    extras_require={
        'tools': [
            'semanticscholar>=0.8.4'
        ],
        'coding': [
            'llm-sandbox @ git+https://github.com/antonkulaga/llm-sandbox.git'
        ]
    },
    keywords=['python', 'llm', 'science', 'review', 'agents', 'AI'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    package_data={
        'just_agents': ['config/*.yaml'],
        'just_agents_coding': ['config/*.yaml'],
        
    },
    include_package_data=True,
)

# Using LLM to Q&A Over Documents

Suppose you possess various text documents (such as PDFs, txt, code etc.) and want to ask questions questions about what's inside these docs. Since LLMs understand text tasks quite well we can use them as a solution for this purpose.



## Prerequisites

1. python 3.11
2. OpenAI APIKey. Please generate a **.env** file with the key. This file must be saved at this level with the following structure.
    ```bash
    OPENAI_API_KEY=your_key   
    ```
## Installation
After cloning this repo, please install requirements.
```python
pip install -r requirements.txt
```
## Usage
There are two ways to use the code, running the streamlit application
```bash
streamlit run app.py
```
Or there is a notebook index.ipynb

## Knowledge



![QA_Over_Documents](https://github.com/csernac0/llm-qa-over-docs/assets/30326740/fddc8946-14d5-46e5-a93d-3b7ee13f1c83)



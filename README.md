# End-to-end-PythonBook-Chatbot-Generative-AI

# How to run?

### Steps:

Clone the repository 

```bash
Project repo: https://github.com/
```

### Step 1: Create aconda environment after opening the repository

```bash
conda create -p venv python=3.10 -y
```

```bash
conda activate venv
```


### Step 2: Install requirements

```bash
pip install -r requirements.txt
```



## What is Retrieval-Augmented Generation?
##### Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.

-----------------------

## Why is Retrieval-Augmented Generation important?
#### LLMs are a key artificial intelligence (AI) technology powering intelligent chatbots and other natural language processing (NLP) applications. The goal is to create bots that can answer user questions in various contexts by cross-referencing authoritative knowledge sources. Unfortunately, the nature of LLM technology introduces unpredictability in LLM responses. Additionally, LLM training data is static and introduces a cut-off date on the knowledge it has.

Known challenges of LLMs include:

Presenting false information when it does not have the answer.
Presenting out-of-date or generic information when the user expects a specific, current response.
Creating a response from non-authoritative sources.
Creating inaccurate responses due to terminology confusion, wherein different training sources use the same terminology to talk about different things.
You can think of the Large Language Model as an over-enthusiastic new employee who refuses to stay informed with current events but will always answer every question with absolute confidence. Unfortunately, such an attitude can negatively impact user trust and is not something you want your chatbots to emulate!

RAG is one approach to solving some of these challenges. It redirects the LLM to retrieve relevant information from authoritative, pre-determined knowledge sources. Organizations have greater control over the generated text output, and users gain insights into how the LLM generates the response.




![RAG_ARCH](https://github.com/user-attachments/assets/113fe8f1-613c-4c66-9ff8-15bf7b2e2ce9)













--------------------------------------------------------


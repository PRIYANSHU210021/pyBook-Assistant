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


















## **Detailed Explanation, what is mean of lines of codes**


---

## 🧩 **1️⃣ Data Extraction (PDF Loader Part)**

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

* Ye dono import hote hain PDF files se text extract karne aur split karne ke liye.
* `DirectoryLoader` → ek poori folder me jitne bhi PDF files hain, sabko load karta hai.
* `PyPDFLoader` → har PDF page ka text nikalta hai line-by-line.

```python
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
```

* Ye function folder me PDFs leta hai aur unka saara textual content ek list me return karta hai.
* Har element ek “document” hota hai, jisme page content aur metadata hota hai (like page number).

✅ **Presentation line:**

> "Sabse pehle humne apne Python book ke PDFs ko load karke uska raw textual data extract kiya using LangChain loaders."

---

## 🧩 **2️⃣ Text Splitting (Chunking Step)**

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(extracted_data)
```

* Large text ko hum directly embeddings me nahi bhej sakte, isliye hum usse chhote chhote parts (chunks) me todte hain.
* `chunk_size=1000` → har chunk me max 1000 characters honge.
* `chunk_overlap=200` → har chunk ke end ke 200 characters agle chunk me bhi repeat honge taaki context na toote.

✅ **Presentation line:**

> "Large PDF ko chhoti chhoti meaningful text chunks me divide kiya, taaki LLM context samajh sake aur query ka exact answer de sake."

---

## 🧩 **3️⃣ Embedding Creation (Gemini Embeddings)**

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

* Embeddings ka matlab hota hai **text ko numbers ke vector form me convert karna**.
* Ye vector machine ko semantic similarity samajhne me help karta hai — jaise “car” aur “automobile” similar hain.
* Humne **Google Gemini’s embedding model (`text-embedding-004`)** use kiya, jo har text ko 768-dimension vector me convert karta hai.

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", 
    google_api_key=os.environ["GEMINI_API_KEY"]
)
```

✅ **Presentation line:**

> "Phir humne har text chunk ko Gemini ke embedding model se numeric vectors me convert kiya — jisse semantic meaning preserve rahe."

---

## 🧩 **4️⃣ Vector Database (Pinecone Setup)**

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(name="pybookreader", dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```

* **Pinecone** ek **vector database** hai jahan hum embeddings store karte hain.
* Jab user koi question poochta hai, Pinecone nearest matching vectors find karta hai using **cosine similarity**.
* Humne ek index banaya — “pybookreader”, jisme sab embeddings store hongi.

✅ **Presentation line:**

> "Embeddings ko Pinecone vector database me store kiya, taaki future me query karte waqt similar context retrieve kiya ja sake."

---

## 🧩 **5️⃣ Document Insertion into Index**

```python
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
```

* Ye step har chunk ke embedding ko Pinecone index me push karta hai (called **upsert** operation).
* Matlab: “Store kar do ye vector along with its original text.”

✅ **Presentation line:**

> "Har text chunk ko uske embedding ke saath Pinecone me upsert kiya, jisse humara knowledge base ready ho gaya."

---

## 🧩 **6️⃣ Retriever Setup**

```python
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
retrieved_docs = retriever.invoke("what is decorator?")
```

* **Retriever** ek bridge hai jo user ke query se relevant documents nikalta hai.
* `k=3` → Top 3 most similar chunks retrieve karega.
* Ye context hi aage LLM ko diya jata hai answer generate karne ke liye.

✅ **Presentation line:**

> "Retriever user ke query ke basis pe sabse relevant 3 text chunks nikalta hai jo question se semantically milte hain."

---

## 🧩 **7️⃣ LLM Setup (Gemini Flash Model)**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

* Ye model actual **answer generation** karta hai — similar to GPT but by Google.
* “Flash” version fast and cost-effective hai.

✅ **Presentation line:**

> "Gemini 2.5 Flash model ko humne LLM ke roop me use kiya answer generate karne ke liye."

---

## 🧩 **8️⃣ System Prompt (Custom Instruction)**

```python
system_prompt = (
    "You have to behave like a **Python Programming Expert and Tutor**..."
)
```

* Ye model ko **role** batata hai (system message).
* Isme instructions diye gaye ki:

  * Sirf document ke context se hi answer dena.
  * Agar answer context me nahi hai → clearly bolna “I could not find the answer...”.
  * Simple explanation aur code examples dena.

✅ **Presentation line:**

> "Prompt me clear instruction diya gaya ki model sirf document ke base pe answer kare aur hallucination avoid kare."

---

## 🧩 **9️⃣ Runnable Chain (Pipeline Creation)**

```python
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'input': RunnablePassthrough()
})
main_chain = parallel_chain | prompt | llm | parser
```

* Ye **LangChain Runnables** use karta hai — matlab ek modular pipeline:

  * **retriever → prompt → LLM → parser**
  * `RunnableParallel` se context aur input parallel process hote hain.
  * `RunnablePassthrough()` → user ka raw input as-it-is bhejta hai.
  * `format_docs()` → retrieved docs ko ek readable text me convert karta hai.

✅ **Presentation line:**

> "LangChain Runnables ke through humne ek modular pipeline banayi jisme retrieval aur generation parallelly handle hota hai."

---

## 🧩 **🔟 Final Step: Query Execution**

```python
response = main_chain.invoke('what is Generators')
```

* Ab jab user query deta hai →

  1. Retriever Pinecone se related context nikalta hai.
  2. Ye context + user query prompt me jata hai.
  3. Gemini model based on context answer generate karta hai.

✅ **Presentation line:**

> "Aakhri step me user query LLM tak jaati hai, retriever se context aata hai, aur model accurate, context-based answer deta hai."

---

## 🧾 **✨ Summary**

> “Toh overall, ye project ek complete **RAG pipeline** hai — jisme humne apne PDFs se knowledge extract karke usko Pinecone me store kiya, Gemini embeddings se semantic search enable kiya, aur Gemini LLM se context-based answers generate karvaye. Ye approach hallucination-free aur document-grounded responses ke liye best hai.”















--------------------------------------------------------


# %%
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, CSVLoader
from  langchain_text_splitters import RecursiveCharacterTextSplitter 
from pathlib import Path
import os
import json
from typing import TypedDict,List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import pandas as pd
from langchain_groq import ChatGroq



# %% [markdown]
# 

# %%
load_dotenv()
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# %%
load_dotenv()
llm2 = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    temperature=0
)

# %%
class State(TypedDict):
    df: pd.DataFrame
    similarity: float
    target_col: str
    route: str
    filepath:str
    filetype:str
    question:str
    answer:str
    
    documents: list
    schema_text: str
    prediction:List
   

class Similarity(BaseModel):
    similarity:float

# %%
import pandas as pd
import json

# Load dataset
df = pd.read_csv (
   r"C:\Users\aksha\OneDrive\Documents\Desktop\churn_analysis_rag\Bank_Churn_Classification_Dataset.csv"
)

# Extract schema
schema = {
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "num_columns": len(df.columns)
}

# Save schemag
with open("schema.json", "w") as f:
    json.dump(schema, f, indent=4)

print("Schema saved successfully!")

# %% [markdown]
# 

# %%
import pandas as pd
import os
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader
)

def data_ingestion(state: State):

    file_path =  state["filepath"]

    ext = os.path.splitext(file_path)[1].lower()

    # CSV
    if ext == ".csv":

        df = pd.read_csv(file_path)

        schema_text = f"""
        Columns:
        {list(df.columns)}

        Data Types:
        {df.dtypes.astype(str).to_dict()}

        Sample Rows:
        {df.head(5).to_dict()}
        """

        documents = CSVLoader(file_path).load()

        return {
            "filetype": ".csv",
            "documents": documents,
            "schema_text": schema_text
        }

    # PDF
    elif ext == ".pdf":

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        sample_text = "\n".join(
            doc.page_content
            for doc in documents[:3]
        )

        schema_text = f"""
        PDF Content Sample:

        {sample_text[:4000]}
        """
        

        return {
            "filetype": ".pdf",
            "documents": documents,
            "schema_text": schema_text
        }

    else:
        raise ValueError(
            f"Unsupported file type: {ext}"
        )

# %%
import json

with open("schema.json", "r") as f:
    training_schema = json.load(f)

training_schema_text = f"""
Training Dataset Schema:

Columns:
{training_schema['columns']}

Data Types:
{training_schema['dtypes']}
"""

# %%
from pydantic import BaseModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

class Similarity(BaseModel):
    similarity: float


def schema_checker(state: State):

    checker = llm2.with_structured_output(
        Similarity
    )

    result = checker.invoke([
        SystemMessage(
            content="""
            Compare the training dataset schema
            with the uploaded dataset.

            Return a similarity score between
            0 and 1.
            

        Consider:
        - Column names
        - Data types
        - Semantic meaning of fields

        
        1.0 = nearly identical schema
        0.0 = completely unrelated schema
    
            """
        ),
        HumanMessage(
            content=f"""
            Training Schema:

            {training_schema_text}

            Uploaded Dataset:

            {state['schema_text']}
            """
        )
    ])

    return {
        "similarity": result.similarity
    }
print("Similarity node executed")

# %%
def route_after_similarity(state: State):

    if state["similarity"] < 0.8:
        return "train"

    return "predict"

# %%
import joblib

best_model = joblib.load("best_model.pkl")
transformer = joblib.load("transformer.pkl")
def predict_node(state:State):
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from langchain_core.prompts import ChatPromptTemplate
    import pandas as pd
    import joblib
    import os

    file_path = state["filepath"]
    name, extension = os.path.splitext(file_path)
    file_type = extension
    

    if file_type==".csv":
        df = pd.read_csv(state["filepath"])


    elif file_type==".pdf":
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Combine all pages
        pdf_text = " ".join(
            doc.page_content
            for doc in documents
        )
        """
        # Convert text into ML features
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        X = vectorizer.fit_transform([text])"""

        with open("schema.json", "r") as f:
            schema = json.load(f)

        columns = schema["columns"]
        dtypes = schema["dtypes"]

        response = llm2.invoke([
        SystemMessage(
            content=f"""
            Extract data from the PDF.

            Return ONLY valid JSON.

            Required columns:
            {columns}

            Required data types:
            {dtypes}

            If a value is missing, return null.

            Do not return explanations.
            """
        ),
        HumanMessage(content=pdf_text)
        ])

        record = json.loads(response.content)

        df = pd.DataFrame([record])
        df = df[columns]
        for col, dtype in dtypes.items():

            if dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif dtype == "str":
                df[col] = df[col].astype(str)
    # Prediction
                
    X = df.drop(columns=["Churn"], errors="ignore")
    x = transformer.transform(X)

    predictions = best_model.predict(x)
    df["Churn"]=predictions

    return{ 
        "predictions": predictions.tolist(),
        "df":df
        
        
    }

                

# %%
def training_node(state:State):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from langchain_core.prompts import ChatPromptTemplate
    import pdfplumber

    file_path = state["filepath"]
    name, extension = os.path.splitext(file_path)
    file_type = extension
    

    if file_type==".csv":
        final_df = pd.read_csv(state["filepath"])

    elif file_type=='.pdf':
        
        text = ""

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # Use only a chunk initially
        chunk = text[:5000]


        prompt = ChatPromptTemplate.from_template(
            """
        You are a data engineer.

        The following text was extracted from a PDF that originally came from a CSV file.

        Instructions:
        1. Reconstruct the tabular dataset.
        2. Return ONLY valid JSON.
        3. Detect all columns.
        4. Remove columns named Output, Label, Target, Chunk, Churn.
        5. Preserve numeric values as numbers.
        6. Do not include explanations.

        Text:
        {text}
        """
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "text": chunk
            }
        )

        content = response.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        records = json.loads(content)

        final_df = pd.DataFrame(records)

        for col in final_df.columns:
            try:
                final_df[col] = pd.to_numeric(
                    final_df[col],
                    errors="coerce"
                )
            except Exception:
                pass

   
    X = final_df.select_dtypes(include=['number'])
    if X.shape[1] == 0:
        raise ValueError(
            "No numeric columns found for clustering"
        )        
    

    scaler = StandardScaler()

    X = X.fillna(X.mean())
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=2, random_state=42)

    final_df["Cluster"] = model.fit_predict(X_scaled)

    return {
        "df":final_df,
        "filetype":file_type
    }   



    
    

# %%

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

def rag_node(state: State):

    df = state["df"]
    question = state["question"]

    # Convert DataFrame rows into documents
    documents = []

    for idx, row in df.iterrows():

        content = "\n".join(
            [
                f"{col}: {row[col]}"
                for col in df.columns
            ]
        )

        documents.append(
            Document(
                page_content=content,
                metadata={"row": idx}
            )
        )

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a data analyst.

Use ONLY the provided dataset context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "answer": response.content
    }

# %%
graph=StateGraph(State)
graph.add_node("data_ingestion",data_ingestion)
graph.add_node("schema_checker",schema_checker)
graph.add_node("training_node", training_node)
graph.add_node("predict_node", predict_node)
graph.add_node("rag_node",rag_node)
graph.add_edge(START,"data_ingestion")
graph.add_edge("data_ingestion","schema_checker")
graph.add_conditional_edges(
    "schema_checker",
    route_after_similarity,
    {
        "train": "training_node",
        "predict": "predict_node"
    }
)
graph.add_edge("training_node", "rag_node")
graph.add_edge("predict_node", "rag_node")
graph.add_edge("rag_node",END)
g=graph.compile()


# # %%
# result = g.invoke({
#     "filepath": r"C:\Users\aksha\OneDrive\Documents\Desktop\churn_analysis_rag\Bank_Churn_Classification_Dataset.csv",
#     "question": "which feature effects the most"
# })
# print(result.content)
def get_answer(filepath, question):

    result = g.invoke({
        "filepath": filepath,
        "question": question
    })

    return result["answer"]

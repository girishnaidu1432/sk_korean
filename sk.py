import os
import pickle
import warnings
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import AzureChatOpenAI

warnings.filterwarnings("ignore")

def process_pdf(file_path, faiss_db_path, pickle_file_path, embeddings):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    st.write("Splits:", splits)

    if os.path.exists(faiss_db_path):
        db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(splits)
        db.save_local(faiss_db_path)
        st.write("Database already existed; data added.")
    else:
        db = FAISS.from_documents(splits, embeddings)
        db.save_local(faiss_db_path)
        st.write("Database not present; creating new database.")

    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            current_splits = pickle.load(file)
            st.write("File already existed; adding splits to it.")
    else:
        current_splits = []
        st.write("File does not exist; creating new file.")

    current_splits.extend(splits)

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(current_splits, file)

def retrieve_and_process_context(faiss_db_path, pickle_file_path, embeddings, llm, question_answering_prompt, user_input):
    vectorstore = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)

    with open(pickle_file_path, 'rb') as file:
        current_splits = pickle.load(file)

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(current_splits)
    keyword_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb, keyword_retriever],
                                           weights=[0.5, 0.5])

    docs = ensemble_retriever.invoke(user_input)

    for i in docs:
        st.write("Metadata:", i.metadata)
        st.write("Page Content:", i.page_content)

    st.write('-' * 100)

    result = document_chain.invoke(
        {
            "context": docs,
            "messages": [
                HumanMessage(content=user_input)
            ],
        }
    )

    st.write("Result:", result)

def main():
    st.title("PDF Processing and Context Retrieval")

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint="https://amrxgenai.openai.azure.com/",
        api_key="14560021aaf84772835d76246b53397a",
        openai_api_version="2023-05-15",
    )

    llm = AzureChatOpenAI(
        api_key="14560021aaf84772835d76246b53397a",
        azure_endpoint="https://amrxgenai.openai.azure.com/",
        api_version="2024-02-15-preview",
        deployment_name="gpt"
    )

    SYSTEM_TEMPLATE = """
    "You are a knowledgeable and helpful assistant designed to assist users effectively. "
    "Even if the content is insufficient, you must generate a response by hypothesizing or reasoning creatively. "
    "Always use the provided content if available, and craft responses drawing from the context or inferred information."
    "Strictly,Please do not Hallucinate or provide false information."
    
    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    st.header("Upload a PDF to Process")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        file_path = os.path.join("uploaded_pdfs", uploaded_file.name)

        os.makedirs("uploaded_pdfs", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded file: {uploaded_file.name}")

        faiss_db_path = f"{uploaded_file.name}_faiss"
        pickle_file_path = f"{uploaded_file.name}_splits.pkl"

        process_pdf(
            file_path=file_path,
            faiss_db_path=faiss_db_path,
            pickle_file_path=pickle_file_path,
            embeddings=embeddings
        )

        st.header("Retrieve and Process Context")
        user_input = st.text_input("Enter your query:", "What is this document about?")
        if st.button("Retrieve Information"):
            retrieve_and_process_context(
                faiss_db_path=faiss_db_path,
                pickle_file_path=pickle_file_path,
                embeddings=embeddings,
                llm=llm,
                question_answering_prompt=question_answering_prompt,
                user_input=user_input
            )

if __name__ == "__main__":
    main()

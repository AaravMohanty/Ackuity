from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter()
embeddings = OpenAIEmbeddings()


# loads and splits text from a file and adds a set of roles to it
def load_text(file_path: str | Path, roles: list[str]) -> list[Document]:
    docs = TextLoader(file_path).load_and_split(text_splitter)
    for doc in docs:
        doc.metadata["roles"] = roles
    return docs


database_folder = "faiss_db"


make_database = False

if make_database:

    pub_doc1 = load_text(
        "./example_docs/public/pub_doc1.txt",
        ["all"],
    )
    pub_doc2 = load_text(
        "./example_docs/public/pub_doc2.txt",
        ["employee"],
    )
    secrets1 = load_text(
        "./example_docs/private/secrets1.txt",
        ["admin"],
    )
    secrets2 = load_text(
        "./example_docs/private/secrets2.txt",
        ["admin", "manager"],
    )

    all_docs = pub_doc1 + pub_doc2 + secrets1 + secrets2

    vectorstore = FAISS.from_documents(all_docs, embedding=embeddings)

    vectorstore.save_local(database_folder)

else:
    vectorstore = FAISS.load_local(
        database_folder, embeddings=embeddings, allow_dangerous_deserialization=True
    )


retriever = vectorstore.as_retriever(
    search_kwargs={
        "score_threshold": 0.5,
        # "k": 1,
    },
)

from pathlib import Path
from dotenv import load_dotenv

# import faiss

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

# from langchain_core.tools import retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser


from pprint import pprint


load_dotenv()


text_splitter = RecursiveCharacterTextSplitter()


# loads and splits text from a file and adds a set of roles to it
def load_text(file_path: str | Path, roles: list[str]) -> list[Document]:
    docs = TextLoader(file_path).load_and_split(text_splitter)
    for doc in docs:
        doc.metadata["roles"] = roles
    return docs


# returns only the docs which the user has roles to access
def filter_documents(docs: list[Document], roles: list[str]) -> list[Document]:
    filtered_docs = []
    for doc in docs:
        # authorized roles
        for role in doc.metadata["roles"]:
            # roles the user has
            if role in roles:
                filtered_docs.append(doc)
                break
    return filtered_docs


def parse_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


user1_roles = ["admin", "manager", "employee", "all"]
user2_roles = ["manager", "all"]
user3_roles = ["employee", "all"]
user4_roles = ["manager", "employee", "all"]


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


prompt = hub.pull("rlm/rag-prompt")

print(prompt)

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()


def build_chain(roles: list[str]) -> RunnableSequence:
    filtered_docs = filter_documents(all_docs, roles=roles)
    vectorstore = FAISS.from_documents(filtered_docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "score_threshold": 0.5,
            # "k": 1,
        },
    )

    rag_chain = (
        {"context": retriever | parse_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    # print(type(rag_chain))

    return rag_chain


def query(question: str, roles: list[str]) -> str:
    filtered_docs = filter_documents(all_docs, roles=roles)
    vectorstore = FAISS.from_documents(filtered_docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "score_threshold": 0.5,
            # "k": 1,
        },
    )

    rag_chain = (
        {"context": retriever | parse_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return rag_chain.invoke(question)


user1 = build_chain(user1_roles)
user2 = build_chain(user2_roles)
user3 = build_chain(user3_roles)
user4 = build_chain(user4_roles)

questions = ["who is hilary lloyd?", "who is margaret?"]

for question in questions:
    print("User 1:", question)
    print(user1.invoke(question))
    print()
    print("User 2:", question)
    print(user2.invoke(question))
    print()
    print("User 3:", question)
    print(user3.invoke(question))
    print()
    print("User 4:", question)
    print(user4.invoke(question))
    print()
    print()

# print(query("what is the root of lorem ipsum?", user1_roles))
# print(query("lorem ipsum relation to web dev", user1_roles))

# print("User 1:")
# print(query("who is hilary lloyd?", user1_roles))
# print()
# print("User 3:")
# print(query("who is hilary lloyd?", user3_roles))


# chain = model | parser

# pprint(retriever.invoke("What is the root of lorem ipsum?"))
# pprint(retriever.invoke("historical facts"))
# pprint(retriever.invoke("How has lorem ipsum changed over the years?"))
# pprint(retriever.invoke("How is lorem ipsum used in web pages?"))

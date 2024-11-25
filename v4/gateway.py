from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


from database_stuff import retriever


prompt = hub.pull("rlm/rag-prompt")

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()


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


# concatenates the contents of a list of documents into a single string
def parse_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def query(question: str, roles: list[str]) -> str:

    def filter_results(docs: list[Document]) -> list[Document]:
        return filter_documents(docs, roles)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(filter_results) | parse_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | parser
    )

    return rag_chain.invoke(question)

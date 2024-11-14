import streamlit as st
import csv
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

# Initialize environment and ChromaDB Client
load_dotenv()
client = chromadb.Client()

# Paths for user data and sample documents
USER_DATA_FILE = "user_data.csv"
PUBLIC_DOC_PATH = "C:/Ackuity/public/pub_doc1.txt"
PRIVATE_DOCS_PATH = ["C:/Ackuity/private/secrets1.txt", "C:/Ackuity/private/secrets2.txt"]

# Load user data from the CSV file for data persistence
def load_user_data():
    # create dictionary to store username as keys and passwords as their value
    user_data = {}
    try:
        # open file in read mode. with is essentially a try with resources
        with open(USER_DATA_FILE, mode="r") as file:
            # CSV reader object to read file line by line
            reader = csv.reader(file)
            # iterate through each row
            for row in reader:
                if len(row) == 2:
                    # first element of row will be assigned to username and second to password
                    username, password = row
                    # adds username and password pair to the dictionary
                    user_data[username] = password
    except FileNotFoundError:
        pass  # Returns an empty dictionary if the file doesn't exist
    return user_data

# Save user data to the CSV file
def save_user_data(data):
    with open(USER_DATA_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        # loops thru data dictionary and writes username, password to file
        for username, password in data.items():
            writer.writerow([username, password])

# Load initial user data from CSV file
user_data = load_user_data()

# Function to create a new user account
def create_account(username, password):
    user_data[username] = password # assign key and value in user data dictionary
    save_user_data(user_data) # write dict to file

# Function to validate user login
def validate_login(username, password):
    return user_data.get(username) == password

def process_and_store_documents():
    # create chunker and vectorstore that will have all the embeded docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

    # Load and process public and private documents
    pub_doc = TextLoader(PUBLIC_DOC_PATH).load()
    private_docs = [TextLoader(path).load() for path in PRIVATE_DOCS_PATH]

    # Split public and private docs into chunks
    pub_chunks = text_splitter.split_documents(pub_doc)
    private_chunks = [text_splitter.split_documents(doc) for doc in private_docs]

    # Prepare the text and metadata for embedding
    # create a texts list with all the content from pub chunks
    # page content function contains the text in that chunk
    texts = [chunk.page_content for chunk in pub_chunks]
    # create a metadata list for the chunks
    # "user 1" is allowed to access public document
    metadatas = [{"allowed_users": "user1"}]

    for chunk in private_chunks[0]:
        texts.append(chunk.page_content)
        metadatas.append({"allowed_users": "user2"})

    for chunk in private_chunks[1]:
        texts.append(chunk.page_content)
        metadatas.append({"allowed_users": "user3"})

    # Adds the texts and metadatas to the vectorstore.  
    # The embeddings of the texts are automatically generated 
    # and stored as part of the vectorstore
    vectorstore.add_texts(texts, metadatas)

    return vectorstore  # Return the Chroma vector store

# Function to filter results based on user permissions
def filter_results(username, results):
    return [
        res for res in results
        if "allowed_users" in res.metadata and username in res.metadata["allowed_users"]
    ]

# Function to query documents based on user prompt
def query_documents(prompt, username):
    # Retrieve the vectorstore from the session state
    vectorstore = st.session_state.vectorstore
    results = vectorstore.similarity_search(query=prompt, k=1) # conduct similarity search
    return filter_results(username, results)


# Streamlit State Management. Set log in to false at first
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# if user is not logged in allow them to log in or create account   
if not st.session_state.logged_in:
    with st.expander("Create a new account"):
        new_username = st.text_input("Choose a username:")
        new_password = st.text_input("Choose a password", type="password")
        if st.button("Create Account"):
            if new_username and new_password:
                # call create account function w username and password typed in
                create_account(new_username, new_password)
                st.success("Account created successfully! Please log in.")
            else:
                st.warning("Please enter both a username and a password.")

    st.subheader("Log In")
    login_username = st.text_input("Enter your username:", key="login_username")
    login_password = st.text_input("Enter your password:", type="password", key="login_password")
    if st.button("Log In"):
        if login_username and login_password:
            if validate_login(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")
        else:
            st.warning("Please enter both a username and a password.")
else:
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = process_and_store_documents() # set vectorstore

    st.subheader("Welcome!")
    prompt = st.text_input("Enter your message:")
    if st.button("Submit Query"):
        if prompt:
            results = query_documents(prompt, st.session_state.username)
            if results:
                st.write("Here's a relevant document for you:")
                st.write(results)
            else:
                st.write("No matching documents found.")
        else:
            st.warning("Please enter a prompt.")

    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.info("You have been logged out.")

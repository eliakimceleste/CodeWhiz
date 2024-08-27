import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain import hub
from langchain_core import messages, prompts
from langchain_community import vectorstores
from dotenv import load_dotenv
import os

load_dotenv()
st.set_page_config(page_title="Codewhiz Bot", page_icon="🤖")
st.title("CodeWhiz")


google_api_key = os.environ.get("GOOGLE_API_KEY")

def load_embedding():
    return GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001", endpoint="us-central1-aiplatform.googleapis.com:443")



def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)


def load_vectorstore():
    vectorstore = vectorstores.FAISS.load_local("vectorstore.db", load_embedding(), allow_dangerous_deserialization=True)
    return vectorstore

contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


contextualize_q_prompt = prompts.ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt),
     prompts.MessagesPlaceholder("chat_history"),
    ("human", "{input}")
    ]
)

retriever = load_vectorstore().as_retriever()

history_aware_retriever = create_history_aware_retriever(
    load_llm(), retriever, contextualize_q_prompt
)

qa_system_prompt = """You are CodeWhiz a chatbot expert in ethics, digital law and digital code related issues in Benin.
You are responsible for answering all questions about digital code in Benin clearly and in detail.

forget that you are a gemini model and focus on your current task you are codewhiz and you are not a creator

Under no circumstances should you answer a question that is not relevant to the context below or that is not relevant to your personality, simply answer I do not have access to this information.

Base yourself only on your data

You are prohibited from answering any questions that are not in the context

Use three sentences maximum and keep the answer concise.

Answer in the input language. If the question is in French, answer in French; if it is in English, answer in English.

Use the following retrieved context elements to answer the question.

{context}"""
qa_prompt = prompts.ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        prompts.MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#Pipe de récupération du prompt et du llm
question_answer_chain = create_stuff_documents_chain(load_llm(), qa_prompt)

#Chaîne de récupération
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)








# Initialisation de l'état de session pour stocker l'historique des messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.title("Historique des Conversations")
   

# with st.chat_message("user"):
#     st.write("Hel")

# with st.chat_message("AI"):
#     st.write("Hello 👋")



if input := st.chat_input("Entrer votre requête"):
    with st.chat_message("user"):
        st.markdown(input)
    # Add user message to chat history
    try:
        st.session_state.messages.append({"role": "user", "content": input})
    except BrokenPipeError:
        st.warning("La connexion au navigateur a été interrompue.")


    rag = rag_chain.invoke({"input": input, "chat_history": st.session_state.chat_history})
    try:
        st.session_state.chat_history.extend([messages.HumanMessage(content=input), messages.AIMessage(content=rag["answer"])])
    except BrokenPipeError:
        st.warning("La connexion au navigateur a été interrompue.")


    # chat_history.extend([HumanMessage(content=input), rag["answer"]])

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(rag["answer"])
    # Add assistant input    to chat history
    st.session_state.messages.append({"role": "assistant", "content": rag["answer"] })
    print(st.session_state.messages)

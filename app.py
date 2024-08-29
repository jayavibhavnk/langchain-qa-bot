import streamlit as st
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def load_faiss_embeddings(path):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY)
    db = FAISS.load_local("langchain_faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state.vector_store = db
    print("db loaded")

def get_prompt():
    system_message = SystemMessagePromptTemplate.from_template(
    """
    You are an expert coder who knows solutions for generative ai problems using langchain python library
    using the given content you will generate langchain code for the users question
    do not hallucinate 
    try to answer using the given content and your own knowledge
    if the solution is not possible, give partial solutions 
    give code followed by explanation
    Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
    {context}
    """
    )
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    return system_message, human_message

def get_conversation_chain(vector_store: FAISS, system_message: str, human_message: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model="GPT-4o", openai_api_key=st.secrets.OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def query_with_link(query):
    vector_store = st.session_state.vector_store
    new_db = vector_store.similarity_search(query)
    relevant_links = [i.metadata['source'] for i in new_db]
    rel_links = []
    for i in relevant_links:
        if i not in rel_links:
            rel_links.append(i + "\n")
    links = '\n'.join(rel_links)
    response_from_chatgpt = query_from_doc(query)
    final_response = response_from_chatgpt + "\n\nHere are some of the relevant links: \n\n" + links
    return final_response

def query_from_doc(text):
    response = st.session_state.conversation({"question": text, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history = response['chat_history']
    return response['answer']

def main():
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about Langchain"}]

    st.title("Langchain Documentation")
    st.subheader("Ask anything about langchain")

    if st.session_state.vector_store is None:
        load_faiss_embeddings("db_faiss")

    system_message_prompt, human_message_prompt = get_prompt()

    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, system_message_prompt, human_message_prompt)

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                msg = query_with_link(prompt)
                st.write(msg)
                message = {"role": "assistant", "content": msg}
                st.session_state.messages.append(message)
                # Update chat history to maintain conversation state
                st.session_state.chat_history.append((prompt, msg))

if __name__ == "__main__":
    main()

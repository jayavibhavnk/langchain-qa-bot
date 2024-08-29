import streamlit as st
import time
import openai
from PIL import Image
from io import BytesIO
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
client = OpenAI()

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY
# HF_KEY = st.secrets.HF_KEY
openai.api_key=OPENAI_API_KEY
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

st.set_page_config(
        page_title="Langchain QA",
        page_icon="✍️",
        # layout="wide",
        initial_sidebar_state="expanded",
    )

st.sidebar.title("Generation Type:")

customization_options = {
        "Generation_type": st.sidebar.radio("Generation Type", ["Text", "Image"])
    }


st.title("💬 Langchian QA")

def get_image_from_api(text):
    # Replace with your API function call
    import requests

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": "{}".format(text),
    })

    return image_bytes

def query_langchain(query):
      
    llm = ChatOpenAI(model="gpt-4o-mini")

    store = FAISS.load_local(
    "langchain_faiss_index", embeddings, allow_dangerous_deserialization=True
    )
  
    retriever = store.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    res = rag_chain.invoke(query)

    return res

def query_openai(query):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": query}
    ],
    n = 1
    )

    return(completion.choices[0].message.content)

if customization_options['Generation_type'] == "Text":
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Thinking..."):
                msg = query_langchain(prompt)

                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)

if customization_options['Generation_type'] == "Image":
    try:
        st.write("Enter Image Generation Prompt:")
        prompt = st.text_input("Prompt: ")
        if prompt:
            with st.spinner("Generating Image..."):
                img = get_image_from_api(prompt)
                # st.write("Generated Image:")
                # print(img)
                if img:
                    image = Image.open(BytesIO(img))
                    st.image(image, caption="Generated Image", use_column_width=True)
    except:
        pass

import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from streamlit-extras.add_vertical_space import add_vertical_space
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

###
#from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.chat_models import ChatOpenAI
###

from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
load_dotenv()

#Sidebar contents
with st.sidebar:
    st.title('🙂 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
                
    ''')
    #add_vertical_space(5)
    st.write('Made with 🧡 by Muskan')



def main():
    st.header("Chat with PDF 💭")

    #upload a PDF file
    pdf= st.file_uploader("Upload your pdf", type='pdf')


    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        #st.write(pdf_reader)

        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)


        ### embeddings
        
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            #st.write('Embeddings loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings(
            openai_api_key='Enter your open api key'
        )
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            #st.write('Embeddings computation completed')
                
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs =VectorStore.similarity_search(query=query, k=4)
            llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key='Enter your open api key')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
        




        ########
            
        #docs =VectorStore.similarity_search(query=query, k=3)
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm=ChatOpenAI(model="gpt-3.5-turbo-instruct", openai_api_key='Enter your open api key'),
        #     chain_type="stuff",
        #     retriever=VectorStore.as_retriever(),
        #     get_chat_history=lambda h:h,
        #     verbose=True
        # )
        # response=qa.run({"question":query})


    

    
if __name__ == '__main__':
    main()


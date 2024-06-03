import streamlit as st
import os
import doc_splitter_vectorise
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import secret_key



#loading llm model
def load_llm_model():
    global llm
    REPO_ID = "tiiuae/falcon-7b-instruct"
    MAX_NEW_TOKENS = 200
    hugging_face_token = secret_key.hugging_face_token
    llm = HuggingFaceHub(huggingfacehub_api_token =hugging_face_token,  repo_id=REPO_ID,
                model_kwargs={
                    "temperature": 0.1,
                    "max_new_tokens":MAX_NEW_TOKENS,
                },
            )


def qa(input_qa):
    print("in qa")
    load_llm_model()
    global llm
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=doc_splitter_vectorise.vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": input_qa})
    result["result"]
    print(result["result"])
    main_split = result["result"].split('Helpful Answer')
    print(len(main_split()))
    st.write("Reply: ", main_split[1])

def main():
    st.header("Chat with PDF")
    input_qa = st.text_input("Ask a question related to pdf")

    if input_qa:
        qa(input_qa) 

    with st.sidebar:
        st.title('Upload pdf')

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                print("Here")
                docs = doc_splitter_vectorise.get_pdf_text(pdf_docs)
                print("Done with splitting", len(docs))
                doc_splitter_vectorise.get_vectordb(docs)
                st.success("Done")   


if __name__ == "__main__":
    main()
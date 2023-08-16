# Streamlit imports.
import os
import pypdf
import numpy as np
import streamlit as st
import streamlit_toggle as tog

# QA Custom classes.
from src.load_document_st import LoadDocument
from src.get_embeddings import GetEmbeddings
from src.qa_context import QAWithContext

# Set page components.
ss = st.session_state
st.set_page_config(page_title='QA Over Documents')
st.header('QA Over Documents')
ENABLE_DETAILS = False
SET_API_KEY = False
enabled_info = tog.st_toggle_switch(
    label="View process details",
    key="Key1",
    default_value=False,
    label_after = False,
    inactive_color = '#D3D3D3',
    active_color="#11567f",
    track_color="#29B5E8"
)

# Save or update OpenAI Api Key.
def save_env():
    api_key = ss.get('api_key')
    with open('.env', 'w') as f:
        f.write('OPENAI_API_KEY=' + api_key.strip())

change_api_key = st.button('Save or update OpenAI API key')
if change_api_key:
    SET_API_KEY = True

if SET_API_KEY:
    api_key = st.text_input(
        'OpenAI API key', 
        type='password', 
        key='api_key',
        on_change=save_env,
        label_visibility="collapsed"
    )


# Updload files. Multiple files are supported.
st.write('')
st.write('### Upload your documents')
uploaded_files = st.file_uploader(
    'Upload your document', 
    accept_multiple_files=True,
    type=['txt', 'py', 'pdf'],
    label_visibility="collapsed"
)

def get_text_from_file(file):
    if '.pdf' in file.name:
        f = file.read()
        pdf_reader = pypdf.PdfReader(file.read())
        text = ''
        print('pags',len(pdf_reader.pages))
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            text += page.extract_text()
        print(text[0:1000])
        return text
    else:
        return file.getvalue().decode('utf-8')
# Catch user actions.
if enabled_info:
    ENABLE_DETAILS = True

if uploaded_files:
    list_uploaded_files = [
        [
            get_text_from_file(file),
            file.name
        ] for file in uploaded_files
    ]
    obj_load_doc = LoadDocument()
    lgc_documents = obj_load_doc.get_lgc_documents(list_uploaded_files)

    if ENABLE_DETAILS:
        st.text_area(
            'Text loaded', 
            value=lgc_documents
        )
    obj_embeddings = GetEmbeddings('vector_store_st')

    chunk_size_limit = 1000
    max_chunk_overlap = 500

    (
        splitted_doc, total_word_count, total_token_count, total_token_cost
    ) = obj_embeddings.calc_estimated_cost(
        lgc_documents, chunk_size_limit, max_chunk_overlap
    )
    if ENABLE_DETAILS:
        st.text_area(
            'First embeddings details', 
            value=f'Chunk size: {chunk_size_limit}\
                \nOvelap: {max_chunk_overlap}\
                \nTotal word count: {total_word_count}\
                \nEmbedding Cost: ${total_token_cost} MXN\
            '
        )

    vector_store_small = obj_embeddings.get_embeddings_st(
        lgc_documents, chunk_size_limit, max_chunk_overlap, 
        'vector_store_small_chunk'
    )

    chunk_size_limit = 1200
    max_chunk_overlap = 800

    (
        splitted_doc, total_word_count, total_token_count, total_token_cost
    ) = obj_embeddings.calc_estimated_cost(
        lgc_documents, chunk_size_limit, max_chunk_overlap
    )

    if ENABLE_DETAILS:
        st.text_area(
            'First embeddings details', 
            value=f'Chunk size: {chunk_size_limit}\
                \nOvelap: {max_chunk_overlap}\
                \nTotal word count: {total_word_count}\
                \nEmbedding Cost: ${total_token_cost} MXN\
            '
        )

    vector_store_large = obj_embeddings.get_embeddings_st(
        lgc_documents, chunk_size_limit, max_chunk_overlap, 
        'vector_store_large_chunk'
    )

    obj_qa_context = QAWithContext('vector_store_st')

    st.write('')
    st.write('### Ask a question about your data')
    user_question = st.text_input("Ask a question about your Data:", label_visibility="collapsed")

    if user_question:
        vector_store, n_k_args = obj_qa_context.define_vector_store_to_use(
            user_question
        )

        result, info_log = obj_qa_context.ask_chat_gpt_w_multiq(
            user_question, vector_store, n_k_args
        )
        info_log = info_log.replace('Generated queries: ', '').strip()
        arr = [i.strip().replace("'",'') for i in info_log[1:-1].split(",")]

        info_log = """Generates multiple queries 
            from different perspectives for the given question.
            \n\n""" + str(arr[0]) + "\n" + str(arr[1]) + "\n" + str(arr[2])
        st.info(info_log, icon="ℹ️")
        st.write(result['answer'])

        
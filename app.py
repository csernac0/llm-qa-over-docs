import streamlit as st
import  streamlit_toggle as tog
from src.load_document_st import LoadDocument
from src.get_embeddings import GetEmbeddings
from src.qa_context import QAWithContext

st.set_page_config(page_title='QA Over Documents')
st.header('QA Over Documents')
uploaded_files = st.file_uploader(
    'Upload your document', accept_multiple_files=True,
    type=['txt', 'py', 'pdf']
)
tog.st_toggle_switch(label="Guide throughout solution", 
                    key="Key1", 
                    default_value=False, 
                    label_after = False, 
                    inactive_color = '#D3D3D3', 
                    active_color="#11567f", 
                    track_color="#29B5E8"
                    )
if uploaded_files:
    list_uploaded_files = [
        [
            file.getvalue().decode('utf-8'),
            file.name
        ] for file in uploaded_files
    ]

    obj_load_doc = LoadDocument()
    lgc_documents = obj_load_doc.get_lgc_documents(list_uploaded_files)

    st.write("Docs: ", lgc_documents)

    obj_embeddings = GetEmbeddings('vector_store_st')

    chunk_size_limit = 1000
    max_chunk_overlap = 500
    st.write(
        f"""Creating embeddings for chunk_size: {chunk_size_limit}, 
        ovelap: {max_chunk_overlap}"""
    )

    (
        splitted_doc, total_word_count, total_token_count, total_token_cost
    ) = obj_embeddings.calc_estimated_cost(
        lgc_documents, chunk_size_limit, max_chunk_overlap
    )

    st.write(f"""Total word count: {total_word_count}""")
    st.write(f"""Total tokens: {total_token_count}""")
    st.write(f"""Embedding Cost: ${total_token_cost} MXN""")

    vector_store_small = obj_embeddings.get_embeddings_st(
        lgc_documents, chunk_size_limit, max_chunk_overlap, 
        'vector_store_small_chunk'
    )

    chunk_size_limit = 1200
    max_chunk_overlap = 800
    st.write(
        f"""Creating embeddings for chunk_size: {chunk_size_limit}, 
        ovelap: {max_chunk_overlap}"""
    )

    (
        splitted_doc, total_word_count, total_token_count, total_token_cost
    ) = obj_embeddings.calc_estimated_cost(
        lgc_documents, chunk_size_limit, max_chunk_overlap
    )

    st.write(f"""Total word count: {total_word_count}""")
    st.write(f"""Total tokens: {total_token_count}""")
    st.write(f"""Embedding Cost: ${total_token_cost} MXN""")

    vector_store_large = obj_embeddings.get_embeddings_st(
        lgc_documents, chunk_size_limit, max_chunk_overlap, 
        'vector_store_large_chunk'
    )

    obj_qa_context = QAWithContext('vector_store_st')

    user_question = st.text_input("Ask a question about your Data:")

    if user_question:
        vector_store, n_k_args = obj_qa_context.define_vector_store_to_use(
            user_question
        )

        result = obj_qa_context.ask_chat_gpt_w_multiq(
            user_question, vector_store, n_k_args
        )
        st.write(result['answer'])
        
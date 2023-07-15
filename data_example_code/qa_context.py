import os
import logging
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display, Markdown

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQAWithSourcesChain


class QAWithContext(object):
    def __init__(self, documents_dir):
        """
        Instanciate OpenAI api key from .env file.
        Instanciate documents dir.
        """
        self.documents_dir = documents_dir
        load_dotenv()
        os.environ[
            "OPENAI_API_KEY"
        ] = os.getenv("OPENAI_API_KEY")

    def print_result(self, result):
        """
        Format results.
        
        Parameters:
        -----------
            result : json
                Results of QA. Keys: ['question', 'answer', 'source_documents']
        Returns:
        -----------
            output_text: Markdown 
                Formated text.
        """
        output_text = f"""
          ### Question: 
          {result['question']}
          ### Answer: 
          {result['answer']}
          ### All relevant sources:
          {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
        """
        display(Markdown(output_text))

    def embedding_result_similarity(self, vector_store_dir, str_question):
        """
        Returns similarity scores for the stored vector embeddings & 
        user question.
        
        Parameters:
        -----------
            vector_store_dir : string
                Name of folder where is located vector store.
            str_question : string
                User question.
        Returns:
        -----------
            vector_store: angchain.vectorstores.FAISS
                Embedding vectors as FAISS object.
            min_similarity: float
                Min similarity score between top 3 more similar documents.
            mean_similarity: float
                Mean similarity score between top 3 more similar documents.
        """
        vector_store = FAISS.load_local(
            self.documents_dir + '/' + vector_store_dir,
            OpenAIEmbeddings()
        )
        search_result = vector_store.similarity_search_with_score(
            str_question, k=3
        )

        search_result = pd.DataFrame(search_result)
        search_result.columns = ['doc', 'similarty']
        search_result = search_result.sort_values(
            by='similarty', ascending=True
        ).reset_index()
        del search_result['index']
        min_similarity = search_result.similarty.min()
        mean_similarity = search_result.head(3).similarty.mean()
        return [vector_store, min_similarity, mean_similarity]

    def define_vector_store_to_use(self, str_question):
        """
        For two different document chunks it selects the one with 
        the closest similarity.
        
        Parameters:
        -----------
            str_question : string
                User question.
        Returns:
        -----------
            vector_store: angchain.vectorstores.FAISS 
                The most appropriate embedding for the given user question.
        """
        (
            vector_store_large, min_similarity_large, mean_similarity_large
        ) = self.embedding_result_similarity(
            'vector_store_large_chunk', str_question
        )
        (
            vector_store_small, min_similarity_small, mean_similarity_small
        ) = self.embedding_result_similarity(
            'vector_store_small_chunk', str_question
        )
        print(
            'Min similarity in large chunk embeds:', 
            min_similarity_large
        )
        print(
            'Min similarity in small chunk embeds:', 
            min_similarity_small
        )
        if min_similarity_large<=min_similarity_small:
            print('Select large chunk embeds')
            vector_store = vector_store_large
            n_k_args = 4
        else:
            print('Select small chunk embeds')
            embeds_to_use = vector_store_small
            n_k_args = 8
            
        return [vector_store, n_k_args]

    def generate_prompt(self):
        """
        Generate promp fro the LLM model.
        
        Returns:
        -----------
            output_text: Markdown 
                Formated text.
        """
        system_template="""
            Use the following context 
            to answer the users question.
            Take note of the sources and include 
            them in the answer in the format: 
                "SOURCES: source1 source2", 
                use "SOURCES" in capital letters regardless 
                of the number of sources.
            If you don't know the answer, 
            just say that "I don't know" 
            and don't try to make up an answer.
            ----------------
            {summaries}
        """
        messages = [
            SystemMessagePromptTemplate.from_template(
                system_template
            ),
            HumanMessagePromptTemplate.from_template(
                "{question}"
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain_type_kwargs = {"prompt": prompt}
        return chain_type_kwargs

    def ask_chat_gpt(self, str_query, vector_store, n_k_args):
        """
        Format results.
        
        Parameters:
        -----------
            result : json
                Results of QA. Keys: ['question', 'answer', 'source_documents']
        Returns:
        -----------
            output_text: Markdown 
                Formated text.
        """
        chain_type_kwargs = self.generate_prompt()
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k", 
            temperature=0, 
            max_tokens=400
        )
        
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_kwargs={"k": n_k_args},
                search_type="mmr"
            ),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        
        result = chain(str_query)
        return result

    def ask_chat_gpt_w_multiq(self, str_query, vector_store, n_k_args):
        """
        Format results.
        
        Parameters:
        -----------
            result : json
                Results of QA. Keys: ['question', 'answer', 'source_documents']
        Returns:
        -----------
            output_text: Markdown 
                Formated text.
        """
        chain_type_kwargs = self.generate_prompt()
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k", 
            temperature=0, 
            max_tokens=400
        )
        
        # llm, and vector_store.as_retriever can be configured.
        logging.basicConfig()
        logging.getLogger(
            'langchain.retrievers.multi_query'
        ).setLevel(logging.INFO)
        retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=vector_store.as_retriever(
                search_kwargs={"k": n_k_args},
                search_type="mmr"
            ),
            llm=llm
        )
        multi_q_docs = retriever_from_llm.get_relevant_documents(
            query=str_query
        )

        embeddings = OpenAIEmbeddings()
        vector_store_multi_q = FAISS.from_documents(
            multi_q_docs, embeddings
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store_multi_q.as_retriever(
                search_kwargs={"k": n_k_args},
                search_type="mmr"
            ),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        
        result = chain(str_query)
        return result

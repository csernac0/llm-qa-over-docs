import os
import logging
import tiktoken
import numpy as np
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class GetEmbeddings(object):
    def __init__(self, documents_dir):
        """
        Instanciate OpenAI api key from .env file.
        Instanciate documents dir.
        """
        load_dotenv(find_dotenv())
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.documents_dir = documents_dir

    def split_text_chunks(
        self, documents, chunk_size_limit, max_chunk_overlap
    ):
        """
        Split complete document text into chunks. 
        The lenght of each chunk is determined by chunk_size_limit.
        Also an overlap between chunks is permited.
        
        Parameters
        ----------
        documents : langchain.docstore.document.Document
            Langchain document object.
        chunk_size_limit : int
            Lenght of each chunk in characters.
        max_chunk_overlap : int
            Character overlap.
        Returns
        -------
        splitted_doc: langchain.text_splitter.CharacterTextSplitter
            Document into chunks, including metadata.
        """
        # Set logger.
        logging.getLogger(
            "langchain.text_splitter"
        ).setLevel(logging.CRITICAL)
        # Instanciate text splitter object with desired params.
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size_limit, 
            chunk_overlap=max_chunk_overlap
        )
        # Split text into chunks.
        splitted_doc = text_splitter.split_documents(
            documents
        )
        if len(splitted_doc)==1:
            # Instanciate text splitter object with desired params.
            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=chunk_size_limit, 
                chunk_overlap=max_chunk_overlap
            )
            # Split text into chunks.
            splitted_doc = text_splitter.split_documents(
                documents
            )
        print('Splitted doc:', len(splitted_doc))
        return splitted_doc

    def get_embeddings_st(
        self, documents, chunk_size_limit, max_chunk_overlap, dir_to_store
    ):
        """
        Create chunks and then get embedding for each chunk. 
        If embeddings already exists, then load them.
        
        Parameters
        ----------
        documents : langchain.docstore.document.Document
            Langchain document object.
        chunk_size_limit : int
            Lenght of each chunk in characters.
        max_chunk_overlap : int
            Character overlap.
        dir_to_store : string
            Path to store.
        Returns
        -------
        vector_store: angchain.vectorstores.FAISS
            Embedding vectors as FAISS object.
        """
        
        # If not exists, create them.
        splitted_doc = self.split_text_chunks(
            documents, chunk_size_limit, max_chunk_overlap
        )
        # Use OpenAI embeddings service.
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(
            splitted_doc, embeddings
        )
        print('Saved')
        # If path not exists, create it.
        if not os.path.isdir(self.documents_dir + '/' + dir_to_store):
            os.makedirs(self.documents_dir + '/' + dir_to_store)
            print(
                "created folder : ", 
                self.documents_dir + '/' + dir_to_store
            )
        # Save embeddings store.
        vector_store.save_local(self.documents_dir + '/' + dir_to_store)
        return vector_store

    def get_embeddings(
        self, documents, chunk_size_limit, max_chunk_overlap, dir_to_store
    ):
        """
        Create chunks and then get embedding for each chunk. 
        If embeddings already exists, then load them.
        
        Parameters
        ----------
        documents : langchain.docstore.document.Document
            Langchain document object.
        chunk_size_limit : int
            Lenght of each chunk in characters.
        max_chunk_overlap : int
            Character overlap.
        dir_to_store : string
            Path to store.
        Returns
        -------
        vector_store: angchain.vectorstores.FAISS
            Embedding vectors as FAISS object.
        """
        
        if os.path.exists(self.documents_dir + '/' + dir_to_store):
            # If vector store exists then load them.
            print('Vector store already created.')
            vector_store = FAISS.load_local(
              self.documents_dir + '/' + dir_to_store,
              OpenAIEmbeddings()
          )
        else:
            # If not exists, create them.
            splitted_doc = self.split_text_chunks(
                documents, chunk_size_limit, max_chunk_overlap
            )
            # Use OpenAI embeddings service.
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(
                splitted_doc, embeddings
            )
            print('Saved')
            # If path not exists, create it.
            if not os.path.isdir(self.documents_dir + '/' + dir_to_store):
                os.makedirs(self.documents_dir + '/' + dir_to_store)
                print(
                    "created folder : ", 
                    self.documents_dir + '/' + dir_to_store
                )
            # Save embeddings store.
            vector_store.save_local(self.documents_dir + '/' + dir_to_store)
        return vector_store

    def calc_estimated_cost(
        self, documents, chunk_size_limit, max_chunk_overlap
    ):
        """
        Use this function in case you want to estimate embedding cost.
        
        Parameters
        ----------
        documents : langchain.docstore.document.Document
            Langchain document object.
        chunk_size_limit : int
            Lenght of each chunk in characters.
        max_chunk_overlap : int
            Character overlap.
        """
        splitted_doc = self.split_text_chunks(
            documents, chunk_size_limit, max_chunk_overlap, 
        )
        # Create a GPT-4 encoder instance.
        enc = tiktoken.encoding_for_model("gpt-4")
        # Calculate costs.
        total_word_count = sum(
            len(doc.page_content.split()) for doc in splitted_doc
        )
        total_token_count = sum(
            len(enc.encode(doc.page_content)) for doc in splitted_doc
        )
        total_token_cost = np.round(
            (total_token_count*0.0004/1000)*18,
            2
        )
        # Print costs.
        print(f"""Total word count: {total_word_count}""")
        print(f"""Total tokens: {total_token_count}""")
        print(f"""Embedding Cost: ${total_token_cost} MXN""")
        return [
            splitted_doc, total_word_count, total_token_count, total_token_cost
        ]

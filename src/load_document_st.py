import os
import re
import pathlib
from langchain.docstore.document import Document
import streamlit as st

class LoadDocument(object):

    def get_lgc_documents(self, documents_loaded):
        """
        Read txt files located in the given path, 
        then convert texts into document store langchain.
        This ssobject allows us to keep text and metadata as well.

        Returns
        -------
        lg_documents: langchain.docstore.document.Document
            The object has text and metadata of the file as well.
        """

        # Convert docs into langchain docs.
        lg_documents = [
            Document(
                page_content=file[0],
                metadata={
                    "source": file[1]
                }
            )
            for file in documents_loaded
        ]
        return lg_documents

import os
import re
import pathlib
from langchain.docstore.document import Document


class LoadDocument(object):
    def __init__(self, documents_dir, type_of_file):
        """
        Instanciate document dir.
        """
        self.documents_dir = documents_dir
        self.type_of_file = type_of_file

    def convert_path_to_doc_url(self, doc_name):
        """
        Convert folder and file name into an url.
        
        Parameters
        ----------
        doc_name : string
            Document name.
        document_folder : string
            Folder name.
        Returns
        -------
        url_name: string
            Url for document.
        """
        # Cast json to dataframe.
        # Convert path string into url.
        url_name = re.sub(
            f"{self.documents_dir}/(.*)\.[\w\d]+", f"/\\1",
            str(doc_name)
        )
        return url_name

    def get_lgc_documents(self):
        """
        Read txt files located in the given path, 
        then convert texts into document store langchain.
        This object allows us to keep text and metadata as well.

        Returns
        -------
        lg_documents: langchain.docstore.document.Document
            The object has text and metadata of the file as well.
        """
        # Get data path.
        data_path = pathlib.Path(
            os.path.join(self.documents_dir)
        )
        # Get all docs in path.
        document_f = list(
            data_path.glob('**/*' + self.type_of_file)
        )
        print(document_f)
        # Convert docs into langchain docs.
        lg_documents = [
            Document(
                page_content=open(file, "r").read(),
                metadata={
                    "source": self.convert_path_to_doc_url(
                        file
                    )
                }
            )
            for file in document_f
        ]
        return lg_documents

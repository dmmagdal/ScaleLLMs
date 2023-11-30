# flan_t5_qna_langchain.py
# Build a simple QnA bot with Flan T5 and langchain using a given set 
# of documents to reference.
# Windows/MacOS/Linux
# Python 3.10


from langchain.document_loaders import TextLoader               # load text files
from langchain.text_splitter import CharacterTextSplitter       # text splitter
from langchain.embeddings import HuggingFaceEmbeddings          # to use HuggingFace models
from langchain.vectorstores import FAISS                        # vector DB/store
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader    # load pdf files
from langchain.indexes import VectorstoreIndexCreator           # vectorize db with ChromaDB
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader    # load urls into document loader



def main():
	# Load the data via the document loader.
    loader = 

    # Chunk the text based on a chunk size.

    # Fetch numerical embeddings for the text.

    # Storing the embeddings to a vector store.

    # Similarity search with Flan T5.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
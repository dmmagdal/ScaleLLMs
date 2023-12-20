# flan_t5_qna_langchain.py
# Build a simple QnA bot with Flan T5 and langchain using a given set 
# of documents to reference.
# Windows/MacOS/Linux
# Python 3.10


import os
from langchain.document_loaders import TextLoader				# load text files
from langchain.text_splitter import CharacterTextSplitter		# text splitter
from langchain.embeddings import HuggingFaceEmbeddings			# to use HuggingFace models
from langchain.vectorstores import FAISS						# vector DB/store
from langchain.chains.question_answering import load_qa_chain
# from langchain import HuggingFaceHub
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader	# load pdf files
from langchain.indexes import VectorstoreIndexCreator			# vectorize db with ChromaDB
from langchain.chains import RetrievalQA						# combines a Retriever with QnA chain to do question answering
from langchain.document_loaders import UnstructuredURLLoader	# load urls into document loader


from langchain.document_loaders import PyPDFLoader						# load pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter		# text splitter
from langchain.embeddings import HuggingFaceEmbeddings					# to use HuggingFace models
from langchain.vectorstores import FAISS								# vector DB/store
# from langchain import HuggingFaceHub									# get (llm) model from huggingface hub
from langchain.llms import HuggingFaceHub									# get (llm) model from huggingface hub
from langchain.chains.question_answering import load_qa_chain			# loads a chain that you can use to do QA over a set of documents, but it uses ALL of those documents
from langchain.chains import RetrievalQA								# combines a Retriever with QnA chain to do question answering



def main():
	# API token.
	with open('.env') as f:
		os.environ["HUGGINGFACEHUB_API_TOKEN"] = f.read()

	###################################################################
	# Load data
	###################################################################
	# Use PyPDFLoader to load PDF file.

	# Load the data via the document loader.
	pdf_path = "What Hedge Funds Really Do - An Introduction to Portfolio Management.pdf"
	loader = PyPDFLoader(pdf_path)
	# txt_path = "What_Hedge_Funds_Really_Do.txt"
	# loader = TextLoader(txt_path)

	# Split PDF into pages.
	pages = loader.load_and_split()

	###################################################################
	# Chunk data
	###################################################################
	# Use the models to generate embedding vectors have maximum limits 
	# on the text fragments provided as input. If we are using these 
	# models to generate embeddings for our text data, it becomes 
	# important to chunk the data to a specific size before passing the
	# data to these models. We use the RecursiveCharacterTextSplitter 
	# here to split the data which works by taking a large text and 
	# splitting it based on a specified chunk size. It does this by 
	# using a set of characters.

	# Chunk the text based on a chunk size.
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1024,
		chunk_overlap=64,
		separators=['\n\n', '\n', '(?=>\. )', ' ', ''],
	)
	docs = text_splitter.split_documents(pages)

	###################################################################
	# Embed data
	###################################################################
	# In order to numerically represent unstructured data like text, 
	# documents, images, audio, etc., we need embeddings. The numerical
	# form captures the contextual meaning of what we are embedding. 
	# Here, we use the HuggingFaceHubEmbeddings object to create 
	# embeddings for each document.

	# Fetch numerical embeddings for the text.
	embeddings = HuggingFaceEmbeddings()	# Original. Uses some other model in sentence-transformers by default.
	# embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')	# Specify using BERT for embeddings. Gives very different outputs.

	###################################################################
	# Store embeddings
	###################################################################
	# Now we need a Vector Store for our embeddings. Here we are using 
	# FAISS. FAISS, short for Facebook AI Similarity Search, is a 
	# powerful library designed for efficient searching and clustering 
	# of dense vectors that offers a range of algorithms that can 
	# search through sets of vectors of any size, even those that may 
	# exceed the available RAM capacity.

	# Storing the embeddings to a vector store.
	db = FAISS.from_documents(docs, embeddings)

	###################################################################
	# Search embeddings
	###################################################################
	# We connect here to the hugging face hub to fetch the Flan-T5 
	# model.
	# We can define a host of model settings for the model, such as 
	# temperature and max_length.
	# The load_qa_chain function provides a simple method for feeding 
	# documents to an LLM. By utilizing the chain type as “stuff”, the 
	# function takes a list of documents, combines them into a single 
	# prompt, and then passes that prompt to the LLM.

	# Similarity search with Flan T5.
	llm = HuggingFaceHub(
		repo_id="google/flan-t5-base", 
		# repo_id="google/flan-t5-large", # Gives very different outputs.
		model_kwargs={"temperature": 1, "max_length": 1000000}
	)
	chain = load_qa_chain(llm, chain_type="stuff")

	# QUERYING TEST.
	query = "Explain in detail what is CAPM:"
	docs = db.similarity_search(query)
	output = chain.run(input_documents=docs, question=query)
	print('-' * 72)
	print(f"QUERY: {query}")
	print(f"RESPONSE: {output}")
	print()

	###################################################################
	# Querying
	###################################################################
	# Use the RetrievalQAChain to retrieve documents using a Retriever 
	# and then uses a QA chain to answer a question based on the 
	# retrieved documents. It combines the language model with the 
	# VectorDB’s retrieval capabilities.

	# Creating QA Chain with Flan-T5 Model.
	qa = RetrievalQA.from_chain_type(
		llm=llm, 
		chain_type="stuff",
		retriever=db.as_retriever(search_kwargs={"k": 3})
	)

	# Querying Our PDF.
	query = "What is the fundamental law of active portfolio management?"
	output = qa.run(query)
	print(f"QUERY: {query}")
	print(f"RESPONSE: {output}")
	print()

	query = "What goes into the Book Value of an asset's evaluation?"
	output = qa.run(query)
	print(f"QUERY: {query}")
	print(f"RESPONSE: {output}")
	print('-' * 72)

	###################################################################
	# Afterward/Conclusion
	###################################################################

	# Real world applications.
	# In the present age of data inundation, there is a constant 
	# challenge of obtaining relevant information from an overwhelming 
	# amount of textual data. Traditional search engines often fail to 
	# give accurate and context-sensitive responses to specific queries
	# from users. Consequently, an increasing demand for sophisticated 
	# natural language processing (NLP) methodologies has emerged, with
	# the aim of facilitating precise document question answering (DQA)
	# systems. A document querying system, just like the one we built, 
	# could be extremely useful to automate interaction with any kind 
	# of document like PDF, excel sheets, html files amongst others. 
	# Using this approach, a lot of context-aware extract valuable 
	# insights from extensive document collections.

	# Conclusion.
	# In this article, we began by discussing how we could leverage 
	# LangChain to load data from a PDF document. Extend this 
	# capability to other document types such as CSV, HTML, JSON, 
	# Markdown, and more. We further learned ways to carry out the 
	# splitting of the data based on a specific chunk size which is a 
	# necessary step before generating the embeddings for the text. 
	# Then, fetched the embeddings for the documents using 
	# HuggingFaceHubEmbeddings. Post storing the embeddings in a vector
	# store, we combined Retrieval with our LLM model ‘Flan-T5’ in 
	# question answering. The retrieved documents and an input question
	# from the user were passed to the LLM to generate an answer to the
	# asked question.

	# Key takeaways.
	# -> LangChain offers a comprehensive framework for seamless 
	#	interaction with LLMs, external data sources, prompts, and user 
	#	interfaces.  It allows for the creation of unique applications 
	#	built around an LLM by “chaining” components from multiple 
	#	modules.
	# -> Flan-T5 is a commercially available open-source LLM. It is a 
	#	variant of the T5 (Text-To-Text Transfer Transformer) model 
	#	developed by Google Research.
	# -> A vector store stores data in the form of high-dimensional 
	#	vectors. These vectors are mathematical representations of 
	#	various features or attributes. Design the vector stores to 
	#	efficiently manage dense vectors and provide advanced 
	#	similarity search capabilities.
	# -> The process of building a document-based question-answering 
	#	system using LLM model and Langchain entails fetching and 
	#	loading a text file, dividing the document into manageable 
	#	sections, converting these sections into embeddings, storing 
	#	them in a vector database and creating a QA chain to enable 
	#	question answering on the document.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
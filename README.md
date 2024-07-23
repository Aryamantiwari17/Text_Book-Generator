RAPTOR Textbook Processing and Question Answering System
This project involves extracting content from textbooks, chunking and embedding the data, creating a hierarchical RAPTOR index, and implementing a retrieval system with re-ranking for answering questions based on the indexed content.

Repository Structure
css
Copy code
.
├── README.md
├── requirements.txt
├── main.py
├── extract_textbook_content.py
├── chunk_text.py
├── embed_chunks.py
├── cluster_embeddings.py
├── build_raptor_tree.py
├── store_raptor_in_milvus.py
├── query_expansion.py
├── hybrid_retrieval.py
├── rerank_documents.py
├── answer_question.py
├── milvus_utils.py
└── textbooks
    ├── fesc111.pdf
    ├── fesc101.pdf
    └── fesc102.pdf
Dependencies
List all necessary dependencies in the requirements.txt file:

Copy code
nltk
PyPDF2
numpy
sentence-transformers
scikit-learn
rank-bm25
transformers
torch
google-generativeai
pymilvus
Setup Instructions
Clone the Repository

sh
Copy code
git clone <your-repository-url>
cd <repository-directory>
Install Dependencies

sh
Copy code
pip install -r requirements.txt
Download NLTK Data

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
Configure Google Generative AI

Set your Google Generative AI API key:

python
Copy code
import google.generativeai as genai
genai.configure(api_key="YOUR_GOOGLE_GENERATIVE_AI_API_KEY")
Prepare Textbooks

Ensure your textbooks are placed in the textbooks directory.

Running the System
Run the Main Script

sh
Copy code
python main.py
Detailed Description of Components
main.py
The main script coordinates the workflow:

Connects to Milvus.
Processes textbooks by extracting content, chunking, and embedding the data.
Creates RAPTOR trees and stores them in Milvus.
Implements a query loop for user interaction to answer questions.
extract_textbook_content.py
Extracts text content from PDF textbooks using PyPDF2.

chunk_text.py
Chunks the extracted text into manageable sizes for embedding, considering token limits.

embed_chunks.py
Generates embeddings for text chunks using SBERT.

cluster_embeddings.py
Clusters the embeddings using Gaussian Mixture Models.

build_raptor_tree.py
Builds a hierarchical RAPTOR tree from the clustered embeddings.

store_raptor_in_milvus.py
Stores the RAPTOR tree in Milvus.

query_expansion.py
Expands user queries using stemming and synonyms for better retrieval performance.

hybrid_retrieval.py
Combines SBERT and DPR embeddings for hybrid retrieval from Milvus.

rerank_documents.py
Re-ranks retrieved documents based on cosine similarity with the query.

answer_question.py
Generates answers to user queries based on the re-ranked documents using Google Generative AI.

milvus_utils.py
Utility functions for connecting to Milvus, creating collections, and managing indexes.

Textbook Titles and Links
Include the titles and links to the textbooks used for content extraction:

FESC 111 Textbook

Download Link
FESC 101 Textbook

Download Link
FESC 102 Textbook

Download Link
User Interface (Optional)
If a user interface is developed, provide instructions on how to access and use it.

Submitting the Project
Once everything is set up and running, submit the GitHub repository link by replying to the assessment invite email.


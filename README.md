# Textbook Question Answering System

This system uses advanced natural language processing and machine learning techniques to answer questions based on textbook content. It employs a RAPTOR (Recursive Application of Patterns To Organize Representation) indexing structure for efficient retrieval and a state-of-the-art language model for generating accurate answers.

## Table of Contents
1. [System Overview](#system-overview)
2. [Setup Instructions](#setup-instructions)
3. [Usage Guide](#usage-guide)
4. [Textbooks Used](#textbooks-used)
5. [System Components](#system-components)
6. [Dependencies](#dependencies)
7. [Known Limitations](#known-limitations)
8. [Contributing](#contributing)
9. [License](#license)

## System Overview

The system processes PDF textbooks, chunks the content, builds a RAPTOR index, and stores embeddings in a Milvus vector database. It uses a hybrid retrieval approach combining SBERT and DPR embeddings, re-ranks results, and generates answers using Google's Gemini Pro model.

## Setup Instructions

1. Clone this repository:
git clone https://github.com/yourusername/textbook-qa-system.git
cd textbook-qa-system

3. Install dependencies:
pip install -r requirements.txt
Copy
4. Set up Milvus:
- Follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
- Start the Milvus server

4. Configure API keys:
- Create a `.env` file in the project root
- Add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

5. Prepare textbooks:
- Place PDF textbooks in the `textbooks/` directory

## Usage Guide

1. Process textbooks and build the RAPTOR index:
python build_index.py
Copy
2. Start the question-answering interface:
python main.py
Copy
3. Enter your questions when prompted. Type 'quit' to exit.

## Textbooks Used

- FESC111 (link to textbook if publicly available)
- FESC101 (link to textbook if publicly available)
- FESC102 (link to textbook if publicly available)

## System Components

1. **Content Extraction**: Uses PyPDF2 to extract text from PDF textbooks.
2. **Data Chunking**: Splits text into manageable chunks using NLTK.
3. **RAPTOR Indexing**: 
- Generates embeddings using SentenceTransformer
- Clusters embeddings with GaussianMixture
- Builds a hierarchical RAPTOR tree structure
4. **Retrieval**:
- Implements hybrid retrieval using SBERT and DPR embeddings
- Performs query expansion with NLTK and WordNet
5. **Re-ranking**: Uses cosine similarity with SBERT embeddings
6. **Question Answering**: Utilizes Google's Gemini Pro model for answer generation

## Dependencies

- nltk
- PyPDF2
- sentence-transformers
- scikit-learn
- torch
- transformers
- google-generativeai
- pymilvus

For exact versions, see `requirements.txt`.

## Known Limitations

- The system's performance depends on the quality and relevance of the input textbooks.
- Very long textbooks may require significant processing time during indexing.
- The system may struggle with highly specialized or technical questions outside the scope of the textbooks.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

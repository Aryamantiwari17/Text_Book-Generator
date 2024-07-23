import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
import os
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRContextEncoder, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from pymilvus import MilvusClient
from pymilvus import model
from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus import connections, Collection, utility
from google.generativeai.types import GenerationConfig
from google.generativeai.types import GenerationConfig, SafetySettingDict, HarmCategory, HarmBlockThreshold



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

genai.configure(api_key="Enter your API-key")

sbert_model = SentenceTransformer('all-mpnet-base-v2')  # This model produces 768-dimensional embeddings
dpr_question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
dpr_context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
stemmer = PorterStemmer()

def extract_textbook_content(file_paths):
    all_text = []
    for pdf_path in file_paths:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        all_text.append(text)
    return all_text

def chunk_text(text, tokenizer, max_chunk_size=6048, overlap=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_tokens = tokenizer(sentence)
        sentence_length = len(sentence_tokens)

        if current_size + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = sentence_tokens
            current_size = sentence_length
        else:
            current_chunk.extend(sentence_tokens)
            current_size += sentence_length

        while current_size > max_chunk_size:
            chunks.append(' '.join(current_chunk[:max_chunk_size]))
            current_chunk = current_chunk[max_chunk_size-overlap:]
            current_size = len(current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def truncate_chunks(chunks, max_length=1024):
    return [chunk[:max_length] for chunk in chunks]

def embed_chunks(chunks, model):
    return model.encode(chunks)

def cluster_embeddings(embeddings, min_clusters=2, max_clusters=10):
    n_samples = len(embeddings)
    n_clusters = min(n_samples, max_clusters)
    
    if n_samples < min_clusters:
        print(f"Warning: Only {n_samples} samples available. Clustering skipped.")
        return [0] * n_samples
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gmm.fit_predict(embeddings)

def summarize_cluster(cluster_texts, max_length=100):
    combined_text = " ".join(cluster_texts)
    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content(f"Summarize the following text in less than {max_length} characters: {combined_text}")
    
    if response.parts:
        summary = response.parts[0].text
        return summary[:max_length].strip()
    else:
        return "Summary not available"

def build_raptor_tree(embeddings, clusters, texts, depth=0, max_depth=3):
    if depth == max_depth or len(texts) <= 1:
        return {"type": "leaf", "texts": texts, "embedding": np.mean(embeddings, axis=0)}
    
    cluster_summaries = []
    for i in range(max(clusters) + 1):
        cluster_texts = [text for j, text in enumerate(texts) if clusters[j] == i]
        summary = summarize_cluster(cluster_texts)
        cluster_summaries.append(summary)
    
    summary_embeddings = embed_chunks(cluster_summaries, sbert_model)
    sub_clusters = cluster_embeddings(summary_embeddings)
    
    children = []
    for i in range(max(sub_clusters) + 1):
        child_indices = [j for j, c in enumerate(sub_clusters) if c == i]
        child_embeddings = [summary_embeddings[j] for j in child_indices]
        child_texts = [cluster_summaries[j] for j in child_indices]
        children.append(build_raptor_tree(child_embeddings, sub_clusters, child_texts, depth+1, max_depth))
    
    return {"type": "internal", "summary": summarize_cluster(cluster_summaries),
            "embedding": np.mean(summary_embeddings, axis=0), "children": children}

def store_raptor_in_milvus(raptor_tree, collection, book_idx, node_idx=0):
    if raptor_tree["type"] == "leaf":
        text = " ".join(raptor_tree["texts"])
        if len(text) < 100:  # Only store chunks that are at least 100 characters long
            return
        data = [{
            "id": str(f"{book_idx}_{node_idx}"),
            "embedding": raptor_tree["embedding"].tolist(),
            "text": text,
            "type": "leaf"
        }]
    else:
        data = [{
            "id": str(f"{book_idx}_{node_idx}"),
            "embedding": raptor_tree["embedding"].tolist(),
            "text": raptor_tree["summary"],
            "type": "internal"
        }]
    
    try:
        collection.insert(data)
    except Exception as e:
        print(f"Error inserting data: {e}")
        print(f"Problematic data: {data}")
    
    if raptor_tree["type"] == "internal":
        for i, child in enumerate(raptor_tree["children"]):
            store_raptor_in_milvus(child, collection, book_idx, f"{node_idx}_{i}")

def connect_to_milvus():
    try:
        connections.connect(host='127.0.0.1', port='19530')
        print("Connected to Milvus successfully")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', max_length=100, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
        FieldSchema(name='text', dtype=DataType.VARCHAR, description='text content', max_length=65535),
        FieldSchema(name='type', dtype=DataType.VARCHAR, description='node type', max_length=20)
    ]
    schema = CollectionSchema(fields=fields, description='RAPTOR tree collection')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def query_expansion(query):
    expanded_query = []
    for word in nltk.word_tokenize(query):
        expanded_query.append(word)
        expanded_query.append(stemmer.stem(word))
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_query.append(lemma.name())
        except LookupError as e:
            print(f"Warning: {e}")
            print("Continuing without WordNet expansion for this word.")
    return ' '.join(set(expanded_query))


def get_collection_index_params(client, collection_name):
    try:
        index_info = client.list_indexes(collection_name)
        if index_info:
            return index_info[0]['params']
        else:
            print(f"No index found for collection {collection_name}")
            return None
    except Exception as e:
        print(f"Error getting index params: {e}")
        return None
    
def query_milvus(collection, query_vector, top_k=5):
    try:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "type"]
        )
        return results
    except Exception as e:
        print(f"Error in query_milvus: {e}")
        return []
    
    

def hybrid_retrieval(query, collection, top_k=50):
    try:
        # SBERT embedding
        sbert_embedding = sbert_model.encode(query)
        print(f"SBERT embedding shape: {sbert_embedding.shape}")
        sbert_results = query_milvus(collection, sbert_embedding, top_k)
        print(f"SBERT results: {len(sbert_results)}")

        # DPR embedding
        with torch.no_grad():
            question_embedding = dpr_question_encoder(**dpr_tokenizer(query, return_tensors="pt")).pooler_output
        dpr_embedding = question_embedding.squeeze().cpu().numpy()
        print(f"DPR embedding shape: {dpr_embedding.shape}")
        dpr_results = query_milvus(collection, dpr_embedding, top_k)
        print(f"DPR results: {len(dpr_results)}")

        # Combine results
        combined_results = []
        for hits in sbert_results + dpr_results:
            for hit in hits:
                combined_results.append((hit.entity.get('text'), hit.distance))

        # Remove duplicates and sort
        unique_results = list(set(combined_results))
        unique_results.sort(key=lambda x: x[1])

        return [text for text, _ in unique_results[:top_k]]
    except Exception as e:
        print(f"Error in hybrid_retrieval: {e}")
        return []


def rerank_documents(query, documents):
    query_embedding = sbert_model.encode([query])[0]
    doc_embeddings = sbert_model.encode(documents)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]
    return [documents[i] for i in ranked_indices]


def initialize_model():
    model = genai.GenerativeModel('gemini-pro')
    config = GenerationConfig(
        temperature=0.2,  # Slightly increased for more creative responses
        top_p=0.95,
        top_k=40,
        candidate_count=1,
        max_output_tokens=8192,  # Increased to allow for longer answers
    )
    safety_settings = [
        SafetySettingDict({
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
        }),
        SafetySettingDict({
            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
        }),
        SafetySettingDict({
            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
        }),
        SafetySettingDict({
            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
        }),
    ]
    return model, config, safety_settings


def generate_content(model, config, safety_settings, prompt):
    try:
        response = model.generate_content(
            prompt,
            generation_config=config,
            safety_settings=safety_settings,
            stream=False
        )
        return response
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def answer_question(query, context):
    model, config, safety_settings = initialize_model()
    
    # Truncate context if it's too long
    max_context_length = 30000  # Adjust as needed, keeping in mind Gemini's token limit
    if len(context) > max_context_length:
        context = context[:max_context_length]
    
    prompt = f"""Given the following context, please provide a comprehensive answer to the question. If the context doesn't contain enough information, state that and provide the best possible answer based on the available information.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = generate_content(model, config, safety_settings, prompt)
        
        if response.prompt_feedback.block_reason:
            print(f"Response blocked. Reason: {response.prompt_feedback.block_reason}")
            return "The response was blocked due to safety concerns. Please try rephrasing your question."
        
        if response.candidates:
            if response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text
                print(answer)
                print("\n" + "_"*80)
                return answer
            else:
                print("Response has no content parts.")
                return "No answer generated. The response had no content."
        else:
            print("No candidates in the response.")
            return "No answer generated. There were no response candidates."

    except Exception as e:
        print(f"Error generating answer: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        return f"An error occurred: {str(e)}"


def check_vector_dimensions(client, collection_name):
    schema = client.describe_collection(collection_name)
    vector_field = next((f for f in schema['fields'] if f['name'] == 'vector'), None)
    if vector_field:
        milvus_dim = vector_field['params']['dim']
        print(f"Milvus vector dimension: {milvus_dim}")
        
        # Check SBERT dimension
        sbert_dim = sbert_model.get_sentence_embedding_dimension()
        print(f"SBERT embedding dimension: {sbert_dim}")
        
        # Check DPR dimension
        with torch.no_grad():
            dpr_dim = dpr_question_encoder(**dpr_tokenizer("test", return_tensors="pt")).pooler_output.shape[-1]
        print(f"DPR embedding dimension: {dpr_dim}")
        
        if milvus_dim != sbert_dim or milvus_dim != dpr_dim:
            print("Warning: Vector dimensions do not match!")
    else:
        print("Warning: Vector field not found in Milvus schema")


def create_index_if_not_exists(collection_name):
    try:
        collection = Collection(collection_name)
        index_info = collection.index()
        if not index_info:
            print(f"Creating index for collection {collection_name}")
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            print("Index created successfully")
        else:
            print(f"Index already exists for collection {collection_name}")
        
        # Load the collection
        collection.load()
        print(f"Collection {collection_name} loaded successfully")
    except Exception as e:
        print(f"Error creating index or loading collection: {e}")
        raise


def load_collection(client, collection_name):
    try:
        client.load_collection(collection_name)
        print(f"Collection {collection_name} loaded successfully")
    except Exception as e:
        print(f"Error loading collection: {e}")



def main():
    collection_name = "textbook_raptor"
    embedding_dim = 768  # Dimension of your SBERT and DPR embeddings

    try:
        # Connect to Milvus
        connect_to_milvus()

        # Create or recreate the collection
        collection = create_milvus_collection(collection_name, embedding_dim)
        print(f"Collection '{collection_name}' created successfully")

        # Process textbooks and store RAPTOR trees
        textbook_files = [
            "/home/aryaman/Downloads/new_project/fesc111.pdf",
            "/home/aryaman/Downloads/new_project/fesc101.pdf",
            "/home/aryaman/Downloads/new_project/fesc102.pdf"

        ]
        textbook_contents = extract_textbook_content(textbook_files)

        for idx, content in enumerate(textbook_contents):
            if content:
                print(f"Processing textbook {idx + 1}")
                chunks = chunk_text(content, nltk.word_tokenize, max_chunk_size=500)
                chunks = truncate_chunks(chunks, max_length=512)
                
                vectors = sbert_model.encode(chunks)
                
                if len(vectors) < 2:
                    print(f"Not enough valid vectors generated for textbook {idx + 1}. Skipping this textbook.")
                    continue

                clusters = cluster_embeddings(vectors)
                raptor_tree = build_raptor_tree(vectors, clusters, chunks)
                store_raptor_in_milvus(raptor_tree, collection, idx)
            else:
                print(f"No content found for textbook {idx + 1}")

        print("Processing complete. RAPTOR trees stored in Milvus.")

        # Load the collection for searching
        collection.load()
        
        
        while True:
            query = input("Enter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break

            expanded_query = query_expansion(query)
            print(f"Expanded query: {expanded_query}")

            try:
                retrieved_docs = hybrid_retrieval(expanded_query, collection)
                print(f"Retrieved {len(retrieved_docs)} documents")

                if not retrieved_docs:
                    print("No relevant documents found. Please try a different query.")
                    continue

                reranked_docs = rerank_documents(query, retrieved_docs)
                print(f"Reranked {len(reranked_docs)} documents")

                # Use more documents for context
                context = " ".join(reranked_docs[:10])  # Increased from 3 to 10
                print(f"Context length: {len(context)} characters")

                # If context is still too short, try to expand it
                if len(context) < 1000:
                    context += " ".join(reranked_docs[10:])  # Add all remaining documents
                    print(f"Expanded context length: {len(context)} characters")

                print("Generating answer:")
                answer = answer_question(query, context)
                
            except Exception as e:
                print(f"Error during search process: {e}")
                print("Please try again with a different query.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Disconnect from Milvus
        connections.disconnect("default")
        print("Disconnected from Milvus")

if __name__ == "__main__":
    main()

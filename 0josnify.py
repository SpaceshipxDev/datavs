import json
import faiss
import numpy as np
import google.generativeai as genai

# Configure the Google Generative AI API
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")

# Load the FAISS index
index = faiss.read_index("1index.index")

# Load the JSON texts
with open('1index.json', 'r') as f:
    texts = json.load(f)

# Load and process the input JSON
with open('0idealized.json', 'r') as f:
    idealized_data = json.load(f)

def json_to_concatenated_string(data):
    result = []
    for key, sub_dict in data.items():
        for sub_key, value in sub_dict.items():
            result.append(f"{sub_key}: {value}")
    return " ".join(result)

# Convert the JSON to a concatenated string
concatenated_string = json_to_concatenated_string(idealized_data)

def query_similar_texts(query, top_k=2):
    # Embed the query text
    r = genai.embed_content(model="models/text-embedding-004", content=query[:3000])
    query_embedding = np.array(r["embedding"]).astype("float32").reshape(1, -1)
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the results
    results = [texts[idx] for idx in indices[0]]
    
    return results

# Example query with the concatenated string
query_text = concatenated_string
similar_texts = query_similar_texts(query_text)

# Print the results
for i, result in enumerate(similar_texts):
    print(f"Result {i+1}:")
    print(f"{result}")
    print("\n" + "-"*80 + "\n")

#import google.generativeai as genai
import faiss
import numpy as np
import json
import google.generativeai as genai 
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")

index = faiss.read_index("1index.index")
with open('1index.json', 'r') as f:
    texts = json.load(f)

def query_similar_texts(query, top_k=2):
    r = genai.embed_content(model="models/text-embedding-004", content=query[:3000])
    query_embedding = np.array(r["embedding"]).astype("float32").reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx in indices[0]:
        results.append(texts[idx])
    
    return results

# Example query
query_text = """

The user seeks to understand complex concepts, such as those related to math, physics, and chemistry, with
 a focus on clarity and thorough explanation. """
similar_texts = query_similar_texts(query_text)

# Print the results
for i, result in enumerate(similar_texts):
    print(f"Result {i+1}:")
    print(f"{result}")
    print("\n" + "-"*80 + "\n")

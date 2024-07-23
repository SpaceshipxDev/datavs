import json
import faiss
import numpy as np
import google.generativeai as genai
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")
import time 


index = faiss.read_index("data/1index.index")
with open('data/1index.json', 'r') as f:
    texts = json.load(f)
with open('0idealized.json', 'r') as f:
    idealized_data = json.load(f)


def json2strings(data):
    result = []
    for key, sub_dict in data.items():
        for sub_key, value in sub_dict.items():
            result.append(f"{sub_key}: {value}")
    return result

# Convert the JSON to a list of idea strings
idea_strings = json2strings(idealized_data)

def query_similar_texts(query, top_k=2):
    r = genai.embed_content(model="models/text-embedding-004", content=query[:3000])
    query_embedding = np.array(r["embedding"]).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[idx] for idx in indices[0]]
    return results


for idea in idea_strings:
    print("\n\n\n\n" + "-"*80 + "\n" + f"Querying for idea: {idea}\n")
    with open("0idealized_further.txt", "a") as f: 
        f.write("\n" + "-"*80 + "\n" + f"\n\n\nQuerying for idea: {idea}\n")
    similar_texts = query_similar_texts(idea)
    
    for i, result in enumerate(similar_texts):
        print(f"Result {i+1}:")
        print(f"{result}")
        with open("0idealized_further.txt", "a") as f: 
            f.write(f"\n\nResult {i+1}: \n" + f"{result}")
    time.sleep(3) 

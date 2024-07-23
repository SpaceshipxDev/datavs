import google.generativeai as genai 
import faiss 
import numpy as np
import json 
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")

texts = ["i love nasa", "i am gay", "elon musk is so smart"]
embeddings = []
for text in texts: 
    r = genai.embed_content(model="models/text-embedding-004", content=text) 
    embeddings.append(r["embedding"])
    print("Finished embedding: ", r["embedding"][:42])
embeddings = np.array(embeddings).astype("float32") 

dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension) 
index.add(embeddings) 
faiss.write_index(index, "1index.index") 
print(index) 

text_ref = {i: text for i, text in enumerate(texts)} 
with open("1text_ref.json", "w") as f: 
    json.dump(text_ref, f) 



#dickmove 
index = faiss.read_index("1index.index") 
with open("1text_ref.json", "r") as f: 
    text_ref = json.load(f) 

query = genai.embed_content(model="models/text-embedding-004", content="im bi").get("embedding") 
query = np.array(query).astype("float32").reshape(1, -1) 

k = 3
D, I = index.search(query, k) 
for i in range(k): 
    print(f"text: {text_ref[str(I[0][i])]}")
    print(f"distance: {D[0][i]}") 







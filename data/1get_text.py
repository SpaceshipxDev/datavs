import google.generativeai as genai
import faiss
import numpy as np
import json

# Configure the generative AI API
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")

# Load the conversation data
with open('0conversations.json', 'r') as f:
    data = json.load(f)

texts = []
embeddings = []

for conversation in data:
    meta = f"\n\n---------\nConversation title: {conversation['title']}"
    print(meta)

    text = ""
    for key, value in conversation['mapping'].items():
        if value.get('message') and value['message']['author']['role'] == 'user':
            parts = value["message"]["content"]["parts"]
            content = ""
            for part in parts:
                if isinstance(part, str):
                    content += part + " "
                elif isinstance(part, dict) and part.get("text"):
                    content += part["text"] + " "
            content = content.strip()
            text += "\nUsr Msg: " + content + " "
            print(f"user message: {content}")
    
    texts.append(text.strip())

    r = genai.embed_content(model="models/text-embedding-004", content=text[:3000])
    embeddings.append(r["embedding"])
    print("finished embedding: ", r["embedding"][:42])

# Convert embeddings to numpy array
embeddings = np.array(embeddings).astype("float32")


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "1index.index")
print(index)

with open('1index.json', 'w') as f:
    json.dump(texts, f)

print("Texts saved to 1index.json")

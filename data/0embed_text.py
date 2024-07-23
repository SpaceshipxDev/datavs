import os
import google.generativeai as genai
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")

r = genai.embed_content(
    model="models/text-embedding-004", 
    content="how r u, if ur gay raise ur hand", 
)
print(r["embedding"][:30])
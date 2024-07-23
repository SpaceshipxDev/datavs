

import os
import google.generativeai as genai
genai.configure(api_key="AIzaSyDbIkubtbmabCNcAImr53wDNIhr1W5dlME")
from google.generativeai.types import HarmCategory, HarmBlockThreshold

safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
}
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config 
)

chat = model.start_chat()
with open("data/0naturallang.txt", "r") as f: 
    info = f.read() 



responses = chat.send_message(
    f"""
    here is a collection of every interaction i've had with chatgpt. 
    try to extract every idea i had, breaking it down into main themes, and specific ideas underneath each theme. 
    Be specific and try to use my original way of expression/terms. 
    Your construction of themes, and the grouping of ideas under such themes, should be very careful and thought-through. 
    
    here it is:#{info[:800000]}#
    """, 
    safety_settings=safety_settings, 
    stream=True
)
for r in responses: 
    print(r.text) 
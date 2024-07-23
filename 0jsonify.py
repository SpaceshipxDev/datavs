

import os, json, re
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



response = chat.send_message(
    f"""
    Given the following text with extracted ideas from chatgpt interactions, structure the information into valid JSON format. Each theme should be a key, and underneath each theme, the specific ideas should be individual keys with their descriptions as values. 
    Don't change any content. 
    ***ENSURE to output in FULL JSON ONLY. 

    Here is the extracted ideas from chatgpt interactions: ###
    ##
 Extracted Ideas from ChatGPT Interactions:

**Theme 1: AI and Technology
**

* **AI as a Revolution:**  AI is seen as a transformative
 force, akin to the Industrial Revolution or the invention of the internet. The user believes AI will significantly impact society and lead to a more abundant future.
*
 **AI-Driven Product Development:** The user emphasizes the importance of creating impactful products with lasting influence, like Apple's iPhone, as opposed to focusing solely on
 transient forms of entertainment like music.
* **AI and User Experience:** The user appreciates the subtle but impactful features that AI can bring to products, such as personalized comments on YouTube and advanced search functionalities. 
* **AI's
 Potential Limitations:** The user acknowledges that AI may not solve all societal problems, as humans' innate intellectual limitations will likely persist even with the development of Superintelligence.
* **AI and Social Interactions:** The user recognizes the potential of AI to
 create personalized media experiences and assist with social interactions, but also acknowledges that AI may not be able to fully replicate human connection and nuance.
* **AI and Learning:** The user finds great value in using first principles to solve problems and believes AI-powered learning assistants can significantly enhance the educational experience.

**Theme 
2: Personal Growth and Development**

* **The Power of Free Will:** The user prioritizes the importance of exercising free will by pursuing passions and making independent choices, viewing it as a crucial aspect of human fulfillment.
* **Charisma and Manipulation:** The user recognizes the power of charisma and manipulation in social interactions
, viewing it as a skill that can be learned and honed. The user admires figures like Steve Jobs and Robert Friedland as models of this type of influence.
* **Personal Identity and Authenticity:** The user emphasizes the importance of being authentic and embracing one's true self, rejecting the notion of seeking validation through external
 validation or conforming to societal expectations.
* **Embracing Vulnerability:** The user acknowledges that being open to vulnerability and sharing personal experiences can create deeper connections with others.
* **Self-Improvement and Mastery:** The user values continuous improvement and seeks methods for honing his communication skills and developing a more charismatic presence.
*
 **Intellectual Growth:** The user emphasizes the importance of continuous learning and the need to engage in intellectual pursuits to prevent mental atrophy.

**Theme 3: Music and Art**

* **Music as a Reflection of Consciousness:** The user views music as a powerful form of artistic expression that can capture human emotions, experiences,
 and the meaning of technology.
* **Hedonic Desires in Music:** The user appreciates the darker, more hedonistic themes explored in music by artists like The Weeknd and XXXTentacion. 
* **Music for the Enlightened:** The user recognizes that great music, like that of Dylan and the Beatles
, transcends fleeting trends and resonates with deeper values and aspirations.
* **Impactful Music:** The user believes that music, like great songs with billions of listeners, can have a profound and lasting impact on culture and society. 

**Theme 4: Social Dynamics and Relationships**

* **The Importance of Connection
:** The user acknowledges that meaningful relationships require shared experiences and a mutual understanding, even if intellectual connections may not be present.
* **The Complexity of Attraction:** The user explores the nature of attraction, recognizing that it goes beyond physical appearance and can be influenced by a sense of connection and personal experience. 
* **
Social Norms and Conformity:** The user rejects the idea of conforming to societal expectations or trying to be "socially coherent," emphasizing the importance of authenticity and free will in social interactions.
* **Navigating Social Interactions:** The user seeks guidance on how to approach and engage with his crush, exploring strategies for building
 rapport, initiating conversations, and expressing his interest in a confident and charismatic manner. 

**Theme 5: Coding and Development**

* **First Principles Thinking:** The user appreciates the power of first principles thinking in solving problems and enjoys tackling novel challenges.
* **AI-Powered Tools:** The user recognizes the
 potential of AI to revolutionize coding and development, creating tools that can automate tasks and enhance productivity. 
* **Accessibility in Technology:** The user seeks to make technology more accessible to individuals with disabilities, such as those with impaired vision, by developing tools that automate tasks. 
* **Technical Challenges:** The user
 encounters various technical challenges while working on his projects, such as errors related to asynchronous programming, JSON formatting, and DOM manipulation.

**Theme 6: Academic and Intellectual Pursuits**

* **Understanding Concepts:** The user seeks to understand complex concepts, such as those related to math, physics, and chemistry, with
 a focus on clarity and thorough explanation. 
* **Effective Learning Strategies:** The user values the importance of  effective learning strategies, such as summarizing lecture notes and using first principles to understand complex concepts. 
* **Dystopian Fiction:** The user experiments with creative writing, using the dystopian novel "1
984" as inspiration for creating a fictional world with advanced technology but societal control.
* **Advanced Vocabulary:** The user seeks to expand his vocabulary, particularly in areas related to academic writing and speculative fiction. 

**Theme 7: General Observations and Reflections**

* **The Value of Impact:** The
 user believes that creating impactful things that have a lasting influence on society, like great products or music, is a significant contribution to the world.
* **The Importance of Passion:** The user emphasizes the need to pursue passions and activities that bring a sense of fulfillment.
* **The Nature of Reality:** The user questions
 the nature of reality and explores the concept of simulations, potentially considering the existence of creators or a deeper, underlying truth.
* **The Power of Curiosity:** The user embraces a sense of curiosity and a desire to explore the world and uncover new knowledge and experiences.

**Additional Notes:**

* The user's
 language and tone are often informal and conversational, reflecting a casual and relatable approach to both personal and intellectual discussions. 
* The user exhibits a strong interest in AI and its potential impact on society, as well as its ability to enhance human experience and productivity. 
* The user often explores concepts related to free will
, personal identity, and social interactions, demonstrating a desire for deeper understanding of human nature and existence.
* The user's thoughts on the nature of reality, the existence of creators, and the implications of advanced technology reflect a philosophical and inquisitive mind. 

This analysis reveals the user's multifaceted interests and provides
 a glimpse into their unique perspective on AI, technology, personal growth, and the world around them. 
    ###
    """, 
    safety_settings=safety_settings, 
)

data = response.text 
print(data) 

json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
match = json_pattern.search(data) 

if match: 
    actual_json = match.group(1) 
    try: 
        json_data = json.loads(actual_json) 
        with open("0idealized.json", "w") as f: 
            json.dump(json_data, f, indent=4) 
        print("done") 
    except json.JSONDecodeError as e: 
        print("invalid json:", e) 
else: 
    print("no json found, model fucked up")
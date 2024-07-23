import json

with open('0conversations.json', 'r') as f:
    data = json.load(f)

for conversation in data:
    meta = f"\n\n---------\nConversation title: {conversation['title']}"
    print(meta)
    with open("0naturallang.txt", "a") as f:
        f.write(meta)

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
            print(f"user message: {content}")
            with open("0naturallang.txt", "a") as f:
                f.write(f"\nusr msg: {content}")
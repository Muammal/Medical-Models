import ollama

def ask_llm(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def get_disease_info(disease_name):
    prompt = f"""
Provide a detailed explanation about the skin disease '{disease_name}'.

Include:
- description
- causes
- precautions
- risks
- treatment options
"""
    return ask_llm(prompt)

def chatbot_response(disease_name, user_query):
    prompt = f"The detected skin disease is '{disease_name}'. {user_query}"
    return ask_llm(prompt)

# Test mode
if __name__ == "__main__":
    disease = input("Enter disease name: ")
    info = get_disease_info(disease)
    print(info)
    while True:
        q = input("\nAsk about this disease: ")
        if q.lower() == "exit":
            break
        print(chatbot_response(disease, q))
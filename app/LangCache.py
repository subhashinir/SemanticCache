import time
from langcache import LangCache
import requests

api_key = "<LangCache API Key>"
GROQ_API_KEY=""


lang_cache =  LangCache( 
    server_url="<LangCache URL",
    cache_id="<Cache Id>",
    api_key=api_key,
)


def llm(query, user="user1"):
    # --- Semantic Cache Check ---
    start_cache = time.time()

    result = lang_cache.search(
        prompt=query,
        similarity_threshold=0.95,
    )
    end_cache = time.time()

    #print(result)

    if result :
        
        for entry in result.data:
            print("Cache Hit!")
            print(f"Time for cache lookup: {end_cache - start_cache:.4f} seconds")
            print("Cache Response:::")
            print(f"Prompt: {entry.prompt}")
            print(f"Score: {entry.similarity}")
            

    print("Cache Miss!")


    # --- Groq Llama-3-8B Inference Call ---
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",  # Or another supported Groq model
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    start_llm = time.time()
    response = requests.post(url, json=payload, headers=headers)
    end_llm = time.time()


    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None

    response_json = response.json()
    response_text = response_json["choices"][0]["message"]["content"]

    # --- Store in Semantic Cache ---
    save_response = lang_cache.set(
        prompt=query,
        response=response_text,
    )
    #print('Cache Set!')
    print("LLM Response:")
    print(response_text)
    print(f"Time to get response from LLM API: {end_llm - start_llm:.4f} seconds")


# All types of queries 

# print("----#####################################---")
# print("Query:  ----what is the capital of France---")
# llm("what is the capital of France")
# print("----#####################################---")
# print("----#####################################---")


# print("Query:  ----capital of France---")
# llm("capital of France")
# print("----#####################################---")
# print("----#####################################---")

# print("Query:  ----France capital---")
# llm("France capital")
# print("----#####################################---")
# print("----#####################################---")

# print("Query:  ----Is paris the capital of Framce---")
# llm("Is paris the capital of Framce")
# print("----#####################################---")
# print("----#####################################---")

# print("Query:  ----France capital was always---")
# llm("France capital was always")
# print("----#####################################---")
# print("----#####################################---")

# print("Query:  ----France capital never changed from---")
# llm("France capital never changed from")
# print("----#####################################---")
# print("----#####################################---")

# print("Query:  ----what is the capital of France, give me the answer in one word---")
# llm("what is the capital of France, give me the answer in one word")
# print("----#####################################---")

# print("----#####################################---")
# print("Query:  ----Brief history on Capital of France---")
# llm("Brief history on Capital of France")
# print("----#####################################---")
# print("----#####################################---")

# print("----#####################################---")
# print("Query:  ----Brief history on Paris---")
# llm("Brief history on Paris")
# print("----#####################################---")
# print("----#####################################---")

# print("----#####################################---")
# print("Query:  ----Brief history on France---")
# llm("Brief history on France")
# print("----#####################################---")
# print("----#####################################---")

print("----#####################################---")
print("Query:  ----How does Redis Langcache work---")
llm("How does Redis Langcache work")
print("----#####################################---")
print("----#####################################---")

print("----#####################################---")
print("Query:  ----How does Redis Langcache work, explain strictly only in 2 lines---")
llm("How does Redis Langcache work, explain strictly only in 2 lines")
print("----#####################################---")
print("----#####################################---")



# print("----#####################################---")
# print("Query:  ----what is not the capital of France---")
# llm("what is not the capital of France")
# print("----#####################################---")
# print("----#####################################---")


from difflib import SequenceMatcher
from langcache import LangCache
from openai import OpenAI
client = OpenAI(api_key="")


# =======================================================
# CALL open ai
# =======================================================
def llm_function(prompt):
    print("calling openai")
    prompt = prompt + " Provide answer in two lines."
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = completion.choices[0].message.content
    print("openai response: --------------- ", content)
    return content


# =======================================================
# CONFIGURATION
# =======================================================
# Weighting for confidence scoring
W_EXACT = 0.10
W_FUZZY = 0.20
W_SEMANTIC = 0.70

# Minimum overall confidence required to accept a cache hit
CONFIDENCE_THRESHOLD = 0.65

ENABLE_STORE_ON_MISS = True


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_confidence(prompt, entry):
    """
    Combine exact + fuzzy + semantic into one confidence score.
    """

    # Exact match score
    exact_score = 1.0 if prompt.strip() == entry.prompt.strip() else 0.0

    # Fuzzy score
    fuzzy_score = fuzzy_ratio(prompt, entry.prompt)

    # Semantic score from LangCache
    semantic_score = entry.similarity or 0.0

    # Combined weighted confidence
    confidence = (
        W_EXACT * exact_score +
        W_FUZZY * fuzzy_score +
        W_SEMANTIC * semantic_score
    )

    return confidence, exact_score, fuzzy_score, semantic_score


# =======================================================
# MAIN WORKFLOW
# =======================================================
def get_cached_or_generate(prompt_text, llm_function, api_key):

    with LangCache(
        server_url="<LangCache server URL>",
        cache_id="<Cache ID>",
        api_key=api_key
    ) as lang_cache:

        print("\nSearching cache…")

        search_response = lang_cache.search(prompt=prompt_text, similarity_threshold=0.7,)
        print("Search Response:", search_response, "\n")

        best_entry = None
        best_confidence = 0
        best_breakdown = None

        # --------------------------------------------------
        # Evaluate ALL entries with unified scoring
        # --------------------------------------------------
        if search_response:
            for entry in search_response.data:

                confidence, exact_score, fuzzy_score, semantic_score = \
                    compute_confidence(prompt_text, entry)

                print(f"Evaluating candidate:")
                print(f"  Cached Prompt: {entry.prompt}")
                print(f"  Scores → exact={exact_score:.2f}, fuzzy={fuzzy_score:.2f}, semantic={semantic_score:.2f}")
                print(f"  Combined Confidence = {confidence:.3f}\n")

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_entry = entry
                    best_breakdown = (exact_score, fuzzy_score, semantic_score)

        # ================================================================
        # SELECT BEST MATCH BASED ON CONFIDENCE
        # ================================================================
        if best_entry and best_confidence >= CONFIDENCE_THRESHOLD:
            exact_score, fuzzy_score, semantic_score = best_breakdown

            print("=== CACHE HIT: CONFIDENCE-BASED MATCH ===")
            print(f"Selected Prompt: {best_entry.prompt}")
            print(f"Confidence Score: {best_confidence:.3f}")
            print(f"Breakdown:")
            print(f"  Exact:    {exact_score:.2f}")
            print(f"  Fuzzy:    {fuzzy_score:.2f}")
            print(f"  Semantic: {semantic_score:.2f}")
            print("\nResponse:", best_entry.response)

            return best_entry.response

        # ================================================================
        # CACHE MISS → CALL LLM + STORE NEW RESULT
        # ================================================================
        print("=== CACHE MISS → CALLING LLM ===")
        llm_response = llm_function(prompt_text)
        print("LLM Response:", llm_response)

        if ENABLE_STORE_ON_MISS:
            print("Storing LLM response into LangCache…")
            lang_cache.set(prompt=prompt_text, response=llm_response)

        return llm_response


# # =======================================================
# # SAMPLE DUMMY LLM FUNCTION (replace with real LLM)
# # =======================================================
# def dummy_llm(prompt):
#     return f"[LLM generated response for: {prompt}]"


# =======================================================
# EXECUTION EXAMPLE
# =======================================================
if __name__ == "__main__":
    apiKey = "LangCache API key>""

    user_prompt = input("Enter your prompt:")
    response = get_cached_or_generate(user_prompt, llm_function, apiKey)

    print("\nFINAL OUTPUT: ------- ", response)

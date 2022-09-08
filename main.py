from search import run_tool

# Any grants with abstracts less than min_words will be removed. Default = 75
min_words = 75

# Query are tokens which represent the topic area. This is a string with tokens seperated
# by commas. Minimum tokens = 5
query = "machine learning, artificial intelligence, AI, computer vision"

df = run_tool(query, min_words)



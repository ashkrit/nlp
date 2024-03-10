from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = [r for r in ddgs.text("python programming", max_results=5)]
    for r in results:
        print(r)
        print()
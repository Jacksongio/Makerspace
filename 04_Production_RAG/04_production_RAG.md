# Assignment_Introduction_to_LCEL_and_langraph_langchain_powered_rag.ipynb
## Question 1:
### What is the embedding dimension, given that we're using text-embedding-3-small?
    - The embedding dimension for text-embedding-3-small is 1536
## Question 2:
###LangGraphs graph-based approach lets us visualize and manage complex flows naturally. How could we extend our current implementation to handle edge cases?
    - Insert a context-validation node immediately after retrieval that, if state["context"] is empty, branches to an ask_for_clarification fallback instead of proceeding to generation. Then tack on a fact-check node after generation that, on low confidence or failed verification, loops back into retrieval with a refined query or escalates to human review.
## Activity #1
### While there's nothing specifically wrong with the chunking method used above - it is a naive approach that is not sensitive to specific data formats. Brainstorm some ideas that would split large single documents into smaller documents.
    1. Section Header Splitting
    2. Semantic-Shift detection
    3. Pattern-Aware splitting.
# Langsmith_and_Evaluation.ipynb
## Activity 1
![Alt text](image.png)
## Activity #2
### Complete the prompt so that your RAG application answers queries based on the context provided, but does not answer queries if the context is unrelated to the query.
- Prompt:  Answer the query using only the provided context.  If the context is empty, unrelated, or does not contain the answer, reply with exactly: “I don’t know.”  Do not add or invent any information beyond what’s in the context.
## Question #1 (LangSmith and Evaluation.ipynb)
### What conclusions can you draw about the above results? Describe in your own words what the metrics are expressing.
- The results show the base RAG pipeline reliably used the retrieved context and achieves a strong average string-match score, completing every query in about 1.17 seconds at virtually 0 cost. However, only 33% of the outputs pass the subjective dopeness check. indicating while correctness and efficiency are high, the responses could be made more engaging or stylistically polished.


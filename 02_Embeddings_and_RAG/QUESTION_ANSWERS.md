# Question #1:
### The default embedding dimension of text-embedding-3-small is 1536, as noted above.
1. Is there any way to modify this dimension?
2. What technique does OpenAI use to achieve this?
---
**Answer**: You cannot modify the dimension of the text-embedding-3-small model as its baked into the model's weights. The techniques that OpenAI utilizes is that they get a fixed-length embedding, pool these token represenations and then pass that pooled vector through a learned linear projection layer (for which this one is set to 1536)

# Question #2:
### What are the benefits of using an async approach to collecting our embeddings?
---
**Answer**:  Using an async approach lets you issue multiple embedding requests in parallel without blocking your application. Which in turn improves throughput and reduces overall latency.

# Question #3:
### When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?
---
**Answer**:  You can achieve more reproducible outputs by setting temperature to 0 and fixing the top_p to 1, which will then remove stochastic token sampling. Also you can specify the exact model version and paramters identical on all calls to ensure as many constants.

# Question #4:
### What prompting strategies could you use to make the LLM have a more thoughtful, detailed response? What is that strategy called?
---
**Answer**:  You can instruct the model to articulate its reasoning  by including cues like "think through this step by step" or "explain in detail" which unpacks its thought process. THis process is known as Chain of thought prompting.
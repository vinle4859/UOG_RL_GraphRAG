from src.rag.pipeline import RAGPipeline

pipe = RAGPipeline()
pipe.index()                          # reads data/raw/*.txt, chunks & embeds
result = pipe.query("What is PPO?")
print(result.answer)
print(result.retrieved_chunks)        # evidence chunks
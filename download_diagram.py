import urllib.request
import urllib.parse

dot_code = """
digraph G {
  node [shape=box, style=filled, fillcolor=lightblue, fontname="Arial"];
  edge [fontname="Arial", fontsize=10];
  rankdir=TB;

  ENV [label=".env file", fillcolor=lightyellow];
  CFG [label="config.py", fillcolor=lightgreen];
  LOG [label="logger.py", fillcolor=lightgreen];
  DC [label="data_cleaner.py"];
  NR [label="news_retriever.py"];
  CHK [label="chunker.py"];
  EMB [label="embeddings.py"];
  FM [label="faiss_manager.py"];
  RET [label="retriever.py"];
  LLM [label="llm_engine.py", fillcolor=pink];
  PIP [label="pipeline.py", fillcolor=gold];
  APP [label="app.py", fillcolor=orange];
  VAL [label="validators.py"];
  PRO [label="prompts.py"];

  ENV -> CFG [label=" load_dotenv"];
  CFG -> LOG [label=" LOGS_DIR"];
  CFG -> DC [label=" CORPUS_DIR"];
  CFG -> NR [label=" NEWS_API_KEY, URLs"];
  CFG -> CHK [label=" CHUNK_SIZE, OVERLAP"];
  CFG -> EMB [label=" EMBEDDING_MODEL"];
  CFG -> FM [label=" FAISS_INDEX_DIR"];
  CFG -> RET [label=" TOP_K, THRESHOLD"];
  CFG -> LLM [label=" LLM_PROVIDER, keys"];

  LOG -> NR [label=" get_logger"];
  LOG -> DC [label=" get_logger"];
  LOG -> CHK [label=" get_logger"];
  LOG -> EMB [label=" get_logger"];
  LOG -> FM [label=" get_logger"];
  LOG -> RET [label=" get_logger"];
  LOG -> LLM [label=" get_logger"];
  LOG -> PIP [label=" get_logger"];
  LOG -> APP [label=" get_logger"];

  EMB -> RET [label=" embed_query"];
  FM -> RET [label=" search"];

  PIP -> VAL [label=" validate_inputs"];
  PIP -> NR [label=" retrieve_news"];
  PIP -> DC [label=" clean & dedup"];
  PIP -> CHK [label=" chunk_articles"];
  PIP -> EMB [label=" embed_texts"];
  PIP -> FM [label=" FAISSManager"];
  PIP -> RET [label=" retrieve & classify"];
  PIP -> LLM [label=" generate_insight"];
  PIP -> PRO [label=" templates"];

  APP -> PIP [label=" run_pipeline"];
  APP -> LOG [label=" get_logger"];
}
"""

encoded_graph = urllib.parse.quote(dot_code)
url = f"https://quickchart.io/graphviz?graph={encoded_graph}"

req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0'}
)

try:
    print(f"Downloading from QuickChart...")
    with urllib.request.urlopen(req) as response, open(r"F:\data\GEN AI PROJECTS\market research insight\dependency_map.jpg", 'wb') as out_file:
        out_file.write(response.read())
    print("Success! Saved as dependency_map.jpg")
except Exception as e:
    print(f"Error: {e}")

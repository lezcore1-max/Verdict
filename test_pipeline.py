import os
from orchestration.paper_runner import run_paper
model_name = "gemini-3.1-flash-lite"

gemini_key = os.getenv("GEMINI_API_KEY", "")
tavily_key = os.getenv("TAVILY_API_KEY", "")

# Ensure we use an absolute path so it matches UI logic
pdf_path = os.path.abspath("attention.pdf")
db_path = os.path.abspath("verdict.db")

print("🚀 Launching VERDICT Pipeline Test Script...")

# Basic callback to print what Streamlit would see
def my_status(msg):
    print(f"[UI STATUS] {msg}")

paper_id = run_paper(
    pdf_path=pdf_path,
    db_path=db_path,
    model_name=model_name,
    api_key=gemini_key,
    tavily_key=tavily_key,
    status_callback=my_status
)

print(f"🎉 Pipeline finished successfully. Paper ID: {paper_id}")

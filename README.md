# VERDICT: Multi-Agent AI System for Claim Verification

VERDICT is a powerful, multi-agent AI pipeline designed to extract, decompose, and mathematically verify falsifiable claims from dense scientific literature and text. Built with Streamlit and powered by Gemini 3.1 Flash Lite, it utilizes an advanced system of 6 specialized AI agents to rigorously debate and judge claims using the **Dempster-Shafer Theory** and the **Sequential Probability Ratio Test (SPRT)**.

## 🤖 The 6-Agent Pipeline

The system breaks down complex cognitive reasoning into specialized roles:
1. **Agent 1 (Claim Extractor):** Reads the source document and extracts structured, falsifiable claims, tracking logical dependencies between them.
2. **Agent 2 (Hypothesis Decomposer):** Takes a complex claim and breaks it down into granular, atomic sub-hypotheses.
3. **Agent 3 (Evidence Hunter):** Scours the web (via Tavily) and local vector databases (ChromaDB RAG) to find evidence supporting the claim.
4. **Agent 4 (Evidence Judge):** Rigorously evaluates the found evidence and assigns a statistical p-value representing its strength and relevance.
5. **Agent 5 (Devil's Advocate):** Specifically searches for evidence that *contradicts* the claim to ensure unbiased evaluation.
6. **Agent 6 (Verdict Synthesizer):** Aggregates the mathematical probabilities, resolves inter-agent disagreement, and writes a nuanced, final verdict (Support, Contradiction, or Uncertainty).

## 📊 The Math Behind the Magic

Instead of relying on LLM hallucination, VERDICT proves its conclusions mathematically:
* **Dempster-Shafer Theory (DST):** Explicitly models *uncertainty*. Instead of a simple "true/false", the system maintains dynamic probabilities for Support, Contradiction, and Uncertainty.
* **Sequential Probability Ratio Test (SPRT):** Calculates a running likelihood ratio as evidence rolls in, guaranteeing statistically significant conclusions and preventing the system from over-analyzing claims that are already proven.

## 🚀 Setup & Deployment

VERDICT requires two API keys to function:
- `GEMINI_API_KEY`: For the LLM reasoning (Gemini 3.1 Flash Lite).
- `TAVILY_API_KEY`: For web search capabilities.

### Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lezcore1-max/Verdict.git
   cd Verdict
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your keys:
   ```env
   GEMINI_API_KEY="your_gemini_key_here"
   TAVILY_API_KEY="your_tavily_key_here"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run frontend/app.py
   ```

### Deploying to Streamlit Community Cloud
1. Connect your GitHub repository to Streamlit Community Cloud.
2. Set the main file path to `frontend/app.py`.
3. In the Streamlit dashboard, go to **Advanced Settings -> Secrets** and securely add your API keys:
   ```toml
   GEMINI_API_KEY="your_gemini_key_here"
   TAVILY_API_KEY="your_tavily_key_here"
   ```

## 🛡️ API Safeguards
The system is hard-capped with a strict 4.1-second delay between Gemini API calls to ensure it mathematically respects the free-tier limit of 15 Requests Per Minute (RPM), preventing quota exhaustion during deep multi-agent debates.

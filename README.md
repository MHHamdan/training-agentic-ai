training-agentic-ai

Overview
This repository collects practical agents under the agents/ folder. All agents share a single .env file and a single Python virtual environment at the repository root to keep setup simple and reduce resource usage.

Repository structure
- agents/
  - customer-support-agent/
  - legal-document-review/
- requirements.txt
- venv/            (not committed)
- .env             (not committed)
- .env.example

Getting started
1) Create and activate the shared virtual environment at the root
   python -m venv venv
   source venv/bin/activate
2) Install dependencies once at the root
   pip install -r requirements.txt
3) Create your environment file
   cp .env.example .env
   # Add your GOOGLE_API_KEY and any optional values

Run the agents
- Customer Support Agent
  cd agents/customer-support-agent
  export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
  streamlit run src/ui/app.py --server.port 8502

- Legal Document Review
  cd agents/legal-document-review
  streamlit run app.py --server.port 8501

Notes
- Keep running the apps from inside their own directories after activating the shared environment at the root.
- Do not commit .env or the venv directory.

Roadmap (incremental)
- Legal Document Review: basic RAG app for PDFs
- Customer Support Agent: conversation flow with simple escalation
- Next steps
  - Shared knowledge base assistant
  - Lightweight coordination between agents
  - Background worker for notifications

Contributing
- Place each agent under agents/<agent-name>
- Keep README files short and direct
- Use the shared .env and venv

Repository
- GitHub: https://github.com/MHHamdan/training-agentic-ai


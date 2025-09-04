📜 AI Legal Contracts Chatbot

This repository contains an AI-powered Legal Contracts QnA Chatbot built using Agentic RAG (Retrieval-Augmented Generation).
It enables users to ask questions related to legal contracts and get accurate, context-aware responses using NLP and modern AI models.

The chatbot is designed for:

📑 Contract Analysis – understand clauses, terms, and conditions.

⚖️ Legal QnA – ask queries about legal concepts in documents.

🤖 AI Assistance – augment legal workflows with intelligent search + reasoning.

⚙️ Installation

Clone the repository:

git clone https://github.com/iamshivamkumarsharma/AI-Legal-Contracts-Chatbot.git
cd AI-Legal-Contracts-Chatbot/


Create a conda environment (Python 3.10 or later) and install requirements:

conda create -n legal-bot python=3.11
conda activate legal-bot
pip install -r requirements.txt

💻 Usage with CLI

Go to the multi_agent directory and create a .env file:

cd multi_agent/


In .env, add your keys (replace with actual values):

GROQ_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_api_key
LANGSMITH_PROJECT=Legal_Contracts_QnA


Run the chatbot:

python chatbot.py


Exit anytime by typing:

exit

🌐 Usage (Local Deployment)

From the root directory, run:

python app.py


A FastAPI server will start.

Open: http://127.0.0.1:8000

Go to /docs → click “Try it out”, enter your query, and execute to get responses.

🐳 Docker Deployment (Recommended)

Build the docker image:

docker build -t legal_contracts_companion .


Run the container with your .env file:

docker run -d --env-file .env -p 8000:8000 legal_contracts_companion


Open in browser:

http://127.0.0.1:8000


Check running containers and logs:

docker ps -a
docker logs <container_id>

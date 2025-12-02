# Medical Chatbot with LLMs, LangChain, Pinecone, Flask & AWS

This repo demonstrates an end‑to‑end medical Q&A chatbot: load PDFs with treatment, chunk & embed with LangChain, index in Pinecone for semantic search, and serve a Flask UI. A GitHub Actions + AWS (ECR + EC2) pipeline is included for containerized deployment.


---

## Tech Stack
- **Python 3.10**
- **LangChain** (document loading, splitting, retrieval)
- **AZURE OpenAI embeddings** or **GROQ** → Pinecone
- **Pinecone** (vector DB)
- **Flask** (web app)
- **Docker**, **AWS ECR/EC2**, **GitHub Actions** (CI/CD)

---

## Quickstart (Local)

### 1) Clone the repo
```bash
git clone https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS
```

### 2) Create & activate a Conda env
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3) Install requirements
```bash
pip install -r requirements.txt
```

### 4) Configure environment variables
Create a `.env` file in the project root:
```ini
PINECONE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Optional if using Azure OpenAI instead of OpenAI:
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_ENDPOINT=...
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small   # or -large
```

### 5) (One‑time) Create the Pinecone index
**The index dimension must match your embedding model.**
- `text-embedding-3-small` → **1536**
- `text-embedding-3-large` → **3072**

Example (Pinecone Python v3):
```python
from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "medicalbot"

if index_name not in [i.name for i in pc.list_indexes().indexes]:
    pc.create_index(
        name=index_name,
        dimension=1536,              # 1536 for small, 3072 for large
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
```
Update the index name in your code if needed.

### 6) Build the vector store (embed & upsert)
```bash
python store_index.py
```
If you hit a Pinecone 4MB payload limit, reduce the batch size and avoid storing the full text in metadata (see **Troubleshooting** below).

### 7) Run the app
```bash
python app.py
```
Open **http://localhost:8080** in your browser.

---
## Chat interface 
![System Architecture](https://github.com/Immortal-Pi/CareGuideAI/blob/main/assets/NLQV_architecture.jpeg)

![System Architecture](https://github.com/Immortal-Pi/CareGuideAI/blob/main/assets/NLQV_architecture.jpeg)

## Project Structure (typical)
```
.
├─ app.py                     # Flask app entrypoint
├─ store_index.py             # Loads docs, creates embeddings, pushes to Pinecone
├─ .env                       # API keys (never commit!)
├─ requirements.txt
├─ src/
│  ├─ helper.py               # loaders, splitters, document utils
│  ├─ model_loader.py         # embedding/model wiring
│  ├─ prompt.py               # prompts 
|  └─ logging.py              # logging module
├─ templates/
│  └─ index.html              # Flask Jinja2 template
├─ static/
│  ├─ style.css
│  └─ doctor.png
└─ data/                      # PDFs or corpus
```

---

## Deployment: AWS CI/CD (ECR + EC2 + GitHub Actions)

### 1) AWS prerequisites
- **IAM user** with minimally required permissions (for simplicity here):
  - `AmazonEC2FullAccess`
  - `AmazonEC2ContainerRegistryFullAccess`
- **ECR repository** (e.g., `011528265658.dkr.ecr.us-east-1.amazonaws.com/careguideai`)
- **EC2 instance** (Ubuntu) with Docker installed:
  ```bash
  sudo apt-get update -y && sudo apt-get upgrade -y
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker ubuntu
  newgrp docker
  ```

### 2) Self-hosted runner (optional pattern used by many templates)
In your GitHub repo: **Settings → Actions → Runners → New self-hosted runner** → follow the Linux instructions on your EC2.

### 3) GitHub Secrets (repo → Settings → Secrets and variables → Actions)
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION       # e.g., us-east-1
ECR_REPO                 # e.g., 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
PINECONE_API_KEY
OPENAI_API_KEY
```
Add any Azure/OpenAI extras if you use them.

### 4) High-level deployment flow
1. GitHub Actions builds a Docker image.
2. Pushes the image to **ECR**.
3. On EC2, pull the latest image from ECR and run the container.

Typical EC2 commands after login:
```bash
eval $(aws ecr get-login --no-include-email --region $AWS_DEFAULT_REGION)
IMAGE_URI="$ECR_REPO:latest"
docker pull "$IMAGE_URI"
# Stop old container if running
(docker rm -f medicalbot || true)
# Run container (map ports and pass env)
docker run -d --name medicalbot \
  -p 80:5000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e PINECONE_API_KEY=$PINECONE_API_KEY \
  "$IMAGE_URI"
```

> For production: use an ALB or Nginx reverse proxy with HTTPS (ACM certs), an SSM Parameter Store for secrets, and least‑privilege IAM.

---

## Troubleshooting & Gotchas

### LangChain ≥ 0.2 import changes
- Use community packages:
  ```python
  from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_core.documents import Document
  ```

### Pydantic v2: "arbitrary type" or schema errors
- If a Pydantic model holds a non‑Pydantic object, allow it:
  ```python
  from pydantic import BaseModel, ConfigDict
  class MyModel(BaseModel):
      model_config = ConfigDict(arbitrary_types_allowed=True)
  ```

### `Document.metadata` must be a **dict**
- Avoid `{"source", src}` (a **set**). Use `{"source": src}`.

### Pinecone dimension mismatch
- Error like `Vector dimension 1536 does not match index 3072` → create index with the correct dim or switch to the matching embedding model.

### Pinecone request too large (HTTP 400, limit 4MB)
- Reduce `batch_size` in `from_documents`/`add_texts` (e.g., `batch_size=4`).
- Don’t store full text as metadata (`text_key=None` or store only a short `snippet`).
- Use smaller chunks (e.g., 500–800 chars) if needed.

### OpenAI/Azure embeddings
- `text-embedding-3-small` (1536 dims) is cheaper/faster; `text-embedding-3-large` (3072 dims) is more accurate. You can also request fewer dims with a `dimensions` parameter if your DB cap is smaller.

---

## Development Tips
- Put PDFs under `data/` and confirm they load.
- Start small: index a single PDF and test retrieval before bulk loading.
- Add logging around embedding/upsert phases to spot payload or API errors quickly.

---

## License
MIT (update if your project uses a different license).


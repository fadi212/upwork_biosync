# biosync_ml


## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create a .env File
Create a `.env` file in the root directory and add your OpenAI and Pinecone API credentials:

```bash
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1-aws
```


### 3️⃣ Install Dependencies
Run the following command to install all required packages:

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the FastAPI Server
Start the FastAPI application:

```bash
python main.py
```

The API will be accessible at:
- 📌 [http://localhost:8190](http://localhost:8000)
- 📌 Open the Swagger UI at [http://localhost:8190/docs](http://localhost:8000/docs) to test the API.

---

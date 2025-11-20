# Using GCP Credits for Your Project

## Quick Setup

### Step 1: Enable Vertex AI API

1. Go to [GCP Console](https://console.cloud.google.com/)
2. Select your project (or create one)
3. Go to **APIs & Services** → **Library**
4. Search for "Vertex AI API"
5. Click **Enable**

### Step 2: Authenticate

**Option A: Using gcloud CLI**
```bash
# Install gcloud CLI if you haven't
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

**Option B: Using Service Account**
1. Go to **IAM & Admin** → **Service Accounts**
2. Create new service account
3. Grant "Vertex AI User" role
4. Create and download JSON key
5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   ```

### Step 3: Install Required Packages

```bash
pip install google-cloud-aiplatform langchain-google-vertexai
```

### Step 4: Use in Your Code

```python
from websocietysimulator.llm.vertex_ai_llm import VertexAILLM
from websocietysimulator import Simulator

# Initialize simulator
simulator = Simulator(data_dir="your_data_dir", device="auto", cache=True)
simulator.set_task_and_groundtruth(
    task_dir="./example/track1/yelp/tasks",
    groundtruth_dir="./example/track1/yelp/groundtruth"
)

# Use Vertex AI with your GCP project
simulator.set_llm(VertexAILLM(
    project_id="your-gcp-project-id",
    location="us-central1",  # or us-east1, europe-west1, etc.
    model="gemini-2.5-pro"  # or gemini-1.5-pro, gemini-1.5-flash
))

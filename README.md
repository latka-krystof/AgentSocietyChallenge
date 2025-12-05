# CS245 Team 7 -- AgentSociety Challenge Extension

This repository contains our implementation of LLM agents for user behavior simulation, developed as part of a class project. The agents combine different strategies including **memory**, **planning**, and **reasoning** to simulate realistic user behavior on review platforms.

## Directory Structure

- **`websocietysimulator/`**: Core library containing the simulation framework, agent base classes, LLM clients, and evaluation tools.
- **`agents/`**: Contains our agent implementations:
  - `ModelingAgent_memory_planning_and_reasoning.py` - **Final combined agent** (Memory + Planning + Reasoning)
  - `ModelingAgent_Vader.py` - Vader agent (Planning + Reasoning with VADER sentiment analysis)
  - `ModelingAgent_baseline.py` - Baseline agent
  - `ModelingAgent_planning.py` - Planning-only agent
  - `ModelingAgent_planning_and_reasoning.py` - Planning + Reasoning agent
  - `ModelingAgent_memory_and_reasoning.py` - Memory + Reasoning agent
  - `ModelingAgent_memory.py` - Memory-only agent
  - `ModelingAgent_reasoning.py` - Reasoning-only agent
- **`data_process.py`**: Script to process raw Yelp/Amazon/Goodreads datasets into the required format.

---

## Setup Instructions

### 1. Install Dependencies

The repository is organized using [Python Poetry](https://python-poetry.org/). Follow these steps to install the library:

1. Clone the repository:
   ```bash
   git clone <this_repo>
   cd websocietysimulator
   ```

2. Install dependencies:
   - **Option 1: Install dependencies using Poetry** :
     ```bash
     poetry install && \
     poetry shell
     ```
   - **Option 2: Install dependencies using pip** :
     ```bash
     pip install websocietysimulator
     ```
   - **Option 3: Install dependencies using conda**:
     ```bash
     conda create -n websocietysimulator python=3.11 && \
     conda activate websocietysimulator && \
     pip install -r requirements.txt && \
     pip install .
     ```

3. Verify the installation:
   ```python
   import websocietysimulator
   ```

### 2. Google Cloud Authentication

Our agents use Google Cloud Vertex AI (Gemini 2.5 Pro) for LLM inference. You need to:

1. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth application-default login
   ```

2. **Set your GCP project**:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```
   Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID.

3. **Set the environment variable**:
   
   **Temporary (current session only)**:
   ```bash
   export GCP_PROJECT_ID=YOUR_PROJECT_ID
   ```
   
   **Permanent (add to your shell profile)**:
   
   For bash:
   ```bash
   echo 'export GCP_PROJECT_ID=YOUR_PROJECT_ID' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   For zsh:
   ```bash
   echo 'export GCP_PROJECT_ID=YOUR_PROJECT_ID' >> ~/.zshrc
   source ~/.zshrc
   ```
   
   **For conda environment** (recommended):
   ```bash
   conda env config vars set GCP_PROJECT_ID=YOUR_PROJECT_ID -n websocietysimulator
   conda activate websocietysimulator
   ```

4. **Verify authentication**:
   ```bash
   echo $GCP_PROJECT_ID
   gcloud auth application-default print-access-token
   ```

### 3. Data Preparation

1. Download the raw Yelp dataset from [Yelp Dataset](https://www.yelp.com/dataset).

2. Extract the dataset:
   ```bash
   tar -xvf yelp_dataset.tar
   ```

3. Process the dataset:
   ```bash
   python data_process.py --input <path_to_raw_dataset> --output ./dataset --yelp_only
   ```
   
   The `--yelp_only` flag processes only Yelp data (Amazon and Goodreads files are optional).

4. Verify the processed dataset structure:
   ```
   ./dataset/
   ├── item.json
   ├── review.json
   └── user.json
   ```

---

## Running the Agents

### Running the Final Combined Agent

The final agent (`ModelingAgent_memory_planning_and_reasoning.py`) combines memory, planning, and reasoning strategies. It will automatically run comparisons against all other agents:

```bash
conda activate websocietysimulator
export GCP_PROJECT_ID=YOUR_PROJECT_ID  # If not set permanently
python agents/ModelingAgent_memory_planning_and_reasoning.py
```

This will:
1. Run the combined agent (Memory + Planning + Reasoning)
2. Run all comparison agents:
   - Baseline
   - Planning Only
   - Planning + Reasoning
   - Memory + Reasoning
   - Memory Only
   - Reasoning Only
3. Generate evaluation results for each agent
4. Display a comprehensive comparison summary

### Running the Vader Agent

The Vader agent (`ModelingAgent_Vader.py`) combines planning and reasoning strategies with VADER sentiment analysis. It will automatically run comparisons against other agents:

```bash
conda activate websocietysimulator
export GCP_PROJECT_ID=YOUR_PROJECT_ID  # If not set permanently
python agents/ModelingAgent_Vader.py
```

This will:
1. Run the Vader agent (Planning + Reasoning with VADER sentiment analysis)
2. Run all comparison agents:
   - Baseline
   - Planning Only
3. Generate evaluation results for each agent
4. Display a comprehensive comparison summary

### Running Individual Agents

You can also run individual agents for testing:

```bash
# Baseline agent
python agents/ModelingAgent_baseline.py

# Planning-only agent
python agents/ModelingAgent_planning.py

# Planning + Reasoning agent
python agents/ModelingAgent_planning_and_reasoning.py

# Memory + Reasoning agent
python agents/ModelingAgent_memory_and_reasoning.py
```

---

## Viewing Results

After running an agent, results are saved in timestamped directories:

```
./memory_planning_reasoning_yelp_YYYYMMDD_HHMMSS/
├── MemoryPlanningAndReasoning_outputs.json      # Agent outputs
├── MemoryPlanningAndReasoning_evaluation.json   # Evaluation metrics
├── Baseline_outputs.json
├── Baseline_evaluation.json
├── PlanningOnly_outputs.json
├── PlanningOnly_evaluation.json
└── ... (similar files for all agents)
```

### Evaluation Metrics

Each `*_evaluation.json` file contains:
- **Preference Estimation**: How well the agent predicts user preferences
- **Review Generation**: Quality of generated reviews
- **Overall Quality**: Combined metric

### Output Format

Each `*_outputs.json` file contains the agent's predictions:
```json
[
  {
    "task_id": "...",
    "stars": 4.0,
    "review": "Great experience! The service was excellent..."
  },
  ...
]
```

### Comparison Summary

The script automatically prints a comparison summary showing:
- Metrics for each agent strategy
- Improvements vs baseline
- Comparison of combined strategies

---

## Agent Architecture

Our agents are built using three key components that can be combined in different ways:

1. **Planning Module** (`PlanningOnlyModule`): 
   - Creates a deterministic plan for gathering context
   - Executes structured steps: fetch user, business, reviews
   - Provides a systematic approach to information gathering

2. **Memory Module** (`MemoryUserProfile`):
   - Stores and retrieves relevant reviews from memory
   - Builds user preference profiles
   - Enables agents to leverage historical context and past interactions

3. **Reasoning Module** (`ReasoningCOTWithReflection`):
   - Uses Chain of Thought with reflection
   - Determines rating and generates review text based on gathered context
   - Applies logical reasoning to synthesize information and produce outputs

Different agent implementations combine these components in various ways:
- **Baseline**: Minimal implementation without these modules
- **Planning-only**: Uses only the planning module
- **Reasoning-only**: Uses only the reasoning module
- **Memory-only**: Uses only the memory module
- **Combined agents**: Use two or more components together (e.g., Planning + Reasoning, Memory + Reasoning, Memory + Planning + Reasoning)
- **Vader agent**: Uses Planning + Reasoning with VADER sentiment analysis for enhanced rating prediction

The typical agent workflow:
1. Executes the planning structure to gather data (if planning module is used)
2. Stores and retrieves relevant information from memory (if memory module is used)
3. Uses the reasoning module to determine rating and generate review text
4. Returns both rating and review text

---

## Requirements

- Python 3.11
- Google Cloud account with Vertex AI API enabled
- At least 16GB RAM for dataset processing
- Processed Yelp dataset in `./dataset/` directory

---

## License

This project is licensed under the MIT License.

## References

- [Yelp Dataset](https://www.yelp.com/dataset)
- [Amazon Dataset](https://amazon-reviews-2023.github.io/)
- [Goodreads Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)

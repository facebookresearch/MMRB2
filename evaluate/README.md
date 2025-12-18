# MMRB2 Evaluation

This directory contains evaluation scripts for the MMRB2 benchmark.

## Structure

```
evaluate/
├── generate_judgements/      # Part 1: Generate judgements using LLM judges
│   ├── evaluate.py           # Single-process evaluation
│   ├── multi_gpu_evaluate.py # Multi-GPU/process evaluation
│   ├── run_gpt4o.sh          # GPT-4o script
│   ├── run_gemini25flash.sh  # Gemini 2.5 Flash script
│   ├── run_qwen3.sh          # Qwen3-VL-8B script
│   └── outputs/              # Sample outputs
└── compute_scores/           # Part 2: Compute accuracy from judgement files
    └── compute_accuracy.py   # Scoring script
```

---

## Part 1: Generate Judgements (optional)

Generate pairwise judgements using any reward models. We provide example implementations for a few multimodal LLM-as-a-judge, but you can easily add your own LLM as the base model for judges.

Note that you don't necessarily need an LLM-as-a-judge, you can also use any other models, as long as you process the results in the correct format. Here we mainly provide the codes for using any LLM as judge.

### Example Evaluators

| Evaluator | Type | Description |
|-----------|------|-------------|
| `gpt4o-pairwise` | API | GPT-4o via Azure OpenAI |
| `gemini25flash-pairwise` | API | Gemini 2.5 Flash |
| `qwen3vl8b-pairwise` | Local | Qwen3-VL-8B (requires GPU) |

### Setup

1. **Set up API keys:**

   For OpenAI/Azure:
   ```bash
   export AZURE_OPENAI_ENDPOINT="your-endpoint"
   export AZURE_OPENAI_API_KEY="your-key"
   ```
   
   For Google:
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```

2. **Run evaluation:**
   ```bash
   cd generate_judgements
   
   # Run GPT-4o evaluation
   ./run_gpt4o.sh
   
   # Or run Gemini 2.5 Flash
   ./run_gemini25flash.sh
   
   # Or run local Qwen3 (requires 8 GPUs)
   ./run_qwen3.sh
   ```

### Manual Usage

```bash
# Single process
python evaluate.py \
    --evaluator_name gpt4o-pairwise \
    --task_type image \
    --pairs_path /path/to/t2i.json \
    --output_path ./outputs/task1_gpt4o-pairwise.json

# Multi-process (8 parallel workers)
python multi_gpu_evaluate.py \
    --evaluator_name gpt4o-pairwise \
    --task_type image \
    --pairs_path /path/to/t2i.json \
    --output_path ./outputs/task1_gpt4o-pairwise.json \
    --n_gpu 8
```

### Task Types

| Task | Type Flag | Benchmark File |
|------|-----------|----------------|
| Text-to-Image | `image` | t2i.json |
| Image Editing | `edit` | edit.json |
| Interleaved | `interleaved` | interleaved.json |
| Visual Reasoning | `reasoning` | reasoning.json |

### Adding Your Own LLM as judge

You can add your own model as a judge by following these steps:

#### For API-based Models

1. **Add your API client** in `evaluator_apis/model_apis/your_provider/`:

```python
# evaluator_apis/model_apis/your_provider/your_model.py
class YourModelClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate(self, messages: list) -> str:
        # Call your API and return the response text
        ...
```

2. **Create your evaluator class** in `evaluator_apis/llm_judges/api_pairwise_evaluator.py`:

```python
class YourModelPairwiseEvaluator(APIPairwiseEvaluator):
    def __init__(self, device_id: int = None):
        # Initialize your model client
        self.client = YourModelClient(api_key="...")
    
    @property
    def evaluator_name(self):
        return "your-model_pairwise_evaluator"
    
    def generate_response(self, prompt: List[List[str]]) -> str:
        # Convert prompt to your API format and get response
        ...
```

3. **Register your evaluator** in `evaluator_apis/evaluators.py`:

```python
from .llm_judges import YourModelPairwiseEvaluator

EVALUATORS = {
    # ... existing evaluators ...
    "your-model-pairwise": {
        "class": YourModelPairwiseEvaluator,
        "type": EvaluatorTypes.PAIRWISE,
        "is_api_based": True,
        "capabilities": [
            EvaluatorCapabilities.IMAGE,
            EvaluatorCapabilities.EDIT,
            EvaluatorCapabilities.INTERLEAVED,
            EvaluatorCapabilities.REASONING,
        ],
    },
}
```

#### For Local Models

1. **Add your model** in `evaluator_apis/llm_judges/local_models.py`:

```python
class LocalModelManager:
    MODEL_CONFIGS = {
        # ... existing models ...
        "your-local-model": {
            "huggingface_id": "your-org/your-model",
        },
    }
```

2. **Create your evaluator class** in `evaluator_apis/llm_judges/local_pairwise_evaluator.py`:

```python
class YourLocalModelPairwiseEvaluator(LocalPairwiseEvaluator):
    def __init__(self, device_id: int = None):
        super().__init__(model_name="your-local-model", device_id=device_id)
```

3. **Register in `evaluators.py`** with `"is_api_based": False`.

#### Output Format

Your evaluator must return judgements in this JSON format:

```json
{
  "better_response": "A" or "B",
  "reasoning": "...",  // optional
  ...  // any other metadata
}
```

---

## Part 2: Compute Scores

Compute accuracy by comparing judgements against ground truth labels.

### Usage

```bash
cd compute_scores

# Evaluate a single task
python compute_accuracy.py --task image \
    --predictions ../generate_judgements/outputs/sample_task1_image.json

# Evaluate all 4 tasks
python compute_accuracy.py --task all \
    --predictions ../generate_judgements/outputs/sample_task1_image.json \
                  ../generate_judgements/outputs/sample_task2_edit.json \
                  ../generate_judgements/outputs/sample_task3_interleaved.json \
                  ../generate_judgements/outputs/sample_task4_reasoning.json
```

### Scoring Logic

For each pair:
1. **Forward**: Compare `forward.judgement` with ground truth `chosen`
2. **Reverse**: Flip `reverse.judgement` (A↔B), then compare with `chosen`
3. **Overall accuracy** = (forward_correct + reverse_correct) / total

### Output Format

```
==================================================
SUMMARY
==================================================
Task                      Accuracy     Missing
--------------------------------------------------
task1_image                53.20%      0
task2_edit                 55.50%      0
task3_interleaved          57.50%      0
task4_reasoning            47.50%      0
--------------------------------------------------
Overall                    53.42%
==================================================
```

---

## Requirements

```
torch
transformers
openai
google-genai
json-repair
tqdm
Pillow
```

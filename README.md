# imsearch_eval

Abstract, extensible framework for benchmarking vector databases and models across different datasets for Image Search and caption generation.

## Installation

### Core Framework Only

Install just the core framework (no adapters):

```bash
pip install imsearch_eval
```

### With Specific Adapters

**Triton adapters** (for Triton Inference Server):
```bash
pip install imsearch_eval[triton]
```

**Weaviate adapters** (includes Triton, as WeaviateAdapter uses TritonModelUtils):
```bash
pip install imsearch_eval[weaviate]
```

**All adapters**:
```bash
pip install imsearch_eval[all]
```

**Development dependencies**:
```bash
pip install imsearch_eval[dev]
```

### From Source

```bash
git clone https://github.com/waggle-sensor/imsearch_eval
cd imsearch_eval
pip install -e .  # Core only
# Or with extras:
pip install -e ".[triton]"
pip install -e ".[weaviate]"
pip install -e ".[all]"
```

## Quick Start

```python
from imsearch_eval import BenchmarkEvaluator
from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
import tritonclient.grpc as TritonClient

# Initialize clients
weaviate_client = WeaviateAdapter.init_client(host="127.0.0.1", port="8080")
triton_client = TritonClient.InferenceServerClient(url="triton:8001")

# Create adapters
vector_db = WeaviateAdapter(
    weaviate_client=weaviate_client,
    triton_client=triton_client
)
model_provider = TritonModelProvider(triton_client=triton_client)

# Use in evaluator (requires a BenchmarkDataset implementation)
evaluator = BenchmarkEvaluator(
    vector_db=vector_db,
    model_provider=model_provider,
    dataset=dataset,  # Your BenchmarkDataset implementation
    collection_name="my_collection",
    query_method="clip_hybrid_query"
)
```

## Architecture

The framework is organized into two main components:

1. **Framework** (`imsearch_eval/framework/`): Abstract interfaces and evaluation logic (dataset-agnostic)
2. **Adapters** (`imsearch_eval/adapters/`): Shared concrete implementations for vector databases and models

### Package Structure

```
imsearch_eval/
├── framework/                    # Abstract interfaces and evaluation logic
│   ├── interfaces.py            # VectorDBAdapter, ModelProvider, Query, BenchmarkDataset, etc.
│   ├── model_utils.py           # ModelUtils abstract interface
│   └── evaluator.py            # BenchmarkEvaluator class
│
└── adapters/                     # Shared concrete implementations
    ├── __init__.py             # Exports all adapters
    ├── triton.py               # TritonModelProvider, TritonModelUtils
    └── weaviate.py             # WeaviateAdapter, WeaviateQuery
```

## Framework Components

### Interfaces (`imsearch_eval.framework.interfaces`)

- **`VectorDBAdapter`**: Abstract interface for vector databases
  - Methods: `init_client()`, `search()`, `create_collection()`, `delete_collection()`, `insert_data()`, `close()`
- **`ModelProvider`**: Abstract interface for model providers
  - Methods: `get_embedding()`, `generate_caption()`
- **`Query`**: Abstract interface for query classes (used by vector DB adapters)
  - Method: `query(near_text, collection_name, limit, query_method, **kwargs)` - Generic query method
  - Each vector DB implementation can define its own query types via `query_method` parameter
- **`ModelUtils`**: Abstract interface for model utilities (in `imsearch_eval.framework.model_utils`)
  - Methods: `calculate_embedding()`, `generate_caption()`
- **`BenchmarkDataset`**: Abstract interface for benchmark datasets
- **`DataLoader`**: Abstract interface for loading data into vector DBs
- **`Config`**: Abstract interface for configuration/hyperparameters
- **`QueryResult`**: Container for query results

### Available Adapters

#### Triton Adapters (`imsearch_eval.adapters.triton`)

- **`TritonModelUtils`**: Triton-based implementation of `ModelUtils` interface
- **`TritonModelProvider`**: Triton inference server model provider

**Dependencies**: `tritonclient[grpc]`

#### Weaviate Adapters (`imsearch_eval.adapters.weaviate`)

- **`WeaviateQuery`**: Implements `Query` interface for Weaviate
  - Generic `query()` method routes to specific methods based on `query_method` parameter
  - Also provides Weaviate-specific methods: `hybrid_query()`, `colbert_query()`, `clip_hybrid_query()`
- **`WeaviateAdapter`**: Implements `VectorDBAdapter` interface for Weaviate
  - Uses `WeaviateQuery` internally for search operations

**Dependencies**: `weaviate-client`, `tritonclient[grpc]` (for embedding generation)

### Evaluator (`imsearch_eval.framework.evaluator`)

- **`BenchmarkEvaluator`**: Main evaluation class that works with any combination of adapters and benchmark datasets
- Computes metrics: NDCG, precision, recall, accuracy
- Supports parallel query processing

## Usage

### Dataset schema (required columns)

Your `BenchmarkDataset.load()` must return a pandas `DataFrame`. **Column names can differ**, but the *meaning* of the fields below must stay constant because they’re used to compute metrics.

`BenchmarkEvaluator` gets the required column names from your `BenchmarkDataset`:
- `get_query_column()` → query text
- `get_query_id_column()` → query/group id (unique id for each unique query)
- `get_relevance_column()` → relevance label (1/0)
- `get_metadata_columns()` → optional metadata copied into the per-query stats output

#### Required semantic fields

- **Query text**: The text sent to `VectorDBAdapter.search(...)`.
- **Query id**: A stable identifier used to group rows belonging to the same query.
- **Relevance label**: Binary label for each row/item (1 = relevant, 0 = not relevant). Used for precision/recall/NDCG.

#### Optional (but common) fields for image search

- **Image**: A file path/URL/bytes you use when building embeddings or generating captions (consumed by your `BenchmarkDataset` / adapter, not the core evaluator).
- **Ranking score(s)**: If your search results include a score column like `rerank_score`, `clip_score`, `score`, or `distance`, the evaluator will use the first one it finds to compute NDCG.
- **License / rights_holder**: Useful when combining datasets, otherwise optional.
- **Additional metadata**: Any extra fields you want to use for result breakdowns (e.g., animalspecies category). These do **not** change the metrics; they’re just copied into the results table.

### Basic Usage Pattern

1. **Import adapters**:
   ```python
   from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider
   ```

2. **Initialize clients**:
   ```python
   import tritonclient.grpc as TritonClient
   
   weaviate_client = WeaviateAdapter.init_client(host="127.0.0.1", port="8080")
   triton_client = TritonClient.InferenceServerClient(url="triton:8001")
   ```

3. **Create adapters**:
   ```python
   vector_db = WeaviateAdapter(
       weaviate_client=weaviate_client,
       triton_client=triton_client
   )
   model_provider = TritonModelProvider(triton_client=triton_client)
   ```

4. **Create benchmark dataset** (you need to implement this):
   ```python
   from imsearch_eval import BenchmarkDataset
   import pandas as pd
   
   class MyBenchmarkDataset(BenchmarkDataset):
       def load(self, split="test", **kwargs) -> pd.DataFrame:
           # Load your dataset
           return dataset_df
       
       def get_query_column(self) -> str:
           return "query"
       
       def get_query_id_column(self) -> str:
           return "query_id"
       
       def get_relevance_column(self) -> str:
           return "relevant"
       
       def get_metadata_columns(self) -> list:
           return ["category", "type"]
   ```

5. **Create evaluator and run**:
   ```python
   from imsearch_eval import BenchmarkEvaluator
   
   dataset = MyBenchmarkDataset()
   
   evaluator = BenchmarkEvaluator(
       vector_db=vector_db,
       model_provider=model_provider,
       dataset=dataset,
       collection_name="my_collection",
       query_method="clip_hybrid_query"  # Query type for WeaviateQuery
   )
   
   results, stats = evaluator.evaluate_queries(split="test")
   ```

### Query Method Parameter

The `query_method` parameter in `BenchmarkEvaluator` is passed to the `Query.query()` method:

- **For Weaviate**: `query_method` can be `"clip_hybrid_query"`, `"hybrid_query"`, or `"colbert_query"`
- **For other vector DBs**: Implement your own query types in your `Query` implementation
- The `Query.query()` method routes to the appropriate implementation based on `query_method`

### Model Names

The `ModelProvider` and `ModelUtils` interfaces accept `model_name` parameters:

- **Embedding models**: `"clip"`, `"colbert"`, `"align"` (for TritonModelProvider)
- **Caption models**: `"gemma3"`, `"qwen2_5"` (for TritonModelProvider)
- Other implementations can define their own model names

## Extending the Framework

### Adding a New Vector Database

1. **Create a Query class** implementing the `Query` interface:
   ```python
   from imsearch_eval import Query
   import pandas as pd
   
   class MyVectorDBQuery(Query):
       def query(self, near_text, collection_name, limit=25, query_method="vector", **kwargs):
           # Implement your query logic
           # query_method can be "vector", "keyword", "hybrid", etc.
           return pd.DataFrame(results)
   ```

2. **Create an adapter** implementing `VectorDBAdapter`:
   ```python
   from imsearch_eval import VectorDBAdapter, QueryResult
   
   class MyVectorDBAdapter(VectorDBAdapter):
       @classmethod
       def init_client(cls, **kwargs):
           # Initialize your vector DB client
           return client
       
       def __init__(self, client=None, **kwargs):
           if client is None:
               client = self.init_client(**kwargs)
           self.client = client
           self.query_instance = MyVectorDBQuery(client)
       
       def search(self, query, collection_name, limit=25, query_method="vector", **kwargs):
           df = self.query_instance.query(query, collection_name, limit, query_method, **kwargs)
           return QueryResult(df.to_dict('records'))
       
       # Implement other required methods...
   ```

### Adding a New Model Provider

1. **Create ModelUtils implementation** (optional but recommended):
   ```python
   from imsearch_eval.framework.model_utils import ModelUtils
   
   class MyModelUtils(ModelUtils):
       def calculate_embedding(self, text, image=None, model_name="default"):
           # Your embedding implementation
           return embedding
       
       def generate_caption(self, image, model_name="default"):
           # Your caption generation
           return caption
   ```

2. **Create ModelProvider**:
   ```python
   from imsearch_eval import ModelProvider
   
   class MyModelProvider(ModelProvider):
       def __init__(self, **kwargs):
           self.model_utils = MyModelUtils(**kwargs)
       
       def get_embedding(self, text, image=None, model_name="default"):
           return self.model_utils.calculate_embedding(text, image, model_name)
       
       def generate_caption(self, image, model_name="default"):
           return self.model_utils.generate_caption(image, model_name)
   ```

### Adding a New Dataset

Create a benchmark dataset implementing `BenchmarkDataset`:

```python
from imsearch_eval import BenchmarkDataset
import pandas as pd

class MyBenchmarkDataset(BenchmarkDataset):
    def load(self, split="test", **kwargs) -> pd.DataFrame:
        # Load your dataset
        return dataset_df
    
    def get_query_column(self) -> str:
        return "query"
    
    def get_query_id_column(self) -> str:
        return "query_id"
    
    def get_relevance_column(self) -> str:
        return "relevant"
    
    def get_metadata_columns(self) -> list:
        return []
```

## How It Works

### Abstract Interface Pattern

The framework uses abstract interfaces to ensure consistency and extensibility:

1. **Framework defines interfaces** (`imsearch_eval.framework.interfaces`, `imsearch_eval.framework.model_utils`):
   - `VectorDBAdapter`, `ModelProvider`, `Query`, `ModelUtils`, `BenchmarkDataset`, etc.
   - These define the contract that all implementations must follow

2. **Adapters implement interfaces** (`imsearch_eval.adapters`):
   - `TritonModelUtils` implements `ModelUtils`
   - `TritonModelProvider` implements `ModelProvider` and uses `TritonModelUtils`
   - `WeaviateQuery` implements `Query`
   - `WeaviateAdapter` implements `VectorDBAdapter` and uses `WeaviateQuery`

3. **Users use adapters**:
   - Import from `imsearch_eval.adapters` (e.g., `from imsearch_eval.adapters import WeaviateAdapter, TritonModelProvider`)
   - Use the abstract interfaces, not concrete implementations
   - Easy to swap implementations without changing benchmark code

### Example: Using the Query Interface

```python
from imsearch_eval.adapters import WeaviateQuery
import tritonclient.grpc as TritonClient

# WeaviateQuery implements the Query interface
triton_client = TritonClient.InferenceServerClient(url="triton:8001")
query_instance = WeaviateQuery(weaviate_client, triton_client)

# Use the generic query() method
results = query_instance.query(
    near_text="search query",
    collection_name="my_collection",
    limit=25,
    query_method="clip_hybrid_query"  # Weaviate-specific query type
)

# Or use Weaviate-specific methods directly
results = query_instance.clip_hybrid_query("search query", "my_collection", limit=25)
```

### Example: Using the ModelUtils Interface

```python
from imsearch_eval.adapters import TritonModelProvider, TritonModelUtils
import tritonclient.grpc as TritonClient

# TritonModelUtils implements the ModelUtils interface
triton_client = TritonClient.InferenceServerClient(url="triton:8001")
model_utils = TritonModelUtils(triton_client)

# Use the abstract methods
embedding = model_utils.calculate_embedding("text", image=None, model_name="clip")
caption = model_utils.generate_caption(image, model_name="gemma3")

# Or use via ModelProvider
model_provider = TritonModelProvider(triton_client)
embedding = model_provider.get_embedding("text", image=None, model_name="clip")
caption = model_provider.generate_caption(image, model_name="gemma3")
```

## Key Features

- **Dataset-Agnostic**: Works with any dataset by implementing `BenchmarkDataset`
- **Extensible**: Easy to add new vector databases, models, and datasets
- **Abstract Interfaces**: Clean separation between evaluation logic and implementations
- **Reusable**: Framework code can be shared across all benchmarks
- **Consistent**: Same evaluation metrics and methodology
- **Type Safe**: Abstract interfaces ensure all implementations provide required functionality
- **Flexible**: Each implementation can define its own query types and model names

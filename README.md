# AI-HEXAGON

> ⚠️ **Early Development**: This project is currently in its early development phase and not accepting external architecture submissions yet. Star/watch the repository to be notified when we open for contributions.

📊 **[View Live Leaderboard & Results](https://ai-hexagon.dev)**

AI-HEXAGON is an objective benchmarking framework designed to evaluate neural network architectures independently of natural language processing tasks. By isolating architectural capabilities from training techniques and datasets, it enables meaningful and efficient comparisons between different neural network designs.

## 🎯 Motivation

Traditional neural network benchmarking often conflates architectural performance with training techniques and dataset biases. This makes it challenging to:

-   Isolate true architectural capabilities
-   Iterate quickly on design changes
-   Compare models fairly

**AI-HEXAGON solves these challenges by:**

-   **🔍 Pure Architecture Focus**: Tests that evaluate only the architecture, removing confounding factors like tokenization and dataset-specific optimizations
-   **⚡ Rapid Iteration**: Enable quick testing of architectural changes without large-scale training
-   **🛠️ Flexible Testing**: Support both standard benchmarking and custom test suites

## 🌟 Key Features

-   **📊 Pure Architecture Evaluation**: Tests fundamental capabilities independently
-   **⚖️ Controlled Environment**: Fixed parameter budget and raw numerical inputs
-   **📐 Clear Metrics**: Six independently measured fundamental capabilities
-   **🔍 Transparent Implementation**: Clean, framework-agnostic code
-   **🤖 Automated Testing**: GitHub Actions for fair, manipulation-proof evaluation
-   **📈 Live Results**: Real-time benchmarking results at [ai-hexagon.dev](https://ai-hexagon.dev)

## 📐 Metrics (The Hexagon)

Each architecture is evaluated on six fundamental capabilities:

| Metric                       | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| 🧠 **Memory Capacity**       | Store and recall information from training data |
| 🔄 **State Management**      | Maintain and manipulate internal hidden states  |
| 🎯 **Pattern Recognition**   | Recognize and extrapolate sequences             |
| 📍 **Position Processing**   | Handle positional information within sequences  |
| 🔗 **Long-Range Dependency** | Manage dependencies over long sequences         |
| 📏 **Length Generalization** | Process sequences longer than training examples |

## 📁 Project Structure

```
ai-hexagon/
├── ai_hexagon/
│   └── modules/          # Common neural network modules
└── results/              # Model implementations and results
    ├── suite.json        # Default test suite configuration
    └── transformer/
        ├── model.py      # Transformer implementation
        └── modules/      # Custom modules (if needed)
```

## ⚙️ Parameter Budget

The default suite enforces a 4MB parameter limit for fair comparisons:

| Precision | Parameter Limit |
| --------- | --------------- |
| Complex64 | 0.5M params     |
| Float32   | 1M params       |
| Float16   | 2M params       |
| Int8      | 4M params       |

## 🤝 Contributing

We welcome contributions once the project is ready for external input. To contribute:

1. **Fork**: Create your own fork of the project
2. **Install**: Run `poetry install` (optionally with `--with dev,cuda12`) to get the `ai-hex` command
3. **Implement**: Add your model in `results/your_model_name/`
4. **Document**: Include comprehensive docstrings and references
5. **Submit**: Create a pull request following our guidelines
6. **Wait**: CI will automatically evaluate your model and update the leaderboard

Use `ai-hex tests list` to see available tests, `ai-hex tests show test_name` to view test schema, and `ai-hex suite run ./path/to/model.py` to run your model against the suite.

### 🔧 Technical Stack: JAX and Flax

We chose JAX and Flax for their:

-   **🧩 Functional Design**: Clear architecture definitions with immutable state
-   **⚡ Custom Operations**: Comprehensive support through `jax.numpy`
-   **🎯 Reproducibility**: First-class random number handling

### 📝 Code Style: Using `einops`

We mandate `einops` for complex tensor operations to enhance readability. Compare:

```python
# Traditional approach - hard to understand the transformation
x = x.reshape(batch, x.shape[1], x.shape[-2]*2, x.shape[-1]//2)
x = x.transpose(0, 2, 1, 3)

# Using einops - crystal clear intent
x = rearrange(x, 'b t (h d) c -> b (h t) (d c)')
```

### 📖 Example Model Implementation

```python
import flax.linen as nn
from einops import rearrange

class Transformer(nn.Module):
    """
    Transformer Decoder Stack architecture from 'Attention Is All You Need'.
    Reference: https://arxiv.org/abs/1706.03762
    """
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        # Architecture implementation
        return x
```

## 🔍 Test Suite Configuration

Test suites use a JSON configuration format:

```json
{
    "name": "General 1M",
    "description": "General architecture performance evaluation",
    "metrics": [
        {
            "name": "Memory Capacity",
            "description": "Information storage and recall capability",
            "tests": [
                {
                    "weight": 1.0,
                    "test": {
                        "name": "hash_map",
                        "seed": 0,
                        "key_length": 8,
                        "value_length": 64,
                        "num_pairs_range": [32, 65536],
                        "vocab_size": 1024
                    }
                }
            ]
        }
    ]
}
```

---

📈 Results are automatically generated via GitHub Actions to ensure fairness. The leaderboard is updated in real-time at [ai-hexagon.dev](https://ai-hexagon.dev).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

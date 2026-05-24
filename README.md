# NeuralNets: Foundations from Scratch 🧠

Welcome to **NeuralNets**! The core objective of this repository is to build and understand the fundamental concepts behind neural networks, backpropagation, and deep learning from the ground up.

This project is inspired by and follows Andrej Karpathy's highly acclaimed **[Zero to Hero](https://karpathy.ai/zero-to-hero.html)** deep learning series.

---

## 🚀 Key Learning Milestones

### 1. Chapter 1: Building Micrograd (Autograd & MLP)
In this section, we implement **Micrograd**, a tiny scalar-valued autograd engine, and use it to construct a Multi-Layer Perceptron (MLP) trained to perform binary classification.

* **Autograd DAG**: Building a directed acyclic graph (DAG) dynamically as operations are performed.
* **Backpropagation & Chain Rule**: Automating gradient calculation through local derivatives using the multivariate chain rule.
* **Gradient Accumulation**: Resolving the classic "gradient overwriting" bug using `+=` to correctly handle multiple inputs/connections (multivariate paths).
* **Neural Net API**: Building modular abstractions for:
  * `Neuron`: A single artificial neuron ($y = \tanh(\sum w_i x_i + b)$).
  * `Layer`: A fully connected layer of neurons.
  * `MLP`: A stack of sequential layers.
* **Optimization Loop**: Implementing the complete training cycle:
  1. Forward pass (predictions and Mean Squared Error Loss).
  2. Zero gradients.
  3. Backward pass (autograd).
  4. Parameter updates (Gradient Descent step).

---

## 📂 Project Structure

* **`Zero-to-hero/`**: Dedicated directory for lecture chapters.
  * **`Chapter-1/`**:
    * 📓 **[NeuralNets_BackProp.ipynb](file:///c:/Users/sujat/projects/NeuralNets/Zero-to-hero/Chapter-1/NeuralNets_BackProp.ipynb)**: The main notebook containing a detailed step-by-step walkthrough, visualized computational graphs, bug resolution, and the MLP training loop.
    * 📓 **[Micrograd_Part1.ipynb](file:///c:/Users/sujat/projects/NeuralNets/Zero-to-hero/Chapter-1/Micrograd_Part1.ipynb)**: Scratchpad/part 1 notebook for basic functions.

---

## 🛠️ Setup and Installation

To run these notebooks locally, set up your Python virtual environment and install the required dependencies:

### 1. Clone the repository and navigate into it:
```bash
git clone <your-repo-url>
cd NeuralNets
```

### 2. Activate the virtual environment:
* **Windows**:
  ```powershell
  .venv\Scripts\activate
  ```
* **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Requirements:
Make sure you have the following libraries installed in your environment:
* `numpy`
* `matplotlib`
* `graphviz` (requires Graphviz binary installed on your system)
* `torch` (used to cross-reference and verify gradients)

---

## 🔍 Core Concept: Scalar autograd under the hood

The heart of backpropagation in our custom autograd engine relies on the `Value` class, which handles local math operations. When a forward operation is performed, it stores a pointer to a `_backward` function:

```python
# Addition implementation in Value class
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
```

Calling `.backward()` performs a topological sort of the graph and executes the `_backward` chain rule updates in reverse topological order, automatically computing gradients for all parameters.
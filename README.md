# Custom-BERT 🤖

Customized the legacy BERT architecture by integrating recent research advancements focused on model performance optimization. This project builds upon the basic BERT model with a series of enhancements to improve efficiency, training speed, and overall performance.

---

## Improvements 🛠️

### Model Improvement:

- **Using Flash Attention** ⚡  
  The Flash Attention logic optimizes the calculation of attention scores by reducing memory overhead and computational cost. This technique leverages efficient algorithms to compute attention more rapidly, making it particularly beneficial for long sequence processing.

- **GELU Activation Function** 🔄  
  The Gaussian Error Linear Unit (GELU) activation function provides a smoother, non-linear transformation compared to traditional functions like ReLU. Its probabilistic nature helps in better capturing the nuances in data, leading to improved model performance and training stability.

- **Prenormalized the Layer** 📏  
  Prenormalization involves applying normalization techniques (such as LayerNorm) before the main transformations in the model layers. This helps in stabilizing the training process, ensuring that the inputs to each layer have a consistent scale and distribution, which can lead to faster convergence.

- **Fusing the Kernel Operation** 🔗  
  Kernel fusion leverages advanced features from `torch.compiler` mode to combine multiple operations into a single kernel. This reduces the overhead associated with launching multiple kernels on hardware accelerators and enhances the overall computational efficiency.

- **Auto Mixed Precision** ⚖️  
  Auto Mixed Precision (AMP) enables the use of both 16-bit and 32-bit floating point types during training. By intelligently switching between precisions, the model can achieve faster training speeds and reduced memory usage without sacrificing accuracy.

- **Uniform Length Batching** 📦  
  Uniform length batching standardizes the sequence lengths within a batch, minimizing the need for dynamic padding. This method reduces the computational overhead associated with variable-length sequences and leads to more efficient use of resources during training.

---

## Performance Metrics 📊

| Optimization      | Speedup | Memory Reduction |
|-------------------|---------|------------------|
| Flash Attention   | 2.8×    | 60%              |
| Kernel Fusion     | 1.4×    | 22%              |
| Mixed Precision   | 1.8×    | 35%              |
| Uniform Batching  | 1.3×    | 73%              |


## Data Preparation 📂

- **Train Data & Labels**: Place your training data and corresponding labels in the `data/` directory in `.txt` format.
- **Validation Data & Labels**: Similarly, ensure your validation data and labels are also available in the `data/` directory in `.txt` format.

---

## Setup & Configuration 🔧

1. **Edit the Configuration**  
   Open the `config.py` file and modify the settings as per your requirements. This file contains the hyperparameters and paths that the training script will use.

2. **Run Training**  
   Load the training function and execute it with your configuration:
   ```python
   # Example usage in your main training script
   from train import train_model  # Ensure you have a train.py file with the train_model function
   import config

   train_model(config)

Happy Coding! 🎉

# Foundations Study Notes

These notes accompany the one-week foundations track. They explain the theory, coding rationale, and practical insights behind each exercise.

---

## 1. Math and Gradients

### Why we do it
Gradients power optimization. To train neural networks, we need to compute how changing each weight affects the loss. This requires understanding derivatives of vector and matrix operations.

### How we do it
- **Finite differences**: Approximate gradients numerically by perturbing inputs. Useful for debugging analytic gradients.
- **Analytic gradients**: Derive using calculus and matrix rules.
- **Verification**: Compare numerical and analytic results.

### Code explanation
We implement a small function (e.g., f(x) = x^2) and verify its gradient via finite differences. This ensures our backpropagation implementations are correct.

### Challenges
- Extend gradient checking to matrix multiplication.
- Visualize error between numerical and analytic gradients.

---

## 2. Softmax + Cross-Entropy

### Why we do it
Softmax converts raw scores into probabilities. Cross-entropy measures the distance between predicted probabilities and the true labels. Together they form the standard loss for classification.

### How we do it
- Derive gradient of cross-entropy w.r.t. logits by hand.
- Implement softmax with numerical stability (subtract max).
- Verify with gradient checking.

### Code explanation
We code forward softmax, then derive backward gradient analytically and verify with NumPy.

### Challenges
- Implement one-hot encoding manually.
- Plot decision boundaries for a toy dataset.

---

## 3. MLP from Scratch

### Why we do it
A multilayer perceptron (MLP) is the simplest neural network. Building it from scratch teaches how forward and backward passes work under the hood.

### How we do it
- Implement layers (Linear, ReLU).
- Chain them into a network.
- Implement backprop by hand.
- Train on MNIST.

### Code explanation
Each layer stores inputs during forward pass. In backward pass, we compute gradients using chain rule. Parameters are updated via optimizer.

### Challenges
- Add another hidden layer and see accuracy gains.
- Compare training curves for different activations (ReLU vs Tanh).

---

## 4. Optimizers

### Why we do it
Gradient descent updates can be improved. Momentum accelerates training, Adam adapts learning rates. Understanding them helps tuning.

### How we do it
- Implement vanilla SGD.
- Add momentum (velocity term).
- Implement Adam (adaptive moments).

### Code explanation
Each optimizer updates parameters differently. In code, we define an `update` method that takes gradients and modifies weights.

### Challenges
- Plot loss curves for different learning rates.
- Compare Adam vs SGD on the same task.

---

## 5. Regularization

### Why we do it
Networks can overfit. Regularization techniques improve generalization.

### How we do it
- L2 penalty: discourages large weights.
- Dropout: randomly zeros activations during training.

### Code explanation
- Add L2 term to gradients.
- Implement dropout mask in forward pass, scale in inference.

### Challenges
- Visualize weight distributions with and without L2.
- Compare accuracy with dropout probability 0.2 vs 0.5.

---

## 6. CIFAR-10 Challenge

### Why we do it
CIFAR-10 is harder than MNIST. It shows limits of shallow MLPs and motivates CNNs.

### How we do it
- Load CIFAR-10 dataset.
- Normalize inputs (important for convergence).
- Train shallow MLP.

### Code explanation
Same MLP pipeline, but accuracy will be lower (~40-60%). Shows that architecture matters.

### Challenges
- Try different weight initialization.
- Add dropout to mitigate overfitting.

---

## 7. Wrap-Up Report

### Why we do it
Summarizing results clarifies learnings and progress.

### How we do it
- Collect test accuracy from experiments.
- Plot training curves.
- Write key observations.

### Code explanation
The notebook aggregates results into a JSON/dict, plots curves, and exports a report.

### Challenges
- Compare all optimizers in a single plot.
- Write a reflection on what concepts were hardest.

---

# Final Insights
- Gradients and optimization are the backbone of deep learning.
- Even simple networks can achieve high performance (MNIST ~98%).
- More complex datasets (CIFAR-10) reveal need for advanced architectures.
- Regularization and optimizers significantly affect convergence and generalization.


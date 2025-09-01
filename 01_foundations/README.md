# 🧠 Foundations — Study Guide (1 Week)

This repo contains the **Foundations module** of the LLM Research Playbook.  
Focus: **80% coding, 20% theory** — implement neural networks & optimizers from scratch to build strong fundamentals.

---

## 🎯 Learning Outcomes
- Understand and compute gradients for neural networks.
- Implement a minimal **MLP** and train it on MNIST/CIFAR-10.
- Write and compare optimizers (SGD, Momentum, Adam).
- Debug models using gradient checking & regularization techniques.

---

## 📅 1-Week Plan

| Day | Focus | Coding Deliverables |
|-----|-------|---------------------|
| 1   | Math refresh, gradient checking | `grad_check.py` |
| 2   | Softmax + cross-entropy gradients | Verified gradient script |
| 3   | Build 2-layer MLP | `mlp_from_scratch.py` |
| 4   | Implement optimizers (SGD, Adam) | `optim_sgd_adam.py` |
| 5   | Add regularization (L2, dropout) | Accuracy table |
| 6   | Train MLP on CIFAR-10 | ≥60% accuracy baseline |
| 7   | Wrap-up, report results | `foundations_report.md` |

---

## 📚 Core Resources

### Books / Papers
- [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning) — Shalev-Shwartz & Ben-David (2014)  
- [Deep Learning (Goodfellow et al., 2016)](https://www.deeplearningbook.org/) — Optimization & regularization chapters  

### Courses / Lectures
- [Andrew Ng’s Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)  
- [DeepLearning.AI Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)  
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) — Optimization & backprop lectures  

### Tutorials
- [NumPy neural network from scratch (3Blue1Brown inspired)](https://victorzhou.com/blog/intro-to-neural-networks/)  
- [Gradient Checking in NumPy](https://cs231n.github.io/neural-networks-3/#gradcheck)  

---

## 🏗️ Repo Structure

```
foundations/
├── README.md                # Study guide & resources
├── src/
│   ├── mlp_from_scratch.py  # MLP implementation
│   ├── optim_sgd_adam.py    # Optimizers
│   └── grad_check.py        # Gradient checker
├── exercises/
│   ├── softmax_derivation.md
│   ├── mnist_training.ipynb
│   └── cifar10_training.ipynb
└── foundations_report.md    # Final summary + results
```

---

## ✅ Success Metrics
- [ ] Softmax gradient derivation verified numerically.  
- [ ] MLP achieves **≥97% accuracy on MNIST**.  
- [ ] Shallow MLP achieves **≥60% accuracy on CIFAR-10**.  
- [ ] Short report with accuracy tables + plots.  

---

## 🚀 Next Steps
Once you complete this module:
1. Push your code & report to GitHub.  
2. Move to **Transformers from Scratch** module (next step in the playbook).  

---

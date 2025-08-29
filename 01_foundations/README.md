# Foundations


## Overview
Math (linear algebra, probability), optimization, and ML/DL basics to build the mental model for modern LLMs.

## Learning Outcomes
- Confident with vector/matrix calculus, gradients, and optimizers.
- Can implement and debug a minimal MLP/optimizer from scratch.

## Key Papers/Books
- *Understanding Machine Learning: From Theory to Algorithms* (Shalev-Shwartz & Ben-David, 2014)
- *Deep Learning* (Goodfellow et al., 2016) — Chapters on optimization & regularization

## Tutorials/Courses
- Andrew Ng ML, DeepLearning.AI
- Stanford CS231n lectures on optimization & backprop

## Code Starters
- `src/mlp_from_scratch.py`: linear → ReLU → linear + cross-entropy
- `src/optim_sgd_adam.py`: implement SGD, Momentum, Adam

## Exercises
- [ ] Derive gradient for softmax cross-entropy (by hand, then verify numerically)
- [ ] Implement MLP training on MNIST/CIFAR-10
- [ ] **Deliverables**: small report + accuracy table
- **Success Metrics**: Reproduce ~>97% on MNIST (MLP) or ≥60% on CIFAR-10 (shallow net)

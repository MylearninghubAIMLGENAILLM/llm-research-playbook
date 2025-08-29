# Siamese Network Architecture

A **Siamese Network** is a neural architecture composed of two or more **weight-sharing** subnetworks that learn to map inputs into an embedding space where **similar inputs are close** and **dissimilar inputs are far apart**. It is widely used for **verification** (same/not-same), **metric learning**, **few-shot learning**, **face/signature verification**, and **image retrieval**.

---

## 1) High-Level Idea

```
 x₁ ─▶ [ Encoder fθ ] ─▶ z₁ ─┐
                             │  distance d(z₁, z₂) ─▶ loss( d, y )
 x₂ ─▶ [ Encoder fθ ] ─▶ z₂ ─┘
         (shared weights)
```

- Two inputs **x₁, x₂** are passed through the **same encoder** \(f_θ\) producing embeddings **z₁, z₂**.
- A **distance/similarity head** computes \( d(z₁, z₂) \) (e.g., Euclidean, cosine).
- A **metric-learning loss** (e.g., *contrastive*, *triplet*) pushes positive pairs closer and negative pairs apart.

---

## 2) Core Components

1. **Shared Encoder \(f_θ\)**  
   - CNN (images), Transformer/bi-LSTM (text), MLP (tabular), etc.  
   - Weight sharing ensures **consistent embedding space**.

2. **Embedding Normalization (optional)**  
   - L2-normalization \( \tilde{z} = z / \|z\| \) stabilizes cosine-based training.

3. **Similarity/Distance Head**  
   - **Cosine similarity:** \( s = \frac{z_1 \cdot z_2}{\|z_1\|\|z_2\|} \)  
   - **Euclidean distance:** \( d = \| z_1 - z_2 \|_2 \)

4. **Loss Function** (choose one)  
   - **Contrastive Loss (Hadsell et al.)**  
     For binary label \(y \in \{0,1\}\), margin \(m>0\):  
     \[
     \mathcal{L} =
     y \cdot d^2 +
     (1-y) \cdot \max(0, m - d)^2
     \]
     Interpreting \(y=1\) as *similar* (pull together) and \(y=0\) as *dissimilar* (push apart).

   - **Triplet Loss (FaceNet)**  
     For anchor \(a\), positive \(p\), negative \(n\):  
     \[
     \mathcal{L} = \max\big(0, \|f(a)-f(p)\|_2^2 - \|f(a)-f(n)\|_2^2 + \alpha \big)
     \]
     Requires **triplet mining** (semi-hard/hard negatives).

   - **Binary Cross-Entropy on Similarity**  
     Compute a similarity score \(s\) and apply a sigmoid + BCE with target \(y\).

---

## 3) Data Construction

- **Positive pairs:** two samples from the same class/identity.  
- **Negative pairs:** two samples from different classes/identities.  
- **Balance:** aim for a balanced batch of positives/negatives.  
- **Mining:** for triplet loss, select informative negatives (semi-hard > random).

---

## 4) PyTorch Reference Implementation (Contrastive Loss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Encoder (example: small CNN for images) ---
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        h = self.backbone(x)               # [B, 128, 1, 1]
        h = h.view(x.size(0), -1)          # [B, 128]
        z = self.fc(h)                     # [B, out_dim]
        z = F.normalize(z, p=2, dim=1)     # L2 normalize (optional)
        return z

# --- Siamese wrapper ---
class Siamese(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.f = encoder

    def forward(self, x1, x2):
        z1 = self.f(x1)
        z2 = self.f(x2)
        return z1, z2

# --- Contrastive Loss ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        # Euclidean distance
        d = torch.norm(z1 - z2, p=2, dim=1)
        # y=1 for similar, y=0 for dissimilar
        loss_pos = y * (d ** 2)
        loss_neg = (1 - y) * torch.clamp(self.margin - d, min=0) ** 2
        return (loss_pos + loss_neg).mean()

# --- Example training step ---
def train_step(model, loss_fn, optimizer, batch):
    x1, x2, y = batch  # y in {0,1}, shape [B]
    model.train()
    optimizer.zero_grad()
    z1, z2 = model(x1, x2)
    loss = loss_fn(z1, z2, y.float())
    loss.backward()
    optimizer.step()
    return loss.item()
```

**Dataset for Paired Samples**

```python
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        # Pre-index items by label for sampling
        self.by_label = {}
        for idx, y in enumerate(labels):
            self.by_label.setdefault(int(y), []).append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import random
        x1 = self.data[idx]
        y1 = int(self.labels[idx])

        # 50% positive, 50% negative pair
        make_positive = random.random() < 0.5 and len(self.by_label[y1]) > 1

        if make_positive:
            j = idx
            while j == idx:
                j = random.choice(self.by_label[y1])
            y = 1
        else:
            # choose a different class
            neg_classes = [c for c in self.by_label.keys() if c != y1]
            c2 = random.choice(neg_classes)
            j = random.choice(self.by_label[c2])
            y = 0

        x2 = self.data[j]

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2, y
```

---

## 5) TensorFlow/Keras Variant (Cosine + BCE)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

def build_encoder(out_dim=128, input_shape=(28,28,1)):
    inputs = keras.Input(shape=input_shape)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.MaxPool2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(out_dim)(x)
    x = tf.linalg.l2_normalize(x, axis=-1)
    return keras.Model(inputs, x, name="encoder")

encoder = build_encoder()
x1 = keras.Input(shape=(28,28,1))
x2 = keras.Input(shape=(28,28,1))

z1 = encoder(x1)
z2 = encoder(x2)

cos_sim = tf.reduce_sum(z1 * z2, axis=-1, keepdims=True)
prob = L.Activation("sigmoid")(cos_sim)

model = keras.Model([x1, x2], prob)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

---

## 6) Training Tips & Tricks

- **Batching:** larger batches improve negative variety; for triplets, use *batch-all* or *batch-hard* mining.
- **Normalization:** L2-normalize embeddings when using cosine similarity.
- **Margins:** tune margin \(m \in [0.2, 1.0]\) for contrastive/triplet; watch for collapsed embeddings.
- **Augmentations:** crucial in vision (random crops, flips, color jitter); keep **paired** transforms consistent if needed.
- **Evaluation:** use **ROC-AUC**, **EER**, **F1** for verification; **Recall@K**, **mAP** for retrieval.
- **Indexing:** for large-scale retrieval, build ANN indices (FAISS, ScaNN).

---

## 7) Variants & Extensions

- **Triplet / N-pairs / InfoNCE** (NT-Xent) losses for stronger metric learning.
- **Siamese with Cross-Attention**: fuse features post-encoder for structured comparisons.
- **Siamese + Prototypical Networks** for few-shot classification.
- **Self-Supervised Pretraining** (SimCLR, MoCo) then fine-tune Siamese head.

---

## 8) Common Pitfalls

- **Uninformative pairs** → use hard/semi-hard mining; maintain class balance.
- **Overfitting identities** → regularize, augment, and validate on unseen identities.
- **Non-stationary labels** in verification scenarios → periodically refresh mined negatives.

---

## 9) Minimal Checklist

- [ ] Define encoder \(f_θ\) and embedding dim.  
- [ ] Choose distance (cosine vs L2).  
- [ ] Pick loss (contrastive/triplet/BCE).  
- [ ] Build balanced pair/triplet sampler.  
- [ ] Monitor ROC-AUC/EER on a validation split.  
- [ ] Export embeddings + build ANN index for retrieval use-cases.

---

## 10) References (starter set)

- *Dimensionality Reduction by Learning an Invariant Mapping* — Hadsell, Chopra, LeCun (2006).  
- *FaceNet: A Unified Embedding for Face Recognition and Clustering* — Schroff et al. (2015).  
- *Deep Metric Learning Using Triplet Network* — Hoffer & Ailon (2015).  
- *In Defense of the Triplet Loss for Person Re-Identification* — Hermans et al. (2017).  
- *SimCLR: A Simple Framework for Contrastive Learning of Visual Representations* — Chen et al. (2020).

---

**License:** CC-BY 4.0. Feel free to copy/modify.

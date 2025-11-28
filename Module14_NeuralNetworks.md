# Module 14: Neural Networks

## 1. General Knowledge

**Neural Networks (NN)** were first introduced in the 1980s and later revived after 2010 under the name **Deep Learning (DL)**.  
They are widely used in **image classification**, **text modelling**, and many other fields.

### Types of Neural Networks:
- Feed-Forward NN (FNN)  
- Convolutional NN (CNN)  
- Recurrent NN (RNN)

In this chapter, we focus on **FNN** and **CNN**.

---

## 2. Single Layer Neural Network

A **Feed-Forward Neural Network (FNN)** sends information in one direction  
**from input → hidden → output layers**.  
A *single layer NN* contains **only one hidden layer**.

### Layer Components:
- **Input Layer** — contains independent predictor variables.
- **Hidden Layer** — nodes are called **activations** (number chosen by the user).
- **Output Layer** — one or more output variables.

### Mathematical Formulation

Let:

\[
X = (X_1, X_2, …, X_p)
\]

Hidden activations:

\[
A_k = h_k(X) = g(w_{k0} + \sum_{j=1}^{n} w_{kj}X_j)
\]

where:

- `g(.)` = **activation function**
- `w` = **weights**
- `w_{k0}` = **bias**

Output function:

\[
F(X) = \beta_0 + \sum_{k=1}^{K} \beta_k A_k
\]

### Common Activation Functions:
- Sigmoid:  `g(z)=e^z / (1+e^z)`
- ReLU:  
```
g(z)=0 if z<0
g(z)=1 if z>=0
```

---

## 3. Multi-Layer Neural Networks

We use multiple layers instead of one large layer.  
For **categorical output (0–9)**:

- 10 output neurons  
- One‑hot encoding

### Model Formulas

\[
A_k^{(1)} = g(w^{(1)}_{k0} + \sum_{j=1}^{p} w^{(1)}_{kj}X_j)
\]

\[
A_l^{(2)} = g(w^{(2)}_{l0} + \sum_{k=1}^{K_2} w^{(2)}_{lk} A_k^{(1)})
\]

\[
Z_m = \beta_{m0} + \sum_{l=1}^{K_2} \beta_{ml} A_l^{(2)}
\]

Loss (**Cross Entropy**):

\[
- \sum_{i=1}^{n} \sum_{m=0}^{9} y_{im} \log(f_m(X_i))
\]

Softmax Output:

\[
f_m(X) = \frac{e^{Z_m}}{\sum_{l=0}^{9}e^{Z_l}}
\]

---

## 4. Neural Network Estimation

Techniques:

- Gradient Descent / Adam
- Regularization
- Stochastic Training

### Gradient Descent

\[
\theta^{(m+1)} = \theta^{(m)} - \rho \nabla R(\theta^{(m)})
\]

| Method | Update Frequency | Pros | Cons |
|---|---|---|---|
| Case Updating | Per‑sample | Fast, escapes minima | Noisy |
| Batch | Full dataset | Smooth & stable | Slow |
| Mini‑Batch | Small groups | Best trade‑off | Needs tuning |

---

## 5. Regularization

### Ridge Regularization

\[
\min \sum(y_i - f(X_i))^2 + \lambda \sum \theta_j^2
\]

### Dropout Learning

Randomly drop neurons — prevents overfitting.

**Why it works:**

- No single‑neuron dependency  
- Information distributed  
- More robust  
- Works like ensemble learning  

---

## 6. Tuning the NN

| Type | Includes |
|---|---|
| Model Hyperparameters | Layers, neurons, activation |
| Algorithm Hyperparameters | Learning rate, batch size, dropout, epochs |

Good tuning improves training time & accuracy.


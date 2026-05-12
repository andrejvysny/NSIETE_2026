# Backpropagation — Detailed Explanation & Implementation Guide

This document explains backpropagation in the context of our custom NumPy-based neural network framework (week 3 task). It builds on the forward pass from week 2 and introduces the backward pass, loss functions, and gradient checking.

---

## Table of Contents

1. [Big Picture: What Is Backpropagation?](#1-big-picture)
2. [Prerequisites: The Forward Pass Recap](#2-forward-pass-recap)
3. [The Chain Rule — Core of Backpropagation](#3-chain-rule)
4. [Loss Functions: Where Gradients Start](#4-loss-functions)
5. [Backward Pass Through Activations](#5-activations-backward)
6. [Backward Pass Through the Linear Layer](#6-linear-backward)
7. [Backward Pass Through the Model](#7-model-backward)
8. [Gradient Checking — Verifying Correctness](#8-gradient-checking)
9. [Full Walkthrough: End-to-End Example](#9-full-walkthrough)
10. [Common Pitfalls](#10-common-pitfalls)

---

## 1. Big Picture: What Is Backpropagation? <a name="1-big-picture"></a>

Backpropagation = "backward propagation of errors". It computes **how much each weight contributed to the error** by propagating the gradient of the loss backward through the network using the **chain rule of calculus**.

The training loop is:

```
1. Forward pass:   X → Layer1 → Layer2 → ... → Y_hat (prediction)
2. Compute loss:   L = Loss(Y_hat, Y_true)
3. Backward pass:  Compute dL/dW for every weight W in the network
4. Update weights: W = W - learning_rate * dL/dW
```

Week 3 focuses on steps 2-3. Step 4 (optimization) comes later.

### Why Separate Linear and Activation?

Our framework separates the linear transformation (`z = W @ a + b`) from the activation (`a = sigma(z)`). This makes backpropagation modular — each module only needs to know its own local derivative, and we chain them together.

---

## 2. Prerequisites: The Forward Pass Recap <a name="2-forward-pass-recap"></a>

### Convention: Shapes

- Input `X`: shape `(features, m)` where `m` = number of samples
- Weights `W`: shape `(out_features, in_features)`
- Bias `b`: shape `(out_features, 1)` — broadcasts across samples
- Output `z`: shape `(out_features, m)`

### Linear Forward

```python
def forward(self, input):
    self.fw_inputs = input          # Save for backward! shape (in_features, m)
    self.m = input.shape[1]         # Number of samples
    net = self.W @ input + self.b   # shape (out_features, m)
    return net
```

Key detail: **`self.fw_inputs` is saved** because we need it during backward pass.

### Activation Forward (Sigmoid example)

```python
def forward(self, input):
    self.fw_input = input           # Save for backward!
    return 1.0 / (1.0 + np.exp(-input))
```

Again: **input is saved** for computing the derivative later.

### Model Forward

The model iterates over all modules in order:

```python
def forward(self, input):
    for name, module in self.modules.items():
        input = module(input)  # output of one layer = input of next
    return input
```

---

## 3. The Chain Rule — Core of Backpropagation <a name="3-chain-rule"></a>

### Single Variable Chain Rule

If `y = f(g(x))`, then:

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

### Neural Network Chain Rule

Consider a simple 2-layer network:

```
z1 = W1 @ X + b1       (linear)
a1 = sigmoid(z1)        (activation)
z2 = W2 @ a1 + b2       (linear)
a2 = sigmoid(z2)        (activation)
L  = Loss(a2, Y)        (loss)
```

To compute `dL/dW1`, we chain:

```
dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
```

This is why it's called "back" propagation — we start from the loss and work backward through each layer.

### What Gets Passed Between Layers

Each layer receives a gradient **from above** (closer to the loss) and:

1. Computes gradients for its **own parameters** (dW, db) — these are stored for the optimizer
2. Computes the gradient **to pass further back** (to the layer below) — this is the return value

---

## 4. Loss Functions: Where Gradients Start <a name="4-loss-functions"></a>

The loss function is the **starting point** of backpropagation. Its backward method computes `dL/d(Y_hat)` — the gradient of the loss with respect to the network's output.

### Loss vs Cost

- **Loss** = error for a single sample: `L(y_hat, y)`
- **Cost** = average error over all samples: `J = (1/m) * sum(L_i)`

In our framework, `forward()` computes the **per-sample loss**, and we take `np.mean()` outside for the cost.

### Squared Error Loss (SE)

```
L = (y_hat - y)^2
```

**Forward** (per-sample):

```python
def forward(self, input, target):
    return (input - target) ** 2       # shape (1, m)
```

**Backward** (derivative of SE w.r.t. y_hat):

```
dL/d(y_hat) = 2 * (y_hat - y) / m
```

```python
def backward(self, input, target):
    m = input.shape[1]
    return 2 * (input - target) / m    # shape (1, m)
```

The `/m` is here because we want the gradient of the **cost** (mean), not just the sum. This is the convention used in gradient_check where `np.mean(loss)` is used.

### Binary Cross-Entropy Loss (BCE)

```
L = -[y * log(y_hat) + (1 - y) * log(1 - y_hat)]
```

**Forward**:

```python
def forward(self, input, target):
    return -(target * np.log(input) + (1 - target) * np.log(1 - input))
```

**Backward** (derivative of BCE w.r.t. y_hat):

```
dL/d(y_hat) = (-y/y_hat + (1-y)/(1-y_hat)) / m
```

```python
def backward(self, input, target):
    m = input.shape[1]
    return ((-target / input) + (1 - target) / (1 - input)) / m
```

**Note**: BCE requires `y_hat` in (0, 1), so the final layer should be Sigmoid.

---

## 5. Backward Pass Through Activations <a name="5-activations-backward"></a>

Each activation receives `da` — the gradient of the loss w.r.t. its **output**. It must return `dz` — the gradient w.r.t. its **input**. The rule is:

```
dz = da * activation_derivative(input)
```

This is element-wise multiplication (`*`), not matrix multiplication.

### Sigmoid Backward

Sigmoid: `sigma(z) = 1 / (1 + exp(-z))`

Derivative: `sigma'(z) = sigma(z) * (1 - sigma(z))`

```python
def backward(self, da):
    s = self.forward(self.fw_input)     # recompute sigmoid output
    dz = da * s * (1 - s)              # element-wise
    return dz
```

**Derivation:**

```
d/dz [1/(1+e^(-z))] = e^(-z) / (1+e^(-z))^2
                     = [1/(1+e^(-z))] * [1 - 1/(1+e^(-z))]
                     = sigma(z) * (1 - sigma(z))
```

### Tanh Backward

Tanh: `tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)`

Derivative: `tanh'(z) = 1 - tanh(z)^2`

```python
def backward(self, da):
    t = self.forward(self.fw_input)     # recompute tanh output
    dz = da * (1 - t ** 2)             # element-wise
    return dz
```

**Derivation:**

```
d/dz [tanh(z)] = 1 - tanh^2(z)

(can be derived from quotient rule on the exponential form)
```

### ReLU Backward

ReLU: `relu(z) = max(0, z)`

Derivative: `relu'(z) = 1 if z > 0, else 0`

```python
def backward(self, da):
    dz = da * (self.fw_input > 0).astype(float)  # element-wise
    return dz
```

**Intuition**: ReLU passes the gradient through unchanged where input was positive, and kills it where input was negative (or zero). This is the "dying ReLU" problem.

---

## 6. Backward Pass Through the Linear Layer <a name="6-linear-backward"></a>

This is the most critical piece. The linear layer computes `z = W @ a_prev + b`.

It receives `dz` (gradient w.r.t. its output) and must compute three things:

### 6.1 Gradient w.r.t. Weights: `dW`

```
dW = (1/m) * dz @ a_prev^T
```

**Derivation:**

- `z = W @ a_prev + b`, so `dL/dW = dL/dz * dz/dW = dz @ a_prev^T`
- The `1/m` averages over the batch

```python
self.dW = (1 / self.m) * dz @ self.fw_inputs.T    # shape (out, in)
```

### 6.2 Gradient w.r.t. Bias: `db`

```
db = (1/m) * sum(dz, axis=1, keepdims=True)
```

**Derivation:**

- `dz/db = 1`, so `dL/db = dz`, summed over samples
- `keepdims=True` preserves shape `(out, 1)`

```python
self.db = (1 / self.m) * np.sum(dz, axis=1, keepdims=True)  # shape (out, 1)
```

### 6.3 Gradient to Pass Back: `da_prev`

```
da_prev = W^T @ dz
```

**Derivation:**

- `dz/d(a_prev) = W`, so `dL/d(a_prev) = W^T @ dz`

```python
da_prev = self.W.T @ dz    # shape (in, m)
return da_prev
```

### Full Linear Backward

```python
def backward(self, dz):
    self.dW = (1 / self.m) * dz @ self.fw_inputs.T
    self.db = (1 / self.m) * np.sum(dz, axis=1, keepdims=True)
    return self.W.T @ dz
```

### Shape Verification

Given: `W` shape `(out, in)`, `fw_inputs` shape `(in, m)`, `dz` shape `(out, m)`:

| Expression                       | Shapes                 | Result                    |
| -------------------------------- | ---------------------- | ------------------------- |
| `dz @ fw_inputs.T`               | `(out, m) @ (m, in)`   | `(out, in)` = same as `W` |
| `sum(dz, axis=1, keepdims=True)` | sum over `m`           | `(out, 1)` = same as `b`  |
| `W.T @ dz`                       | `(in, out) @ (out, m)` | `(in, m)` = same as input |

---

## 7. Backward Pass Through the Model <a name="7-model-backward"></a>

The model's backward pass iterates over modules in **reverse order**:

```python
def backward(self, dz):
    for name, module in reversed(self.modules.items()):
        dz = module.backward(dz)
```

This is because in the forward pass:

```
X → Dense_1 → Tanh_1 → Dense_2 → Tanh_2 → ... → Sigmoid → Y_hat
```

The backward pass goes:

```
dL/dY_hat ← Sigmoid ← Dense_4 ← Tanh_3 ← Dense_3 ← Tanh_2 ← Dense_2 ← Tanh_1 ← Dense_1
```

Each module's `backward()` receives the gradient from the layer above and returns the gradient to pass to the layer below.

---

## 8. Gradient Checking — Verifying Correctness <a name="8-gradient-checking"></a>

Gradient checking numerically approximates the gradient and compares it to your analytical backward pass. It's slow but essential for debugging.

### The Method

For each weight `W[i][j]`:

```
J_plus  = Loss(forward(W[i][j] + epsilon))
J_minus = Loss(forward(W[i][j] - epsilon))
grad_approx = (J_plus - J_minus) / (2 * epsilon)
```

This is the **centered difference** formula, accurate to O(epsilon^2).

### Comparison

```python
numerator = ||grad_backward - grad_approx||_2
denominator = ||grad_backward||_2 + ||grad_approx||_2
difference = numerator / denominator
```

- `difference < 2e-7` → Correct implementation
- `difference > 2e-7` → Bug in backward pass

### What gradient_check Does (from `utils.py`)

1. Iterates over all layers that have `W` and `dW` attributes
2. For each weight, perturbs it by `+epsilon` and `-epsilon`
3. Runs full forward pass and computes mean loss for each perturbation
4. Compares numerical gradient with `layer.dW[i][j]` (from backward pass)

**Important**: You must run forward + backward BEFORE calling gradient_check, because it reads `layer.dW` values computed by your backward pass.

---

## 9. Full Walkthrough: End-to-End Example <a name="9-full-walkthrough"></a>

### Network Architecture

```python
mlp = Model()
mlp.add_module(Linear(2, 3), 'Dense_1')    # 2→3
mlp.add_module(Tanh(), 'Tanh_1')
mlp.add_module(Linear(3, 4), 'Dense_2')    # 3→4
mlp.add_module(Tanh(), 'Tanh_2')
mlp.add_module(Linear(4, 5), 'Dense_3')    # 4→5
mlp.add_module(Tanh(), 'Tanh_3')
mlp.add_module(Linear(5, 1), 'Dense_4_out') # 5→1
mlp.add_module(Sigmoid(), 'Sigmoid')
```

### Step-by-Step

```python
# 1. Dataset
X, Y = dataset_Circles(m=128, radius=0.7, noise=0.0)
# X shape: (2, 128), Y shape: (1, 128)

# 2. Forward pass
Y_hat = mlp.forward(X)
# Y_hat shape: (1, 128)

# 3. Compute loss
loss_fn = BCELoss()     # or SELoss()
L = loss_fn.forward(Y_hat, Y)
# L shape: (1, 128) — per-sample loss

# 4. Backward through loss — get dL/dY_hat
dY_hat = loss_fn.backward(Y_hat, Y)
# dY_hat shape: (1, 128)

# 5. Backward through model — computes all dW, db
mlp.backward(dY_hat)
# After this, every Linear layer has .dW and .db filled

# 6. Verify
gradient_check(mlp, loss_fn, X, Y)
```

### What Happens Inside `mlp.backward(dY_hat)`

The gradient flows backward through each module:

```
dY_hat  shape (1, 128)   — from loss.backward()
  ↓ Sigmoid.backward(dY_hat)
dz4     shape (1, 128)   — dY_hat * sigmoid'(z)
  ↓ Dense_4_out.backward(dz4)
        computes: dW4 = (1/m) * dz4 @ a3.T    shape (1, 5)
                  db4 = (1/m) * sum(dz4)        shape (1, 1)
da3     shape (5, 128)   — W4.T @ dz4
  ↓ Tanh_3.backward(da3)
dz3     shape (5, 128)   — da3 * (1 - tanh(z3)^2)
  ↓ Dense_3.backward(dz3)
        computes: dW3 = (1/m) * dz3 @ a2.T    shape (5, 4)
                  db3 = (1/m) * sum(dz3)        shape (5, 1)
da2     shape (4, 128)   — W3.T @ dz3
  ↓ Tanh_2.backward(da2)
dz2     shape (4, 128)   — da2 * (1 - tanh(z2)^2)
  ↓ Dense_2.backward(dz2)
        computes: dW2 = (1/m) * dz2 @ a1.T    shape (4, 3)
                  db2 = (1/m) * sum(dz2)        shape (4, 1)
da1     shape (3, 128)   — W2.T @ dz2
  ↓ Tanh_1.backward(da1)
dz1     shape (3, 128)   — da1 * (1 - tanh(z1)^2)
  ↓ Dense_1.backward(dz1)
        computes: dW1 = (1/m) * dz1 @ X.T     shape (3, 2)
                  db1 = (1/m) * sum(dz1)        shape (3, 1)
dX      shape (2, 128)   — W1.T @ dz1  (not used, but returned)
```

---

## 10. Common Pitfalls <a name="10-common-pitfalls"></a>

### 1. Forgetting to Save Forward Inputs

Every module must save its input during `forward()` (`self.fw_input` / `self.fw_inputs`) because the backward pass needs them. If you forget, backward will fail or use stale values.

### 2. Element-wise vs Matrix Multiply

- Activation backward: `da * derivative` — element-wise `*`
- Linear backward: `dz @ fw_inputs.T` — matrix multiply `@`

Mixing these up is the #1 source of shape errors.

### 3. The `/m` Factor

The `1/m` division in Linear backward and Loss backward averages over samples. If you put it in both places or neither place, gradient check will fail. In our framework:

- Loss backward divides by `m`
- Linear backward's `dW` and `db` divide by `m`

Wait — that's dividing by `m` twice? No. The loss backward's `/m` converts per-sample loss gradient to cost gradient. The linear layer's `/m` is for averaging the weight gradient over the batch. Both are needed because:

- `dL/dY_hat` = `(1/m) * d(sum_of_losses)/dY_hat` — the loss backward
- `dW = (1/m) * dz @ a.T` — the weight update averages across samples

Actually, be careful here. Look at what `gradient_check` does: it computes `np.mean(loss_fn(A, Y))`. This means the cost is `(1/m) * sum(L_i)`. So the gradient of cost w.r.t Y_hat should include the `1/m`. Then in Linear, the `dW = dz @ a.T` already sums over samples (matrix multiply sums over the `m` dimension), so we divide by `m` to get the average.

**Bottom line**: Check gradient_check to see what convention it uses, then make your `/m` placements consistent.

### 4. Reversed Order in Model Backward

`reversed(self.modules.items())` — if you forget `reversed`, gradients flow in the wrong direction and everything breaks silently.

### 5. BCE Numerical Stability

`log(0)` = `-inf`. If your sigmoid outputs exactly 0 or 1, BCE explodes. In practice, clip predictions:

```python
input = np.clip(input, 1e-12, 1 - 1e-12)
```

### 6. Recomputing vs Caching Activation Output

In Sigmoid/Tanh backward, you can either:

- Cache the output during forward: `self.fw_output = output` and reuse it
- Recompute by calling `self.forward(self.fw_input)` again

Both work. Caching is faster, recomputing is simpler. Just be consistent.

---

## Summary Table

| Module      | Forward                          | Backward (returns)            | Stores                  |
| ----------- | -------------------------------- | ----------------------------- | ----------------------- |
| **Linear**  | `z = W @ a + b`                  | `da_prev = W.T @ dz`          | `dW`, `db`, `fw_inputs` |
| **Sigmoid** | `a = 1/(1+e^(-z))`               | `dz = da * a*(1-a)`           | `fw_input`              |
| **Tanh**    | `a = tanh(z)`                    | `dz = da * (1-a^2)`           | `fw_input`              |
| **ReLU**    | `a = max(0,z)`                   | `dz = da * (z>0)`             | `fw_input`              |
| **SELoss**  | `L = (y_hat-y)^2`                | `dL = 2(y_hat-y)/m`           | —                       |
| **BCELoss** | `L = -[y*log(a)+(1-y)*log(1-a)]` | `dL = (-y/a + (1-y)/(1-a))/m` | —                       |
| **Model**   | iterate forward                  | iterate **reversed**          | —                       |

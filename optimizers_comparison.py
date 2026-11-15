"""
Assignment 8051: Momentum and RMSProp Optimizers
-------------------------------------------------
Compares plain Gradient Descent, SGD with Momentum and RMSProp on the same
neural network (2-layer MLP, implemented from scratch in NumPy) trained on the
scikit-learn Digits dataset (8x8 images, 10 classes).

Compared on: loss curves, convergence speed, final accuracy.
Run:  python optimizers_comparison.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
HIDDEN = 64


# ----------------------------- Data ---------------------------------
def load_data():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def one_hot(y, k=10):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


# ----------------------------- Model --------------------------------
class MLP:
    """Input -> Hidden (ReLU) -> Output (Softmax)."""

    def __init__(self, n_in, n_hidden, n_out, seed=SEED):
        rng = np.random.default_rng(seed)  # same seed => same initial weights
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)),
            "b2": np.zeros(n_out),
        }

    def forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ p["W2"] + p["b2"]
        z2 -= z2.max(axis=1, keepdims=True)
        e = np.exp(z2)
        probs = e / e.sum(axis=1, keepdims=True)
        return probs, (X, z1, a1)

    def loss(self, probs, Y):
        return -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))

    def backward(self, probs, cache, Y):
        X, z1, a1 = cache
        n = X.shape[0]
        dz2 = (probs - Y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)


# --------------------------- Optimizers -----------------------------
class GradientDescent:
    """Plain (vanilla) gradient descent: w = w - lr * g"""
    name = "Gradient Descent"

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Momentum:
    """SGD with momentum: v = beta*v + g ; w = w - lr * v"""
    name = "SGD + Momentum"

    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, {}

    def step(self, params, grads):
        for k in params:
            self.v[k] = self.beta * self.v.get(k, 0) + grads[k]
            params[k] -= self.lr * self.v[k]


class RMSProp:
    """RMSProp: s = rho*s + (1-rho)*g^2 ; w = w - lr * g / (sqrt(s)+eps)"""
    name = "RMSProp"

    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, {}

    def step(self, params, grads):
        for k in params:
            self.s[k] = self.rho * self.s.get(k, 0) + (1 - self.rho) * grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.s[k]) + self.eps)


# --------------------------- Training -------------------------------
def train(optimizer, X_tr, y_tr, X_te, y_te, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model = MLP(X_tr.shape[1], HIDDEN, 10)          # identical init for all
    Y_tr = one_hot(y_tr)
    rng = np.random.default_rng(SEED)               # identical batch order
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    start = time.time()

    for _ in range(epochs):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            probs, cache = model.forward(X_tr[b])
            grads = model.backward(probs, cache, Y_tr[b])
            optimizer.step(model.params, grads)

        probs, _ = model.forward(X_tr)
        hist["loss"].append(model.loss(probs, Y_tr))
        hist["train_acc"].append((model.predict(X_tr) == y_tr).mean())
        hist["test_acc"].append((model.predict(X_te) == y_te).mean())

    hist["time"] = time.time() - start
    return hist


def epochs_to_reach(loss_curve, target):
    for i, l in enumerate(loss_curve, start=1):
        if l <= target:
            return i
    return None


# ----------------------------- Main ---------------------------------
def main():
    X_tr, X_te, y_tr, y_te = load_data()

    optimizers = [
        GradientDescent(lr=0.01),
        Momentum(lr=0.01, beta=0.9),
        RMSProp(lr=0.001, rho=0.9),
    ]
    results = {o.name: train(o, X_tr, y_tr, X_te, y_te) for o in optimizers}

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, h in results.items():
        axes[0].plot(h["loss"], label=name)
        axes[1].plot(h["train_acc"], label=name)
        axes[2].plot(h["test_acc"], label=name)
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].set_yscale("log")
    axes[1].set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[2].set(title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)

    # ---- Summary table ----
    target = 0.1   # convergence threshold on training loss
    lines = [f"{'Optimizer':<18}{'Final loss':>12}{'Train acc':>11}{'Test acc':>10}"
             f"{'Epochs to loss<=' + str(target):>20}{'Time(s)':>9}"]
    lines.append("-" * len(lines[0]))
    for name, h in results.items():
        e = epochs_to_reach(h["loss"], target)
        lines.append(f"{name:<18}{h['loss'][-1]:>12.4f}{h['train_acc'][-1]:>11.4f}"
                     f"{h['test_acc'][-1]:>10.4f}{(str(e) if e else 'not reached'):>20}"
                     f"{h['time']:>9.2f}")
    table = "\n".join(lines)
    print(table)
    with open("results.txt", "w") as f:
        f.write(table + "\n")
    print("\nSaved: loss_curves.png, results.txt")


if __name__ == "__main__":
    main()

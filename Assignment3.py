# THIS IS THE CODE FROM KAGGLE, SOME DIRECTORIES MAY NOT APPEAR HERE#
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


train_file = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_train.pkl"
test_file  = "/kaggle/input/fii-nn-2025-homework-3/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data, train_labels = [], []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = [image.flatten() for image, _ in test]

X_raw = np.array(train_data, dtype=np.float32) / 255.0
y_idx = np.array(train_labels, dtype=np.int64)
X_test_raw = np.array(test_data, dtype=np.float32) / 255.0


# standardize + one-hot
mu = X_raw.mean(axis=0, keepdims=True)
sd = X_raw.std(axis=0, keepdims=True) + 1e-6
X = (X_raw - mu) / sd
X_test = (X_test_raw - mu) / sd

num_classes = 10
Y = np.zeros((y_idx.shape[0], num_classes), dtype=np.float32)
Y[np.arange(y_idx.shape[0]), y_idx] = 1.0

X_train, X_val, Y_train, Y_val, y_train, y_val = train_test_split(
    X, Y, y_idx, test_size=0.1, random_state=42, stratify=y_idx
)
print("Train:", X_train.shape, "| Val:", X_val.shape, "| Test:", X_test.shape)

# relu + softmax
def init_params(input_size=784, hidden_size=100, output_size=10, seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)  # He init
    b1 = np.zeros((1, hidden_size), dtype=np.float32)
    W2 = rng.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size), dtype=np.float32)
    return W1, b1, W2, b2

def relu(Z): return np.maximum(0.0, Z)
def relu_deriv(Z): return (Z > 0.0).astype(np.float32)

def softmax(Z):
    Zs = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Zs, dtype=np.float32)
    return expZ / (np.sum(expZ, axis=1, keepdims=True) + 1e-12)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def cross_entropy(y_true_onehot, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred + eps), axis=1))

def accuracy_from_probs(y_true_idx, probs):
    return (np.argmax(probs, axis=1) == y_true_idx).mean()

def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]
    dZ2 = (A2 - Y).astype(np.float32)
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def clip_grads_(grads, max_norm=5.0):
    for g in grads:
        gn = np.linalg.norm(g)
        if gn > max_norm:
            g *= (max_norm / (gn + 1e-12))

def cosine_lr(epoch, total_epochs, lr_max=1e-3, lr_min=1e-5):
    import math
    cos = (1 + math.cos(math.pi * epoch / total_epochs)) / 2.0
    return lr_min + (lr_max - lr_min) * cos


def train(
    X_train, Y_train, y_train,
    X_val, Y_val, y_val,
    epochs=90, batch_size=256,
    lr_max=1e-3, lr_min=1e-5, weight_decay=1e-4, seed=42
):
    np.random.seed(seed)
    W1, b1, W2, b2 = init_params(784, 100, 10, seed=seed)

    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mB1 = np.zeros_like(b1); vB1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mB2 = np.zeros_like(b2); vB2 = np.zeros_like(b2)

    t = 0
    n = X_train.shape[0]
    num_batches = (n + batch_size - 1) // batch_size
    best_val_acc = 0.0
    best_snapshot = None

    for epoch in range(1, epochs+1):
        lr = cosine_lr(epoch-1, epochs, lr_max=lr_max, lr_min=lr_min)

        idx = np.random.permutation(n)
        X_train = X_train[idx]; Y_train = Y_train[idx]; y_train = y_train[idx]

        # loop mini-batch
        for bi in range(num_batches):
            t += 1
            s, e = bi * batch_size, min((bi + 1) * batch_size, n)
            Xb = X_train[s:e]; Yb = Y_train[s:e]

            Z1, A1, Z2, A2 = forward(Xb, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(Xb, Yb, Z1, A1, A2, W2)

            # doar weights
            if weight_decay > 0.0:
                dW1 += weight_decay * W1
                dW2 += weight_decay * W2

            # stabilitate
            clip_grads_([dW1, db1, dW2, db2], max_norm=5.0)

            # am corectat bias-ul la adam
            for P, g, m, v in [(W1,dW1,mW1,vW1),(b1,db1,mB1,vB1),(W2,dW2,mW2,vW2),(b2,db2,mB2,vB2)]:
                m[:] = 0.9 * m + 0.1 * g
                v[:] = 0.999 * v + 0.001 * (g*g)
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                P -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        _, _, _, train_probs = forward(X_train, W1, b1, W2, b2)
        train_loss = cross_entropy(Y_train, train_probs)
        if weight_decay > 0.0:
            train_loss += 0.5 * weight_decay * (np.sum(W1*W1)+np.sum(W2*W2)) / X_train.shape[0]
        train_acc  = accuracy_from_probs(y_train, train_probs)

        _, _, _, val_probs = forward(X_val, W1, b1, W2, b2)
        val_loss = cross_entropy(Y_val, val_probs)
        if weight_decay > 0.0:
            val_loss += 0.5 * weight_decay * (np.sum(W1*W1)+np.sum(W2*W2)) / X_val.shape[0]
        val_acc  = accuracy_from_probs(y_val, val_probs)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc*100:.2f}% || "
              f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}% | lr={lr:g}")

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_snapshot = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    if best_snapshot is not None:
        W1, b1, W2, b2 = best_snapshot

    return (W1, b1, W2, b2)


W1, b1, W2, b2 = train(
    X_train, Y_train, y_train,
    X_val,   Y_val,   y_val,
    epochs=90,
    batch_size=256,
    lr_max=1e-3,
    lr_min=1e-5,
    weight_decay=5e-4,
    seed=42
)

_, _, _, val_probs = forward(X_val, W1, b1, W2, b2)
final_val_loss = cross_entropy(Y_val, val_probs)
final_val_acc  = accuracy_from_probs(y_val, val_probs)
print(f"\nFinal Val Loss: {final_val_loss:.4f} | Final Val Acc: {final_val_acc*100:.2f}%")

_, _, _, A_test = forward(X_test, W1, b1, W2, b2)
predictions = np.argmax(A_test, axis=1)

pd.DataFrame({
    "ID": np.arange(0, len(predictions)),
    "target": predictions
}).to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")

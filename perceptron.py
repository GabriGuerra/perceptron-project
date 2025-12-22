
---

# 3) perceptron.py

```python
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def ativacao(valor):
    """Step activation function: returns 1 if valor >= 0 else 0"""
    return 1 if valor >= 0 else 0

def treinar_perceptron(X, y, taxa=0.1, epocas=100, seed=42):
    """
    Train Perceptron.
    Returns weights, bias and number of epochs until convergence.
    """
    np.random.seed(seed)
    n_amostras, n_atributos = X.shape
    pesos = np.random.uniform(-1, 1, n_atributos)
    bias = np.random.uniform(-1, 1)

    for epoca in range(epocas):
        erro_total = 0
        for i in range(n_amostras):
            soma = np.dot(X[i], pesos) + bias
            saida = ativacao(soma)
            erro = y[i] - saida
            if erro != 0:
                pesos += taxa * erro * X[i]
                bias += taxa * erro
                erro_total += abs(erro)
        if erro_total == 0:
            break

    return pesos, bias, epoca+1

def prever(X, pesos, bias):
    """Predict binary classes for samples X."""
    return np.array([ativacao(np.dot(x, pesos) + bias) for x in X])

def avaliar_modelo(y_true, y_pred, nome_modelo):
    """Print accuracy and plot confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    print(f"--- {nome_modelo} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"True:     {y_true}")
    print(f"Predicted:{y_pred}")
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {nome_modelo}')
    plt.show()

def executar(X, y, nome, taxa=0.1, epocas=100):
    """Train, predict and evaluate, printing results."""
    pesos, bias, epoca = treinar_perceptron(X, y, taxa, epocas)
    y_pred = prever(X, pesos, bias)
    print(f"Weights: {pesos}")
    print(f"Bias: {bias:.4f}")
    print(f"Epochs until convergence: {epoca}")
    avaliar_modelo(y, y_pred, nome)

if __name__ == "__main__":
    X_logicas = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_or = np.array([0,1,1,1])
    y_and = np.array([0,0,0,1])

    executar(X_logicas, y_or, "Logical Gate OR")
    executar(X_logicas, y_and, "Logical Gate AND")

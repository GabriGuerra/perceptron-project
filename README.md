# Perceptron - Logical Gates and Iris Classification | Perceptron - Portas Lógicas e Classificação Iris

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/GabriGuerra/perceptron-project.svg)](https://github.com/your-username/perceptron-project/issues)

---

## Description / Descrição

**EN:**  
This project implements the Perceptron algorithm for binary classification problems, including logical gates (AND, OR, XOR) and the Iris dataset (distinguishing Setosa species from others). It demonstrates fundamental concepts of neural networks, supervised learning, and model evaluation.

**PT:**  
Este projeto implementa o algoritmo Perceptron para problemas de classificação binária, incluindo portas lógicas (AND, OR, XOR) e o conjunto de dados Iris (diferenciando a espécie Setosa das demais). Demonstra conceitos fundamentais de redes neurais, aprendizado supervisionado e avaliação de modelos.

---

## Technologies / Tecnologias

- Python 3  
- NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  

---

## Features / Funcionalidades

- Train Perceptron with automatic weight adjustment  
- Predict and evaluate using accuracy and confusion matrix  
- Graphical visualization of data and results  
- Modular and well-documented code for easy maintenance  

---

## How to run / Como executar

Clone the repository / Clone o repositório:

bash
git clone https://github.com/your-username/perceptron-project.git
cd perceptron-project
Install dependencies / Instale as dependências:

pip install -r requirements.txt


Run the Jupyter notebook for examples and visualization / Rode o notebook Jupyter para exemplos e visualização:

jupyter notebook perceptron_notebook.ipynb


Or run the Python script for quick tests / Ou execute o script Python para testes rápidos:

python perceptron.py

File structure / Estrutura dos arquivos

perceptron.py — Perceptron algorithm implementation

perceptron_notebook.ipynb — Notebook with examples and graphs

requirements.txt — Project dependencies

README.md — This documentation

Results / Resultados

Perceptron successfully classifies AND and OR logical gates, but not XOR (non-linearly separable problem)

On the Iris dataset, high accuracy in distinguishing Setosa vs other species, demonstrating good performance on linearly separable problems
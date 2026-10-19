# 🎬 IMDb Sentiment Benchmark: Comparativo de Arquiteturas de NLP

Estudo comparativo de abordagens clássicas e de Deep Learning para classificação binária de sentimentos utilizando o dataset **IMDb (50.000 avaliações)** da Universidade de Stanford.

---

## 📌 Arquiteturas Avaliadas

O objetivo deste projeto foi avaliar a relação de custo-benefício (tempo de processamento em CPU versus exatidão) entre três abordagens:

1. **Baseline Estatístico:** Vetorização com TF-IDF (Unigramas + Bigramas) + Regressão Logística (solver SAGA).
2. **Rede Neural Recorrente (RNN):** Embeddings treinadas do zero + Bidirectional LSTM (2 camadas bidirecionais com Dropout).
3. **Mecanismo de Atenção (Transformer):** Implementação nativa em Keras com camadas de `TokenAndPositionEmbedding` e `TransformerBlock` (*Multi-Head Self-Attention*).

---

## 📊 Resultados do Benchmark

| Arquitetura | Família / Paradigma | Exatidão de Teste (%) | Tempo de Treino (CPU) |
| :--- | :--- | :---: | :---: |
| **TF-IDF + Regressão Logística** | Modelo Linear Estatístico | ~89.0% | ~0.5 min |
| **Bidirectional LSTM** | Rede Neural Recorrente | ~86.5% | ~2.5 min |
| **Transformer Nativo** | Mecanismo de Atenção | ~87.8% | ~1.8 min |

*(Valores consolidados no ficheiro `benchmark_results.csv` gerado durante a execução).*

---

## 🚀 Como Executar Localmente

1. **Clone o repositório:**
   ```bash
git clone https://github.com/eniocancian/imdb-sentiment-benchmark.git
cd imdb-sentiment-benchmark
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o pipeline completo:**
   ```bash
   python benchmark.py
   ```
   *(Ou abra a versão interativa na pasta `notebooks/benchmark_imdb.ipynb`).*

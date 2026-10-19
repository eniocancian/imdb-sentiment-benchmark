#Benchmark de NLP no Dataset IMDb (50k Reviews)
#Comparativo entre:
# 1. Abordagem Clássica: TF-IDF + Regressão Logística
# 2. Rede Neural Recorrente: Bidirectional LSTM
# 3. Mecanismo de Atenção: Transformer Nativo (Multi-Head Self-Attention)

import os
import shutil
import tarfile
import time
import urllib.request
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Configurações de Ambiente e Reprodutibilidade
# ---------------------------------------------------------
# Força o uso da CPU para reprodutibilidade uniforme
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(42)
tf.random.set_seed(42)

# Hiperparâmetros Globais
MAX_FEATURES = 20000     # Tamanho do vocabulário para Deep Learning
SEQUENCE_LENGTH = 128    # Comprimento de corte do texto (tokens)
BATCH_SIZE = 32
EPOCHS = 2

# Diretórios
BASE_DIR = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "aclImdb_v1.tar.gz")
DATASET_DIR = os.path.join(BASE_DIR, "aclImdb")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Dicionário de resultados
resultados_benchmark = {}


# =========================================================
# 1. Download e Preparação do Dataset IMDb
# =========================================================
def preparar_dados():
    print("=" * 60)
    print("     ETAPA 1: VERIFICAÇÃO E PREPARAÇÃO DO DATASET IMDb")
    print("=" * 60)
    if not os.path.exists(TRAIN_DIR):
        if not os.path.exists(DATASET_PATH):
            print("Baixando dataset aclImdb_v1.tar.gz (~84MB)...")
            url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            urllib.request.urlretrieve(url, DATASET_PATH)
        
        print("Extraindo arquivos...")
        with tarfile.open(DATASET_PATH, "r:gz") as tar:
            tar.extractall(path=BASE_DIR)

    # Remove pasta não-supervisionada que atrapalha o carregador de classes
    unsup_dir = os.path.join(TRAIN_DIR, 'unsup')
    if os.path.exists(unsup_dir):
        try:
            shutil.rmtree(unsup_dir)
        except OSError:
            pass
    print("Dataset pronto para uso!\n")


def carregar_textos_puros(split_dir):
    """Carrega arquivos brutos de texto para o modelo clássico (scikit-learn)."""
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        sub_dir = os.path.join(split_dir, label_type)
        if not os.path.exists(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(sub_dir, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    return texts, labels


# =========================================================
# 2. Modelo 1: TF-IDF + Regressão Logística (Baseline)
# =========================================================
def benchmark_modelo_linear():
    print("=" * 60)
    print("     MODELO 1/3: TF-IDF + REGRESSÃO LOGÍSTICA (BASELINE)")
    print("=" * 60)
    start_time = time.time()

    print("Carregando textos para o Scikit-Learn...")
    X_train_raw, y_train = carregar_textos_puros(TRAIN_DIR)
    X_test_raw, y_test = carregar_textos_puros(TEST_DIR)

    print("Extraindo características com TF-IDF (Unigramas + Bigramas)...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    print("Treinando Regressão Logística (solver SAGA)...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
    lr_model.fit(X_train_vec, y_train)

    y_pred = lr_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred) * 100
    tempo_min = (time.time() - start_time) / 60

    resultados_benchmark["TF-IDF + Reg. Logística"] = {"acc": acc, "time": tempo_min}
    print(f"-> Concluído! Acurácia: {acc:.2f}% | Tempo Total: {tempo_min:.2f} min\n")


# =========================================================
# 3. Preparação dos Datasets para Keras / Deep Learning
# =========================================================
def preparar_datasets_keras():
    print("Configurando pipelines tf.data e TextVectorization...")
    train_ds = keras.utils.text_dataset_from_directory(
        TRAIN_DIR, batch_size=BATCH_SIZE, label_mode="binary",
        validation_split=0.2, subset="training", seed=42
    )
    val_ds = keras.utils.text_dataset_from_directory(
        TRAIN_DIR, batch_size=BATCH_SIZE, label_mode="binary",
        validation_split=0.2, subset="validation", seed=42
    )
    test_ds = keras.utils.text_dataset_from_directory(
        TEST_DIR, batch_size=BATCH_SIZE, label_mode="binary"
    )

    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH
    )
    # Adaptar no conjunto de treino
    vectorize_layer.adapt(train_ds.map(lambda x, y: x))

    def vetorizar(texto, label):
        return vectorize_layer(texto), label

    autotune = tf.data.AUTOTUNE
    v_train = train_ds.map(vetorizar, num_parallel_calls=autotune).prefetch(autotune)
    v_val = val_ds.map(vetorizar, num_parallel_calls=autotune).prefetch(autotune)
    v_test = test_ds.map(vetorizar, num_parallel_calls=autotune).prefetch(autotune)

    return v_train, v_val, v_test


# =========================================================
# 4. Modelo 2: Bidirectional LSTM
# =========================================================
def benchmark_bilstm(v_train, v_val, v_test):
    print("=" * 60)
    print("     MODELO 2/3: REDE NEURAL RECORRENTE (BiLSTM)")
    print("=" * 60)
    start_time = time.time()

    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int64),
        layers.Embedding(input_dim=MAX_FEATURES, output_dim=128),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(v_train, validation_data=v_val, epochs=EPOCHS, verbose=1)

    eval_res = model.evaluate(v_test, verbose=0)
    acc = eval_res[1] * 100
    tempo_min = (time.time() - start_time) / 60

    resultados_benchmark["BiLSTM (Recorrente)"] = {"acc": acc, "time": tempo_min}
    print(f"-> Concluído! Acurácia: {acc:.2f}% | Tempo Total: {tempo_min:.2f} min\n")


# =========================================================
# 5. Modelo 3: Transformer Nativo (Multi-Head Self-Attention)
# =========================================================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(x) + positions


def benchmark_transformer(v_train, v_val, v_test):
    print("=" * 60)
    print("     MODELO 3/3: TRANSFORMER NATIVO (SELF-ATTENTION)")
    print("=" * 60)
    start_time = time.time()

    embed_dim = 128
    num_heads = 2
    ff_dim = 128

    inputs = layers.Input(shape=(SEQUENCE_LENGTH,))
    x = TokenAndPositionEmbedding(SEQUENCE_LENGTH, MAX_FEATURES, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(v_train, validation_data=v_val, epochs=EPOCHS, verbose=1)

    eval_res = model.evaluate(v_test, verbose=0)
    acc = eval_res[1] * 100
    tempo_min = (time.time() - start_time) / 60

    resultados_benchmark["Transformer Nativo"] = {"acc": acc, "time": tempo_min}
    print(f"-> Concluído! Acurácia: {acc:.2f}% | Tempo Total: {tempo_min:.2f} min\n")


# =========================================================
# 6. Relatório Consolidado e Exportação
# =========================================================
def exibir_e_salvar_relatorio():
    print("=" * 60)
    print("          RELATÓRIO COMPARATIVO FINAL DE BENCHMARK")
    print("=" * 60)

    print("\n[ 1. Acurácia de Teste (%) ]")
    for modelo, metricas in resultados_benchmark.items():
        barra = "█" * int(metricas["acc"] // 3)
        print(f"{modelo:<25} | {barra} {metricas['acc']:.2f}%")

    print("\n[ 2. Tempo de Execução (Minutos na CPU) ]")
    for modelo, metricas in resultados_benchmark.items():
        t_barra = "█" * max(1, int(metricas["time"] * 2))
        print(f"{modelo:<25} | {t_barra} {metricas['time']:.2f} min")

    # Salva os resultados em CSV para documentação
    csv_file = os.path.join(BASE_DIR, "benchmark_results.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("Arquitetura,Acuracia_Teste_Porcentagem,Tempo_CPU_Minutos\n")
        for modelo, metricas in resultados_benchmark.items():
            f.write(f"{modelo},{metricas['acc']:.2f},{metricas['time']:.2f}\n")

    print("\n" + "=" * 60)
    print(f"Arquivo de métricas salvo em: {csv_file}")
    print("=" * 60)


# =========================================================
# Execução do Pipeline
# =========================================================
if __name__ == "__main__":
    preparar_dados()
    benchmark_modelo_linear()
    
    # Prepara datasets para Keras uma única vez
    v_train, v_val, v_test = preparar_datasets_keras()
    
    benchmark_bilstm(v_train, v_val, v_test)
    benchmark_transformer(v_train, v_val, v_test)
    
    exibir_e_salvar_relatorio()
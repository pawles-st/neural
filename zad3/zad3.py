import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

iris = load_iris()
X, Y = iris.data, iris.target

# Podział danych na treningowe i testowe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standaryzacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding etykiet
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

def create_model(architecture, activation, optimizer):
    model = Sequential()
    model.add(Input(shape = (X_train.shape[1],)))

    for n_neurons in architecture:
        model.add(Dense(n_neurons, activation=activation))
    
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Eksperymenty z różnymi konfiguracjami
activations = ["relu", "tanh", "sigmoid"]
optimizers = ["adam", "sgd"]
architectures = [
    [3, 3],
    [3, 3, 3, 3, 3, 3, 3],
    [8, 16, 16, 8],
    [128],
    [64, 64]
]

histories = []
for activation in activations:
    for optimizer in optimizers:
        for architecture in architectures:
            architecture_str = [str(arch) for arch in architecture]
            # optimizer_str = "adam" if optimizer == "adam" else optimizer.name
            architecture_str = f"[4->{'-'.join(architecture_str)}->3]"
            model_name = f"{architecture_str: <25} | {optimizer: <7} | {activation.title(): <10}"

            model = create_model(architecture=architecture, activation=activation, optimizer=optimizer)
            try:
                history = model.fit(X_train, Y_train, validation_split=0.2, epochs=50, verbose=0)
                print(model_name)
                histories.append((model_name, model, history))
            except Exception as e:
                print(f"Failed to train {model_name} model: {e}")

data = []

for i, (model_name, model, history) in enumerate(histories):
    data.append(go.Scatter(
        x=list(range(1, len(history.history['val_accuracy']) + 1)),
        y=history.history['val_accuracy'],
        mode='lines',
        name=model_name,
        line=dict(width=2)
    ))

fig = go.Figure(data=data)

# Dodanie tytułów i osi
fig.update_layout(
    title="Porównanie dokładności modeli",
    xaxis_title="Epoki",
    yaxis_title="Dokładność walidacji",
    legend_title="Model",
    template="plotly_dark",
    height=800
)
fig.show()

# Testowanie i wyświetlenie metryk
for i, (model_name, model, _) in enumerate(histories):
    Y_pred = np.argmax(model.predict(X_test), axis=-1)
    Y_true = np.argmax(Y_test, axis=-1)
    print(f"Config: {model_name}")
    # print(confusion_matrix(Y_true, Y_pred))
    print(classification_report(Y_true, Y_pred, zero_division=True))
    print("-" * 50)

final_accuracies = [
    (model_name, history.history['val_accuracy'][-1])
    for model_name, model, history in histories
]
final_accuracies.sort(key=lambda x: x[1], reverse=True)

bar_data = [
    go.Bar(
        x=[model_name for model_name, _ in final_accuracies],
        y=[accuracy for _, accuracy in final_accuracies],
        text=[f"{accuracy:.2f}" for _, accuracy in final_accuracies],
        textposition='auto',
        marker=dict(color='skyblue')
    )
]

bar_fig = go.Figure(data=bar_data)

# Dodanie tytułów i osi dla wykresu słupkowego
bar_fig.update_layout(
    title="Końcowa dokładność modeli",
    xaxis_title="Model",
    yaxis_title="Dokładność walidacji",
    template="plotly_white",
    height=600
)

# Wyświetlenie wykresu słupkowego
bar_fig.show()

bar_fig.write_html("bar_fig_all.html")
fig.write_html("fig_all.html")

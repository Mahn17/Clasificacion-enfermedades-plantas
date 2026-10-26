#Manuel Alejandro Heredia Nogales
"""Enfermedades de plantas — transferencia de aprendizaje con MobileNetV2

Dataset plant_leaves (https://www.tensorflow.org/datasets/catalog/plant_leaves)
  * 4502 fotografías de 6000x4000 px (6.8 GB)
  * 22 clases: 12 especies de plantas, hojas sanas o enfermas

Modelo: MobileNetV2 preentrenada con ImageNet
  Fase 1. Extracción de características: la base se congela y solo se
          entrena la nueva cabeza de clasificación.
  Fase 2. Ajuste fino: se descongelan las últimas capas de la base y se
          entrena con una tasa de aprendizaje muy pequeña.

Al terminar se guarda el modelo en formato .h5 para convertirlo a
TensorFlow.js (tensorflowjs_converter) y usarlo desde el celular (app/).
"""

# =============================================================================
# IMPORTACIONES

import os
import glob
import json
import time
import datetime
import zoneinfo

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
tf.config.threading.set_intra_op_parallelism_threads(N_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(2)

zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")
tiempo_inicial = time.time()


def hora():
    return datetime.datetime.now(zona_hermosillo).strftime("%H:%M:%S")


print(f"Hermosillo, {hora()}")
print(f"TensorFlow {tf.__version__}, {N_CPUS} CPUs")

# =============================================================================
# 0. CONFIGURACIÓN

DIR_BASE       = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS      = os.path.join(DIR_BASE, "datos", "plant_leaves")
DIR_RESULTADOS = os.path.join(DIR_BASE, "resultados")
DIR_MODELO     = os.path.join(DIR_BASE, "modelo")
DIR_APP        = os.path.join(DIR_BASE, "app")
for d in (DIR_RESULTADOS, DIR_MODELO, os.path.join(DIR_APP, "muestras")):
    os.makedirs(d, exist_ok=True)

TAMANO_IMG  = (224, 224)   # tamaño de entrada de MobileNetV2 con pesos de ImageNet
TAMANO_LOTE = 32
EPOCAS_FE   = 15           # fase 1: extracción de características
EPOCAS_FT   = 20           # fase 2: ajuste fino
LR_FE       = 1e-3
LR_FT       = 1e-5         # muy pequeña para no destruir lo aprendido en ImageNet
CAPA_FT     = 100          # se descongela MobileNetV2 desde esta capa (tiene 154)
SEMILLA     = 42

keras.utils.set_random_seed(SEMILLA)

# Mismo orden de etiquetas que tfds: el prefijo del archivo (0001 ... 0022) - 1
CLASES = [
    ("Mango (P0) healthy",              "Mango - sana"),
    ("Arjun (P1) healthy",              "Arjun - sana"),
    ("Alstonia Scholaris (P2) healthy", "Alstonia - sana"),
    ("Gauva (P3) healthy",              "Guayaba - sana"),
    ("Jamun (P5) healthy",              "Jamun - sana"),
    ("Jatropha (P6) healthy",           "Jatrofa - sana"),
    ("Pongamia Pinnata (P7) healthy",   "Pongamia - sana"),
    ("Basil (P8) healthy",              "Albahaca - sana"),
    ("Pomegranate (P9) healthy",        "Granada - sana"),
    ("Lemon (P10) healthy",             "Limón - sana"),
    ("Chinar (P11) healthy",            "Chinar - sana"),
    ("Mango (P0) diseased",             "Mango - enferma"),
    ("Arjun (P1) diseased",             "Arjun - enferma"),
    ("Alstonia Scholaris (P2) diseased","Alstonia - enferma"),
    ("Gauva (P3) diseased",             "Guayaba - enferma"),
    ("Bael (P4) diseased",              "Bael - enferma"),
    ("Jamun (P5) diseased",             "Jamun - enferma"),
    ("Jatropha (P6) diseased",          "Jatrofa - enferma"),
    ("Pongamia Pinnata (P7) diseased",  "Pongamia - enferma"),
    ("Pomegranate (P9) diseased",       "Granada - enferma"),
    ("Lemon (P10) diseased",            "Limón - enferma"),
    ("Chinar (P11) diseased",           "Chinar - enferma"),
]
NOMBRES  = [nombre for _, nombre in CLASES]
N_CLASES = len(CLASES)

# Colores (sana / entrenamiento = azul, enferma / validación = naranja)
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8985"

# =============================================================================
# 1. CARGA DEL DATASET (rutas de las imágenes y etiquetas)

rutas = np.array(sorted(glob.glob(os.path.join(DIR_DATOS, "*", "*", "*.JPG"))))
if len(rutas) == 0:
    raise SystemExit(f"No hay imágenes en {DIR_DATOS}. Ejecuta primero: python descargar_datos.py")

etiquetas = np.array([int(os.path.basename(r)[:4]) - 1 for r in rutas])

print(f"\nImágenes encontradas: {len(rutas)}  (el dataset completo tiene 4502)")
print(f"Clases: {N_CLASES}")

unique_elements, count_elements = np.unique(etiquetas, return_counts=True)
conteo = np.zeros(N_CLASES, dtype=int)
conteo[unique_elements] = count_elements

print("\nImágenes por clase:")
for nombre, n in zip(NOMBRES, conteo):
    print(f"  {nombre:20s} {n:4d}")
print(f"Desbalance: la clase más grande tiene {conteo.max() / max(conteo.min(), 1):.1f} veces "
      f"más imágenes que la más pequeña")

# =============================================================================
# 2. EXPLORACIÓN DEL DATASET

# Distribución de clases (agrupadas por especie)
orden = sorted(range(N_CLASES), key=lambda i: (NOMBRES[i].split(" - ")[0], NOMBRES[i]))
colores = [AZUL if "sana" in NOMBRES[i] else NARANJA for i in orden]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh([NOMBRES[i] for i in orden], conteo[orden], color=colores, height=0.75)
ax.invert_yaxis()
ax.set_xlabel("Número de imágenes")
ax.set_title("Distribución de clases — plant_leaves", loc="left")
ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AZUL), plt.Rectangle((0, 0), 1, 1, color=NARANJA)],
          labels=["sana", "enferma"], loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "distribucion_clases.png"), dpi=150)
plt.close()


def cargar_imagen(ruta, etiqueta):
    datos = tf.io.read_file(ruta)
    # ratio=8: libjpeg decodifica directamente a 1/8 del tamaño (6000x4000 -> 750x500).
    # Es mucho más rápido y usa mucha menos memoria que decodificar 24 megapíxeles.
    imagen = tf.io.decode_jpeg(datos, channels=3, ratio=8)
    imagen = tf.image.resize(imagen, TAMANO_IMG, antialias=True)
    return imagen, etiqueta


# Un ejemplo de cada clase
plt.figure(figsize=(18, 11))
for k, i in enumerate(orden):
    candidatas = np.where(etiquetas == i)[0]
    if len(candidatas) == 0:
        continue
    imagen, _ = cargar_imagen(rutas[candidatas[0]], i)
    plt.subplot(4, 6, k + 1)
    plt.imshow(imagen.numpy().astype("uint8"))
    plt.title(NOMBRES[i], fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "ejemplos_dataset.png"), dpi=100)
plt.close()

# =============================================================================
# 3. DIVISIÓN EN TRAIN / VALIDACIÓN / TEST (70 / 15 / 15, estratificada)

X_train, X_test, y_train, y_test = train_test_split(
    rutas, etiquetas, test_size=0.30, stratify=etiquetas, random_state=SEMILLA)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.50, stratify=y_test, random_state=SEMILLA)

print(f"\nTrain={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# =============================================================================
# 4. PIPELINE DE DATOS (tf.data)

AUTOTUNE = tf.data.AUTOTUNE

# El aumento de datos se aplica en el pipeline y no dentro del modelo, así el
# modelo exportado a TensorFlow.js solo contiene capas de inferencia.
aumento_datos = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2, value_range=(0, 255)),
], name="aumento_datos")


def crear_dataset(rutas, etiquetas, entrenamiento=False):
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    # Las imágenes ya reducidas a 224x224 se guardan en memoria (cache):
    # solo la primera época lee y decodifica los 6.8 GB de JPEG.
    ds = ds.map(cargar_imagen, num_parallel_calls=AUTOTUNE).cache()
    if entrenamiento:
        ds = ds.shuffle(len(rutas), seed=SEMILLA)
    ds = ds.batch(TAMANO_LOTE)
    if entrenamiento:
        ds = ds.map(lambda x, y: (aumento_datos(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


train_ds = crear_dataset(X_train, y_train, entrenamiento=True)
val_ds   = crear_dataset(X_val, y_val)
test_ds  = crear_dataset(X_test, y_test)

# Imágenes aumentadas de un mismo ejemplo
imagen, _ = cargar_imagen(X_train[0], y_train[0])
plt.figure(figsize=(10, 7))
for k in range(9):
    plt.subplot(3, 3, k + 1)
    aumentada = aumento_datos(tf.expand_dims(imagen, 0), training=True)
    plt.imshow(np.clip(aumentada[0].numpy(), 0, 255).astype("uint8"))
    plt.axis("off")
plt.suptitle(f"Aumento de datos — {NOMBRES[y_train[0]]}")
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "aumento_datos.png"), dpi=100)
plt.close()

# =============================================================================
# 5. CLASES PONDERADAS (para dataset desbalanceado)

pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
pesos_clases = dict(zip(np.unique(y_train).tolist(), pesos))
print("\nPesos de las clases:", {NOMBRES[k]: round(v, 2) for k, v in pesos_clases.items()})

# =============================================================================
# 6. MODELO: MobileNetV2 + NUEVA CABEZA DE CLASIFICACIÓN

base_model = keras.applications.MobileNetV2(
    input_shape=TAMANO_IMG + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False
print(f"\nMobileNetV2 tiene {len(base_model.layers)} capas")

entradas = keras.Input(shape=TAMANO_IMG + (3,), name="imagen")
# MobileNetV2 espera pixeles en [-1, 1]. Se usa una capa Rescaling en lugar de
# preprocess_input porque TensorFlow.js sí la reconoce.
x = layers.Rescaling(1. / 127.5, offset=-1, name="escala")(entradas)
# training=False mantiene las BatchNormalization en modo inferencia (también en el ajuste fino)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="promedio_global")(x)
x = layers.Dropout(0.2, name="dropout")(x)
salidas = layers.Dense(N_CLASES, activation="softmax", name="clases")(x)

modelo = keras.Model(entradas, salidas, name="plantas_mobilenetv2")
modelo.summary()

metricas = ["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]

# =============================================================================
# 7. FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=metricas,
)

callbacks_fe = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

print(f"\nFase 1 (extracción de características), Hermosillo {hora()}")
t = time.time()
historial_fe = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCAS_FE,
    class_weight=pesos_clases,
    callbacks=callbacks_fe,
    verbose=2,
)
print(f"Tiempo fase 1: {(time.time() - t)/60:.2f} mins")

# =============================================================================
# 8. FASE 2: AJUSTE FINO

base_model.trainable = True
for capa in base_model.layers[:CAPA_FT]:
    capa.trainable = False

# Hay que recompilar para que el cambio en trainable tenga efecto
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FT),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=metricas,
)
print(f"\nVariables entrenables en el ajuste fino: {len(modelo.trainable_variables)}")

ruta_pesos = os.path.join(DIR_MODELO, "mejores_pesos.h5")
callbacks_ft = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    # En Keras 2.15 restore_best_weights solo actúa si el entrenamiento se detiene
    # antes de tiempo, así que se guarda explícitamente la mejor época
    keras.callbacks.ModelCheckpoint(ruta_pesos, monitor="val_loss", save_best_only=True,
                                    save_weights_only=True, verbose=1),
]

epoca_inicio_ft = len(historial_fe.epoch)
print(f"\nFase 2 (ajuste fino), Hermosillo {hora()}")
t = time.time()
historial_ft = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoca_inicio_ft + EPOCAS_FT,
    initial_epoch=epoca_inicio_ft,
    class_weight=pesos_clases,
    callbacks=callbacks_ft,
    verbose=2,
)
print(f"Tiempo fase 2: {(time.time() - t)/60:.2f} mins")

modelo.load_weights(ruta_pesos)

# =============================================================================
# 9. GRÁFICAS DE ACCURACY Y LOSS

acc      = historial_fe.history["accuracy"]     + historial_ft.history["accuracy"]
val_acc  = historial_fe.history["val_accuracy"] + historial_ft.history["val_accuracy"]
loss     = historial_fe.history["loss"]         + historial_ft.history["loss"]
val_loss = historial_fe.history["val_loss"]     + historial_ft.history["val_loss"]
epocas   = np.arange(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.plot(epocas, acc,    color=AZUL,    linewidth=2, label="entrenamiento")
plt.plot(epocas, val_acc, color=NARANJA, linewidth=2, label="validación")
plt.axvline(epoca_inicio_ft + 0.5, color=GRIS, linestyle="--", linewidth=1.5, label="inicio ajuste fino")
plt.title("Accuracy")
plt.xlabel("época")
plt.ylabel(r"$Acc(\omega, b)$")
plt.ylim([0, 1.05])
plt.grid(True, color="#e1e0d9")
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.plot(epocas, loss,    color=AZUL,    linewidth=2, label="entrenamiento")
plt.plot(epocas, val_loss, color=NARANJA, linewidth=2, label="validación")
plt.axvline(epoca_inicio_ft + 0.5, color=GRIS, linestyle="--", linewidth=1.5, label="inicio ajuste fino")
plt.title("Loss (sparse categorical crossentropy)")
plt.xlabel("época")
plt.ylabel(r"$Loss(\omega, b)$")
plt.ylim(bottom=0)
plt.grid(True, color="#e1e0d9")
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "graficas_entrenamiento.png"), dpi=150)
plt.close()

# =============================================================================
# 10. EVALUACIÓN, REPORTE Y MATRIZ DE CONFUSIÓN

print("\nEvaluación final [loss, accuracy, top3]:")
evaluacion = {}
for nombre, ds in (("Train", crear_dataset(X_train, y_train)), ("Val", val_ds), ("Test", test_ds)):
    evaluacion[nombre] = modelo.evaluate(ds, verbose=0)
    print(f"  {nombre:5s}: {np.round(evaluacion[nombre], 4)}")

prediccion_test = modelo.predict(test_ds, verbose=0)
pred_test = np.argmax(prediccion_test, axis=1)

reporte = classification_report(y_test, pred_test, labels=np.arange(N_CLASES),
                                target_names=NOMBRES, digits=3, zero_division=0)
print("\n--- Reporte de clasificación (test) ---")
print(reporte)

with open(os.path.join(DIR_RESULTADOS, "reporte_clasificacion.txt"), "w", encoding="utf-8") as f:
    f.write("Evaluación [loss, accuracy, top3]\n")
    for nombre, valores in evaluacion.items():
        f.write(f"  {nombre:5s}: {np.round(valores, 4).tolist()}\n")
    f.write("\nReporte de clasificación (test)\n")
    f.write(reporte)

con = confusion_matrix(y_test, pred_test, labels=np.arange(N_CLASES))
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=con, display_labels=NOMBRES)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
# Con 22 clases los ceros saturan la figura; solo se muestran las celdas con datos
for texto in disp.text_.ravel():
    if texto.get_text() == "0":
        texto.set_text("")
ax.set_title(f"Matriz de confusión (test) — accuracy = {evaluacion['Test'][1]:.3f}")
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "matriz_confusion.png"), dpi=150)
plt.close()

# =============================================================================
# 11. GUARDAR EL MODELO Y LOS ARCHIVOS PARA LA APLICACIÓN MÓVIL

# Formato .h5 (Keras 2) para tensorflowjs_converter
ruta_h5 = os.path.join(DIR_MODELO, "modelo_plantas.h5")
modelo.save(ruta_h5, include_optimizer=False)
print(f"\nModelo guardado en {ruta_h5}")

with open(os.path.join(DIR_APP, "clases.json"), "w", encoding="utf-8") as f:
    json.dump([{"tfds": tfds_nombre, "nombre": nombre} for tfds_nombre, nombre in CLASES],
              f, ensure_ascii=False, indent=1)

# Una imagen de prueba por clase para probar la app
muestras = []
for i in range(N_CLASES):
    candidatas = X_test[y_test == i]
    if len(candidatas) == 0:
        continue
    imagen = tf.io.decode_jpeg(tf.io.read_file(candidatas[0]), channels=3, ratio=8)
    nombre_archivo = os.path.basename(candidatas[0]).replace(".JPG", ".jpg")
    tf.io.write_file(os.path.join(DIR_APP, "muestras", nombre_archivo),
                     tf.io.encode_jpeg(imagen, quality=85))
    muestras.append({"archivo": f"muestras/{nombre_archivo}", "clase": i})

with open(os.path.join(DIR_APP, "muestras.json"), "w", encoding="utf-8") as f:
    json.dump(muestras, f, indent=1)

print(f"\nTiempo total: {(time.time() - tiempo_inicial)/60:.2f} mins")
print(f"Hermosillo, {hora()}")

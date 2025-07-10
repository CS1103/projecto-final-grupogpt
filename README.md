[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: Desarrollo de un Clasificador Neuronal Multiclase desde C++
## **CS1103 Programación III** · Informe Final

VIDEO DE PRESENTACION: https://drive.google.com/file/d/1Jv4BrKS0BFECl6xLnw_O-fsP1_dKcS__/view?usp=sharing
### **Descripción**

Reconocimiento de Dígitos Manuscritos mediante Redes Neuronales Multicapa

### Contenidos

1. [Datos generales](#datos-generales)  
2. [Requisitos e instalación](#requisitos-e-instalación)  
3. [Investigación teórica](#1-investigación-teórica)  
4. [Diseño e implementación](#2-diseño-e-implementación)  
5. [Ejecución](#3-ejecución)  
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)  
7. [Conclusiones](#5-conclusiones)  
8. [Bibliografía](#6-bibliografía)  
---

### Datos generales

* **Tema**: Clasificación de dígitos manuscritos con redes neuronales en C++
* **Grupo**: `grupogpt`
* **Integrantes**:

  * Rodrigo Miguel Gomez Pacheco - 202410309
  * Javier Alejandro Castro Barreto - 202310081
  * Julio Ruiz Villavicencio - 202410616
---

### Requisitos e instalación

Para compilar y ejecutar correctamente el proyecto, es necesario contar con los siguientes requisitos:

#### 1. Compilador

* GCC 11 o superior (compatible con C++17)

#### 2. Dependencias

* [CMake ≥ 3.18](https://cmake.org/)
* [Eigen 3.4](https://eigen.tuxfamily.org/) (incluida localmente en el repositorio)
* STL estándar de C++

#### 3. Clonación del repositorio

bash
git clone https://github.com/CS1103/projecto-final-grupogpt.git
cd projecto-final-grupogpt

---

### 1. Investigación teórica

### 1.1.1. Fundamentos de Espacios Vectoriales y Transformaciones Lineales

Los espacios vectoriales son estructuras matemáticas que permiten representar datos y operaciones de forma abstracta, y resultan esenciales en disciplinas como la computación, física e ingeniería. Las transformaciones lineales, por su parte, modelan cómo cambian estos vectores bajo distintas operaciones, preservando propiedades como la aditividad y homogeneidad. Según Lay, Lay y McDonald en *Linear Algebra and Its Applications* (2015), la comprensión de estos fundamentos permite construir modelos matemáticos robustos para la manipulación de datos en múltiples dimensiones, siendo base para algoritmos modernos de reconocimiento de patrones y compresión de información.

### 1.1.2. Cálculo de Derivadas Parciales y Gradientes en Optimización

El cálculo multivariable, y en particular las derivadas parciales, permiten analizar cómo una función cambia respecto a cada una de sus variables independientes. Cuando estas derivadas se agrupan en un vector —el gradiente—, se obtiene una herramienta clave para minimizar funciones objetivo mediante métodos iterativos. Boyd y Vandenberghe, en *Convex Optimization* (2004), resaltan el rol del gradiente como dirección de descenso más pronunciada, lo cual es esencial para algoritmos como el descenso de gradiente, utilizados ampliamente en estadística, inteligencia artificial y simulaciones numéricas.

### 1.1.3. Computación Numérica en Ambientes de Bajo Nivel

En el desarrollo de algoritmos de aprendizaje automático desde cero, la manipulación eficiente de estructuras de datos como matrices es esencial. A diferencia de entornos de alto nivel como Python con NumPy o TensorFlow, en C++ la gestión de memoria, operaciones aritméticas y control de flujo deben ser implementadas de forma explícita. Esto obliga a un conocimiento más profundo del álgebra lineal computacional. Según Golub y Van Loan (2013), las operaciones básicas sobre matrices como productos punto, transposición y normalización son la base sobre la cual se construyen modelos de redes neuronales, y su correcta implementación determina el rendimiento numérico y la estabilidad del entrenamiento.

### 1.1.4. Representación de Datos y Tipos Genéricos en Modelos Neuronales

Una red neuronal requiere representar grandes volúmenes de datos numéricos de forma estructurada. El uso de plantillas (`templates`) en C++ permite diseñar estructuras genéricas como tensores o capas densas que funcionen indistintamente sobre tipos de precisión simple (`float`) o doble (`double`). Esta capacidad de abstracción es vital cuando se implementan redes neuronales que deben balancear precisión con rendimiento, como afirman Meyers (2014) y Alexandrescu (2001). En este contexto, cada componente del modelo como los pesos de una capa, los gradientes o las salidas intermedias puede tratarse como una matriz multidimensional, operando bajo un marco matemático riguroso pero optimizado para eficiencia computacional.

### 1.1.5. Dataset MNIST: Referente Clásico para la Clasificación de Dígitos

El dataset MNIST (Modified National Institute of Standards and Technology) es una colección ampliamente utilizada en el ámbito del aprendizaje automático para tareas de clasificación de imágenes. Introducido por LeCun et al. (1998), MNIST contiene un conjunto de 70,000 imágenes en escala de grises de dígitos manuscritos del 0 al 9, donde cada imagen posee una resolución de 28x28 píxeles. De estas, 60,000 se destinan al entrenamiento y 10,000 a la evaluación del modelo.

Cada imagen está acompañada por una etiqueta que indica el dígito correspondiente, lo cual lo convierte en un problema de clasificación multiclase supervisada con 10 clases posibles. Su popularidad se debe a la simplicidad del formato, la limpieza de los datos y su idoneidad para validar arquitecturas básicas de redes neuronales como perceptrones multicapa (MLP) y redes convolucionales (CNN).

### 6. Implementación del Clasificador Neuronal desde Cero

El presente proyecto implementa un sistema de reconocimiento de dígitos manuscritos mediante una red neuronal artificial construida íntegramente en C++. La arquitectura general corresponde a un perceptrón multicapa (MLP), donde se emplean capas densas (`DenseLayer`) y funciones de activación no lineales (`ReLU` y `Softmax`) para el procesamiento y clasificación de entradas.

La implementación se ha dividido modularmente en distintos archivos fuente que encapsulan funcionalidades específicas. La clase `NeuralNetwork` gestiona la secuencia de capas, el proceso de entrenamiento, y la retropropagación del error, apoyándose en interfaces (`ILayer`, `IActivation`, `ILoss`) que promueven un diseño extensible. Asimismo, la clase `Tensor` permite representar vectores y matrices de forma eficiente, facilitando las operaciones algebraicas fundamentales para el forward y backward pass.

Los datos de entrada provienen del dataset MNIST en formato `.csv`, los cuales son procesados mediante la clase `MNISTLoader`, responsable de cargar, normalizar y preparar los tensores de entrada y las etiquetas one-hot para el entrenamiento. El entrenamiento se realiza con descenso de gradiente estocástico (`SGD`), implementado en el optimizador `SGDOptimizer`, que actualiza los pesos tras cada época.

Durante la ejecución, el modelo se entrena con `mnist_train.csv` y se valida con `mnist_test.csv`, mostrando métricas como la precisión (accuracy) por época y la pérdida (loss) acumulada. La estructura modular, combinada con pruebas automatizadas, permite escalar o adaptar fácilmente el modelo a datasets similares o arquitecturas más complejas.

Este proyecto demuestra la viabilidad de desarrollar desde cero una red neuronal funcional en C++, sin el uso de frameworks externos como TensorFlow o PyTorch, contribuyendo así al entendimiento profundo de los mecanismos internos del aprendizaje supervisado.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

El proyecto se ha estructurado siguiendo una arquitectura modular, lo cual facilita la escalabilidad, el mantenimiento del código y la incorporación de futuras mejoras. Se ha aplicado el principio de separación de responsabilidades, distribuyendo las funcionalidades en archivos fuente bien delimitados, cada uno con una responsabilidad específica.

* **Principales módulos del sistema**:

 ### 📄 `neural_network.h`

**Descripción:**  
Este archivo define la clase `NeuralNetwork`, que representa la estructura principal de la red neuronal multicapa (MLP).

**Responsabilidad principal:**  
Gestionar la **secuencia de capas** de la red y coordinar el flujo de datos durante las fases de **propagación hacia adelante (forward)** y **retropropagación del error (backward)**.

**Características clave:**

- Utiliza un vector de punteros inteligentes (`std::unique_ptr`) a la interfaz `ILayer<T>`, lo que permite almacenar distintos tipos de capas (densas, activación, etc.) de forma polimórfica.
- El método `add_layer()` permite construir la red añadiendo capas dinámicamente.
- El método `forward()` propaga una entrada a través de todas las capas y devuelve la salida.
- El método `backward()` recorre las capas en orden inverso, propagando el gradiente hacia atrás.
- `predict()` es simplemente un alias de `forward()`, pensado para la fase de inferencia.
- La red es **modular** y **extensible**, ya que depende solo de la interfaz `ILayer`.

**Relación con otros archivos:**

- Incluye cabeceras como `nn_dense.h`, `nn_activation.h`, `nn_loss.h` y `nn_optimizer.h`, lo que indica que cada capa específica (como `DenseLayer`, `ReLU`, etc.) implementa la interfaz común `ILayer<T>`.
- Utiliza la clase `Tensor2D<T>` como tipo de entrada/salida para representar los datos (probablemente un alias de `Tensor<T, 2>`).

### 📄 `nn_dense.h`

**Descripción:**  
Este archivo implementa la clase `Dense`, que representa una **capa densa (fully connected)** en la red neuronal. Es una de las capas principales del modelo MLP.

**Responsabilidad principal:**  
Realizar el producto matricial entre la entrada y los pesos (`_weights`), sumar los sesgos (`_biases`) y propagar el resultado hacia la siguiente capa. También se encarga de calcular los gradientes y actualizar los parámetros durante el entrenamiento.

**Características clave:**

- Inicializa los pesos con una distribución normal escalada por la dimensión de entrada (He initialization).
- Implementa los métodos:
  - `forward()`: Propagación hacia adelante.
  - `backward()`: Retropropagación del error, calculando gradientes de pesos y sesgos.
  - `update_params()`: Actualiza los parámetros usando un optimizador (como SGD).
- Soporta carga y guardado de pesos (`save_weights` y `load_weights`) para persistencia del modelo.
- Guarda el último input (`_last_input`) para usarlo en la retropropagación.

**Relación con otros archivos:**

- Depende de `nn_interfaces.h`, ya que hereda de la interfaz `ILayer<T>`.
- Utiliza `Tensor2D<T>` para representar entradas, pesos y gradientes.
- Colabora con `IOptimizer` (ej. `SGDOptimizer`) para actualizar sus parámetros.

### 📄 `nn_activation.h`

**Descripción:**  
Este archivo define la clase `ReLU`, una función de activación usada comúnmente en redes neuronales profundas. Implementa la interfaz `ILayer<T>`, lo que le permite integrarse como una capa más dentro de la red.

**Responsabilidad principal:**  
Aplicar la función de activación **ReLU (Rectified Linear Unit)** durante la propagación hacia adelante, y su derivada durante la retropropagación.

**Características clave:**

- En el método `forward()`, reemplaza los valores negativos por cero, conservando los positivos.
- Utiliza una **máscara (`_mask`)** para almacenar qué entradas fueron mayores que cero, lo cual se usa en la fase `backward()` para derivar correctamente.
- En `backward()`, propaga el gradiente solo donde la entrada original fue positiva (según la máscara).

**Relación con otros archivos:**

- Hereda de la interfaz `ILayer<T>`, definida en `nn_interfaces.h`, lo que permite su uso dentro de la red definida en `neural_network.h`.
- Utiliza `Tensor2D<T>` como estructura para manejar las matrices de activación y gradientes.

### 📄 `nn_loss.h`

**Descripción:**  
Este archivo implementa la clase `SoftmaxCrossEntropyLoss`, que combina la función de activación **Softmax** con la función de pérdida **Entropía Cruzada (Cross-Entropy)**. Es ideal para tareas de **clasificación multiclase**, como el reconocimiento de dígitos en MNIST.

**Responsabilidad principal:**  
- Calcular la **pérdida** entre las predicciones (`logits`) y las etiquetas reales codificadas en **one-hot**.
- Calcular el **gradiente** necesario para retropropagación en el entrenamiento.

**Características clave:**

- Realiza el cálculo de **Softmax** de forma numéricamente estable (usando el truco de restar el valor máximo por fila).
- Evita problemas de precisión al calcular logaritmos mediante una constante `epsilon` (`std::numeric_limits<T>::epsilon()`).
- Guarda internamente:
  - `_softmax_outputs`: resultados de la activación softmax.
  - `_last_targets`: etiquetas reales del batch.
- El método `forward()` devuelve la pérdida promedio por batch.
- El método `backward()` devuelve el gradiente de la pérdida respecto a las salidas (`softmax_outputs - targets`), ya que esta combinación permite una derivada simplificada y eficiente.

**Relación con otros archivos:**

- Utiliza la clase `Tensor2D<T>` como estructura principal para representar matrices de entrada, salida y gradiente.
- Se integra con la red definida en `neural_network.h` y se usa típicamente después de la última capa (por ejemplo, después de `Dense` y `Softmax` implícito).

### 📄 `nn_optimizer.h`

**Descripción:**  
Este archivo define dos algoritmos de optimización: **SGD (Stochastic Gradient Descent)** y **Adam**, ambos implementando la interfaz `IOptimizer<T>`. Estos optimizadores actualizan los pesos y sesgos de las capas entrenables usando los gradientes calculados durante la retropropagación.

**Responsabilidad principal:**  
Aplicar reglas de actualización a los parámetros del modelo (pesos y sesgos) con base en los gradientes y una tasa de aprendizaje.

---

#### 🔹 `SGD` (Stochastic Gradient Descent)

- Algoritmo de optimización más simple.
- La clase `SGD` recibe una tasa de aprendizaje (`_lr`) y actualiza los parámetros según la fórmula:  
  \[
  \text{param} = \text{param} - \text{lr} \times \text{grad}
  \]
- Método: `update(Tensor<T,2>& params, const Tensor<T,2>& grads)`

---

#### 🔹 `Adam` (Adaptive Moment Estimation)

- Optimizador avanzado que adapta la tasa de aprendizaje para cada parámetro.
- Utiliza momentos de primer orden (**m**, promedio de los gradientes) y segundo orden (**v**, promedio de los gradientes al cuadrado).
- Aplica corrección de sesgo (`m_hat`, `v_hat`) en cada iteración `t`.
- Internamente maneja vectores `m`, `v` y contador `t` usando `thread_local`, lo que permite mantener el estado del optimizador por hilo.

**Fórmula de actualización usada:**
\[
\theta = \theta - \eta \cdot \frac{m_t}{\sqrt{v_t} + \varepsilon}
\]

**Relación con otros archivos:**

- Se usa desde las capas entrenables como `Dense` mediante el método `update_params()` del `ILayer<T>`.
- Compatible con cualquier estructura de parámetros basada en `Tensor2D<T>`.

### 📄 `mnist_loader.h`

**Descripción:**  
Este archivo implementa una utilidad para cargar datos del conjunto **MNIST** desde archivos `.csv`, formateándolos como tensores numéricos compatibles con la red neuronal.

**Responsabilidad principal:**  
Leer y procesar los datos de imágenes y etiquetas desde el archivo CSV, normalizarlos y representarlos como tensores `Tensor2D<double>` que se usarán como entradas (`images`) y salidas (`labels`) en el entrenamiento del modelo.

**Características clave:**

- La función `load_mnist_csv()`:
  - Recibe la ruta del archivo `.csv` y el número de imágenes a cargar.
  - Convierte las imágenes en escala de grises de 28x28 (784 píxeles) a un tensor normalizado entre 0 y 1.
  - Convierte las etiquetas (dígitos del 0 al 9) a codificación **one-hot** de 10 dimensiones.
- Usa la función auxiliar `split()` para dividir cada línea del CSV.
- Devuelve un `std::pair<Tensor2D_d, Tensor2D_d>` que representa:  
  `⟶ (imágenes_normalizadas, etiquetas_one_hot)`

**Relación con otros archivos:**

- Utiliza la clase `Tensor2D` (definida en `tensor.h`) para almacenar los datos cargados.
- Es utilizada al inicio del entrenamiento para preparar los datos provenientes de `mnist_train.csv` y `mnist_test.csv`.

**Ejemplo de uso en entrenamiento:**

```cpp
auto [train_images, train_labels] = utec::data::load_mnist_csv("mnist_train.csv", 60000);
auto [test_images, test_labels] = utec::data::load_mnist_csv("mnist_test.csv", 10000);
```

### 📄 `tensor.h`

**Descripción:**  
Este archivo implementa una estructura de datos genérica llamada `Tensor`, utilizada para representar arreglos multidimensionales (vectores, matrices, etc.) de cualquier tipo numérico. Constituye la base del cálculo algebraico dentro del modelo neuronal.

**Responsabilidad principal:**  
Proporcionar un contenedor flexible y eficiente para almacenar datos y realizar operaciones matemáticas esenciales como suma, resta, multiplicación escalar, broadcasting, transposición y multiplicación de matrices. Es indispensable para la propagación hacia adelante y hacia atrás en la red neuronal.

**Características clave:**

- Template general `Tensor<T, Rank>` para manejar tensores de cualquier tipo y dimensión (`Rank`).
- Métodos sobrecargados para:
  - Acceso multidimensional con `operator()`.
  - Acceso lineal con `operator[]`.
  - Operaciones aritméticas `+`, `-`, `*`, `/`.
  - Transposición (`transpose_2d()`) y multiplicación matricial (`matmul()`).
- Internamente usa `std::vector<T>` como almacenamiento lineal y `std::array<size_t, Rank>` para la forma (`shape`) y los `strides`.
- Función de impresión `operator<<` para visualización directa de tensores por consola.

**Utilidad en el proyecto:**

- Es el tipo base sobre el cual operan las capas (`Dense`, `ReLU`, etc.) y el optimizador (`SGDOptimizer`).
- Permite calcular gradientes, productos matriciales y mantener consistencia dimensional durante el entrenamiento.
- Aporta abstracción matemática sin depender de librerías externas como Eigen o Armadillo.

**Ejemplo de uso:**

```cpp
Tensor<double, 2> A(3, 4);     // Tensor de 2D con 3 filas y 4 columnas
A.fill(1.0);                   // Llenar con unos
auto B = A.transpose_2d();     // Transponer A
```

### 📄 `common_helpers.h`

**Descripción:**  
Este archivo contiene funciones auxiliares utilizadas para evaluar el rendimiento del modelo y extraer predicciones. Proporciona herramientas prácticas para el flujo de pruebas y validación, especialmente después del entrenamiento.

**Responsabilidad principal:**  
- Determinar la clase predicha a partir del vector de salida de la red.
- Evaluar el modelo completo sobre el conjunto de prueba, mostrando el **accuracy** total.

**Funciones principales:**

- `get_predicted_class(prediction)`:  
  Recibe un tensor de salida (por ejemplo, de tamaño `1x10`) y devuelve el índice con mayor probabilidad, usando un **argmax**.
  
- `evaluate(model, test_images, test_labels)`:  
  - Recorre todas las imágenes de prueba y predice la clase utilizando `model.predict()`.
  - Compara con la etiqueta real (también extraída con argmax).
  - Muestra la **exactitud (accuracy)** como porcentaje en consola.

**Relación con otros archivos:**

- Usa la clase `NeuralNetwork<double>` definida en `neural_network.h`.
- Utiliza la clase `Tensor2D<double>` definida en `tensor.h`.
- Ideal para su uso en el `main()` al final del entrenamiento para validar el desempeño del modelo.

**Ejemplo de uso:**

```cpp
evaluate(model, test_images, test_labels);
// Output: Accuracy: 91.75%
```

### 📄 `image_processor.h`

**Descripción:**  
Este módulo proporciona funcionalidades para procesar imágenes externas (como PNG) y convertirlas en vectores de entrada adecuados para la red neuronal. Está diseñado especialmente para experimentar con imágenes reales de dígitos manuscritos, fuera del dataset MNIST.

**Responsabilidad principal:**  
Leer una imagen desde disco, convertirla a escala de grises, binarizarla, encontrar su contorno, redimensionarla a 28x28 píxeles, centrarla y normalizar sus valores para que pueda ser utilizada como entrada para la red neuronal.

**Funciones principales:**

- `preprocess_image_stb(filepath)`  
  - Carga la imagen usando `stb_image`.
  - Convierte a escala de grises.
  - Binariza con umbral (`thr = 60`).
  - Aplica una dilatación para agrandar el trazo del dígito.
  - Calcula una bounding box del dígito.
  - Recorta la región relevante, la redimensiona a 20×20 y la centra en una imagen de 28×28.
  - Normaliza los valores entre 0 y 1 y devuelve un `Tensor2D<double>` listo para predecir.

- `print_ascii_28x28(tensor)`  
  - Imprime una representación ASCII del tensor 28×28.
  - Útil para verificar visualmente si el dígito fue correctamente procesado.

**Relación con otros módulos:**

- Usa `Tensor2D<double>` definido en `tensor.h`.
- Usa las bibliotecas externas `stb_image.h` y `stb_image_resize.h`.

**Ejemplo de uso:**

```cpp
auto input_tensor = utec::utils::preprocess_image_stb("mi_digito.png");
print_ascii_28x28(input_tensor);
auto prediction = model.predict(input_tensor);
```

**Dependencias externas:**
- `stb_image.h` y `stb_image_resize.h` (de `https://github.com/nothings/stb`) para la lectura y redimensionamiento de imágenes PNG.

- `stb_image.h` y `stb_image_resize.h`: Librerías externas utilizadas para el procesamiento de imágenes en formato PNG.

**Diseño orientado a interfaces**:

El sistema incorpora interfaces genéricas como `ILayer`, `IActivation`, `ILoss` y `IOptimizer`, que permiten desacoplar las implementaciones concretas y seguir principios de diseño como el Open/Closed (abierto a extensión, cerrado a modificación). Este enfoque posibilita extender el sistema con nuevas capas, funciones o métodos de entrenamiento sin alterar la estructura central.

## 📁 Estructura del Proyecto

```plaintext
projecto-final-grupogpt/
├── .gitignore                    # Archivos/directorios ignorados por Git
├── CMakeLists.txt                # Configuración de compilación con CMake
├── README.md                     # Documentación del proyecto
├── mnist_loader.h               # Carga y preprocesamiento del dataset MNIST
├── predict.cpp                  # Ejecuta predicciones sobre nuevas imágenes
├── train.cpp                    # Entrenamiento de la red neuronal
├── stb_image.h                  # Librería externa para cargar imágenes PNG
├── stb_image_resize.h           # Librería externa para redimensionar imágenes PNG

├── Imagenes_Prueba/             # Imágenes PNG para pruebas de predicción
│   └── .gitkeep

├── data/                        # Datos y recursos auxiliares
│   ├── data.zip
│   └── model_architecture.txt   # Estructura textual de la red neuronal entrenada

├── layer_output/                # Pesos y sesgos guardados tras entrenamiento
│   ├── layer0_weights.txt
│   ├── layer0_biases.txt
│   ├── layer2_weights.txt
│   ├── layer2_biases.txt
│   ├── layer4_weights.txt
│   └── layer4_biases.txt

├── nn/                          # Núcleo del sistema de red neuronal
│   ├── neural_network.h         # Clase central que orquesta las capas y el entrenamiento
│   ├── nn_activation.h          # Funciones de activación (ReLU, Softmax)
│   ├── nn_dense.h               # Capas densas totalmente conectadas
│   ├── nn_interfaces.h          # Interfaces base para capas, funciones de pérdida, etc.
│   ├── nn_loss.h                # Función de pérdida: CrossEntropy
│   └── nn_optimizer.h           # Optimizador: Descenso de Gradiente Estocástico (SGD)

├── utils/                       # Módulos auxiliares y utilitarios
│   ├── ascii_view.h             # Visualización de imágenes en formato ASCII
│   ├── common_helpers.h         # Funciones auxiliares para métricas y evaluación
│   ├── image_processor.h        # Conversión de imágenes PNG a tensores
│   └── tensor.h                 # Implementación genérica de tensores (N-dimensionales)
```

## 3. Ejecución

**Estado:**  
El sistema compila y se ejecuta correctamente. Se entrenó y evaluó sobre el dataset MNIST, mostrando predicciones y errores visuales en consola.

```
bash
cd build
cmake ..
make
./train

```

Se prepararon correctamente los archivos mnist_train.csv y mnist_test.csv en formato CSV.

Se ejecutó el programa train, el cual:

    Carga los datos de manera secuencial.

    Inicializa una red neuronal con la siguiente arquitectura:
    784 → 128 → 64 → 10.

    Utiliza codificación one-hot para las etiquetas (Y_train, Y_test).

    Entrena la red durante 2 épocas con batch_size = 100 y learning_rate = 0.001.

    Imprime 2 aciertos y 1 error visualizado en consola mediante caracteres ASCII.

    Guarda la arquitectura y los pesos entrenados en archivos .txt.



### 3.2 Inferencia (`predict`)

**Pasos de ejecución:**

```bash
cd build
./predict
```
Flujo de uso real:

    El programa predict no requiere volver a entrenar la red.

    Carga automáticamente los pesos y arquitectura previamente guardados por el programa train.

Fuentes de entrada para predicción:

    Imágenes del archivo mnist_test.csv (muestras estándar de validación).

    Dibujos propios del usuario almacenados como imágenes dentro de la carpeta Imagenes_Prueba/.

Requisitos para imágenes propias:

    Las imágenes pueden estar en distintos tamaños iniciales, pero:

        El sistema redimensiona automáticamente a 28x28 píxeles.

        El dígito se centra automáticamente en la imagen.

        Se recomienda que el número tenga un grosor suficiente para evitar errores por irregularidades de trazo.

    El preprocesamiento incluye:

        Escalado a escala de grises.

        Normalización de valores.

        Conversión al formato tensorial compatible con la red neuronal.

Salida esperada:

    Visualización en consola de:

        El número predicho por la red.

        La imagen original representada en ASCII, útil para validar visualmente aciertos y errores.


---

## 4. Análisis del rendimiento

### 4.1 Métricas reales observadas

- 🧭 **Épocas ejecutadas:** 2  
- ⏱️ **Tiempo estimado por época:** 600 segundos  
- 🎯 **Precisión obtenida en el test set:** > 90 %  
- 📉 **Función de pérdida:** decreciente por época  

---

### 4.2 Observaciones adicionales

- El sistema mostró **buen rendimiento con las imágenes del dataset original MNIST**.
- Sin embargo, presentó **dificultades al predecir correctamente dígitos como 9 y 7** cuando se utilizaron dibujos propios (inputs externos al dataset).
- Esto sugiere que la red tiene una **sensibilidad particular a ciertas formas o estilos no representados** en los datos de entrenamiento.


---

## 5. Conclusiones

- Se logró implementar exitosamente una red neuronal multicapa en C++ capaz de reconocer dígitos escritos a mano del dataset MNIST, alcanzando una precisión superior al 90 % en el conjunto de prueba.
- El sistema fue dividido en dos programas principales: `train`, encargado del entrenamiento, y `predict`, dedicado exclusivamente a la inferencia a partir de pesos previamente guardados.
- El flujo completo de entrenamiento, evaluación y predicción se ejecuta correctamente, incluyendo la visualización ASCII de los resultados, lo que permite una validación visual inmediata.
- Las predicciones sobre imágenes externas (dibujos propios) evidenciaron limitaciones en la capacidad de generalización del modelo, especialmente en dígitos con trazos irregulares como el 9 y el 7. Esto resalta la importancia de incluir datos más variados o aplicar técnicas de aumento de datos en futuros trabajos.
- En general, el proyecto demuestra el potencial de construir modelos de aprendizaje profundo desde cero sin frameworks externos, reforzando el entendimiento práctico de redes neuronales, procesamiento de datos e ingeniería de software en C++.

---



## 6. Trabajo en equipo



| Tarea                            | Miembro                                   | Rol                                           |
|----------------------------------|--------------------------------------------|-----------------------------------------------|
| Investigación teorica y Implementación del dataset | Rodrigo Miguel Gomez Pacheco - 202410309   | Análisis del dataset y Documentar bases teóricas      |
| Diseño de la arquitectura        | Julio Ruiz Villavicencio - 202410616 | Organización estructural del sistema    |
| Implementación del modelo final  |  Javier Alejandro Castro Barreto - 202310081      | Programación de la red neuronal en C++        |


---



### 7. Bibliografía

[1] G. Strang, *Linear Algebra and Its Applications*, 4th ed. Belmont, CA: Brooks/Cole, 2006.

[2] C. D. Meyer, *Matrix Analysis and Applied Linear Algebra*, SIAM, 2000.

[3] A. Galindo, M. Osorio, and J. Serrano Enciso, “Objetos de aprendizaje para operaciones matriciales en procesamiento digital de imágenes,” *Revista Dilemas Contemporáneos: Educación, Política y Valores*, vol. 7, no. 2, 2020.

[4] V. Cherkassky, M. Fassett, and P. Vassilas, “Efficient Implementation of Matrix Algorithms in C++,” *IEEE Transactions on Education*, vol. 34, no. 2, pp. 148–155, May 1991.

[5] A. Al-Ghuribi and S. Thabit, “A Survey on Matrix Multiplication Algorithms,” *International Journal of Computer Applications*, vol. 58, no. 19, 2012.

---

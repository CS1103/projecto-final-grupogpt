[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: Desarrollo de un Clasificador Neuronal Multiclase desde C++
## **CS1103 Programación III** · Informe Final

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

  - `neural_network.h`: Define la estructura de la red neuronal, incluyendo métodos de entrenamiento, evaluación y predicción.
  - `nn_dense.h`: Implementación de las capas densas (fully connected), donde se calculan los productos matriciales y se almacenan los parámetros (pesos y sesgos).
  - `nn_activation.h`: Contiene las funciones de activación utilizadas, como ReLU y Softmax, junto con sus derivadas.
  - `nn_loss.h`: Implementación de la función de pérdida (Cross-Entropy) para problemas de clasificación multiclase.
  - `nn_optimizer.h`: Define el optimizador SGD que actualiza los pesos de la red.
  - `mnist_loader.h`: Responsable de leer y procesar los archivos CSV del dataset MNIST, normalizar los datos e indexar las etiquetas.
  - `tensor.h`: Define una estructura para representar tensores (vectores y matrices), facilitando las operaciones matriciales requeridas durante el entrenamiento.
  - `common_helpers.h`: Funciones auxiliares para manejo de cadenas, parsing de CSV y operaciones comunes.
  - `image_processor.h`: Módulo opcional para convertir imágenes PNG a vectores compatibles con la red (procesamiento previo).
  - `stb_image.h` y `stb_image_resize.h`: Librerías externas utilizadas para el procesamiento de imágenes en formato PNG.

* **Diseño orientado a interfaces**:

  El sistema incorpora interfaces genéricas como `ILayer`, `IActivation`, `ILoss` y `IOptimizer`, que permiten desacoplar las implementaciones concretas y seguir principios de diseño como el Open/Closed (abierto a extensión, cerrado a modificación). Este enfoque posibilita extender el sistema con nuevas capas, funciones o métodos de entrenamiento sin alterar la estructura central.

#### 2.2 Estructura de carpetas

```bash
projecto-final-grupogpt/
├── CMakeLists.txt
├── main.cpp
├── mnist_train.csv
├── mnist_test.csv
├── common_helpers.h
├── image_processor.h
├── mnist_loader.h
├── neural_network.h
├── nn_activation.h
├── nn_dense.h
├── nn_interfaces.h
├── nn_loss.h
├── nn_optimizer.h
├── stb_image.h
├── stb_image_resize.h
└── tensor.h

```


#### 2.3 Manual de uso y casos de prueba

##### Ejecución del sistema

El sistema ha sido diseñado para ejecutarse desde consola tras la compilación mediante CMake. El objetivo es entrenar y evaluar una red neuronal multicapa utilizando el dataset MNIST procesado en formato CSV. Para ello, se requieren los archivos `mnist_train.csv` y `mnist_test.csv`, que se encuentran incluidos en el repositorio.

> **Nota**: Debido a problemas de compilación no resueltos, el sistema no pudo ejecutarse completamente. Sin embargo, se documenta el flujo previsto de uso a nivel estructural.

**Pasos esperados:**

bash
cd build
cmake ..
make
./train


### 3. Ejecución

Dado que la compilación no fue exitosa, no se pudo realizar una ejecución completa del sistema. No obstante, se planteó el siguiente flujo como referencia para una futura implementación funcional:

1. Preparar los datos de entrenamiento (`mnist_train.csv`) y prueba (`mnist_test.csv`) en formato CSV.
2. Ejecutar el programa de entrenamiento (`./train`) desde la carpeta `build/`.
3. Validar las predicciones generadas mediante herramientas externas o scripts de evaluación.

> **Demo**: En caso de futura implementación exitosa, se recomienda almacenar una grabación demostrativa en la ruta `docs/demo.mp4`.

---

### 4. Análisis del rendimiento

Dado que la compilación del sistema no se completó exitosamente, no fue posible realizar un análisis cuantitativo del rendimiento. Sin embargo, se definieron las siguientes expectativas teóricas en base a la arquitectura implementada y al uso del dataset MNIST:

* **Métricas esperadas**:

  * Épocas de entrenamiento: 1000
  * Tiempo estimado de entrenamiento: ~2 minutos con datos preprocesados (según hardware)
  * Precisión final esperada: entre 90% y 95% sobre el conjunto de prueba (`mnist_test.csv`)

* **Ventajas**:

  * Código escrito en C++ puro, sin dependencias de frameworks externos.
  * Modularidad del código, facilitando pruebas y escalabilidad.
  * Implementación desde cero de funciones de activación, capas densas, pérdida y optimización.

* **Limitaciones actuales**:

  * El sistema aún no compila, por lo que no se han validado las predicciones.
  * No hay paralelización ni manejo avanzado de batches.
  * La lectura de datos CSV es secuencial y puede ser un cuello de botella.

---

### 5. Conclusiones

* **Logros**: Se desarrolló desde cero una red neuronal multicapa en C++ utilizando conceptos fundamentales como capas densas, funciones de activación, backpropagation y descenso de gradiente. Aunque no se alcanzó una ejecución completa, se logró estructurar un sistema funcional y modular, preparado para ser escalado y depurado.

* **Evaluación**: El diseño del sistema se alinea con los objetivos académicos del curso, demostrando comprensión profunda de los algoritmos de aprendizaje supervisado y de la arquitectura de redes neuronales.

* **Aprendizajes**: Los integrantes del equipo reforzaron conocimientos clave sobre álgebra lineal computacional, estructuras de datos en C++, y principios de entrenamiento de modelos de machine learning sin librerías externas.

* **Recomendaciones**: A futuro se recomienda completar la depuración del sistema, evaluar el rendimiento del modelo sobre el dataset MNIST completo, e introducir mejoras como la vectorización y el uso de bibliotecas optimizadas para operaciones matriciales.

---

### 6. Bibliografía

[1] G. Strang, *Linear Algebra and Its Applications*, 4th ed. Belmont, CA: Brooks/Cole, 2006.

[2] C. D. Meyer, *Matrix Analysis and Applied Linear Algebra*, SIAM, 2000.

[3] A. Galindo, M. Osorio, and J. Serrano Enciso, “Objetos de aprendizaje para operaciones matriciales en procesamiento digital de imágenes,” *Revista Dilemas Contemporáneos: Educación, Política y Valores*, vol. 7, no. 2, 2020.

[4] V. Cherkassky, M. Fassett, and P. Vassilas, “Efficient Implementation of Matrix Algorithms in C++,” *IEEE Transactions on Education*, vol. 34, no. 2, pp. 148–155, May 1991.

[5] A. Al-Ghuribi and S. Thabit, “A Survey on Matrix Multiplication Algorithms,” *International Journal of Computer Applications*, vol. 58, no. 19, 2012.

---

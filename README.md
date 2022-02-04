# NeuralNetworkMIMD
Parallelization of a neural network with OpenMP & MPI
Objetivos

El objetivo principal del proyecto que planteamos es el de paralelizar la implementación proporcionada de una red neuronal de tipo perceptron utilizando diferentes aproximaciones: Memoria compartida y paso de mensajes.
Se realizarán análisis de comportamiento de las versiones implementadas, se ejecutarán utilizando diferentes configuraciones, se tomarán medidas correspondientes y se identificarán los diferentes problemas de rendimiento encontrados.
Descripción del problema
El código que facilitamos se corresponde a la implementación en lenguaje C de una red neuronal de tipo Perceptron multicapa,
Una red neuronal es un conjunto de neuronas artificiales interconectadas que utiliza un modelo matemático o computacional de procesamiento de datos basado en una aproximación conexionista para la computación. Una red neuronal de tipo Perceptron consiste, al menos, tres capas de nodos:
• capa de entrada
• capa oculta
• capa de salida
Este tipo de red neuronal tiene una capa de neuronas de entrada, cada una de las cuales está conectada a todas las neuronas de una capa intermedia conocida como capa oculta. Las neuronas de esta capa oculta pueden estar conectadas a todas las neuronas de una otra capa oculta o en las neuronas de una capa de salida.
El comportamiento de una neurona consiste en integrar los valores presentes en sus entradas y activarse o no, dependiendo del resultado. Esta organización de elementos sencillos de procesamiento es particularmente adecuado para la clasificación y/o reconocimiento de patrones.
Para poder utilizar una red neuronal, es necesaria una fase de entrenamiento, en la que se le presentan un conjunto de entradas con una salida conocida para que la red sea capaz de adaptar los pesos de las conexiones entre las neuronas de distintas capas. Así, en el código que proporcionamos encontrará la funcion trainN( responsable de esta fase. Una vez entrenada, la red entra en una fase de uso, donde puede recibir entradas con una salida desconocida, 
que serán clasificadas según los valores determinados en la fase de entrenamiento.

import os
import helpers
from numpy import exp, array, random
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from Perceptron_sin_tf import Perceptron
from Perceptron_con_tf import Perceptron_con_Tensor_Flow

def lanzar():
    observaciones_entradas = [[1, 0],[1, 1],[0, 1],[0, 0]]
    while True:
        os.system('cls')
        print("========================")
        print(" BIENVENIDO AL Manager ")
        print("========================")
        print("[1] Entrenar sin TensorFlow ")
        print("[2] Entrenar con TensorFlow ")
        print("[3] Salir ")
        print("NOTA: Este entrenamiento se plantea con solo dos neuronas")
        print("========================")

        opcion = input("> ")
        helpers.limpiar_pantalla()

        if opcion == '1':
            perceptron = Perceptron()
            predicciones = list()
            print('LOS VALORES QUE QUIERES PREDECIR TIENEN QUE ESTAR ENTRE 0 Y 1 INCLUIDOS')
            for i in range(4):
                valor = -.1
                while valor >= 0 and valor <= 1:
                    print(f'\nPara la observación {observaciones_entradas[i][0]} , {observaciones_entradas[i][1]}')
                    valor = float(input(" > "))
                    if not (valor >= 0 and valor <= 1):
                        helpers.limpiar_pantalla()
                        print('Por favor sigue las indicaciones')
                        print('LOS VALORES QUE QUIERES PREDECIR TIENEN QUE ESTAR ENTRE 0 Y 1 INCLUIDOS')
                predicciones.append(valor)

            helpers.limpiar_pantalla()

            print('¿Que funcion de activación vas as usar?')
            print("[1] Sigmoide")
            print("[2] RELU ")
            f_activacion = int(input("> "))
            helpers.limpiar_pantalla()

            print('¿En cuantas épocas vas a dividir el aprendizaje?')
            epochs = int(input("> "))
            helpers.limpiar_pantalla()

            print('¿Cual va a ser la tasa de aprendizaje?')
            print('RECOMENDAMOS 0.1')
            txAprendizaje = float(input("> "))

            if f_activacion == 2:
                perceptron.aprendizaje(array(observaciones_entradas), predicciones, epochs, 'RELU', txAprendizaje)

            else:
                perceptron.aprendizaje(array(observaciones_entradas), predicciones, epochs, txAprendizaje = txAprendizaje)







'''observaciones_entradas = array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
])


predicciones = array([[0], [1], [0], [0]])

perci = Perceptron()

perci.aprendizaje(observaciones_entradas, predicciones, epochs=10000)

perci.visualizacion()
perci.predicciones(1, 1)

observaciones_entradas = array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
])


predicciones = array([[0], [1], [0], [0]])'''

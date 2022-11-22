#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 10 - La predicción con neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#   TENSORFLOW : 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

from numpy import exp, array, random
import matplotlib.pyplot as plt
class Perceptron():
    def __init__(self):
        #Generación de los pesos en el intervalo [-1;1]
        random.seed(1)
        self.w11 = 2 * random.random() - 1
        self.w21 = 2 * random.random() - 1
        self.wb = 0
        self.pesos_iniciales =[self.w11, self.w21, self.wb]

    #--------------------------------------
    #       FUNCIONES ÚTILES
    #--------------------------------------
    def suma_ponderada(self, X1,W11,X2,W21,B,WB):
        return (B*WB+( X1*W11 + X2*W21))

    def funcion_activacion_sigmoide(self, valor_suma_ponderada):
        return (1 / (1 + exp(-valor_suma_ponderada)))

    def funcion_activacion_relu(self, valor_suma_ponderada):
        return (max(0,valor_suma_ponderada))

    def error_lineal(self, valor_esperado, valor_predicho):
        return (valor_esperado-valor_predicho)

    def calculo_gradiente(self, valor_entrada,prediccion,error):
        return (-1 * error * prediccion * (1-prediccion) * valor_entrada)

    def calculo_valor_ajuste(self, valor_gradiente, tasa_aprendizaje):
        return (valor_gradiente*tasa_aprendizaje)

    def calculo_nuevo_peso (self, valor_peso, valor_ajuste):
        return (valor_peso - valor_ajuste)

    def calculo_MSE(self, predicciones_realizadas, predicciones_esperadas):
        i=0;
        suma=0;
        for prediccion in predicciones_esperadas:
            diferencia = predicciones_esperadas[i] - predicciones_realizadas[i]
            cuadradoDiferencia = diferencia * diferencia
            suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
        return media_cuadratica


    #--------------------------------------
    #    APRENDIZAJE
    #--------------------------------------

    def aprendizaje(self, observaciones_entradas, predicciones, epochs = 10000, f_activacion = 'Sigmoide', txAprendizaje = .1, sesgo = 1):
        self.f_activacion = f_activacion
        self.Grafica_MSE = []
        for epoch in range(epochs):
            print("EPOCH ("+str(epoch)+"/"+str(epochs)+")")
            predicciones_realizadas_durante_epoch = [];
            predicciones_esperadas = [];
            numObservacion = 0
            for observacion in observaciones_entradas:

                #Carga de la capa de entrada
                x1 = observacion[0];
                x2 = observacion[1];

                #Valor de predicción esperado
                valor_esperado = predicciones[numObservacion][0]

                #Etapa 1: Cálculo de la suma ponderada
                valor_suma_ponderada = self.suma_ponderada(x1,self.w11,x2,self.w21,sesgo,self.wb)


                #Etapa 2: Aplicación de la función de activación
                if self.f_activacion == 'Sigmoide':
                    valor_predicho = self.funcion_activacion_sigmoide(valor_suma_ponderada)

                else:
                    valor_predicho = self.funcion_activacion_relu(valor_suma_ponderada)


                #Etapa 3: Cálculo del error
                valor_error = self.error_lineal(valor_esperado,valor_predicho)


                #Actualización del peso 1
                #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
                gradiente_W11 = self.calculo_gradiente(x1,valor_predicho,valor_error)
                valor_ajuste_W11 = self.calculo_valor_ajuste(gradiente_W11,txAprendizaje)
                self.w11 = self.calculo_nuevo_peso(self.w11,valor_ajuste_W11)

                # Actualización del peso 2
                gradiente_W21 = self.calculo_gradiente(x2, valor_predicho, valor_error)
                valor_ajuste_W21 = self.calculo_valor_ajuste(gradiente_W21, txAprendizaje)
                self.w21 = self.calculo_nuevo_peso(self.w21, valor_ajuste_W21)


                # Actualización del peso del sesgo
                gradiente_Wb = self.calculo_gradiente(sesgo, valor_predicho, valor_error)
                valor_ajuste_Wb = self.calculo_valor_ajuste(gradiente_Wb, txAprendizaje)
                self.wb = self.calculo_nuevo_peso(self.wb, valor_ajuste_Wb)

                print("     EPOCH (" + str(epoch) + "/" + str(epochs) + ") -  Observación: " + str(numObservacion+1) + "/" + str(len(observaciones_entradas)))

                #Almacenamiento de la predicción realizada:
                predicciones_realizadas_durante_epoch.append(valor_predicho)
                predicciones_esperadas.append(predicciones[numObservacion][0])

                #Paso a la observación siguiente
                numObservacion = numObservacion+1

            MSE = self.calculo_MSE(predicciones_realizadas_durante_epoch, predicciones)
            self.Grafica_MSE.append(MSE[0])
            print("MSE: "+str(MSE))

        print()
        print()
        print ("¡Aprendizaje terminado!")
        print ("Pesos iniciales: " )
        print ("W11 = "+str(self.pesos_iniciales[0]))
        print ("W21 = "+str(self.pesos_iniciales[1]))
        print ("Wb = "+str(self.pesos_iniciales[2]))

        print ("Pesos finales: " )
        print ("W11 = "+str(self.w11))
        print ("W21 = "+str(self.w21))
        print ("Wb = "+str(self.wb))



    #--------------------------------------
    #       GRÁFICA
    #--------------------------------------
    def visualizacion(self):
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

    #--------------------------------------
    #       PREDICCIÓN
    #--------------------------------------

    def predicciones(self, x1, x2, sesgo = 1):

        #Etapa 1: Cálculo de la suma ponderada
        valor_suma_ponderada = self.suma_ponderada(x1,self.w11,x2,self.w21,sesgo,self.wb)


        #Etapa 2: Aplicación de la función de activación
        if self.f_activacion == 'Sigmoide':
            valor_predicho = self.funcion_activacion_sigmoide(valor_suma_ponderada)
        else:
            valor_predicho = self.funcion_activacion_relu(valor_suma_ponderada)

        print()
        print("--------------------------")
        print ("PREDICCIÓN ")
        print("--------------------------")

        print("Predicción del [" + str(x1) + "," + str(x2)  + "]")
        print("Predicción = " + str(valor_predicho))



observaciones_entradas = array([
                                [1, 0],
                                [1, 1],
                                [0, 1],
                                [0, 0]
                                ])


predicciones = array([[0],[1], [0],[0]])

perci = Perceptron()

perci.aprendizaje(observaciones_entradas, predicciones, epochs =10000)

perci.visualizacion()
perci.predicciones(1,1)

observaciones_entradas = array([
                                [1, 0],
                                [1, 1],
                                [0, 1],
                                [0, 0]
                                ])


predicciones = array([[0],[1], [0],[0]])


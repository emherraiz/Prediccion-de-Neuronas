#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 10 - La predicción con neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB: 3.0.3
#   TENSORFLOW: 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


#-------------------------------------
#    PARÁMETROS DE LA RED
#-------------------------------------
#import tensorflow as tf
# Compact ya que no disponemos de GPU
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

# Desactivamos la version v2
tf.disable_v2_behavior()


class Perceptron_con_Tensor_Flow():
    def __init__(self, valores_entradas_X, valores_a_predecir_Y):
        #-------------------------------------
        #    PARAMETROS GENERALES
        #-------------------------------------

        self.valores_entradas_X = valores_entradas_X
        self.valores_a_predecir_Y = valores_a_predecir_Y


        #Variable TensorFLow correspondiente a los valores de neuronas de entrada
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

        #Variable TensorFlow correspondiente a la neurona de salida (predicción real)
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

    #--------------------------------------
    #       FUNCIONES ÚTILES
    #--------------------------------------

    def activacion(self, neuronas_entrada_X, peso, sesgo):


        '''La suma ponderada es en la práctica una multiplicación de matrices
        entre los valores en la entrada X y los distintos pesos

        Args:
            neuronas_entrada_X (tf.placeholder()): Variable TensorFLow correspondiente a los valores de neuronas de entrada
            peso (tf.Variable()): Creación de una variable TensorFlow de tipo tabla
            sesgo (tf.Variable()): _description_

        Returns:
            Predicción (tf.sigmoid()): Activación de tipo sigmoide que permite calcular la predicción
        '''

        suma = tf.matmul(neuronas_entrada_X, peso)
        suma = tf.add(suma, sesgo)
        return tf.sigmoid(suma)

    def error_MSE(self, prediccion, valores_reales_Y):
        '''Función de error de media cuadrática MSE

        Args:
            prediccion (tf.sigmoid()): Activación de tipo sigmoide que permite calcular la predicción
            valores_reales_Y (tf.placeholder()): Variable TensorFlow correspondiente a la neurona de salida (predicción real)

        Returns:
            tf.reduce_sum(): Reducción del error cuadrático
        '''

        funcion_error = tf.reduce_sum(tf.pow(valores_reales_Y-prediccion,2))
        return funcion_error

    def optimizar(self, funcion_error, txAprendizaje = .1):
        '''Descenso de gradiente

        Args:
            txAprendizaje (float): tasa de aprendizaje
            funcion_error (tf.reduce_sum()): MSE

        Returns:
            tf.train.GradientDescentOptimizer(): Gradiente descendiente
        '''

        gradiente = tf.train.GradientDescentOptimizer(learning_rate = txAprendizaje).minimize(funcion_error)
        return gradiente


    #-------------------------------------
    #    APRENDIZAJE
    #-------------------------------------

    def aprendizaje(self, epochs):
        '''Realizamos el aprendizaje mediante tf.Session()

        Args:
            epochs (int): Numero de épocas
        '''

        #-- Peso --
        #Creación de una variable TensorFlow de tipo tabla
        #que contiene 2 entradas y cada una tiene un peso [2,1]
        #Estos valores se inicializan al azar
        peso = tf.Variable(tf.random_normal([2, 1]), tf.float32)

        #-- Sesgo inicializado a 0 --
        sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

        # Función de activación de tipo sigmoide que permite calcular la predicción
        self.prediccion = self.activacion(self.tf_neuronas_entradas_X, peso, sesgo)

        # Función de error de media cuadrática MSE
        funcion_error = self.error_MSE(self.prediccion, self.tf_valores_reales_Y)

        # Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        optimizador = self.optimizar(funcion_error)

        #Inicialización de la variable
        init = tf.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        self.sesion = tf.Session()
        self.sesion.run(init)

        #Para la realización de la gráfica para la MSE
        self.Grafica_MSE=[]


        #Para cada epoch
        for i in range(epochs):

            #Realización del aprendizaje con actualzación de los pesos
            self.sesion.run(optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y: self.valores_a_predecir_Y})

            #Calcular el error
            MSE = self.sesion.run(funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y: self.valores_a_predecir_Y})

            #Visualización de la información
            self.Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


    def visualizacion(self):
        '''Visualización de la gráfica de MSE
        '''
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

    def verificaciones(self):
        '''Realización de verificaciones
        '''
        print("--- VERIFICACIONES ----")
        for i in range(4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(self.sesion.run(self.prediccion, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))

        self.sesion.close()


cua = Perceptron_con_Tensor_Flow([[1., 0.], [1., 1.], [0., 1.], [0., 0.]], [[0.], [1.], [0.], [0.]])
cua.aprendizaje(1000)
cua.visualizacion()
cua.verificaciones()









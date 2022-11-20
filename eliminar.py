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

# Desactivamos la version v2
tf.disable_v2_behavior()


class Perceptron_con_Tensor_Flow():

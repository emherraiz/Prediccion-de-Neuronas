predicciones_esperadas = [1,0,1]
predicciones_realizadas = [0.25,0.55,0.75]
suma = 0
i=0
for i in range(len(predicciones_esperadas)):

    diferencia = predicciones_esperadas[i]-  predicciones_realizadas [i]
    cuadradoDiferencia = diferencia * diferencia
    print("Diferencia = " + str(predicciones_esperadas[i]) + "-" +  str(predicciones_realizadas [i]) + "=" + str(diferencia))
    print("Diferencia * Diferencia = " + str(diferencia) + "*" +  str(diferencia) + str() + "=" + str(cuadradoDiferencia))
    print("Suma = " + str(suma) + "+" + str(cuadradoDiferencia))
    suma = suma + cuadradoDiferencia

    print("")

media_cuadratica = 1/(len(predicciones_esperadas)) * suma

print("Error media cuadrática ="+str(media_cuadratica))

import cv2, os, csv
import numpy as np  
         



                                       

#constantes
SCORE_ALTO = 0.8
REDUCCION_VERTICAL = 0.5
REDUCCION_HORIZONTAL = 0.30
REDUCCION_LATERAL_PULGAR = 0.33
REDUCCION_LADO = 0.20
REDUCCION_INDICE_PULGAR = 0.20
REDUCCION_MENIQUE_PULGAR = 0.2
ANGULO_NUDILLO = 90
ANGULO_INTERFALANGICO = 125
CONSTANTE_TUMBADA = 0.30
SEGUNDOS_POR_LETRA = 2
MARGEN_MOVIMIENTO = 0.25
CONSTANTE_MOVIMIENTO = 3


#variables
calibrada = False; primerframe = True
segsinicio = 0
segsInicioMovimiento = 0
frames = 0
mensaje = ""
ultimaLetra = ""
contadorDerecha = 0
contadorIzquierda = 0
letraDetectada = ""
letraActual = ""
letraAnterior = ""
finTemporizador = False

#movimiento
contadorMovimientoX = 0
contadorMovimientoY = 0
direccionX = ""
direccionY = ""
margenMovimiento = 0
posicionXInicial = 0   
posicionYInicial = 0
posicionXFinal = 0   
posicionYFinal = 0

#distancias
mano_derecha = [0, 0, 0, 0, 0, 0, 0]  #0 para orientacion horizontal, 1 la vertical, 2 para saber si está de lado o no, 3 para saber si el pulgar está extendido, 4 medida para factor de movimiento, 5 y 6 para dedos pegados
mano_izquierda = [0, 0, 0, 0, 0, 0, 0]  

#estado
estado = {}

##################################################
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
PUNTAS = {4, 8, 12, 16, 20}
##################################################



def prepare(sdl):
    global primerframe, segsinicio
    """Se ejecuta UNA VEZ al cargar el proyecto. Configura widgets y precarga modelos."""

    #No utilizados
    sdl.widget("file1").hide()
    sdl.widget("counter1").hide()
    sdl.widget("slider1").hide()
    sdl.widget("text_input1").hide()
    sdl.widget("text_input2").hide()
    sdl.widget("counter2").hide()
    sdl.widget("file2").hide()
    sdl.widget("select1").hide()
    sdl.widget("select2").hide()

    #Mensaje
    sdl.widget("text1").show()
    sdl.widget("text1").label("Mensaje: ")
    sdl.widget("text1").value(" ")

    #Calibración de la mano
    sdl.widget("led1").label("Mano/s calibrada/s").show()
    sdl.widget("led1").value(False)
    sdl.widget("led2").label("Calibrando mano/s:").show()
    sdl.widget("led2").value(False)

    #Recalibrar
    sdl.widget("toggle1").label("Recalibrar mano/s").show()
    sdl.widget("toggle1").value(False)

    #borrar mensaje
    sdl.widget("toggle2").label("Borrar mensaje").show()

    #detectando letra
    sdl.widget("led3").label("Detectando letra").value(False).show()

    #modelos
    sdl.preload("hands_mp", "hands_mp_v1")

    #variables
    segsinicio = sdl.elapsed()
    primerframe = True

    #debug
    sdl.widget("text2").label("Debugg").show()
    sdl.widget("text2").value("")


def process(frame, sdl):
    global segsinicio, frames, mano_derecha, mano_izquierda, calibrada, contador, ultimaLetra, mensaje, letraDetectada, letraActual, letraAnterior, finTemporizador, estado, primerframe
    """Se ejecuta en cada frame. Retorna imagen BGR anotada para AI imgput."""
    img = frame.copy()
    inferencia = sdl.infer("hands_mp", img, "hands_mp_v1")

    manos = inferencia.get("hands", [])

    ##############################################################################
    for hand in inferencia.get("hands", []):
        lms = hand["landmarks"]
        color = (0, 220, 120) if hand["handedness"] == "Left" else (220, 120, 0)
        for a, b in CONEXIONES:
            pa = (lms[a]["px"], lms[a]["py"])
            pb = (lms[b]["px"], lms[b]["py"])
            cv2.line(img, pa, pb, color, 2)
        for i, lm in enumerate(lms):
            c = (0, 255, 80) if i in PUNTAS else (255, 255, 255)
            cv2.circle(img, (lm["px"], lm["py"]), 5, c, -1)
            cv2.circle(img, (lm["px"], lm["py"]), 5, color, 1)
        x1, y1, x2, y2 = hand["box"]
        score = hand.get("score", 1.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"{hand['handedness']} ({score:.2f})"
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(img, f"Mensaje: {mensaje}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    ##############################################################################
        

    #Calibra las manos para saber distancias medias 
    if(not calibrada):
        calibrarManos(manos, sdl)
   
    elif manos and scoreAltoAlMenosUnaMano(manos):
        
        #calcula el estado
        estado = analizarEstado(sdl, manos)
        
        #debug
        sdl.widget("text2").value(
        f"IndEnt:{estado['indiceDobladoEntero']}    "
        f"CorEnt:{estado['corazonDobladoEntero']}    "
        f"AnuEnt:{estado['anularDobladoEntero']}    "
        f"MenEnt:{estado['meniqueDobladoEntero']} |\n"

        f"IndSem:{estado['indiceDobladoSemi']}    "
        f"CorSem:{estado['corazonDobladoSemi']}    "
        f"AnuSem:{estado['anularDobladoSemi']}    "
        f"MenSem:{estado['meniqueDobladoSemi']} |\n"

        f"IndEst:{estado['indiceEstirado']}    "
        f"CorEst:{estado['corazonEstirado']}    "
        f"AnuEst:{estado['anularEstirado']}    "
        f"MenEst:{estado['meniqueEstirado']} |\n"

        f"GirH:{estado['estaGiradaHorizontalmente']}    "
        f"GirV:{estado['estaGiradaVerticalmente']}    "
        f"Lado:{estado['estaDeLado']}    "
        f"Frente:{estado['estaDeFrente']}    "
        f"Tumb:{estado['estaTumbada']} |\n"

        f"PulLat:{estado['pulgarFuera']}    "
        f"IndPul:{estado['indicePulgarPegados']}    "
        f"MenPul:{estado['meniquePulgarPegados']} |\n"

        f"IndCorCruz:{estado['indiceCorazonCruzados']}    "
        f"IndPulCruz:{estado['indicePulgarCruzados']}    "
        f"Puno:{estado['punoCerrado']}" + "\n Detectando letra:" + letraActual + "  |   Letra Anterior: " + letraDetectada
        )
       
        #LETRAS: detecta la letra, espera X segundos y vuelve a comprobarla, si sigue cumpliendo se apunta.

        #A: girada y con todos los dedos cerrados (menos el meñique porque no detecta bien)
        if estado["estaGiradaHorizontalmente"] and estado["indiceDobladoEntero"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["meniqueDobladoEntero"]:
            letraDetectada = "A0"

        
        #B: girada y con todos los dedos estirados
        elif estaGiradaHorizontalmente(sdl, manos, estado) and estado["estaTumbada"] and not estado["indiceDobladoEntero"] and not estado["anularDobladoEntero"] and not estado["corazonDobladoEntero"] and not estado["meniqueDobladoEntero"]:
            letraDetectada = "B0"

        #C: que esté de lado y con los dedos flexionados de cualquier forma (pero todos)
        elif (estado["estaDeLado"] and (estado["indiceDobladoSemi"] or estado["indiceDobladoEntero"]) and (estado["corazonDobladoSemi"] or estado["corazonDobladoEntero"]) 
                            and (estado["anularDobladoSemi"] or estado["anularDobladoEntero"]) and (estado["meniqueDobladoSemi"] or estado["meniqueDobladoEntero"])):
            letraDetectada = "C0"                                
    
        #D: de lado y anular, corazón y meñique semiflexionados, indice estirado
        elif estado["estaDeLado"] and estado["indiceEstirado"] and (estado["anularDobladoSemi"]) and (estado["corazonDobladoSemi"]) and (estado["meniqueDobladoSemi"]):
            letraDetectada = "D0"

        #E: de frente, pulgar estirado lateral y los dedos semiflexionados
        elif estado["estaDeFrente"] and estado["indiceDobladoSemi"] and estado["anularDobladoSemi"] and estado["corazonDobladoSemi"] and estado["meniqueDobladoSemi"] and estado["pulgarFuera"]:
            letraDetectada = "E0"

        #F: de lado, dedo indice semi y los otros dedos estirados
        elif estado["estaDeLado"] and estado["indiceDobladoSemi"] and estado["corazonEstirado"] and estado["anularEstirado"] and estado["meniqueEstirado"]:
            letraDetectada = "F0"

        #G: girada, tumbada y con los dedos doblados menos el índice
        elif estado["estaGiradaHorizontalmente"] and estado["indiceEstirado"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["meniqueDobladoEntero"]:
            letraDetectada = "G1"
        
        #H: de lado con pulgar, indice y anular estirados
        elif estado["estaDeLado"] and estado["indiceEstirado"] and estado["corazonEstirado"] and estado["anularDobladoEntero"] and estado["meniqueDobladoEntero"]:
            letraDetectada = "H0"
        
        #I: de frente con solo el meñique estirado sin movimiento
        elif (estado["estaDeFrente"] or estado["estaDeLado"]) and estado["indiceDobladoEntero"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["meniqueEstirado"]:
            ########esperar(sdl, manos, estado, True)
            letraDetectada = "I1"
        
        #J: como la i pero con movimiento horizontal
        elif estado["estaDeFrente"] and estado["indiceDobladoEntero"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["meniqueEstirado"]:
           ######## esperar(sdl, manos, estado, True) 
            letraDetectada = "J1"
        
        #K: de lado, dedo indice estirado, anular semi doblado y meñique y corazón doblados
        elif estado["estaDeLado"] and estado["indiceEstirado"] and estado["corazonDobladoSemi"] and estado["anularDobladoEntero"] and estado["meniqueDobladoEntero"] :
            letraDetectada = "K0"

        #L: de frente, pulgar estirado, indice estirado y el resto recogidos
        elif estado["estaDeFrente"] and estado["pulgarFuera"] and estado["indiceEstirado"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["meniqueDobladoEntero"]: 
            letraDetectada = "L1"
        
        #M: girada verticalmente con indice corazon y anular estirados
        elif estado["estaGiradaVerticalmente"] and estado["indiceEstirado"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["meniqueDobladoEntero"] :
            letraDetectada = "M0"
        
        #N: igual que M pero con indice y anular solo y sin movimiento
        elif estado["estaGiradaVerticalmente"] and estado["indiceEstirado"] and estado["anularEstirado"] and estado["corazonDobladoEntero"] and estado["meniqueDobladoEntero"]:
           ######## esperar(sdl, manos, estado, True) 
            letraDetectada = "N1"
        
        #Ñ: como la N pero con movimiento lateral
        elif estado["estaGiradaVerticalmente"] and estado["indiceEstirado"] and estado["anularEstirado"] and estado["corazonDobladoEntero"] and estado["meniqueDobladoEntero"]:
            #####esperar(sdl, manos, estado, True) 
            letraDetectada = "Ñ1"
        
        #O: de frente, meñique, corazón y anular estirados, pulgar cerca del indice
        elif estado["indicePulgarPegados"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["meniqueEstirado"] and not estado["estaDeLado"]:
            letraDetectada = "O0"
        
        #P: pulgar y meñique juntos y indice corazon y anular estirados, sin movimiento
        elif estado["estaDeFrente"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["indiceEstirado"] and estado["meniqueDobladoEntero"]:
            letraDetectada = "P1"

        #Q: girada horizontalmente, no tumbada, y con los cuatro dedos estirados
        elif estado["estaGiradaHorizontalmente"] and not estado["estaTumbada"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["indiceEstirado"] and estado["meniqueEstirado"] :
            letraDetectada = "Q0"
        
        #R: de frente, indice y corazon estirados y cruzados
        elif estado["estaDeFrente"] and estado["anularDobladoEntero"] and estado["corazonEstirado"] and estado["indiceEstirado"] and estado["meniqueDobladoEntero"] and estado["indiceCorazonCruzados"] :
            letraDetectada = "R1"

        #S: como la O pero con el indice flexionado del todo
        elif estado["indiceDobladoEntero"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["meniqueEstirado"]: 
            letraDetectada = "S0"
                
        #T: de lado, pulgar y indice cruzados
        elif estado["estaDeLado"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["meniqueEstirado"] and estado["indicePulgarCruzados"] :
            letraDetectada = "T0"
        
        #U: de frente, indice y corazón estirados, el resto doblados, sin movimiento
        elif estado["estaDeFrente"] and estado["indiceEstirado"] and estado["corazonEstirado"] and estado["meniqueDobladoEntero"] and estado["anularDobladoEntero"] :
            letraDetectada = "U1"

        #V: como la U pero con movimiento
        elif estado["estaDeFrente"] and estado["indiceEstirado"] and estado["corazonEstirado"] and estado["meniqueDobladoEntero"] and estado["anularDobladoEntero"]:
            #######esperar(sdl, manos, estado, True) 
            letraDetectada = "V1"

        #W: como la P pero con movimiento lateral
        elif estado["estaDeFrente"] and estado["meniquePulgarPegados"] and estado["anularEstirado"] and estado["corazonEstirado"] and estado["indiceEstirado"]:
            ######esperar(sdl, manos, estado, True) 
            letraDetectada = "W1"

        #X: tumbada, girada horizontalmente, con todos los dedos doblados excepto el indice, semidoblado
        elif estado["estaTumbada"] and estado["estaGiradaHorizontalmente"] and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["indiceDobladoSemi"] and estado["meniqueDobladoEntero"]:
            #######esperar(sdl, manos, estado, True) 
            letraDetectada = "X1"

        #Y: todos los dedos doblados del todo, meñique semidoblado, de frente o de lado
        elif (estado["estaDeFrente"] or estado["estaDeLado"]) and estado["anularDobladoEntero"] and estado["corazonDobladoEntero"] and estado["indiceDobladoEntero"] and estado["meniqueDobladoSemi"]: 
            letraDetectada = "Y0"

        #Z: meñique estirado, el resto doblados, de lado y con movimiento en x y dirección abajo en y
        elif estado["estaTumbada"] and estado["meniqueEstirado"] and estado["indiceDobladoEntero"] and estado["corazonDobladoEntero"] and estado["anularDobladoEntero"]:
            #####esperar(sdl, manos, estado, True) 
            letraDetectada = "Z1"
        
        else:
            letraDetectada = ""

        #no detecta nada
        if letraDetectada == "":
            letraActual = ""
            sdl.widget("led3").value(False)

        #cambia la letra y reinicia temporizador
        elif letraDetectada != letraActual:
            letraActual = letraDetectada
            segsinicio = sdl.elapsed()

            if letraActual[1] == "1":       #llevarán un 1 las que dependan del movimiento, un 0 las que no
                primerframe = True
                contadorMovimiento(sdl, manos, estado)
             

        #misma letra mantenida
        elif letraActual != "":
            sdl.widget("led3").value(True)
            tiempo = sdl.elapsed() - segsinicio
            if letraActual[1] == "1":
                contadorMovimiento(sdl, manos, estado)      #para llamar a la funcion en cada frame 

            if tiempo >= SEGUNDOS_POR_LETRA:

                if letraActual[1] == "1":
                    mensaje += letraActualMovida(letraActual[0])
                else:
                    mensaje += letraActual[0]

                letraActual = ""
                
                sdl.widget("text1").value(mensaje)
                sdl.widget("led3").value(False)
                mismaLetra = True

    else: 
        sdl.widget("led3").label("Detectando letra").value(False)

    ##BOTONES
    #recalibrar
    if(sdl.widget("toggle1").value):
        recalibrar(sdl)

    #borrar mensaje
    if(sdl.widget("toggle2").value):
        mensaje = ""
        sdl.widget("text1").value(mensaje)
        sdl.widget("toggle2").value(False)

    return img

def analizarEstado(sdl, manos):

    global estado

    if manos and scoreAltoAlMenosUnaMano(manos):

        #orientación de la mano
        estado["estaDeLado"] = estaDeLado(sdl, manos, estado)
        estado["estaGiradaHorizontalmente"] = estaGiradaHorizontalmente(sdl, manos, estado)
        estado["estaGiradaVerticalmente"] = estaGiradaVerticalmente(sdl, manos, estado)

        estado["estaDeFrente"] = (
            not estado["estaDeLado"]
            and not estado["estaGiradaHorizontalmente"]
            and not estado["estaGiradaVerticalmente"]
        )

        estado["estaTumbada"] = estaTumbada(sdl, manos, estado)  

        #dedos doblados completamente si tanto nudillo como intrafalage doblados
        estado["indiceDobladoEntero"] = doblado(sdl, manos, estado, 0, 5, 7, 8, 6, ANGULO_NUDILLO, True) and doblado(sdl, manos, estado, 5, 6, 8, 8, 7, ANGULO_INTERFALANGICO, False)
        estado["corazonDobladoEntero"] = doblado(sdl, manos, estado, 0, 9, 11, 12, 10, ANGULO_NUDILLO, True) and doblado(sdl, manos, estado, 9, 10, 12, 12, 11, ANGULO_INTERFALANGICO, False)
        estado["anularDobladoEntero"] = doblado(sdl, manos, estado, 0, 13, 15, 16, 14,ANGULO_NUDILLO, True) and doblado(sdl, manos, estado, 13, 14, 16, 16, 15, ANGULO_INTERFALANGICO, False)
        estado["meniqueDobladoEntero"] = doblado(sdl, manos, estado, 0, 17, 19, 20, 18, ANGULO_NUDILLO, True) and doblado(sdl, manos, estado, 17, 18, 20, 20, 19, ANGULO_INTERFALANGICO, False)

        #dedos semidoblados si intrafalange doblada o nudillo doblado pero no ambas
        estado["indiceDobladoSemi"] = doblado(sdl, manos, estado, 0, 5, 7, 8, 6, ANGULO_NUDILLO, True) ^ doblado(sdl, manos, estado, 5, 6, 8, 8, 7, ANGULO_INTERFALANGICO, False) 
        estado["corazonDobladoSemi"] = doblado(sdl, manos, estado, 0, 9, 11, 12, 10, ANGULO_NUDILLO, True) ^ doblado(sdl, manos, estado, 9, 10, 12, 12, 11, ANGULO_INTERFALANGICO, False) 
        estado["anularDobladoSemi"] = doblado(sdl, manos, estado, 0, 13, 15, 16, 14,ANGULO_NUDILLO, True) ^ doblado(sdl, manos, estado, 13, 14, 16, 16, 15, ANGULO_INTERFALANGICO, False) 
        estado["meniqueDobladoSemi"] = doblado(sdl, manos, estado, 0, 17, 19, 20, 18, ANGULO_NUDILLO, True) ^ doblado(sdl, manos, estado, 17, 18, 20, 20, 19, ANGULO_INTERFALANGICO, False) 

        #dedos estirados
        estado["indiceEstirado"] = (not estado["indiceDobladoEntero"] and not estado["indiceDobladoSemi"])
        estado["corazonEstirado"] = (not estado["corazonDobladoEntero"] and not estado["corazonDobladoSemi"])
        estado["anularEstirado"] = (not estado["anularDobladoEntero"] and not estado["anularDobladoSemi"])
        estado["meniqueEstirado"] = (not estado["meniqueDobladoEntero"] and not estado["meniqueDobladoSemi"])

        #resto
        estado["pulgarFuera"] = pulgarFuera(sdl, manos, estado)
        estado["indicePulgarPegados"] = dedosPegados(sdl, manos, 4, 8, 5, True)
        estado["meniquePulgarPegados"] = dedosPegados(sdl, manos, 4, 17, 6, estado["estaDeFrente"])
        estado["indiceCorazonCruzados"] = indiceCorazonCruzados(sdl, manos, estado) #para la R
        estado["indicePulgarCruzados"] = indicePulgarCruzados(sdl, manos, estado) #para la T

        estado["punoCerrado"] = (estado["indiceDobladoEntero"] and estado["corazonDobladoEntero"] and estado["anularDobladoEntero"] and estado["meniqueDobladoEntero"])

        return estado    

#funcion para detectar el movimiento en eje x o y
def contadorMovimiento(sdl, manos, estado):
    global margenMovimiento, primerframe, posicionXInicial, mensaje, posicionXFinal, posicionYFinal, posicionYInicial, contadorMovimientoX, contadorMovimientoY, direccionX, direccionY, segsInicioMovimiento
    #mensaje += "\ncontadormovimiento llamada primer frame: " + str(primerframe) + "\n X: " + str(contadorMovimientoX) + "\n Y: " + str(contadorMovimientoY)
    sdl.widget("text1").value(mensaje)
    #primera llamada
    if primerframe:
        segsInicioMovimiento = sdl.elapsed()  
        primerframe = False

        #punto inicial para contar oscilaciones, punto 0
        posicionXInicial = manos[0]["landmarks"][17]["px"]
        posicionYInicial = manos[0]["landmarks"][17]["py"]
        contadorMovimientoX = 0
        contadorMovimientoY = 0
        direccionX = ""
        direccionY = ""        

        #distancia para el margen de movimiento
        distancia = mano_derecha[4] if manos[0]["handedness"] == "Right" else mano_izquierda[4] 
        margenMovimiento = MARGEN_MOVIMIENTO * distancia

    elif sdl.elapsed() - segsInicioMovimiento < SEGUNDOS_POR_LETRA:
        posicionXFinal = manos[0]["landmarks"][17]["px"]
        posicionYFinal = manos[0]["landmarks"][17]["py"]       

        #si el resultado es negativo (se ha movido a la derecha) y la dirección era la misma entonces no se suma en contador, pero si el resultado cambia de signo si que suma
        
        if posicionXInicial - posicionXFinal < -margenMovimiento and not direccionX == "Derecha":
            contadorMovimientoX += 1
            direccionX = "Derecha"
            
        elif posicionXInicial - posicionXFinal > margenMovimiento and not direccionX == "Izquierda":
            contadorMovimientoX += 1
            direccionX = "Izquierda"

        if posicionYInicial - posicionYFinal < -margenMovimiento and not direccionY == "Abajo":
            contadorMovimientoY += 1
            direccionY = "Abajo"

        elif posicionYInicial - posicionYFinal > margenMovimiento and not direccionY == "Arriba":
            contadorMovimientoY += 1
            direccionY = "Arriba"


    return 

def letraActualMovida(letra):

    if letra == "G":
        letraMovida = "X"
    elif letra == "I":
        letraMovida = "J"
    elif letra == "L":
        letraMovida = "LL"
    elif letra == "R":
        letraMovida = "RR"
    elif letra == "U":
        letraMovida = "V"
    elif letra == "P":
        letraMovida = "W"

    return letra[0] if contadorMovimientoX < CONSTANTE_MOVIMIENTO else letraMovida

def calibrarManos(manos, sdl):
    
        #para que python sepa que son globales 
        global primerframe, segsinicio, mano_derecha, mano_izquierda, calibrada, contadorDerecha, contadorIzquierda

        if manos and primerframe and scoreAltoAlMenosUnaMano(manos):
            segsinicio = sdl.elapsed()
            sdl.widget("led2").value(True)
            primerframe = False
        
        # 3 segundos de calibre
        elif manos and not primerframe:
            if sdl.elapsed() - segsinicio < 3.0:            
                for mano in manos:
                    if mano["score"] > SCORE_ALTO:

                        #contador de manos buenas procesadas

                        #calibre distinto para cada mano por si acaso
                        if(mano["handedness"] == "Right"):          
                            mano_derecha[0] += mano["landmarks"][5]["px"] - mano["landmarks"][17]["px"]
                            mano_derecha[1] += mano["landmarks"][12]["py"] - mano["landmarks"][0]["py"]
                            mano_derecha[2] += np.sqrt((mano["landmarks"][12]["py"] - mano["landmarks"][0]["py"])**2 + (mano["landmarks"][12]["px"] - mano["landmarks"][0]["px"])**2)
                            mano_derecha[3] += np.sqrt((mano["landmarks"][4]["py"] - mano["landmarks"][6]["py"])**2 + (mano["landmarks"][4]["px"] - mano["landmarks"][6]["px"])**2)
                            mano_derecha[4] += np.sqrt((mano["landmarks"][17]["py"] - mano["landmarks"][13]["py"])**2 + (mano["landmarks"][17]["px"] - mano["landmarks"][13]["px"])**2)
                            mano_derecha[5] += np.sqrt((mano["landmarks"][4]["py"] - mano["landmarks"][8]["py"])**2 + (mano["landmarks"][4]["px"] - mano["landmarks"][8]["px"])**2)
                            mano_derecha[6] += np.sqrt((mano["landmarks"][5]["py"] - mano["landmarks"][17]["py"])**2 + (mano["landmarks"][5]["px"] - mano["landmarks"][17]["px"])**2)
                            
                            contadorDerecha += 1
                        
                        else:
                            mano_izquierda[0] += mano["landmarks"][5]["px"] - mano["landmarks"][17]["px"]
                            mano_izquierda[1] += mano["landmarks"][12]["py"] - mano["landmarks"][0]["py"]
                            mano_izquierda[2] += np.sqrt((mano["landmarks"][12]["py"] - mano["landmarks"][0]["py"])**2 + (mano["landmarks"][12]["px"] - mano["landmarks"][0]["px"])**2)
                            mano_izquierda[3] += np.sqrt((mano["landmarks"][4]["py"] - mano["landmarks"][6]["py"])**2 + (mano["landmarks"][4]["px"] - mano["landmarks"][6]["px"])**2)
                            mano_izquierda[4] += np.sqrt((mano["landmarks"][17]["py"] - mano["landmarks"][13]["py"])**2 + (mano["landmarks"][17]["px"] - mano["landmarks"][13]["px"])**2)
                            mano_izquierda[5] += np.sqrt((mano["landmarks"][4]["py"] - mano["landmarks"][8]["py"])**2 + (mano["landmarks"][4]["px"] - mano["landmarks"][8]["px"])**2)
                            mano_izquierda[6] += np.sqrt((mano["landmarks"][5]["py"] - mano["landmarks"][17]["py"])**2 + (mano["landmarks"][5]["px"] - mano["landmarks"][17]["px"])**2)
                            contadorIzquierda += 1



            else:
                

                for i in range(len(mano_derecha)):
                    if contadorDerecha > 0:
                        mano_derecha[i] /= contadorDerecha
                
                for i in range(len(mano_izquierda)):
                    if contadorIzquierda > 0:
                        mano_izquierda[i] /= contadorIzquierda
                
                calibrada = True
                sdl.widget("led1").value(True)
                sdl.widget("led2").value(False)

                contadorDerecha = 0
                contadorIzquierda = 0
                
                #para el recalibre
                primerframe = True     
              
def scoreAltoAlMenosUnaMano(manos):
    for mano in manos:
        if mano["score"] > SCORE_ALTO:
            return True
    return False

def recalibrar(sdl):
    global primerframe, segsinicio, mano_derecha, mano_izquierda, calibrada, contadorDerecha, contadorIzquierda

    mano_derecha = [0 for _ in mano_derecha]   
    mano_izquierda = [0 for _ in mano_izquierda]
    contadorDerecha = 0
    contadorIzquierda = 0

    if primerframe:
            segsinicio = sdl.elapsed() 
            primerframe = False
        
    # 3 segundos de calibre
    else:
        if not (sdl.elapsed() - segsinicio < 3.0): 
            primerframe = True
            sdl.widget("led1").value(False)
            calibrada = False
            sdl.widget("toggle1").value(False)

#argumento extra bool, para no calcular semidoblado cuando está de espaldas y evitar resultados raros
def doblado(sdl, manos, estado, a, b, c, d, e, constante, soloestirado):
    #si no esta de lado se calcula viendo el signo de la distancia entre la punta y el nudillo
    
    if (not estado["estaDeLado"]):

        #si está dada la vuelta comparar componente x e y por separado, comparar punta con intrafalange
        if estado["estaGiradaHorizontalmente"] and soloestirado:

            if estado["estaTumbada"]:
                distanciax = manos[0]["landmarks"][d]["px"] - manos[0]["landmarks"][e+1]["px"]
                if distanciax < 0 :
                    return True
                else:
                    return False

            else: 
                distanciay = manos[0]["landmarks"][d]["py"] - manos[0]["landmarks"][e]["py"]

                if not estado["estaGiradaVerticalmente"] and distanciay > 0:
                    return True
                elif  estado["estaGiradaVerticalmente"] and distanciay < 0:
                    return True
                else: 
                    return False
        else:
            distancia = manos[0]["landmarks"][d]["py"] - manos[0]["landmarks"][e-1]["py"]

            if(distancia > 0):
                return True
            else:
                return False

    #si está de lado depende del ángulo del nudillo    
    else:
        A = np.sqrt((manos[0]["landmarks"][a]["px"] - manos[0]["landmarks"][b]["px"])**2 + (manos[0]["landmarks"][a]["py"] - manos[0]["landmarks"][b]["py"])**2)
        B = np.sqrt((manos[0]["landmarks"][b]["px"] - manos[0]["landmarks"][c]["px"])**2 + (manos[0]["landmarks"][b]["py"] - manos[0]["landmarks"][c]["py"])**2)
        C = np.sqrt((manos[0]["landmarks"][c]["px"] - manos[0]["landmarks"][a]["px"])**2 + (manos[0]["landmarks"][c]["py"] - manos[0]["landmarks"][a]["py"])**2)

        #creo que da error si casualmente están en el mismo píxel, para evitar que se caiga el programa   
        if 2*A*B != 0:
            angulo = np.degrees(np.arccos(np.clip((A**2 + B**2 - C**2)/(2*A*B), -1.0, 1.0)))   #para evitar bug con precision de coma flotante, (1.00000002 da error en arccos) 

            if(angulo < constante):
                return True
            else:
                return False
            
        else:
            return False
        
def pulgarFuera(sdl, manos, estado):
    global mano_derecha, mano_izquierda
    distanciaActual = np.sqrt((manos[0]["landmarks"][4]["py"] - manos[0]["landmarks"][6]["py"])**2 + (manos[0]["landmarks"][4]["px"] - manos[0]["landmarks"][6]["px"])**2) 
    distanciaCalibre = mano_derecha[3] if manos[0]["handedness"] == "Right" else mano_izquierda[3] 

    if(distanciaCalibre * REDUCCION_LATERAL_PULGAR < distanciaActual and estado["estaDeFrente"]): #depende de la distancia con el dedo índice si no está de lado
        return True
    else:
        return False

def estaGiradaHorizontalmente(sdl, manos, estado): #posicion x muy cerca pero no y
    global  mano_derecha, mano_izquierda

    distanciaActual = manos[0]["landmarks"][5]["px"] - manos[0]["landmarks"][17]["px"] 
    distanciaCalibre = mano_derecha[0] if manos[0]["handedness"] == "Right" else mano_izquierda[0] 

    #si el resultado es negativo, significa que está girada, si fueran del mismo signo el resultado sería positivo
    if(distanciaActual < distanciaCalibre * REDUCCION_HORIZONTAL and not estado["estaDeLado"]):
        return True
    else:
        return False

def estaGiradaVerticalmente(sdl, manos, estado):
    global mano_derecha, mano_izquierda

    distanciaActual = manos[0]["landmarks"][12]["py"] - manos[0]["landmarks"][0]["py"]
    distanciaCalibre = mano_derecha[1] if manos[0]["handedness"] == "Right" else mano_izquierda[1] 

    #si el resultado es negativo, significa que está girada, si fueran del mismo signo el resultado sería positivo
    if(distanciaActual * distanciaCalibre < 0):
        return True
    else:
        return False

def estaDeLado(sdl, manos, estado):
    global mano_derecha, mano_izquierda

    distanciaActual = np.sqrt((manos[0]["landmarks"][5]["py"] - manos[0]["landmarks"][17]["py"])**2 + (manos[0]["landmarks"][5]["px"] - manos[0]["landmarks"][17]["px"])**2) 
    distanciaCalibre = mano_derecha[2] if manos[0]["handedness"] == "Right" else mano_izquierda[2] 

    if(distanciaActual < distanciaCalibre * REDUCCION_LADO):
        return True
    else:
        return False

#comparando las distancias x e y de los mismos puntos de la mano que estaDeLado, pero not estaDeLado
def estaTumbada(sdl, manos, estado):

    distanciaActualX = abs(manos[0]["landmarks"][5]["px"] - manos[0]["landmarks"][17]["px"])
    distanciaActualY = abs(manos[0]["landmarks"][5]["py"] - manos[0]["landmarks"][17]["py"]) 

    if(abs(distanciaActualY) > abs(distanciaActualX)):
        return True
    else:
        return False

def dedosPegados(sdl, manos, a, b, c, estado):
    global mano_derecha, mano_izquierda

    distanciaActual = np.sqrt((manos[0]["landmarks"][a]["py"] - manos[0]["landmarks"][b]["py"])**2 + (manos[0]["landmarks"][a]["px"] - manos[0]["landmarks"][b]["px"])**2) 
    distanciaCalibre = mano_derecha[c] if manos[0]["handedness"] == "Right" else mano_izquierda[c] 

    if(distanciaActual < distanciaCalibre * REDUCCION_MENIQUE_PULGAR and estado):
        return True
    else:
        return False

#para la R
def indiceCorazonCruzados(sdl, manos, estado):

    distanciaActual = manos[0]["landmarks"][8]["px"] - manos[0]["landmarks"][12]["px"]
    mano = manos[0]["handedness"] 

    if(estado["estaDeFrente"]):
        if(mano == "Right" and distanciaActual < 0):
            return True
        elif mano == "Left" and distanciaActual > 0:
            return True
        else:
            return False
    else:
        return False

#para la T
def indicePulgarCruzados(sdl, manos, estado):

    distanciaActual = manos[0]["landmarks"][8]["px"] - manos[0]["landmarks"][4]["px"]
    mano = manos[0]["handedness"] 

    if(estado["estaDeLado"]):
        if(mano == "Right" and distanciaActual > 0):
            return True
        elif mano == "Left" and distanciaActual < 0:
            return True
        else:
            return False
    else:
        return False

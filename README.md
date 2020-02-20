# RLC-Circuit
# A Python code that describes a RLC Circuit (series connection), with the goal to observe and have a better understanding regarding circuit analysis in both DC and AC regimen. However, do notice that in order for it to run, it's necessary to have numpy and matplotlib.

import math
import numpy as np
import matplotlib.pyplot as plt

Fonte = input("Entre com o tipo de fonte: ")

if Fonte == "DC":
    dt = 1 * pow(10, -6)
    tf = 0.1
    t = np.arange(0, tf, dt)
    Vin = float(input("Entre com a tensão em Volts: "))
    R = float(input("Entre com a resistência em ohms: "))
    Iin = Vin / R
    L = float(input("Entre com a indutância em mH: "))
    L = L * pow(10, -3)
    Xl = 2 * L / dt
    C = float(input("Entre com a capacitância em uF: "))
    C = C*pow(10, -6)
    Xc = dt / (2 * C)
    Vl = np.zeros(t.shape[0])
    Vc = np.zeros(t.shape[0])
    Vr = np.zeros(t.shape[0])
    Il = np.zeros(t.shape[0])
    Ic = np.zeros(t.shape[0])
    Ir = np.zeros(t.shape[0])
    Ileq = np.zeros(t.shape[0])
    Iceq = np.zeros(t.shape[0])

    for i in range(1, t.shape[0]):
        Iceq[i] = (-1 / Xc) * Vc[i - 1] - Ic[i-1]
        Ileq[i] = (1 / Xl) * Vl[i - 1] + Il[i-1]
        a = np.array([[(1 / R) + (1 / Xl), (-1 / Xl)], [(-1 / Xl), (1 / Xl) + (1 / Xc)]])
        b = np.array([[Iin - Ileq[i]], [Ileq[i] - Iceq[i]]])
        x = np.linalg.solve(a, b)
        Vl[i] = x[0] - x[1]
        Vc[i] = x[1]
        Il[i] = Ileq[i] + (Vl[i] / Xl)
        Ic[i] = Iceq[i] + (Vc[i] / Xc)


    plt.xlabel('Tempo (s)')
    plt.ylabel('Tensão (V)')
    plt.title('RLC Série Fonte DC')
    plt.plot(t, Vc)
    plt.plot(t, Vl)
    plt.show()
    plt.xlabel('Tempo (s)')
    plt.ylabel(' Corrente (A)')
    plt.title('RLC Série Fonte DC')
    plt.plot(t, Il)
    plt.plot(t, Ic)
    plt.show()

elif Fonte == "AC":
    dt = 50 * 10 ** (-6)
    tf = 0.1
    t = np.arange(0, tf, dt)
    f = float(input("Entre com a frequência em Hz: "))
    w = 2 * np.pi * f
    Vin = float(input("Entre com a amplitude da tensão em Volts: "))
    Vin = Vin * np.sqrt(2) * np.cos(w * t)
    R = float(input("Entre com a resistência em ohms: "))
    Iin = Vin / R
    L = float(input("Entre com a indutância em mH: "))
    L = L * 10 ** (-3)
    Xl = 2 * L / dt
    C = float(input("Entre com a capacitância em uF: "))
    C = C * 10 ** (-6)
    Xc = dt / (2 * C)
    Vl = np.zeros(t.shape[0])
    Vl2 = np.zeros(t.shape[0])
    Vc = np.zeros(t.shape[0])
    Vr = np.zeros(t.shape[0])
    Il = np.zeros(t.shape[0])
    Ic = np.zeros(t.shape[0])
    Ir = np.zeros(t.shape[0])
    Ileq = np.zeros(t.shape[0])
    Iceq = np.zeros(t.shape[0])

    for i in range(1, t.shape[0]):
        Iceq[i] = (-1 / Xc) * Vc[i-1] - Ic[i-1]
        Ileq[i] = (1 / Xl) * Vl[i-1] + Il[i-1]
        a = np.array([[(1 / R) + (1 / Xl), (-1 / Xl)], [(-1 / Xl), (1 / Xl) + (1 / Xc)]])
        b = np.array([[Iin[i] - Ileq[i]], [Ileq[i] - Iceq[i]]])
        x = np.linalg.solve(a, b)
        Vl[i] = x[0] - x[1]
        Vl2[i] = x[0]
        Vc[i] = x[1]
        Il[i] = Ileq[i] + (Vl[i] / Xl)
        Ic[i] = Iceq[i] + (Vc[i] / Xc)
        Ir[i] = -x[0]/R + Iin[i]
        Vr[i] = Ir[i] * R
        #print(Vin)
        #print(Vl)

    plt.ylabel('Tensão (V)')
    plt.xlabel('Tempo (s)')
    plt.title('RLC Série Fonte AC')
    plt.plot(t, Vc)
    plt.plot(t, Vl)
    plt.show()
    plt.xlabel('Tempo (s)')
    plt.ylabel(' Corrente (A)')
    plt.title('RLC Série Fonte AC')
    plt.plot(t, Il)
    plt.plot(t, Ic)
    plt.show()


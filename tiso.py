#!/usr/bin/python 

import argparse
import math
import cmath 
import numpy as np
import matplotlib.pyplot as plt

def tiso4avdet(z, B, Dz, E):
    x0 = E**2
    result = (2/27)*Dz**3 - 2/3*Dz*x0 + z**3 - z*(B**2 + (1/3)*Dz**2 + x0)
    return result

def tiso4w0(B, Dz, E):
    x0 = (2/3)*1j*math.pi
    x1 = Dz**3
    x2 = E**2
    x3 = 9*Dz*x2
    x4 = 1j*math.sqrt(-(x1 - x3)**2 + (3*B**2 + Dz**2 + 3*x2)**3)
    x5 = -x1 + x3
    result = (1/3)*(-x4 + x5)**(1/3)*cmath.exp(-x0) + (1/3)*(x4 + x5)**(1/3)*cmath.exp(x0)
    return result.real

def tiso4w1(B, Dz, E):
    x0 = (2/3)*1j*math.pi
    x1 = Dz**3
    x2 = E**2
    x3 = 9*Dz*x2
    x4 = 1j*math.sqrt(-(x1 - x3)**2 + (3*B**2 + Dz**2 + 3*x2)**3)
    x5 = -x1 + x3
    result = (1/3)*(-x4 + x5)**(1/3)*cmath.exp(x0) + (1/3)*(x4 + x5)**(1/3)*cmath.exp(-x0)
    return result.real

def tiso4w2(B, Dz, E):
    x0 = Dz**3
    x1 = E**2
    x2 = 9*Dz*x1
    x3 = 1j*math.sqrt(-(x0 - x2)**2 + (3*B**2 + Dz**2 + 3*x1)**3)
    x4 = -x0 + x2
    result = (1/3)*(-x3 + x4)**(1/3) + (1/3)*(x3 + x4)**(1/3)
    return result.real

def tiso4w0dE(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = 3*B**2 + x1 + 3*x2
    x5 = math.sqrt(-x1*(x1 - x3)**2 + x4**3)
    x6 = 1j*x5
    x7 = -Dz**3 + Dz*x3
    x8 = (x6 + x7)**(2/3)
    x9 = (-x6 + x7)**(2/3)
    x10 = 1j*(-x1*(-2*x1 + 18*x2) + x4**2)
    x11 = 2*Dz*x5
    result = E*x0*(x0*x9*(x10 + x11) + x8*(x10 - x11))/(x5*x8*x9)
    return result.real

def tiso4w0dDz(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*x2
    x6 = 3*B**2 + x1 + x5
    x7 = math.sqrt(-x1*x4**2 + x6**3)
    x8 = 1j*x7
    x9 = -Dz**3 + Dz*x3
    x10 = (x8 + x9)**(2/3)
    x11 = (-x8 + x9)**(2/3)
    x12 = x1 - x5
    x13 = 1j*Dz*(x12*x4 - x6**2)
    x14 = x12*x7
    result = -1/3*x0*(x0*x11*(x13 + x14) + x10*(x13 - x14))/(x10*x11*x7)
    return result.real

def tiso4w1dE(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*B**2 + x1 + 3*x2
    x6 = math.sqrt(-x1*x4**2 + x5**3)
    x7 = 1j*x6
    x8 = -Dz**3 + Dz*x3
    x9 = (x7 + x8)**(2/3)
    x10 = (-x7 + x8)**(2/3)
    x11 = 1j*(2*x1*x4 + x5**2)
    x12 = 2*Dz*x6
    result = -E*x0*(-x0*x9*(-x11 + x12) + x10*(x11 + x12))/(x10*x6*x9)
    return result.real

def tiso4w1dDz(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*x2
    x6 = 3*B**2 + x1 + x5
    x7 = math.sqrt(-x1*x4**2 + x6**3)
    x8 = 1j*x7
    x9 = -Dz**3 + Dz*x3
    x10 = (x8 + x9)**(2/3)
    x11 = (-x8 + x9)**(2/3)
    x12 = x1 - x5
    x13 = -1j*Dz*(x12*x4 - x6**2)
    x14 = x12*x7
    result = -1/3*x0*(x0*x10*(x13 + x14) + x11*(x13 - x14))/(x10*x11*x7)
    return result.real

def tiso4w2dE(B, Dz, E):
    x0 = Dz**2
    x1 = E**2
    x2 = 9*x1
    x3 = 3*B**2 + x0 + 3*x1
    x4 = math.sqrt(-x0*(x0 - x2)**2 + x3**3)
    x5 = 1j*x4
    x6 = -Dz**3 + Dz*x2
    x7 = (x5 + x6)**(2/3)
    x8 = (-x5 + x6)**(2/3)
    x9 = 2*x0
    x10 = 18*x1
    x11 = x3**2
    x12 = 2*Dz*x4
    result = E*(x7*(x12 + 1j*(x0*(x10 - x9) - x11)) + x8*(x12 + 1j*(x0*(-x10 + x9) + x11)))/(x4*x7*x8)
    return result.real

def tiso4w2dDz(B, Dz, E):
    x0 = Dz**2
    x1 = E**2
    x2 = 9*x1
    x3 = x0 - x2
    x4 = 3*x1
    x5 = 3*B**2 + x0 + x4
    x6 = math.sqrt(-x0*x3**2 + x5**3)
    x7 = 1j*x6
    x8 = -Dz**3 + Dz*x2
    x9 = (x7 + x8)**(2/3)
    x10 = (-x7 + x8)**(2/3)
    x11 = x5**2
    x12 = x3*(x0 - x4)
    x13 = 1j*Dz
    x14 = x6*(-x0 + x4)
    result = (1/3)*(x10*(x13*(x11 - x12) + x14) + x9*(x13*(-x11 + x12) + x14))/(x10*x6*x9)
    return result.real

def tiso4utot(u1, u2, B, Dz, E):
    x0 = u1 + 2*u2
    x1 = x0**3
    x2 = 2*u1 + u2
    x3 = x2**3
    x4 = Dz**2
    x5 = E**2
    x6 = 2*x0**2
    x7 = 2*x2**2
    result = (16/189)*B**8*(Dz**4 + 9*E**4 + 6*x4*x5)*(x0*x7 + x1 + x2*x6 + x3)/(x0**7*x2**7) + (16/945)*B**6*Dz*(x4 - 9*x5)*(3*x0*x2 + x6 + x7)/(x0**5*x2**5) + (4/15)*B**4*(u1 + u2)*(x4 + 3*x5)/(x1*x3) - u1 - u2
    return result.real

def tiso4utotdu1(u1, u2, B, Dz, E):
    x0 = u1 + 2*u2
    x1 = x0**8
    x2 = 2*u1 + u2
    x3 = x2**8
    x4 = x2**5
    x5 = B**4
    x6 = Dz**2
    x7 = E**2
    x8 = x6 + 3*x7
    x9 = x0**5*x5*x8
    x10 = u1 + u2
    x11 = B**6
    x12 = x0**3
    x13 = x2**3
    x14 = x6 - 9*x7
    x15 = x0**2
    x16 = 2*x15
    x17 = x2**2
    x18 = 2*x17
    x19 = x0*x2
    x20 = Dz*x11*x14*(x16 + x18 + 3*x19)
    x21 = B**8*(Dz**4 + 9*E**4 + 6*x6*x7)
    x22 = x21*(x0*x18 + x12 + x13 + x16*x2)
    result = (1/945)*(16*Dz*x11*x12*x13*x14*(32*u1 + 31*u2) - 756*x0**4*x10*x4*x5*x8 - 1120*x0*x22 - 945*x1*x3 - 1512*x10*x2**4*x9 - 160*x12*x17*x20 - 80*x13*x15*x20 + 80*x19*x21*(7*x15 + 8*x17 + 12*x19) - 560*x2*x22 + 252*x4*x9)/(x1*x3)
    return result

def tiso4utotdE(u1, u2, B, Dz, E):
    x0 = B**4
    x1 = u1 + 2*u2
    x2 = 2*u1 + u2
    x3 = x1**2
    x4 = x2**2
    x5 = 2*x3
    x6 = 2*x4
    result = (8/315)*E*x0*(-12*B**2*Dz*x3*x4*(3*x1*x2 + x5 + x6) + 40*x0*(Dz**2 + 3*E**2)*(x1**3 + x1*x6 + x2**3 + x2*x5) + 63*x1**4*x2**4*(u1 + u2))/(x1**7*x2**7)
    return result

def tiso4utotdDz(u1, u2, B, Dz, E):
    x0 = B**4
    x1 = u1 + 2*u2
    x2 = 2*u1 + u2
    x3 = Dz**2
    x4 = 3*E**2
    x5 = x2**2
    x6 = 2*x5
    x7 = x1**2
    x8 = 2*x7
    result = (8/945)*x0*(6*B**2*x5*x7*(x3 - x4)*(3*x1*x2 + x6 + x8) + 40*Dz*x0*(x3 + x4)*(x1**3 + x1*x6 + x2**3 + x2*x8) + 63*Dz*x1**4*x2**4*(u1 + u2))/(x1**7*x2**7)
    return result


def tisoavdet(z, B, Dz, E):
    x0 = E**2
    result = (2/27)*Dz**3 - 2/3*Dz*x0 + z**3 - z*(B**2 + (1/3)*Dz**2 + x0)
    return result

def tisow0(B, Dz, E):
    x0 = (2/3)*1j*math.pi
    x1 = Dz**3
    x2 = E**2
    x3 = 9*Dz*x2
    x4 = 1j*math.sqrt(-(x1 - x3)**2 + (3*B**2 + Dz**2 + 3*x2)**3)
    x5 = -x1 + x3
    result = (1/3)*(-x4 + x5)**(1/3)*cmath.exp(-x0) + (1/3)*(x4 + x5)**(1/3)*cmath.exp(x0)
    return result.real

def tisow1(B, Dz, E):
    x0 = (2/3)*1j*math.pi
    x1 = Dz**3
    x2 = E**2
    x3 = 9*Dz*x2
    x4 = 1j*math.sqrt(-(x1 - x3)**2 + (3*B**2 + Dz**2 + 3*x2)**3)
    x5 = -x1 + x3
    result = (1/3)*(-x4 + x5)**(1/3)*cmath.exp(x0) + (1/3)*(x4 + x5)**(1/3)*cmath.exp(-x0)
    return result.real

def tisow2(B, Dz, E):
    x0 = Dz**3
    x1 = E**2
    x2 = 9*Dz*x1
    x3 = 1j*math.sqrt(-(x0 - x2)**2 + (3*B**2 + Dz**2 + 3*x1)**3)
    x4 = -x0 + x2
    result = (1/3)*(-x3 + x4)**(1/3) + (1/3)*(x3 + x4)**(1/3)
    return result.real

def tisow0dE(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = 3*B**2 + x1 + 3*x2
    x5 = math.sqrt(-x1*(x1 - x3)**2 + x4**3)
    x6 = 1j*x5
    x7 = -Dz**3 + Dz*x3
    x8 = (x6 + x7)**(2/3)
    x9 = (-x6 + x7)**(2/3)
    x10 = 1j*(-x1*(-2*x1 + 18*x2) + x4**2)
    x11 = 2*Dz*x5
    result = E*x0*(x0*x9*(x10 + x11) + x8*(x10 - x11))/(x5*x8*x9)
    return result.real

def tisow0dDz(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*x2
    x6 = 3*B**2 + x1 + x5
    x7 = math.sqrt(-x1*x4**2 + x6**3)
    x8 = 1j*x7
    x9 = -Dz**3 + Dz*x3
    x10 = (x8 + x9)**(2/3)
    x11 = (-x8 + x9)**(2/3)
    x12 = x1 - x5
    x13 = 1j*Dz*(x12*x4 - x6**2)
    x14 = x12*x7
    result = -1/3*x0*(x0*x11*(x13 + x14) + x10*(x13 - x14))/(x10*x11*x7)
    return result.real

def tisow1dE(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*B**2 + x1 + 3*x2
    x6 = math.sqrt(-x1*x4**2 + x5**3)
    x7 = 1j*x6
    x8 = -Dz**3 + Dz*x3
    x9 = (x7 + x8)**(2/3)
    x10 = (-x7 + x8)**(2/3)
    x11 = 1j*(2*x1*x4 + x5**2)
    x12 = 2*Dz*x6
    result = -E*x0*(-x0*x9*(-x11 + x12) + x10*(x11 + x12))/(x10*x6*x9)
    return result.real

def tisow1dDz(B, Dz, E):
    x0 = (-1)**(1/3)
    x1 = Dz**2
    x2 = E**2
    x3 = 9*x2
    x4 = x1 - x3
    x5 = 3*x2
    x6 = 3*B**2 + x1 + x5
    x7 = math.sqrt(-x1*x4**2 + x6**3)
    x8 = 1j*x7
    x9 = -Dz**3 + Dz*x3
    x10 = (x8 + x9)**(2/3)
    x11 = (-x8 + x9)**(2/3)
    x12 = x1 - x5
    x13 = -1j*Dz*(x12*x4 - x6**2)
    x14 = x12*x7
    result = -1/3*x0*(x0*x10*(x13 + x14) + x11*(x13 - x14))/(x10*x11*x7)
    return result.real

def tisow2dE(B, Dz, E):
    x0 = Dz**2
    x1 = E**2
    x2 = 9*x1
    x3 = 3*B**2 + x0 + 3*x1
    x4 = math.sqrt(-x0*(x0 - x2)**2 + x3**3)
    x5 = 1j*x4
    x6 = -Dz**3 + Dz*x2
    x7 = (x5 + x6)**(2/3)
    x8 = (-x5 + x6)**(2/3)
    x9 = 2*x0
    x10 = 18*x1
    x11 = x3**2
    x12 = 2*Dz*x4
    result = E*(x7*(x12 + 1j*(x0*(x10 - x9) - x11)) + x8*(x12 + 1j*(x0*(-x10 + x9) + x11)))/(x4*x7*x8)
    return result.real

def tisow2dDz(B, Dz, E):
    x0 = Dz**2
    x1 = E**2
    x2 = 9*x1
    x3 = x0 - x2
    x4 = 3*x1
    x5 = 3*B**2 + x0 + x4
    x6 = math.sqrt(-x0*x3**2 + x5**3)
    x7 = 1j*x6
    x8 = -Dz**3 + Dz*x2
    x9 = (x7 + x8)**(2/3)
    x10 = (-x7 + x8)**(2/3)
    x11 = x5**2
    x12 = x3*(x0 - x4)
    x13 = 1j*Dz
    x14 = x6*(-x0 + x4)
    result = (1/3)*(x10*(x13*(x11 - x12) + x14) + x9*(x13*(-x11 + x12) + x14))/(x10*x6*x9)
    return result.real

def tisoutot(u1, u2, B, Dz, E):
    x0 = u1 + 2*u2
    x1 = x0**3
    x2 = 2*u1 + u2
    x3 = x2**3
    x4 = Dz**2
    x5 = E**2
    x6 = 3*x5
    x7 = x0**5
    x8 = x2**5
    x9 = x0**2
    x10 = 2*x9
    x11 = x2**2
    x12 = 2*x11
    x13 = x0**7
    x14 = x2**7
    x15 = Dz**4
    x16 = E**4
    x17 = 6*x4*x5
    x18 = x0**9
    x19 = x2**9
    x20 = x0**4
    x21 = x2**4
    x22 = Dz**6
    x23 = E**6
    x24 = x16*x4
    x25 = x0**6
    x26 = x2**6
    x27 = Dz**8
    x28 = E**8
    x29 = x23*x4
    x30 = x15*x16
    x31 = x22*x5
    x32 = x0**8
    x33 = x2**8
    result = (1024/2119203)*B**20*(161*Dz**10 + 19683*E**10 + 47790*x15*x23 + 5850*x16*x22 + 255*x27*x5 + 91125*x28*x4)*(170*x0*x33 + 770*x1*x26 + 440*x11*x13 + 440*x14*x9 + 34*x18 + 34*x19 + 170*x2*x32 + 1001*x20*x8 + 1001*x21*x7 + 770*x25*x3)/(x0**19*x2**19) + (20480/82648917)*B**18*Dz*(197*x27 - 45927*x28 - 46656*x29 - 8262*x30 - 216*x31)*(117*x0*x14 + 429*x1*x8 + 273*x11*x25 + 117*x13*x2 + 495*x20*x21 + 273*x26*x9 + 429*x3*x7 + 26*x32 + 26*x33)/(x0**17*x2**17) + (256/483327)*B**16*(95*x27 + 5103*x28 + 14580*x29 + 4266*x30 + 276*x31)*(52*x0*x26 + 150*x1*x21 + 108*x11*x7 + 13*x13 + 13*x14 + 52*x2*x25 + 150*x20*x3 + 108*x8*x9)/(x0**15*x2**15) + (512/34749)*B**14*Dz*(-x15*x6 + x22 - 81*x23 - 45*x24)*(77*x0*x8 + 168*x1*x3 + 140*x11*x20 + 77*x2*x7 + 140*x21*x9 + 22*x25 + 22*x26)/(x0**13*x2**13) + (128/104247)*B**12*(261*x15*x5 + 53*x22 + 1215*x23 + 1863*x24)*(9*x0*x21 + 14*x1*x11 + 9*x2*x20 + 14*x3*x9 + 3*x7 + 3*x8)/(x0**11*x2**11) + (128/18711)*B**10*Dz*(x15 - 27*x16 - x17)*(35*x0*x3 + 35*x1*x2 + 45*x11*x9 + 14*x20 + 14*x21)/(x18*x19) + (16/189)*B**8*(x15 + 9*x16 + x17)*(x0*x12 + x1 + x10*x2 + x3)/(x13*x14) + (16/945)*B**6*Dz*(x4 - 9*x5)*(3*x0*x2 + x10 + x12)/(x7*x8) + (4/15)*B**4*(u1 + u2)*(x4 + x6)/(x1*x3) - u1 - u2
    return result.real

def tisoutotdu1(u1, u2, B, Dz, E):
    x0 = u1 + 2*u2
    x1 = x0**20
    x2 = 2*u1 + u2
    x3 = x2**20
    x4 = x2**17
    x5 = B**4
    x6 = Dz**2
    x7 = E**2
    x8 = 3*x7
    x9 = x6 + x8
    x10 = x0**17*x5*x9
    x11 = u1 + u2
    x12 = B**6
    x13 = x0**15
    x14 = x2**15
    x15 = x6 - 9*x7
    x16 = x0**2
    x17 = 2*x16
    x18 = x2**2
    x19 = 2*x18
    x20 = x0*x2
    x21 = Dz*x12*x15*(x17 + x19 + 3*x20)
    x22 = B**8
    x23 = x0**13
    x24 = x2**13
    x25 = Dz**4
    x26 = E**4
    x27 = 6*x6*x7
    x28 = x25 + 9*x26 + x27
    x29 = x0**3
    x30 = x2**3
    x31 = x22*x28*(x0*x19 + x17*x2 + x29 + x30)
    x32 = x0**11
    x33 = 100*x0
    x34 = B**10
    x35 = -x25 + 27*x26 + x27
    x36 = Dz*x2**11*x34*x35
    x37 = x0**4
    x38 = x2**4
    x39 = x2*x29
    x40 = x16*x18
    x41 = 35*x0*x30 + 14*x37 + 14*x38 + 35*x39 + 45*x40
    x42 = B**12
    x43 = x0**9
    x44 = x2**9
    x45 = Dz**6
    x46 = E**6
    x47 = x26*x6
    x48 = 261*x25*x7 + 53*x45 + 1215*x46 + 1863*x47
    x49 = x0**8
    x50 = x0**5
    x51 = x2**5
    x52 = x0*x38
    x53 = x2*x37
    x54 = x16*x30
    x55 = x18*x29
    x56 = x42*x48*(3*x50 + 3*x51 + 9*x52 + 9*x53 + 14*x54 + 14*x55)
    x57 = x2**8
    x58 = x0**7
    x59 = x2**7
    x60 = B**14
    x61 = x25*x8 - x45 + 81*x46 + 45*x47
    x62 = Dz*x59*x60*x61
    x63 = x0**6
    x64 = x2**6
    x65 = x0*x51
    x66 = x2*x50
    x67 = x16*x38
    x68 = x29*x30
    x69 = x18*x37
    x70 = 22*x63 + 22*x64 + 77*x65 + 77*x66 + 140*x67 + 168*x68 + 140*x69
    x71 = Dz**8
    x72 = E**8
    x73 = x46*x6
    x74 = x25*x26
    x75 = x45*x7
    x76 = B**16*(95*x71 + 5103*x72 + 14580*x73 + 4266*x74 + 276*x75)
    x77 = x37*x51
    x78 = x0*x64
    x79 = x2*x63
    x80 = x16*x51
    x81 = x29*x38
    x82 = x30*x37
    x83 = x18*x50
    x84 = x76*(13*x58 + 13*x59 + 52*x78 + 52*x79 + 108*x80 + 150*x81 + 150*x82 + 108*x83)
    x85 = x38*x50
    x86 = B**18*Dz*(-197*x71 + 45927*x72 + 46656*x73 + 8262*x74 + 216*x75)
    x87 = x0*x59
    x88 = x2*x58
    x89 = x16*x64
    x90 = x29*x51
    x91 = x37*x38
    x92 = x30*x50
    x93 = x18*x63
    x94 = x86*(26*x49 + 26*x57 + 117*x87 + 117*x88 + 273*x89 + 429*x90 + 495*x91 + 429*x92 + 273*x93)
    x95 = B**20*(161*Dz**10 + 19683*E**10 + 47790*x25*x46 + 5850*x26*x45 + 91125*x6*x72 + 255*x7*x71)
    x96 = x95*(170*x0*x57 + 440*x16*x59 + 440*x18*x58 + 170*x2*x49 + 770*x29*x64 + 770*x30*x63 + 34*x43 + 34*x44 + 1001*x77 + 1001*x85)
    result = (1/31819833045)*(538748496*Dz*x12*x13*x14*x15*(32*u1 + 31*u2) + 3918170880*Dz*x2**10*x32*x34*x35*x41 + 12189864960*Dz*x58*x60*x61*x64*x70 - 25455866436*x0**16*x11*x4*x5*x9 - 2693742480*x0**14*x14*x21 - 18856197360*x0**12*x24*x31 + 1959085440*x0**10*x36*x41 - 584263680*x0*x96 - 31819833045*x1*x3 - 50911732872*x10*x11*x2**16 + 8485288812*x10*x4 - 5387484960*x13*x2**14*x21 - 37712394720*x2**12*x23*x31 - 292131840*x2*x96 + 15375360*x20*x95*(646*x49 + 782*x57 + 3600*x87 + 3120*x88 + 8470*x89 + 13244*x90 + 15015*x91 + 12628*x92 + 7700*x93) + 2693742480*x22*x23*x24*x28*(7*x16 + 8*x18 + 12*x20) - 653028480*x32*x36*(95*x16*x2 + x18*x33 + 42*x29 + 49*x30) + 39070080*x42*x43*x44*x48*(x30*x33 + 33*x37 + 39*x38 + 92*x39 + 126*x40) - 859541760*x43*x56*x57 - 429770880*x44*x49*x56 + 50561280*x50*x51*x76*(65*x63 + 78*x64 + 280*x65 + 248*x66 + 510*x67 + 600*x68 + 480*x69) + 134041600*x54*x94 + 268083200*x55*x94 - 468840960*x58*x62*(286*x50 + 341*x51 + 1050*x52 + 945*x53 + 1624*x54 + 1568*x55) + 6094932480*x62*x63*x70 - 7884800*x68*x86*(442*x58 + 533*x59 + 2184*x78 + 1911*x79 + 4563*x80 + 6270*x81 + 6105*x82 + 4212*x83) - 252806400*x77*x84 - 505612800*x84*x85)/(x1*x3)
    return result

def tisoutotdE(u1, u2, B, Dz, E):
    x0 = B**4
    x1 = u1 + 2*u2
    x2 = 2*u1 + u2
    x3 = x1**2
    x4 = 2*x3
    x5 = x2**2
    x6 = 2*x5
    x7 = Dz**2
    x8 = E**2
    x9 = x1**3
    x10 = x2**3
    x11 = x1**4
    x12 = x2**4
    x13 = x3*x5
    x14 = x1**8
    x15 = x2**8
    x16 = Dz**4
    x17 = E**4
    x18 = x7*x8
    x19 = x1**5
    x20 = x2**5
    x21 = x1**6
    x22 = x2**6
    x23 = Dz**6
    x24 = E**6
    x25 = 1701*x24
    x26 = x17*x7
    x27 = x16*x8
    x28 = x1**7
    x29 = x2**7
    x30 = x11*x12
    result = (8/3535537005)*E*x0*(6406400*B**16*(17*Dz**8 + 6561*E**8 + 9558*x16*x17 + 780*x23*x8 + 24300*x24*x7)*(34*x1**9 + 170*x1*x15 + 770*x10*x21 + 1001*x11*x20 + 1001*x12*x19 + 170*x14*x2 + 34*x2**9 + 770*x22*x9 + 440*x28*x5 + 440*x29*x3) - 23654400*B**14*Dz*x13*(2*x23 + x25 + 1296*x26 + 153*x27)*(117*x1*x29 + 429*x10*x19 + 26*x14 + 26*x15 + 117*x2*x28 + 429*x20*x9 + 273*x21*x5 + 273*x22*x3 + 495*x30) + 5617920*B**12*x30*(23*x23 + x25 + 3645*x26 + 711*x27)*(52*x1*x22 + 150*x10*x11 + 150*x12*x9 + 108*x19*x5 + 52*x2*x21 + 108*x20*x3 + 13*x28 + 13*x29) - 39070080*B**10*Dz*x21*x22*(x16 + 81*x17 + 30*x18)*(77*x1*x20 + 168*x10*x9 + 140*x11*x5 + 140*x12*x3 + 77*x19*x2 + 22*x21 + 22*x22) + 9767520*B**8*x14*x15*(29*x16 + 405*x17 + 414*x18)*(9*x1*x12 + 14*x10*x3 + 9*x11*x2 + 3*x19 + 3*x20 + 14*x5*x9) - 36279360*B**6*Dz*x1**10*x2**10*(x7 + 9*x8)*(35*x1*x10 + 14*x11 + 14*x12 + 45*x13 + 35*x2*x9) - 134687124*B**2*Dz*x1**14*x2**14*(3*x1*x2 + x4 + x6) + 448957080*x0*x1**12*x2**12*(x7 + 3*x8)*(x1*x6 + x10 + x2*x4 + x9) + 707107401*x1**16*x2**16*(u1 + u2))/(x1**19*x2**19)
    return result

def tisoutotdDz(u1, u2, B, Dz, E):
    x0 = B**4
    x1 = u1 + 2*u2
    x2 = 2*u1 + u2
    x3 = Dz**2
    x4 = E**2
    x5 = 3*x4
    x6 = x3 - x5
    x7 = x1**2
    x8 = 2*x7
    x9 = x2**2
    x10 = 2*x9
    x11 = x1**3
    x12 = x2**3
    x13 = Dz**4
    x14 = E**4
    x15 = x3*x4
    x16 = x1**4
    x17 = x2**4
    x18 = x7*x9
    x19 = x1**8
    x20 = x2**8
    x21 = x1**5
    x22 = x2**5
    x23 = Dz**6
    x24 = E**6
    x25 = x14*x3
    x26 = x13*x4
    x27 = x1**7
    x28 = x2**7
    x29 = x2**6
    x30 = x1**6
    x31 = x16*x17
    x32 = Dz**8
    x33 = E**8
    x34 = x24*x3
    x35 = x13*x14
    x36 = x23*x4
    result = (8/31819833045)*x0*(19219200*B**16*Dz*(161*x32 + 18225*x33 + 19116*x34 + 3510*x35 + 204*x36)*(34*x1**9 + 170*x1*x20 + 770*x11*x29 + 770*x12*x30 + 1001*x16*x22 + 1001*x17*x21 + 170*x19*x2 + 34*x2**9 + 440*x27*x9 + 440*x28*x7) + 985600*B**14*x18*(-8*x3*(-197*x23 + 11664*x24 + 4131*x25 + 162*x26) + 197*x32 - 45927*x33 - 46656*x34 - 8262*x35 - 216*x36)*(117*x1*x28 + 429*x11*x22 + 429*x12*x21 + 26*x19 + 117*x2*x27 + 26*x20 + 273*x29*x7 + 273*x30*x9 + 495*x31) + 16853760*B**12*Dz*x31*(95*x23 + 3645*x24 + 2133*x25 + 207*x26)*(52*x1*x29 + 150*x11*x17 + 150*x12*x16 + 52*x2*x30 + 108*x21*x9 + 108*x22*x7 + 13*x27 + 13*x28) + 58605120*B**10*x29*x30*(-x13*x5 + x23 - 81*x24 - 45*x25 - 6*x3*(-x13 + 15*x14 + 2*x15))*(77*x1*x22 + 168*x11*x12 + 140*x16*x9 + 140*x17*x7 + 77*x2*x21 + 22*x29 + 22*x30) + 29302560*B**8*Dz*x19*x20*(53*x13 + 621*x14 + 174*x15)*(9*x1*x17 + 14*x11*x9 + 14*x12*x7 + 9*x16*x2 + 3*x21 + 3*x22) + 27209520*B**6*x1**10*x2**10*(x13 - 27*x14 - 6*x15 + 4*x3*x6)*(35*x1*x12 + 35*x11*x2 + 14*x16 + 14*x17 + 45*x18) + 202030686*B**2*x1**14*x2**14*x6*(3*x1*x2 + x10 + x8) + 1346871240*Dz*x0*x1**12*x2**12*(x3 + x5)*(x1*x10 + x11 + x12 + x2*x8) + 2121322203*Dz*x1**16*x2**16*(u1 + u2))/(x1**19*x2**19)
    return result


def alambda0(B, Dz, E):
    return tisoutot( tisow1(B, Dz, E), tisow2(B, Dz, E), B, Dz, E)
def alambda1(B, Dz, E):
    return tiso4utot( tiso4w0(B, Dz, E), tiso4w2(B, Dz, E), B, Dz, E)
def alambda2(B, Dz, E):
    return tiso4utot( tiso4w0(B, Dz, E), tiso4w1(B, Dz, E), B, Dz, E)

def dlambda0dDz(B, Dz, E):
    u1 = tisow1(B, Dz, E)
    u2 = tisow2(B, Dz, E)
    return tisoutotdDz(u1, u2, B, Dz, E) + tisoutotdu1(u1, u2, B, Dz, E) * tisow1dDz(B, Dz, E) +  tisoutotdu1(u2, u1, B, Dz, E) * tisow2dDz(B, Dz, E)
def dlambda1dDz(B, Dz, E):
    u1 = tiso4w0(B, Dz, E)
    u2 = tiso4w2(B, Dz, E)
    return tiso4utotdDz(u1, u2, B, Dz, E) + tiso4utotdu1(u1, u2, B, Dz, E) * tiso4w0dDz(B, Dz, E) +  tiso4utotdu1(u2, u1, B, Dz, E) * tiso4w2dDz(B, Dz, E)
def dlambda2dDz(B, Dz, E):
    u1 = tiso4w0(B, Dz, E)
    u2 = tiso4w1(B, Dz, E)
    return tiso4utotdDz(u1, u2, B, Dz, E) + tiso4utotdu1(u1, u2, B, Dz, E) * tiso4w0dDz(B, Dz, E) +  tiso4utotdu1(u2, u1, B, Dz, E) * tiso4w1dDz(B, Dz, E)


def dlambda0dE(B, Dz, E):
    u1 = tisow1(B, Dz, E)
    u2 = tisow2(B, Dz, E)
    return tisoutotdE(u1, u2, B, Dz, E) + tisoutotdu1(u1, u2, B, Dz, E) * tisow1dE(B, Dz, E) +  tisoutotdu1(u2, u1, B, Dz, E) * tisow2dE(B, Dz, E)

def dlambda1dE(B, Dz, E):
    u1 = tiso4w0(B, Dz, E)
    u2 = tiso4w2(B, Dz, E)
    return tiso4utotdE(u1, u2, B, Dz, E) + tiso4utotdu1(u1, u2, B, Dz, E) * tiso4w0dE(B, Dz, E) +  tiso4utotdu1(u2, u1, B, Dz, E) * tiso4w2dE(B, Dz, E)

def dlambda2dE(B, Dz, E):
    u1 = tiso4w0(B, Dz, E)
    u2 = tiso4w1(B, Dz, E)
    return tiso4utotdE(u1, u2, B, Dz, E) + tiso4utotdu1(u1, u2, B, Dz, E) * tiso4w0dE(B, Dz, E) +  tiso4utotdu1(u2, u1, B, Dz, E) * tiso4w1dE(B, Dz, E)


def tisoE(n, B, Dz, E):
    if n == 0:
        return alambda0(B, Dz, E)
    if n == 1:
        return alambda1(B, Dz, E)
    if n == 2:
        return alambda2(B, Dz, E)


def tisoZ(n, B, Dz, E):
    if n == 0:
        return dlambda0dDz(B, Dz, E)
    if n == 1:
        return dlambda1dDz(B, Dz, E)
    if n == 2:
        return dlambda2dDz(B, Dz, E)
    

def tisoXY(n, B, Dz, E):
    if n == 0:
        return dlambda0dE(B, Dz, E)
    if n == 1:
        return dlambda1dE(B, Dz, E)
    if n == 2:
        return dlambda2dE(B, Dz, E)

parser = argparse.ArgumentParser(
                description='prints triplet averages '
                )    
parser.add_argument("-D", "--D", default=1, type=float, help="D value")
parser.add_argument("-E", "--E", default=0, type=float, help="E value")

args = parser.parse_args()
Dz = getattr(args, "D")
E = getattr(args, "E")

Dz = 1
E  = 0.05

tiso_arr = []

for B in np.arange(0.001, 6, 0.01):
    tiso_list = [
        B,
        *[tisoE(i, B, Dz, E) for i in range(3)],
        *[tisoZ(i, B, Dz, E) for i in range(3)],
        *[tisoXY(i, B, Dz, E) for i in range(3)]
    ]

    print(f"{B} {tisoE(0, B, Dz, E)} {tisoE(1, B, Dz, E)} {tisoE(2, B, Dz, E)} {tisoZ(0, B, Dz, E)} {tisoZ(1, B, Dz, E)} {tisoZ(2, B, Dz, E)} {tisoXY(0, B, Dz, E)} {tisoXY(1, B, Dz, E)} {tisoXY(2, B, Dz, E)}")
    # Append the row as a NumPy array
    tiso_arr.append(tiso_list)



# Convert to final 2D NumPy array
tiso_arr = np.array(tiso_arr)

fig=plt.figure()
plt.plot(tiso_arr[:,0],tiso_arr[:,4], label='z0_B')
plt.plot(tiso_arr[:,0],tiso_arr[:,5], label='z1_B')
plt.plot(tiso_arr[:,0],tiso_arr[:,6], label='z2_B')
plt.plot(tiso_arr[:,0],tiso_arr[:,7], label='xy0_B')
#plt.plot(tiso_arr[:,0],tiso_arr[:,8], label='xy1_B')
#plt.plot(tiso_arr[:,0],tiso_arr[:,9], label='xy2_B')

plt.legend()

plt.show()
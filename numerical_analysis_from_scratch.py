#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:42:32 2021

@author: Sofiane SIFAOUI - P2B (TP3)
"""

#==========================EXERCICE 1=========================================
print("\n EXERCICE 1 : Dérivation numérique \n")

from math import sqrt, cos, sin, exp

def d_approx1(f,x:float,h:float)->float:
    """
    Prend en argument une fonction f, un flottant h et un flottant x
    et renvoie la quantité f^1(x,h) = (f(x+h)-f(x-h))/2h qui permet d'approcher numériquement f'(x)
    """
    return (f(x+h)-f(x-h))/(2*h)

def d_approx3(f,x:float,h:float)->float:
    """
    Prend en argument une fonction f, un flottant h et un flottant x
    et renvoie la quantité 4/3*f^1(x,h) - 1/3f^1(x,2h) qui permet d'approcher numériquement f'(x) avec une précision supérieure, ici en o(h³)
    """
    return 4/3*(f(x+h)-f(x-h))/(2*h) - 1/3*(f(x+2*h)-f(x-2*h))/(2*2*h)

def f0(x:float)->float:
    """
    x appartient à [0,10]
    """
    return sqrt(x)+cos(x)

#le calcul à la main de f' donne, si x appartient à [0,10] ouvert à gauche, f'(x) = 1/2sqrt(x) - sin(x)
    
def f0_derivée(x:float)->float:
    """
    Dérivée de la fonction f0
    """
    return 1/(2*sqrt(x))-sin(x)

import matplotlib.pyplot as plt
import numpy as np

X = [x for x in np.arange(0.1,10,0.01)] #on commence à 0.1 puisque la dérivée n'est pas définie en 0
Y = [f0_derivée(x) for x in X]
plt.plot(X,Y)
plt.title("Courbe représentative de la fonction f0_derivée")
plt.grid()
plt.show()

Y1 = [d_approx1(f0,x,h=0.05) for x in X] # on choisit un pas h valant 0.05
plt.plot(X,Y1,color="g")
plt.title("Courbe représentative de la fonction d_approx1 en vert")
plt.grid()
plt.show()

Y2 = [d_approx3(f0,x,h=0.05) for x in X]
plt.plot(X,Y2,color="r")
plt.title("Courbe représentative de la fonction d_approx3 en rouge")
plt.grid()
plt.show()

#Conclusion : On constate que l'on obtient des courbes extrêmement similaires lorsque l'on représente la dérivée exacte (calculée à la main)
# ou lorsque l'on représente la dérivée numérique, évaluée grâce à la quantité f^1(x,h). Evidemment, ici il n'est pas pertinent de superposer les 
# trois courbes obtenues car celles-ci se recouvreraient.


#==============================EXERCICE 2====================================
print("\n EXERCICE 2 : Méthode de Newton-Raphson\n")


#1) Equation de la tangente en un point d'abscisse x_n : y=f(x_n)+f'(x_n)(x-x_n)
# Abscisse de l'intersection de cette tangente avec l'axe des abscisses : on pose y=0. D'où le résultat suvant :
#x_n+1 = x_n-f(x_n)/f'(x_n) 
 
 
#2)

def newt(f,fd,x0:float,n:int):
    """
    Prend en argument deux fonctions : f et sa dérivée fd
    Et deux nombres : x0 correspondant à l'approximation initiale du 0
    n : le nombre  d'iterations à effectuer
    Renvoie en plus de la dernière valeur calculée de la suite (xn), correspondant au résultat de l'équation f(x)=0. 
    """
    y=x0 #initialisation
    x=y-f(y)/fd(y)
    n1=0 
    while abs(x-y)>=10**-5 or n1!=n : #on se donne ici une précision souhaitée sur le zéro de 10**-5 càd que quand l'écart entre 2 valeurs consécutives de la suite (x_n) est inférieur à cette valeur, on considère que l'on a trouvé le zéro.
    #condition d'arrêt : on a atteint la précision sur le zéro souhaite ET on a effectué le nombre d'itérations souhaité
        y=x
        x=x-f(x)/fd(x)
        n1+=1
    return x
 
 
#3)

def h(x): #on souhaite approcher sqrt(2) : on introduit donc une fonction qui a sqrt(2) comme zéro
    return x**2-2

def h1(x): 
    """
    Dérivée de h
    """
    return 2*x


print("L'approximation numérique de sqrt(2) par la méthode de Newton-Raphson à 10**-5 près est : ",newt(h,h1,2,25)) #on choisit un nombre d'itérations de 25 : il ne faut pas que ce dernier soit trop grand pour avoir une éxecution du programme rapide, mais il ne faut pas qu'il soit trop faible au risque de perdre en précision.
 
 
def newt2(f,x0:float,n:int):
    """
    Méthode de Newton avec dérivée numérique. 
    """
    y=x0 #initialisation
    x=y-f(y)/d_approx3(f, y, 10**-5) #on remplace fd par la dérivée numérique vue en exercice 1. On choisit d_approx3 afin de gagner (légèrement) en précision
    n1=0 
    while abs(x-y)>=10**-5 or n1!=n : 
        y=x
        x=x-f(x)/d_approx3(f, x, 10**-5)
        n1+=1
    return x

print("L'approximation numérique de sqrt(2) par la méthode de Newton-Raphson avec dérivée numérique à 10**-5 près est : ",newt2(h,2,25))

#calcul de l'erreur : 
    
print("L'erreur en adaptant la méthode de Newton (dont on suppose connaitre la dérivée) avec une approximation numérique de la dérivée est de : {} %.".format(((newt2(h,2,25)-newt(h,h1,2,25))/newt(h,h1,2,25))*100))
 


#CONCLUSION : On obtient la même approximation de sqrt(2) que ce soit avec la méthode de Newton classique ou avec la méthode de Newton avec dérivée numérique.
# On en conclut donc que notre méthode d'évaluation numérique d'une dérivée vue en exercice 1 est tout à fait performante. En effet, nous ne pourrons pas toujours donner à Python la valeur exacte de la dérivée d'une fonction et nous voyons alors que l'évaluer  numériquement convient tout de même bien !
 

#==============================EXERCICE 3====================================
print("\n EXERCICE 3 : Intégration numérique \n")
from math import pi
#1)

def f1(x:float)->float:
    return 1/(1+x**2)

#Le calcul à la main de l'intégrale de f entre 0 et 1 donne : pi/4

If = (1-0)*((f1(0)+4*f1(1/2)+f1(1))/6)

print("La valeur de l'intégrale de f1 sur [0,1] approchée par la quantité I(f) est : ",If)

#calcul de l'erreur commise :

print("L'erreur commise en approchant l'intégrale de f entre 0 et 1 par I(f) est de {}".format((If-pi/4)/(pi/4))) #L'erreur commise est assez conséquente -> on va alors utiliser la méthode de Simpson composite

print("\n")


def simpson(f,n:int)->float:
    """
    Prend en argument une fonction f et un entier n correspondant au nombre de subdivions de l'intervalle [0,1] à effectuer.
    Renvoie une approximation de l'intégrale entre 0 et 1 de la fonction f par la méthode de Simpson composite
    """
    h=1/n #le pas
    result = 0
    for k in range(0,n):
        result += (1/6)*h*(f(k*h)+f((k+1)*h)+4*f(((2*k+1)*h)/2))
    return result

print("La valeur de l'intégrale de f1 sur [0,1] approchée par la méthode de Simpson composite est de : ",simpson(f1,10**5))
print("L'erreur commise en approchant l'intégrale de f sur [0,1] par la méthode de Simpson composite est de {}".format((simpson(f1,10**5)-pi/4)/(pi/4))) #Diminition conséquente de l'erreur !


#3) 

print("\n")

X=[n for n in range(1,21)]
Y=[simpson(f1,n) for n in X]
plt.plot(X,Y)
plt.title("Graphique représentant la suite n -> simpson(f,n) pour n allant de 1 à 20")
plt.show()

def rectangle(f,a:float,b:float,m:float)->float:
    """
    Prend en arguments une fonction f, des flottants a et b et le nombre m de subdivisions à utiliser
    et renvoie une approximation de l'intégrale I, effectuée par la méthode des rectangles au point milieu (plus précise que rectangle à gauche !)
    """
    somme = 0
    h=(b-a)/m #h représente le pas
    x=(a+(a+h))/2 #méthode des rectangles au point milieu donc au commence au point d'abscisse (a+(a+h))/2
    for i in range(m):
        somme=somme+f(x)*h
        x=x+h #on avance par pas de h
    return somme

print("L'approximation numérique de l'intégrale de la fonction f1 sur [0,1] par la méthode des rectangles au point milieu est de : ",rectangle(f1,0,1,10**5))

#4) 
print("\n")

erreur_simpson = (simpson(f1,10**5)-pi/4)/(pi/4) #erreur commise par la méthode de Simpson

erreur_rectangle = (rectangle(f1,0,1,10**5)-pi/4)/(pi/4) #erreur commise par la méthode des rectangles

if erreur_rectangle > erreur_simpson :
    print("La méthode de Simpson est la plus efficace ! ") #pour l'exemple considéré ici biensur
else:
    print("La méthode des rectangles au point-milieu est plus efficace ! ")


#==============================EXERCICE 4====================================
print("\n EXERCICE 4 : Descente du gradient \n")


def descente_du_gradient(f, x:float, y:float, alpha=10**-3, epsilon=10**-5):
    """
    Recherche le minimum d'une fonction f (de 2 variables) par la méthode de descente de gradient avec dérivée numérique
    Prend en argument deux flottants : x et y les valeurs initiales ; alpha le pas de la descente, que l'on met par défaut à 10**-3
    epsilon est la précision souhaitée (par défaut 10**-5)    
    """
    grad = 1
    X=[]
    Y=[]
    while abs(grad)>epsilon: #condition d'arrêt au point (xn,yn) lorsque la norme du gradient(f(xn,yn)) <= epsilon
        gradx = (f(x+epsilon,y)-f(x-epsilon,y))/(2*epsilon) #c'est l'approximation numérique de la dérivée df/dx
        grady = (f(x, y+epsilon)-f(x, y-epsilon))/(2*epsilon) #c'est l'approximation numérique de la dérivée df/dy
        grad = sqrt(gradx**2+grady**2) # grad correspond à la norme du gradient
        x += -alpha*gradx # petit déplacement selon x dans le sens de la descente du gradient
        y += -alpha*grady # petit déplacement selon y dans le sens de la descente du gradient
        X += [x] #création des listes contenants les valeurs des points (xk,yk)
        Y += [y]
    plt.title("Ligne brisée constituée des points (xk,yk) provenant de la méthode de descente du gradient")    
    plt.plot(X,Y) #on trace la ligne brisée constituée des points (xk,yk)
    return x,y


def f3(x:float,y:float)->float:
    return sin(x**2/2-y**2/4+3)*cos(2*x+1-exp(y))

print("Le minimum de la fonction f3 par la méthode de la descente du gradient se situe au point : ", descente_du_gradient(f3, 3,2))






 

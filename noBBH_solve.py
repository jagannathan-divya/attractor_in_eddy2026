import numpy as np
import scipy
from scipy.optimize import fsolve
import argparse

# Written by Divya Jagannathan on July 9, 2025.
# This is for the case WITHOUT BBH.


def eddyflow(R, Z, sgma):
    # flow field parameters
    a = 0.62
    b = 7.5
    c = 0.69
    r0 = 0.5
    ur = -(1.0/3.0)*(r0-R)*b*R*(1.0 - 2.0*Z)
    ut = a*R*(c+Z*Z)
    uz = (1.0/3.0)*(2.0*r0 - 3.0*R)*b*Z*(1.0 - Z)
    fr = sgma*(1.0/9.0)*R*(-9.0*a*a*((c + Z**2)**2) +
                           (b**2)*(r0-R)*(r0-2.0*R*(1.0-Z+Z*Z)))
    ft = sgma*(2.0/3.0)*(a*b*R)*(R*(Z**3) + (-2*R+r0)
                                 * (Z**2) + 2*c*(r0-R)*Z - c*(r0-R))
    fz = sgma*(1.0/9.0)*(b**2)*Z*(2.0*Z-1.0) * \
        (Z-1.0)*(4*(r0**2) - 9*r0*R + 6*(R**2))
    return [ur, ut, uz, fr, ft, fz]


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--St', type=float, required=True)
args = parser.parse_args()

St = args.St
denR = 0.98
g = 9.81

alp = 1.0/St
sgma = (1.0/denR)

# System of equations


def eqSet(vars):
    Rc, Zc, Omg = vars
    [ur, ut, uz, fr, ft, fz] = eddyflow(Rc, Zc, sgma)
    eq1 = fr + Rc*Omg**2 + alp*ur
    eq2 = ft - alp*Rc*Omg + alp*ut
    eq3 = (sgma-1.0)*g + fz + alp*uz
    return [eq1, eq2, eq3]


# INitial guess
ini_guess = [0.3, 0.4, 0.7]

soln = fsolve(eqSet, ini_guess)

with open("noBBH_trend.txt", "a") as f:
    f.write(f"{St}, {soln[0]}, {soln[1]}, {soln[2]}\n")

print("-----------------------------")
print("*** WITHOUT THE BBH FORCE ***")
print("Parameters:")
print("alpha is ", alp)
print("1/R is ", sgma)
print("-----------------------------")
print("Characteristics of the attractor ")
print("[Rc, Zc, Omega]:", soln)

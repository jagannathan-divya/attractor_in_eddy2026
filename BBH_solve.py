import numpy as np
import scipy
from scipy.optimize import fsolve
import argparse

# Written by Divya Jagannathan on July 9, 2025
# This code solves the set of three nonlinear algebraic equations
# to compute the characteristics of a circular attractor (Rc, Zc, Omg)
# in the ocean eddy with the BBH force INCLUDED in the particle's dynamics.


def eddyflow(R, Z, sgma):
    # flow field parameters
    a = 0.62
    b = 7.5
    c = 0.69
    r0 = 0.5
    ur = -(1.0/3.0)*(r0-R)*b*R*(1.0 - 2.0*Z)
    ut = a*R*(c+Z*Z)
    uz = (1.0/3.0)*(2.0*r0 - 3.0*R)*b*Z*(1.0 - Z)
    # computes sgma*Du/Dt
    fr = sgma*(1.0/9.0)*R*(-9.0*a*a*((c+Z**2)**2) +
                           (b**2)*(r0-R)*(r0-2.0*R*(1.0-Z+Z*Z)))
    ft = sgma*(2.0/3.0)*(a*b*R)*(R*(Z**3) + (-2*R+r0)
                                 * (Z**2) + 2*c*(r0-R)*Z - c*(r0-R))
    fz = sgma*(1.0/9.0)*(b**2)*Z*(2*Z-1)*(Z-1)*(4*(r0**2) - 9*r0*R + 6*(R**2))
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
    eq1 = (Rc*Omg) - ut + ur
    eq2 = (Rc*Omg)*(Omg-alp) + alp*(ur+ut) + (fr + ft)
    eq3 = (sgma-1.0)*g + fz + alp*uz
    return [eq1, eq2, eq3]


# Initial guess
ini_guess = [0.4, 0.55, 0.1]
soln = fsolve(eqSet, ini_guess)

# Write the results to txt file
with open("bbh_trend.txt", "a") as f:
    f.write(f"{St}, {soln[0]}, {soln[1]}, {soln[2]}\n")

print("WITH BBH, THE ATTRACTOR [Rc, Zc, Omega]:", soln)

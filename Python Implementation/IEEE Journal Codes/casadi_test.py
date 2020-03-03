# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:21:47 2020

@author: vxr131730
"""

from casadi import *
opti = casadi.Opti()

x = opti.variable()
y = opti.variable()

opti.minimize(  (y-x**2)**2   )
opti.subject_to( x**2+y**2==1 )
opti.subject_to(       x+y>=1 )

opti.solver('ipopt')


sol = opti.solve()

print(sol.value(x))
print(sol.value(y))
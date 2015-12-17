import numpy as np
import matplotlib.pyplot as plt
from . import ligandbinding as lb

def BindingStatesGraph(Ka, Lfreemin, Lfreemax, npts):
    #graph ratio of macromolecule in each binding state as a function of free ligand concentration
    
    nstates = Ka.size + 1
    
    Lfree = np.linspace(Lfreemin, Lfreemax, npts)
    statefracs = lb.EqstatesFromFreeLigand(Lfree, Ka)
    
    #plot the binding states as a function of free ligand
    lines = plt.plot(Lfree * 1e9, statefracs)
    
    legendtext = np.append('$[M]$', ['$[L_%dM]$' % n for n in range(1,nstates + 1)])
    plt.legend(lines, legendtext, loc = 'best')
    plt.xlabel('$[L]_{free} (nM)$')
    plt.ylabel('Binding state fraction')
    plt.title('Equilibrium binding')
    
    plt.show()

def itcgraph(itcr):
    
    dHdiff = np.diff(itcr.dH) #enthalpy difference for each injection
    MolarRatio = itcr.Ltot / itcr.Mtot
        
    #plot the predicted heats for each injection    
    plt.vlines(np.array(range(1, dHdiff.size + 1)), np.zeros(dHdiff.size), dHdiff) #heat of each injection
    plt.xlabel('Injection number')
    plt.ylabel('Evolved heat per injection (kJ)')
    plt.show()
        
    #plot the total heat as a function of ligand:macromolecule molar ratio    
    plt.plot(MolarRatio, (itcr.dH - itcr.dH[0]) / itcr.Mmoles, '.-') #Molar ratio vs. total evolved heat / moles macromolecule
    plt.xlabel('Molar ratio L:M')
    plt.ylabel('Total evolved heat/macromolecule (kJ/mol)')
    plt.show()
        
    #plot total heat / moles of ligand on last injection vs molar ratio
    plt.plot(MolarRatio[1:], np.diff(itcr.dH) / np.diff(itcr.Lmoles), '.-') #Molar ratio vs. total evolved heat / moles macromolecule
    plt.xlabel('Molar ratio L:M')
    plt.ylabel('Single-injection heat/moles ligand injected (kJ/mol)')
    plt.show()
    
    #plot free ligand concentration for each step of the ITC protocol
    plt.plot(np.array(range(0, dHdiff.size + 1)), itcr.Lfree * 1e9, '.-') #heat of each injection
    plt.xlabel('Injections completed')
    plt.ylabel('$[L]_{free}$ (nM)')
    plt.show()
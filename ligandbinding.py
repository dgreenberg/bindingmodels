import numpy as np
from collections import namedtuple

def EqstatesFromFreeLigand(Lfree, Ka):
    #get binding state ratios as a function of free ligand concentration
    #Lfree and Ka can both be vectors
    
    #convert scalars to numpy arrays if needed etc.      
    L = np.array(Lfree).reshape(-1)
    Kaprod = np.cumprod(np.array(Ka).reshape(-1)) #product coefficients for the Adair-Klotz equation
    
    nstates = Kaprod.size + 1
    nconcentrations = L.size
    
    rc = np.ones((nconcentrations, nstates)) #relative concentrations
    c  = np.ones((nconcentrations, nstates)) #will store normalized concentrations
    s  = np.ones(nconcentrations) #sum of rc. added first term (=1) in advance
    
    for j in range(1, nstates):        
        rc[:, j] = Kaprod[j - 1] * L ** j
        s += rc[:, j]
    
    for i in range(0, nconcentrations):
        c[i, :] = rc[i, :] / s[i]
    
    return c

def sequential_binding_polynomial(Ltot, Mtot, Ka):
    Kaprod = np.cumprod(np.array(Ka).reshape(-1)) #product coefficients for the Adair-Klotz equation
    
    nstates = Ka.size + 1
    nconcentrations = Ltot.size    
    
    P = np.zeros((nconcentrations, nstates + 1)) #polynomial coefficients.
    #highest term is of order nstates, so we have nstates + 1 coefficients.
    #For example, if we can bind up to 4 ligands per macromolecule we have 5 states
    #and a 5-th order polynomial with 6 coefficients.
    #each row goes left-to-right from lower to higher order (starting with constant term)
        
    #we need to solve the equation Z * Ltot = Z * L + Mtot * \sum_{i=1}^{nstates-1} i * L^i * Kaprod[i-1]
    #for the variable L
    #where Z is the partition function:
    #Z = 1 + \sum_{i=1}^{nstates-1} L^i Kaprod[i - 1]    
    P[:, 1:]   += np.append(1.0, Kaprod)                              #Z * L
    P[:, 0:-1] -= Ltot.reshape(nconcentrations, 1) * np.append(1.0, Kaprod) #Z * Ltot. this is an outer product
    P[:, 1:-1] += Mtot.reshape(nconcentrations, 1) * Kaprod * np.array(range(1, nstates))         #Mtot * \sum_{i=1}^{nstates-1} i * L^i * Kaprod[i-1]
    
    return P

def FreeLigandFromTotals(Ltot, Mtot, Ka):
    #calculates free ligand concentration from total ligand and macromolecule concentrations, and Ka's
    #this requires solving a polynomial of order nstates
    #Ka is a vector of association constants for sequential binding
    #Mtot and Ltot can be either scalars or vectors
    
    LT = Ltot.reshape(-1)
    MT = Mtot.reshape(-1)    
    
    nconcentrations = Ltot.size    
    P = sequential_binding_polynomial(LT, MT, Ka)
    
    Lfree = np.zeros(nconcentrations) #free ligand for each concentration of total ligand
    
    for i in range(0, nconcentrations):
        if LT[i] == 0:
            Lfree[i] = 0
        else:
            rts = np.roots(P[i, ::-1]) #note that we reverse the polynomial coefficients to get the order highest to lowest before calling roots()
            ok = np.flatnonzero((np.isreal(rts)) & (rts >= 0) & (rts <= LT[i]))
            assert ok.size == 1, "failed to find valid unique solution for Lfree"
            Lfree[i] = rts[ok[0]].real #explictly take real part to avoid warning; we know the complex is zero from the above.
    
    return Lfree

def EnthalpyDifference(c, dHstates, Mmoles):
    #calculate total enthalpy difference from the ground state
            
    dH = Mmoles * np.dot(c[1:], dHstates) #Jelesarov & Bosshard 1999 eq. 10
    
    return dH
    
def SimulateITC(Ka, dHstates, V0, M0, Lfree0, Linjection, Vinjection, ninjections):
    
    nstates = Ka.size + 1
    
    c0 = EqstatesFromFreeLigand(Lfree0, Ka) #initial binding states    
    Ltotal0 = Lfree0 + M0 * c0[:, 1:].dot(np.array(range(1, nstates))) #initial total ligand concentration
    
    Lmoles = Ltotal0 * V0 + Linjection * Vinjection * np.array(range(0, ninjections + 1)) #total moles of ligand at each step
    Mmoles = np.ones(ninjections + 1) * M0 * V0 #total moles of macromolecule for each time step. this does not change, we only dilute it.
    
    V = V0 + Vinjection * np.array(range(0, ninjections + 1)) #total volume at each step
    
    Ltot = Lmoles / V #total ligand concentration at each step
    Mtot = Mmoles / V #total macromolecule concentration at each step
    
    Lfree = np.zeros(ninjections + 1) #free ligand concentration at each step
    Lfree[0] = Lfree0
    
    c = np.zeros((ninjections + 1, nstates)) #concentration of each binding state at each step        
    c[0,:] = c0    
    
    dH = np.zeros(ninjections + 1) #enthalpy difference of system from ligand-free for each step of the protocol
    
    #determine concentrations of each binding state and free ligand for each step of the protocol:
    for i in range(1, ninjections + 1): #for each injection
        #calculate the free ligand concentration from the total concentrations of ligand and macromolecule
        Lfree[i] = FreeLigandFromTotals(Ltot[i], Mtot[i], Ka)        
        #calculate the binding state fractions from the free ligand concentration
        c[i,:] = EqstatesFromFreeLigand(Lfree[i], Ka)
    
    #determine total evolved heat up to and including each step of the protocol:
    for i in range(0, ninjections + 1): #for each injection and for initial state
        dH[i] = EnthalpyDifference(c[i,:], dHstates, Mmoles[i])
    
    ITCresult = namedtuple('ITCresult', 'Lfree c dH V Mtot Ltot Mmoles Lmoles')
    return ITCresult(Lfree, c, dH, V, Mtot, Ltot, Mmoles, Lmoles)  
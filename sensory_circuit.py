'''

Sensory circuit

@author: Klaus Wimmer

wimmer.klaus@googlemail.com

'''

from brian import *
import random as pyrandom
import numpy
from numpy.random import randn as randn


def make_sensory_circuit():
    
    '''
    Creates the spiking network representing the sensory circuit.
       
    returns:
        groups, connections, subgroups
        
    groups and connections have to be added to the "Network" in order to run the simulation.
    subgroups is used for establishing connections between the sensory and integration circuit; do not add subgroups to the "Network"

    ''' 
    
    # -----------------------------------------------------------------------------------------------
    # Model parameters for the sensory circuit
    # ----------------------------------------------------------------------------------------------- 

    # Populations
    N_E = 1600                                   # Total number of excitatory neurons
    N_E1 = int(0.5 * N_E)                        # Size of excitatory population E1
    N_E2 = N_E1                                  # Size of excitatory population E2
    N_I = 400                                    # Size of inhibitory population I
    N_X = 1000                                   # Size of external population X
    N_X1 = int(0.5 * N_X)                        # Size of external population X1
    N_X2 = int(0.5 * N_X)                        # Size of external population X2
    
    # Connectivity - local recurrent connections
    p = 0.2                                      # Connection probability for (EE, EI, IE, II)
    w_p = 1.3                                    # Relative synaptic strength of connections within populations E1 and E2
    w_m = 2.0 - w_p                              # Relative synaptic strength of connections across populations E1 and E2
    gEE = 0.7589 * nS                            # Weight of excitatory to excitatory synapses
    gEI = 1.5179 * nS                            # Weight of excitatory to inhibitory synapses
    gIE = 12.6491 * nS                           # Weight of inhibitory to excitatory synapses
    gII = gIE                                    # Weight of inhibitory to inhibitory synapses
    dE = (0.5 * ms, 1.5 * ms)                    # Range of uniformly distributed transmission delays of excitatory connections
    dI = (0.1 * ms, 0.9 * ms)                    # Range of uniformly distributed transmission delays of inhibitory connections
                                                  
    # Connectivity - external connections
    p_x = 0.32                                   # Connection probability for external connections
    alpha_x = 0.0
    gextE = 1.7076 * nS                          # Weight of external to excitatory synapses
    gextI = 1.7076 * nS                          # Weight of external to inhibitory synapses
    dXE = (0.5 * ms, 1.5 * ms)                   # Range of uniformly distributed transmission delays of external connections

    # Neuron model
    CmE = 0.25 * nF                              # Membrane capacitance of excitatory neurons
    CmI = 0.25 * nF                              # Membrane capacitance of inhibitory neurons
    gLeakE = 16.7 * nS                           # Leak conductance of excitatory neurons
    gLeakI = 16.7 * nS                           # Leak conductance of inhibitory neurons
    Vl = -70.0 * mV                              # Resting potential
    Vt = -50.0 * mV                              # Spiking threshold
    Vr = -60.0 * mV                              # Reset potential
    tau_refE = 2.0 * ms                          # Absolute refractory period of excitatory neurons
    tau_refI = 1.0 * ms                          # Absolute refractory period of inhibitory neurons
    nu_ext = 12.5 * Hz                           # Firing rate of external Poisson neurons 
    
    # Synapse model
    VrevE = 0 * mV                               # Reversal potential of excitatory synapses
    VrevI = -80 * mV                             # Reversal potential of inhibitory synapses
    tau_decay = 5.0 * ms                         # Decay constants of AMPA-type and GABA-type conductances
    tau_rise = 1.0 * ms                          # Rise constant of AMPA- and GABA-type conductances
    
    # Inputs
                                                 
    
    # -----------------------------------------------------------------------------------------------
    # Set up the model
    # ----------------------------------------------------------------------------------------------- 
           
    # Neuron equations
    eqs = '''
    dV/dt = (-gea*(V-VrevE) - gi*(V-VrevI) - (V-Vl)) * (1.0/tau) + I/Cm: volt
    dgea/dt = (xe-gea)*(1.0/tau_decay)        : 1
    dxe/dt = -xe*(1.0/tau_rise)               : 1     
    dgi/dt = (xi-gi)*(1.0/tau_decay)          : 1
    dxi/dt = -xi*(1.0/tau_rise)               : 1     
    I : nA
    tau : second
    Cm : nF
    '''

    # Set up the sensory circuit
    sensoryE = NeuronGroup(N_E, model=eqs, threshold=Vt, reset=Vr, refractory=tau_refE)
    sensoryI = NeuronGroup(N_I, model=eqs, threshold=Vt, reset=Vr, refractory=tau_refI)
    sensoryE.tau = CmE / gLeakE
    sensoryE.Cm = CmE
    sensoryI.tau = CmI / gLeakI
    sensoryI.Cm = CmI
    sensoryE.I = 0.0
    sensoryI.I = 0.0
    sensoryE1 = sensoryE.subgroup(N_E1)
    sensoryE2 = sensoryE.subgroup(N_E2)
   
    # Connections involving AMPA synapses
    C_SE_SE = Connection(sensoryE, sensoryE, 'xe', delay=True, max_delay=1.5 * ms)
    C_SE_SE.connect_random(sensoryE1, sensoryE1, sparseness=p, weight=lambda:w_p * gEE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dE) 
    C_SE_SE.connect_random(sensoryE2, sensoryE2, sparseness=p, weight=lambda:w_p * gEE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dE) 
    C_SE_SE.connect_random(sensoryE1, sensoryE2, sparseness=p, weight=lambda:w_m * gEE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dE) 
    C_SE_SE.connect_random(sensoryE2, sensoryE1, sparseness=p, weight=lambda:w_m * gEE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dE) 
    C_SE_SI = Connection(sensoryE, sensoryI, 'xe', delay=True, max_delay=1.5 * ms)
    C_SE_SI.connect_random(sparseness=p, weight=lambda:gEI/gLeakI * max(0.0, 1.0 + 0.5 * randn()), delay=dE)

    # Connections involving GABA synapses
    C_SI_SE = Connection(sensoryI, sensoryE, 'xi', delay=True, max_delay=0.9 * ms)
    C_SI_SE.connect_random(sparseness=p, weight=lambda:gIE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dI)
    C_SI_SI = Connection(sensoryI, sensoryI, 'xi', delay=True, max_delay=0.9 * ms)
    C_SI_SI.connect_random(sparseness=p, weight=lambda:gII/gLeakI * max(0.0, 1.0 + 0.5 * randn()), delay=dI)    

    # External inputs
    external = PoissonGroup(N_X, rates = nu_ext)      # unspecific Poisson input
    external1 = external.subgroup(N_X1)
    external2 = external.subgroup(N_X2)
       
    # Connect external inputs
    C_SX_SE = Connection(external, sensoryE, 'xe', delay=True, max_delay=1.5 * ms)
    C_SX_SE.connect_random(external1, sensoryE1, sparseness=p_x * (1.0 + alpha_x), weight=lambda:gextE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dXE)  
    C_SX_SE.connect_random(external2, sensoryE2, sparseness=p_x * (1.0 + alpha_x), weight=lambda:gextE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dXE)
    C_SX_SE.connect_random(external1, sensoryE2, sparseness=p_x * (1.0 - alpha_x), weight=lambda:gextE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dXE)
    C_SX_SE.connect_random(external2, sensoryE1, sparseness=p_x * (1.0 - alpha_x), weight=lambda:gextE/gLeakE * max(0.0, 1.0 + 0.5 * randn()), delay=dXE)
    C_SX_SI = Connection(external, sensoryI, 'xe', delay=True, max_delay=1.5 * ms)
    C_SX_SI.connect_random(sparseness=p_x, weight=lambda:gextI/gLeakI * max(0.0, 1.0 + 0.5 * randn()), delay=dXE)  
       
		# Return the sensory circuit
    groups = {'SE': sensoryE, 'SI': sensoryI, 'SX': external}
    subgroups = {'SE1': sensoryE1, 'SE2': sensoryE2}
    connections = {'C_SX_SE': C_SX_SE, 'C_SX_SI': C_SX_SI, 
                   'C_SE_SE': C_SE_SE, 'C_SE_SI': C_SE_SI, 'C_SI_SE': C_SI_SE, 'C_SI_SI': C_SI_SI}
                     
    return groups, connections, subgroups

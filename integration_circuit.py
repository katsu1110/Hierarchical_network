'''

Integration circuit from:

Wang, X.-J. Probabilistic decision making by slow reverberation in cortical circuits. Neuron, 2002, 36, 955-968.

@author: Klaus Wimmer and Albert Compte

wimmer.klaus@googlemail.com
acompte@clinic.ub.es

'''

from brian import *
import numpy


def make_integration_circuit():
    
    '''
    Creates the spiking network described in Wang 2002.
    
    returns:
        groups, connections, update_nmda, subgroups
        
    groups, connections, and update_nmda have to be added to the "Network" in order to run the simulation.
    subgroups is used for establishing connections between the sensory and integration circuit; do not add subgroups to the "Network"

    ''' 
    
    # -----------------------------------------------------------------------------------------------
    # Model parameters for the integration circuit
    # ----------------------------------------------------------------------------------------------- 

    # Populations
    f_E = 0.15                                   # Fraction of stimulus-selective excitatory neurons
    N = 2000                                     # Total number of neurons
    f_inh = 0.2                                  # Fraction of inhibitory neurons
    NE = int(N * (1.0 - f_inh))                  # Number of excitatory neurons (1600)
    NI = int(N * f_inh)                          # Number of inhibitory neurons (400)
    N_D1 = int(f_E * NE)                         # Size of excitatory population D1
    N_D2 = N_D1                                  # Size of excitatory population D2
    N_DN = int((1.0 - 2.0 * f_E) * NE)           # Size of excitatory population DN

		# Connectivity - local recurrent connections
    w_p = 1.6                                    # Relative recurrent synaptic strength within populations D1 and D2
    w_m = 1.0 - f_E * (w_p - 1.0) / (1.0 - f_E)	 # Relative recurrent synaptic strength of connections across populations D1 and D2 and from DN to D1 and D2
    gEE_AMPA = 0.05 * nS		                     # Weight of AMPA synapses between excitatory neurons
    gEE_NMDA = 0.165 * nS                        # Weight of NMDA synapses between excitatory neurons
    gEI_AMPA = 0.04 * nS                         # Weight of excitatory to inhibitory synapses (AMPA)
    gEI_NMDA = 0.13 * nS                         # Weight of excitatory to inhibitory synapses (NMDA)
    gIE_GABA = 1.3 * nS                          # Weight of inhibitory to excitatory synapses
    gII_GABA = 1.0 * nS                          # Weight of inhibitory to inhibitory synapses
    d = 0.5 * ms                                 # Transmission delay of recurrent excitatory and inhibitory connections
                                                
    # Connectivity - external connections
    gextE = 2.1 * nS                             # Weight of external input to excitatory neurons
    gextI = 1.62 * nS                            # Weight of external input to inhibitory neurons

    # Neuron model
    CmE = 0.5 * nF                               # Membrane capacitance of excitatory neurons
    CmI = 0.2 * nF                               # Membrane capacitance of inhibitory neurons
    gLeakE = 25.0 * nS                           # Leak conductance of excitatory neurons
    gLeakI = 20.0 * nS                           # Leak conductance of inhibitory neurons
    Vl = -70.0 * mV                              # Resting potential
    Vt = -50.0 * mV                              # Spiking threshold
    Vr = -55.0 * mV                              # Reset potential
    tau_refE = 2.0 * ms                          # Absolute refractory period of excitatory neurons
    tau_refI = 1.0 * ms                          # Absolute refractory period of inhibitory neurons
    
    # Synapse model
    VrevE = 0 * mV                               # Reversal potential of excitatory synapses
    VrevI = -70 * mV                             # Reversal potential of inhibitory synapses
    tau_AMPA = 2.0 * ms                          # Decay constant of AMPA-type conductances
    tau_GABA = 5.0 * ms                          # Decay constant of GABA-type conductances
    tau_NMDA_decay = 100.0 * ms                  # Decay constant of NMDA-type conductances
    tau_NMDA_rise = 2.0 * ms                     # Rise constant of NMDA-type conductances
    alpha_NMDA = 0.5 * kHz                       # Saturation constant of NMDA-type conductances
    
    # Inputs
    nu_ext_1 = 2392 * Hz				                 # Firing rate of external Poisson input to neurons in population D1
    nu_ext_2 = 2392 * Hz				                 # Firing rate of external Poisson input to neurons in population D2
    nu_ext = 2400 * Hz				                   # Firing rate of external Poisson input to neurons in population Dn and I
                                                 
    
    # -----------------------------------------------------------------------------------------------
    # Set up the model
    # ----------------------------------------------------------------------------------------------- 

    # Neuron equations
    eqsE = '''
    dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau): volt
    dgea/dt = -gea/(tau_AMPA) : 1
    dgi/dt = -gi/(tau_GABA) : 1
    dspre/dt = -spre/(tau_NMDA_decay)+alpha_NMDA*xpre*(1-spre) : 1
    dxpre/dt= -xpre/(tau_NMDA_rise) : 1
    gen : 1
    tau : second
    '''
    eqsI = '''
    dV/dt = (-gea*(V-VrevE) - gen*(V-VrevE)/(1.0+exp(-V/mV*0.062)/3.57) - gi*(V-VrevI) - (V-Vl)) / (tau): volt
    dgea/dt = -gea/(tau_AMPA) : 1
    dgi/dt = -gi/(tau_GABA) : 1
    gen : 1
    tau : second
    '''

    # Set up the integration circuit
    decisionE = NeuronGroup(NE, model=eqsE, threshold=Vt, reset=Vr, refractory=tau_refE)
    decisionI = NeuronGroup(NI, model=eqsI, threshold=Vt, reset=Vr, refractory=tau_refI)
    decisionE.tau = CmE / gLeakE
    decisionI.tau = CmI / gLeakI      
    decisionE1 = decisionE.subgroup(N_D1)
    decisionE2 = decisionE.subgroup(N_D2)
    decisionE3 = decisionE.subgroup(N_DN)
    
    # Connections involving AMPA synapses
    C_DE_DE_AMPA = Connection(decisionE, decisionE, 'gea', delay = d)             
    C_DE_DE_AMPA.connect_full(decisionE1, decisionE1, weight = w_p * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE2, decisionE2, weight = w_p * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE1, decisionE2, weight = w_m * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE2, decisionE1, weight = w_m * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE3, decisionE1, weight = w_m * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE3, decisionE2, weight = w_m * gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE3, decisionE3, weight = gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE1, decisionE3, weight = gEE_AMPA / gLeakE)
    C_DE_DE_AMPA.connect_full(decisionE2, decisionE3, weight = gEE_AMPA / gLeakE)
    C_DE_DI_AMPA = Connection(decisionE, decisionI, 'gea', weight = gEI_AMPA / gLeakI, delay = d)

    # Connections involving NMDA synapses    
    # Note that due to the all-to-all connectivity, the contribution of NMDA can be calculated efficiently
    selfnmda = IdentityConnection(decisionE, decisionE, 'xpre', weight=1.0, delay = d)    
    E1_nmda = asarray(decisionE1.spre)
    E2_nmda = asarray(decisionE2.spre)
    E3_nmda = asarray(decisionE3.spre)         
    E1_gen = asarray(decisionE1.gen) 
    E2_gen = asarray(decisionE2.gen)
    E3_gen = asarray(decisionE3.gen)
    I_gen = asarray(decisionI.gen)

    # Calculate NMDA contributions in each time step
    @network_operation(when='start')
    def update_nmda():
        sE1 = sum(E1_nmda)
        sE2 = sum(E2_nmda)
        sE3 = sum(E3_nmda)
        E1_gen[:] = gEE_NMDA / gLeakE * (w_p*sE1 + w_m*sE2 + w_m*sE3)
        E2_gen[:] = gEE_NMDA / gLeakE * (w_m*sE1 + w_p*sE2 + w_m*sE3)
        E3_gen[:] = gEE_NMDA / gLeakE * (sE1 + sE2 + sE3)
        I_gen[:] = gEI_NMDA / gLeakI * (sE1 + sE2 + sE3)    
    
    # Connections involving GABA synapses
    C_DI_DE = Connection(decisionI, decisionE, 'gi', weight = gIE_GABA / gLeakE, delay = d)
    C_DI_DI = Connection(decisionI, decisionI, 'gi', weight = gII_GABA / gLeakI, delay = d)
    
    # External inputs
    extinputE1 = PoissonGroup(N_D1, rates = nu_ext_1) 
    extinputE2 = PoissonGroup(N_D2, rates = nu_ext_2)
    extinputE3 = PoissonGroup(N_DN, rates = nu_ext)
    extinputI = PoissonGroup(NI, rates = nu_ext)
   
    # Connect external inputs
    extconnE1 = IdentityConnection(extinputE1, decisionE1, 'gea', weight = gextE / gLeakE)
    extconnE2 = IdentityConnection(extinputE2, decisionE2, 'gea', weight = gextE / gLeakE)
    extconnE3 = IdentityConnection(extinputE3, decisionE3, 'gea', weight = gextE / gLeakE)
    extconnI = IdentityConnection(extinputI, decisionI, 'gea', weight = gextI / gLeakI)
    
		# Return the integration circuit
    groups = {'DE': decisionE, 'DI': decisionI, 'DX1': extinputE1, 'DX2': extinputE2, 'DX3': extinputE3, 'DXI': extinputI}
    subgroups = {'DE1': decisionE1, 'DE2': decisionE2, 'DE3': decisionE3}  
    connections = {'selfnmda': selfnmda,
                   'extconnE1': extconnE1, 'extconnE2': extconnE2, 'extconnE3': extconnE3, 'extconnI': extconnI, 
                   'C_DE_DE_AMPA': C_DE_DE_AMPA, 'C_DE_DI_AMPA': C_DE_DI_AMPA, 'C_DI_DE': C_DI_DE, 'C_DI_DI': C_DI_DI }
                     
    return groups, connections, update_nmda, subgroups

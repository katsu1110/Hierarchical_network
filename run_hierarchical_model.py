'''

Hierarchical network model of perceptual decision making

@author: Klaus Wimmer

wimmer.klaus@googlemail.com

'''


from brian import *
import numpy
import random as pyrandom
from numpy.random import rand as rand
from numpy.random import randn as randn
from scipy.signal import lfilter

from integration_circuit import make_integration_circuit
from sensory_circuit import make_sensory_circuit
 

def get_OU_stim(n, tau):
# UO process in discrete time => AR(1) process
      
    a = int(numpy.exp(-(1.0 / tau)))
    i = lfilter(numpy.ones(1),[1, -a], numpy.sqrt(1-a*a)*randn(int(n)))
         
    return i


if __name__ == '__main__':

    #  initialize  
    defaultclock.reinit()
    clear(True) 


    #------------------------------------------------------------------------------ 
    # Simulation parameters 
    #------------------------------------------------------------------------------ 
    
    connect_seed = 1284                          # seed for random number generators (set before establishing network connectivity)
    stim_seed = 123                              # seed for random number generators (set before generating common part of stimulus)
    init_seed = 8190                             # seed for random number generators (set before generating private part of stimulus)

    # Timing 
    stim_on = 500.0 * ms                         # stimulus onset
    stim_off = 2500.0 * ms                       # stimulus offset   
    stim_duration = stim_off - stim_on           # duration of stimulus interval
    runtime = 3000.0 * ms                        # total simulation time


    #------------------------------------------------------------------------------
    # Construct hierarchical network
    #------------------------------------------------------------------------------ 

    # Set the seed of the random number generator
    numpy.random.seed(connect_seed) 
    pyrandom.seed(connect_seed)

    # Integration circuit
    Dgroups, Dconnections, Dnetfunctions, Dsubgroups = make_integration_circuit()
    
    # get populations from the integrations circuit
    decisionE = Dgroups['DE']
    decisionI = Dgroups['DI']
    decisionE1 = Dsubgroups['DE1']
    decisionE2 = Dsubgroups['DE2']
    decisionE3 = Dsubgroups['DE3']

    # Sensory network  
    Sgroups, Sconnections, Ssubgroups = make_sensory_circuit()

    # get sensory populations
    sensoryE = Sgroups['SE']
    sensoryI = Sgroups['SI']
    sensoryE1 = Ssubgroups['SE1']
    sensoryE2 = Ssubgroups['SE2']
    
    # Feed-forward connections from the sensory to the integration circuit
    
    wSD = 0.0036                                 # Synaptic weight of feed-forward connections from the corresponding stimulus-encoding population 
                                                 # from the sensory circuit (E1 -> D1, E2 -> D2); 
                                                 # synaptic weight (0.09 nS) is given in multiples of the leak conductance of the excitatory neurons in the integration circuit  
                                                 
    C_SE1_DE1 = Connection(sensoryE1, decisionE1, 'gea', weight=wSD, sparseness=0.2, delay=1.0 * ms)
    C_SE2_DE2 = Connection(sensoryE2, decisionE2, 'gea', weight=wSD, sparseness=0.2, delay=1.0 * ms)
	
    # Top-down feedback connections from the integration circuit to the sensory circuit
    
    b_FB = 6.0                                   # Feedback strength
    wDS = 0.004 * b_FB                           # Synaptic weight of feedback connections from the integration circuit to the sensory circuit (D1 -> E1, D2 -> E2);
                                                 # synaptic weight (0.0668 nS) is given in multiples of the leak conductance of the excitatory neurons in the sensory circuit  
    
    C_DE1_SE1 = Connection(decisionE1, sensoryE1, 'xe', weight=wDS, sparseness=0.2, delay=1.0 * ms)
    C_DE2_SE2 = Connection(decisionE2, sensoryE2, 'xe', weight=wDS, sparseness=0.2, delay=1.0 * ms)


    #------------------------------------------------------------------------------
    # Stimulus 
    #------------------------------------------------------------------------------ 
    
    # Stimulus parameters    
    I0 = 0.08 * nA                               # Mean input current for zero-coherence stimulus 
    c = 0.0                                      # Stimulus coherence (between 0 and 1) 
    mu_E1 = +0.25                                # Average additional input current to E1 at highest coherence (c = 1)
    mu_E2 = -0.25                                # Average additional input current to E2 at highest coherence (c = 1)    
    sigma = 1.0																	 # Amplitude of temporal modulations of the stimulus
    sigma_stim = 0.212 * sigma                   # S.d. of modulations of stimulus inputs
    sigma_ind = 0.212 * sigma                    # S.d. of modulations in individual inputs
    tau_stim = 20.0 * ms                         # Correlation time constant of Ornstein-Uhlenbeck process
    
    # Generate stimulus
    # set seed of random number generator (in order to generate a specific stimulus each time)
    numpy.random.seed(stim_seed)    
    pyrandom.seed(stim_seed)

    # "common part" of the stimulus
    z1 = get_OU_stim(stim_duration/ms, tau_stim/ms)
    z1 = numpy.tile(z1,(len(sensoryE1),1))
    z2 = get_OU_stim(stim_duration/ms, tau_stim/ms)
    z2 = numpy.tile(z2,(len(sensoryE2),1))

    # set seed of random number generator (in order to generate a specific stimulus each time)
    numpy.random.seed(init_seed)    
    pyrandom.seed(init_seed)

    # "private part" - part of the stimulus for each neuron, different in each trial  
    zk1 = get_OU_stim(int(stim_duration/ms * len(sensoryE1)), tau_stim/ms)  
#    zk1 = numpy.asarray(zk1).reshape(len(sensoryE1), stim_duration/ms)
    stmdur = len(zk1)/len(sensoryE1)
    zk1 = numpy.asarray(zk1).reshape(len(sensoryE1), stmdur)
    zk2 = get_OU_stim(int(stim_duration/ms * len(sensoryE2)), tau_stim/ms)  
    zk2 = numpy.asarray(zk2).reshape(len(sensoryE2), stmdur)
#    zk2 = numpy.asarray(zk2).reshape(len(sensoryE2), stim_duration/ms)
    
    # stimulus (time series with dt = 1ms)
    # most general case: different input to each neuron in each time step 
    i1 = I0 * (1 + c * mu_E1 + sigma_stim * z1 + sigma_ind * zk1)     
    i2 = I0 * (1 + c * mu_E2 + sigma_stim * z2 + sigma_ind * zk2)     
#    ii = numpy.zeros((len(sensoryI),stim_duration/ms))     # no current input to inh population
    ii = numpy.zeros((len(sensoryI), stmdur))     # no current input to inh population

    # Stimulus-related external current input     
    myclock=Clock(dt=1*ms)
    @network_operation(myclock)
    def update_input():  
        if myclock.t >= stim_on and myclock.t < stim_off:           
            sensoryE1.I = i1[:,int( (myclock.t - stim_on) / (1 * ms))] * amp
            sensoryE2.I = i2[:,int( (myclock.t - stim_on) / (1 * ms))] * amp
            sensoryI.I = ii[:,int( (myclock.t - stim_on) / (1 * ms))] * amp
        else:
            sensoryE1.I = 0 * nA
            sensoryE2.I = 0 * nA
            sensoryI.I = 0 * nA
        
          
    #------------------------------------------------------------------------------
    # Initial conditions and Monitors
    #------------------------------------------------------------------------------

    # --- set seed of random number generator to a different value in each run
    np_seed = int(1587.47)
    numpy.random.seed(np_seed)
    py_seed = int(4736.28) 
    pyrandom.seed(py_seed)

    # ---- set initial conditions (random)
    decisionE.gen = decisionE.gen * (1 + 0.2 * rand(decisionE.__len__()))
    decisionI.gen = decisionI.gen * (1 + 0.2 * rand(decisionI.__len__()))
    decisionE.V = decisionE.V + rand(decisionE.__len__()) * 2 * mV
    decisionI.V = decisionI.V + rand(decisionI.__len__()) * 2 * mV

    # ---- set initial conditions (random)
    sensoryE.V = -50.0 * mV - 2 * mV + rand(sensoryE.__len__()) * 2 * mV
    sensoryI.V = -50.0 * mV - 2 * mV + rand(sensoryI.__len__()) * 2 * mV
    sensoryE.gea = 0.05 * (1 + rand(sensoryE.__len__()) * 0.2)
    sensoryI.gea = 0.05 * (1 + rand(sensoryI.__len__()) * 0.2)

    # record spikes of excitatory neurons
    S_DE1 = SpikeMonitor(decisionE1, record=True)
    S_DE2 = SpikeMonitor(decisionE2, record=True)
    S_SE1 = SpikeMonitor(sensoryE1, record=True)
    S_SE2 = SpikeMonitor(sensoryE2, record=True)

    # record instantaneous populations activity
    R_DE1 = PopulationRateMonitor(decisionE1, bin=20*ms)
    R_DE2 = PopulationRateMonitor(decisionE2, bin=20*ms)
    R_SE1 = PopulationRateMonitor(sensoryE1, bin=20*ms)
    R_SE2 = PopulationRateMonitor(sensoryE2, bin=20*ms)
 
 
    #------------------------------------------------------------------------------
    # Run the simulation
    #------------------------------------------------------------------------------

    # construct network       
    net = Network(Dgroups.values(), Sgroups.values(), Dconnections.values(), Sconnections.values(), 
                  Dnetfunctions, update_input, C_SE1_DE1, C_SE2_DE2, C_DE1_SE1, C_DE2_SE2,
                  S_DE1, S_DE2, S_SE1, S_SE2, R_DE1, R_DE2, R_SE1, R_SE2)
    net.prepare()
    net.run(runtime)        
      
       
    #------------------------------------------------------------------------------
    # Show results (single simulation run similar to Fig. 1b)
    #------------------------------------------------------------------------------

    fig, axs = subplots(7,1, figsize=(4,9))

		# Integration circuit	
    fig.add_axes(axs[0])
    raster_plot(S_DE1,color=(1,0,0),title="Integration circuit",xlabel="",ylabel="")
    xlim(0,runtime/ms)   
    ylim(0,len(decisionE1))
    fig.add_axes(axs[1])
    raster_plot(S_DE2,color=(0,0,1),title="",xlabel="",ylabel="")
    xlim(0,runtime/ms)   
    ylim(0,len(decisionE2))
    fig.add_axes(axs[2])
    plot(R_DE1.times/ms,R_DE1.rate,color=(1,0,0))
    plot(R_DE2.times/ms,R_DE2.rate,color=(0,0,1))
    yticks(range(0,50,10))
    xlim(0,runtime/ms)   
    ylim(0,45)
    ylabel("Rate (sp/s)")

    # Sensory circuit
    fig.add_axes(axs[3])
    raster_plot(S_SE1,color=(1,0,0),title="Sensory circuit",xlabel="",ylabel="")
    xlim(0,runtime/ms)   
    ylim(0,len(sensoryE1))
    fig.add_axes(axs[4])
    raster_plot(S_SE2,color=(0,0,1),title="",xlabel="")
    xlim(0,runtime/ms)   
    ylim(0,len(sensoryE2))
    plot(R_SE2.times/ms,R_SE2.rate,color=(0,0,1))
    fig.add_axes(axs[5])
    plot(R_SE1.times/ms,R_SE1.rate,color=(1,0,0))
    yticks(range(0,30,5))
    xlim(0,runtime/ms)   
    ylim(0,20)
    ylabel("Rate (sp/s)")
    
    # Stimulus
    fig.add_axes(axs[6])
    t = linspace(0., runtime/second, runtime/ms)
#    plot(t, numpy.r_[zeros(stim_on/ms), mean(i1,0)/I0, zeros((runtime-stim_off)/ms)], color=(1,0,0))
#    plot(t, numpy.r_[zeros(stim_on/ms), mean(i2,0)/I0, zeros((runtime-stim_off)/ms)], color=(0,0,1))
    plot(t, numpy.r_[zeros(stmdur/4), mean(i1,0)/I0, zeros(stmdur/4)], color=(1,0,0))
    plot(t, numpy.r_[zeros(stmdur/4), mean(i2,0)/I0, zeros(stmdur/4)], color=(0,0,1))
    xticks(range(0,3))
    yticks(range(0,2,1))
    xlim(0,runtime/second)
    ylim(0,1.5)
    xlabel("Time (s)")
    ylabel("Stimulus")

    for i in [0,1,3,4]:
        axs[i].set_yticklabels([])
    for i in range(0,6):
        axs[i].set_xticklabels([])   
    show()

    # PKA
    fig, axs = subplots(1,1,figsize=(3,3))
    t = linspace(0., runtime/second, runtime/ms)
#    plot(t, numpy.r_[zeros(stim_on/ms), mean(i1,0)/I0, zeros((runtime-stim_off)/ms)], color=(1,0,0))
#    plot(t, numpy.r_[zeros(stim_on/ms), mean(i2,0)/I0, zeros((runtime-stim_off)/ms)], color=(0,0,1))
    plot(t, numpy.r_[zeros(stmdur/4), mean(i1,0)/I0 - mean(i2,0)/I0, zeros(stmdur/4)], color=(1,0,0))
    xticks(range(0,3))
    yticks(range(0,2,1))
    xlim(0,runtime/second)
    ylim(0,1.5)
    xlabel("Time (s)")
    ylabel("Stimulus")
    show()
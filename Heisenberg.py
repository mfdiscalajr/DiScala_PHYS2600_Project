import numpy as np
from pylab import *
from numba import jit
import random as rd
import matplotlib.animation as am
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

@jit(nopython = True)
def MCstep_jit(N, L, T,h, state, acceptedMoves, energy, magnetization):

    randomPositions = L * np.random.random(2*N)
    randomArray = np.random.random(N)

    for k in range(N):
        i = int(randomPositions[2*k])
        j = int(randomPositions[2*k+1])
        energy_i = -np.cos(state[i,j] - state[(i+1)%L,j]) + -np.cos(state[i,j] - state[(i-1)%L,j]) + -np.cos(state[i,j] - state[i,(j-1)%L]) + -np.cos(state[i,j] - state[i,(j+1)%L])
        dtheta = np.random.uniform(-np.pi,np.pi)
        newSpinTemp = state[i,j] + dtheta
        energy_f = -np.cos(newSpinTemp - state[(i+1)%L,j]) + -np.cos(newSpinTemp - state[(i-1)%L,j]) + -np.cos(newSpinTemp - state[i,(j-1)%L]) + -np.cos(newSpinTemp - state[i,(j+1)%L])
        dE = energy_f - energy_i
        if dE <= 0 or np.exp(-dE/T) > randomArray[k]:
            acceptedMoves += 1
            newSpin = state[i, j] + dtheta # spin update
            state[i, j] = newSpin
            energy += dE
            #magnetization += 2*newSpin
        magnetization = updateMag(L,state)

    return state, acceptedMoves, energy, magnetization

@jit(nopython = True)
def updateMag(L,state):
    mcos,msin = 0,0
    for ii in range(L):
        for jj in range(L):
            mcos += np.cos(state[ii,jj])
            msin += np.sin(state[ii,jj])
    magnetization = np.sqrt(mcos**2 + msin**2)
    return magnetization

class Heisenberg2D (object):

    """Class that describes equilibrium statistical mechanics of the two-dimensional Ising model"""

    def __init__(self, L=32, temperature=10.0, field = 0):

        self.L = L
        self.N = L**2

        self.temperature = temperature
        self.field = field

        self.w = zeros(9) # store Boltzmann weights
        self.w[8] = exp(-8.0/self.temperature)
        self.w[4] = exp(-4.0/self.temperature)

        #self.state = ones((self.L, self.L), int) # initially all spins up
        #self.state = np.random.random(self.N) * 2 * np.pi #Initialize as random angles (radians)
        self.state = np.full((self.L, self.L), np.pi/2) #Initialize with all spins pointed 90 degrees
        self.energy = -2 * self.N
        self.magnetization = self.N

        self.reset()

    def increment_T(self, T_increment, reset = True):

        T_new = self.temperature + T_increment

        if T_new <= 0:
            T_new = self.temperature

        # self.w[8] = exp(-8.0/T_new)
        # self.w[4] = exp(-4.0/T_new)

        self.temperature = T_new
        if reset:
            self.reset()

    def increment_h(self, h_increment, reset = True):

        h_new = self.field + h_increment

        # self.w[8] = exp(-8.0/T_new)
        # self.w[4] = exp(-4.0/T_new)

        self.field = h_new
        if reset:
            self.reset()


    def reset(self):

        self.monteCarloSteps = 0
        self.acceptedMoves = 0
        self.energyArray = array([], int)
        self.magnetizationArray = array([], int)


    def monteCarloStep(self):

        N = self.N
        L = self.L
        w = self.w
        T = self.temperature
        h = self.field

        state = self.state
        acceptedMoves = self.acceptedMoves
        energy = self.energy
        magnetization = self.magnetization

        state, acceptedMoves, energy, magnetization = MCstep_jit(N, L, T, h, state, acceptedMoves, energy, magnetization)

        self.state = state
        self.acceptedMoves = acceptedMoves
        self.energy = energy
        self.magnetization = magnetization

        self.energyArray.append(self.energy)
        self.magnetizationArray.append(self.magnetization)
        self.monteCarloSteps += 1

    def steps(self, number = 100):

        self.energyArray = self.energyArray.tolist()
        self.magnetizationArray = self.magnetizationArray.tolist()

        for i in range(number):
            self.monteCarloStep()

        self.energyArray = np.asarray(self.energyArray)
        self.magnetizationArray = np.asarray(self.magnetizationArray)


    # Observables
    def meanEnergy(self):
        return self.energyArray.mean() / self.N

    def specificHeat(self):
        return (self.energyArray.std() / self.temperature)**2 / self.N

    def meanMagnetization(self):
        return self.magnetizationArray.mean()/ self.N
        #return abs((1.0*sum(self.state)/self.N))
        #return abs(1.0* self.state.mean())

    def susceptibility(self):
        return (self.magnetizationArray.std())**2 / (self.temperature * self.N)

    def observables(self):
        print("\nTemperature = ", self.temperature)
        print("Mean Energy = ", self.meanEnergy())
        print("Mean Magnetization = ", self.meanMagnetization())
        print("Specific Heat = ", self.specificHeat())
        print("Susceptibility = ", self.susceptibility())
        print("Monte Carlo Steps = ", self.monteCarloSteps, " Accepted Moves = ", self.acceptedMoves)

    def finish(self):
        filepath = '/home/mike/Desktop/Project/Square_XY/'
        np.save(filepath + 'Final_State' + str(self.L) + '.npy',self.state)

    # Visual snapshot of state
    def plot(self):

        pcolormesh(self.state, edgecolor = 'k', cmap = 'binary')

def T_dep(Length):
    t_incr = -0.1
    T = 2
    model = Heisenberg2D(temperature = T, L = Length)   # Tc = 2.3
    Cp = []
    Chi = []
    M = []
    E = []
    Temp = []

    fig = plt.figure(figsize = (15,15), dpi = 100)
    #fig = Figure()
    ax = fig.add_subplot(111)

    count = 0
    while T > 0.2:

        t_incr = -0.02
        count +=1

        #if count%5 == 0:
        #    model.ShowState(T,fig,ax)

        if count%1 == 0:
            #print('\nShowing State...')
            X, Y = np.meshgrid(np.arange(0,model.L),np.arange(0,model.L))
            U = np.cos(model.state)
            V = np.sin(model.state)
            plt.ion()
            plt.title('T=%.2f'%T +', Scale='+str(model.L)+'x'+str(model.L), fontsize = 18)
            plt.quiver(X,Y,U,V, units = 'width')
            plt.axis('off')
            plt.draw()
            plt.pause(1)
            plt.clf()

        model.steps(number=1000)
        model.reset()
        model.steps(number=10000)
        model.observables()
        Cp.append(model.specificHeat())
        Chi.append(model.susceptibility())
        M.append(model.meanMagnetization())
        E.append(model.meanEnergy())
        Temp.append(T)

        model.increment_T(t_incr)

        T = model.temperature

    Cp = np.array(Cp)
    Chi = np.array(Chi)
    M = np.array(M)
    E = np.array(E)
    Temp = np.array(Temp)

    model.finish() #Saves final state once quenching is completed

    return Temp, E, M, Cp, Chi


import numpy as np
from pylab import *  # plotting library
from numba import jit
import random as rd
import matplotlib.animation as am

@jit(nopython = True)
def MCstep_jit(N, L, T,h, state, acceptedMoves, energy, magnetization):

    randomPositions = L * np.random.random(2*N)
    randomArray = np.random.random(N)

    for k in range(N):

        i = int(randomPositions[2*k])
        j = int(randomPositions[2*k+1])

        dE = 2*state[i, j] * (state[(i+1)%L, j] + state[i-1, j] + state[i, (j+1)%L] + state[i, j-1]) + h*state[i,j]

        if dE <= 0 or np.exp(-dE/T) > randomArray[k]:
            acceptedMoves += 1
            newSpin = -state[i, j] # flip spin
            state[i, j] = newSpin
            energy += dE
            magnetization += 2*newSpin
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

@jit(nopython=True)
def Neighbors(L,SpinSite):
    return [(SpinSite[0], (SpinSite[1]+1)%L), (SpinSite[0],(SpinSite[1]-1)%L), ((SpinSite[0]+1)%L, SpinSite[1]), ((SpinSite[0]-1)%L,SpinSite[1])]

@jit(nopython = True)
def WolffStep(N, L, T,h, state, acceptedMoves, energy, magnetization):
    Padd = 1.0 - np.exp(-2.0/T)
    randomPos = L * np.random.random(2 * N)

    i = int(np.random.choice(randomPos))
    j = int(np.random.choice(randomPos))
    Deck, Cluster = [(i,j)], [(i,j)]
    while len(Deck) > 0:
        RandSpin = Deck.pop()
        nbrs = Neighbors(L,RandSpin)
        for nbr in nbrs:
            if state[nbr[0],nbr[1]] == state[RandSpin[0],RandSpin[1]] and nbr not in Cluster and rd.uniform(0.0,1.0) < Padd:
                Deck.append(nbr)
                Cluster.append(nbr)
    for s in Cluster:
        (ii,jj) = s
        newSpin = -state[ii,jj]
        state[ii,jj] = newSpin
        nbrs2 = Neighbors(L,s)
        for nbr2 in nbrs2:
            if nbr2 not in Cluster:
                (k,l) = nbr2
                energy += -2*state[k, l] * state[ii,jj]
    magnetization = abs(state.sum())
    #magnetization += 2 * len(Cluster) * newSpin
    #magnetization = abs(magnetization)
    acceptedMoves += len(Cluster)

    return state, acceptedMoves, energy, magnetization

@jit(nopython = True)
def Flip(cluster, state, energy, magnetization):
    for s in cluster:
        (ii,jj) = s
        newSpin = -state[ii,jj]
        state[ii,jj] = newSpin
        nbrs = Neighbors(L,s)
        for nbr in nbrs:
            if nbr not in cluster:
                (k,l) = nbr
                energy += -2*state[k,l] * state[ii,jj] + h*state[ii,jj]
    magnetization = abs(state.sum())

    return state,energy,magnetization

def initialize(L):
    nbrlist = [[[]for i in range(L)] for j in range(L)]
    for k in range(L):
        for t in range(L):
            nbrlist[k][t] = [[k,(t+1)%L] , [k,(t-1) %L] , [(k+1)%L,t] , [(k-1)%L,t]]
    return nbrlist

class Ising2D (object):

    """Class that describes equilibrium statistical mechanics of the two-dimensional Ising model"""

    def __init__(self, L=32, temperature=10.0, field = 0):

        #np.random.seed(222)

        self.L = L
        self.N = L**2

        self.temperature = temperature
        self.field = field

        self.w = zeros(9) # store Boltzmann weights
        self.w[8] = exp(-8.0/self.temperature)
        self.w[4] = exp(-4.0/self.temperature)

        self.state = ones((self.L, self.L), int) # initially all spins up
        self.energy = -2 * self.N
        self.magnetization = self.N

        self.nbrlist = initialize(L)

        self.reset()

    def initialize(self):
        nbrlist = {}
        for i in range(self.L):
            for j in range(self.L):
                nbrlist[i,j] = [[(i+1)%self.L,j] , [(i-1)%L,j] , [i,(j+1)%self.L] , [i,(j-1)%self.L]]
        return nbrlist

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
        nbrlist = self.nbrlist

        state = self.state
        acceptedMoves = self.acceptedMoves
        energy = self.energy
        magnetization = self.magnetization

        state, acceptedMoves, energy, magnetization = WolffStep(N, L, T, h, state, acceptedMoves, energy, magnetization)
        #state, acceptedMoves, energy, magnetization = MCstep_jit(N, L, T, h, state, acceptedMoves, energy, magnetization)

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


    # Visual snapshot of state
    def plot(self):

        pcolormesh(self.state, edgecolor = 'k', cmap = 'binary')

def T_dep(Length):
    t_incr = -0.1
    T = 5
    model = Ising2D(temperature = T, L = Length)   # Tc = 2.3
    Cp = []
    Chi = []
    M = []
    E = []
    Temp = []

    while T > 1.0 :

        if T <= 3.0 :
            t_incr = -0.05

        if T <= 2.4 :
            t_incr = -0.01

        if T <= 2.23 :
            t_incr = -0.05

        if T <= 1.8 :
            t_icncr = -0.1

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

    return Temp, E, M, Cp, Chi

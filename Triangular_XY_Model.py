import numpy as np
import random as rd
import math as m
from numba import jit
from matplotlib import gridspec
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

L =10
N = L ** 2
#state = ones((L,L),int)
state = np.full((L,L), 90, int)
#state = np.random.choice([1,-1],size = (L,L))
#state =np.array([[np.random.randint(0,360) for i in range(L)] for j in range(L)])
energy = -2*N
magnetization = N
field = 0
acceptedMoves = 0
Hamiltonian = np.zeros((L,L))

@jit(nopython=True)
def Neighbors(L,SpinSite):
    return [(SpinSite[0],(SpinSite[1]+1)%L) , (SpinSite[0],(SpinSite[1]-1) %L) , ((SpinSite[0]+1)%L,SpinSite[1]) , ((SpinSite[0]-1)%L,SpinSite[1]) , ((SpinSite[0]-1)%L,(SpinSite[1]-1)%L) , ((SpinSite[0]+1)%L,(SpinSite[1]+1)%L)]

@jit(nopython = True)
def getCluster(N, L, T, state):
    Padd = 1.0 - np.exp(-2.0/T)
    randomPos = L * np.random.random(2 * N)
    randomArray = np.random.random(N)

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
    return Cluster

@jit(nopython = True)
def Stepping(N,L,T,h,state,acceptedMoves,energy,magnetization,random = False):
    Cluster = getCluster(N,L,T,state) #get cluster of spins from wolff algorithim

    #dtheta = np.random.uniform(-np.pi,np.pi)
    if random:
        dtheta = np.random.randint(0,360) # generate random angle to change spin
    else:
        dtheta = 1 #degrees
    for s in Cluster:
        (ii,jj) = s
        oldSpin = state[ii,jj]
        newSpin = state[ii,jj] + dtheta
        newSpin  = (newSpin + 180) %(360) - 180 #keeps angles between 0 and 360
        #if newSpin <0:
        #    newSpin += 360
        state[ii,jj] = newSpin
        nbrs = Neighbors(L,s)
        for nbr in nbrs:
            if nbr not in Cluster:
                (k,l) = nbr #spin not in cluster
                energy -= -np.cos(np.radians(state[k, l]) - np.radians(oldSpin))
                energy += -np.cos(np.radians(state[k, l]) - np.radians(newSpin))
    #energy = getEnergy(L,state)
    acceptedMoves += len(Cluster)
    magnetization = updateMag(L,state)
    return state, acceptedMoves, energy, magnetization


def Randomize(N,L,T,h,state,acceptedMoves,energy,magnetization):
    for n in range(1000):
        state,acceptedMoves,energy,magnetization = Stepping(N,L,T,h,state,acceptedMoves,energy,magnetization,random = True)
    acceptedMoves = 0
    energy = initializeE(L,Hamiltonian,state)
    return state,acceptedMoves,energy,magnetization

@jit(nopython = True)
def initializeE(L,Hamiltonian,state):
    for i in range(L):
        for j in range(L):
            Hamiltonian[i,j] = 0
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[i,(j+1)%L]))
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[i,(j-1)%L]))
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[(i+1)%L,j]))
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[(i-1)%L,j]))
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[(i-1)%L,(j-1)%L]))
            Hamiltonian[i,j] -= np.cos(np.radians(state[i,j]) - np.radians(state[(i+1)%L,(j+1)%L]))
    return sum(Hamiltonian)/3

@jit(nopython = True)
def updateMag(L,state):
    mcos,msin = 0,0
    for ii in range(L):
        for jj in range(L):
            mcos += np.cos(state[ii,jj])
            msin += np.sin(state[ii,jj])
    magnetization = np.sqrt(mcos**2 + msin**2)
    return magnetization

def getCorrelation(L,state,corr_r):
    fixed = [5,5]
    corri,corrj = fixed
    correlations = state[(corri - corr_r) %L, corrj] + state[(corri + corr_r)%L, corrj] + state[corri, (corrj-corr_r)%L] + state[corri,(corrj-corr_r)%L]
    return state[corri,corrj] * correlations /4

@jit(nopython = True)
def getEnergy(L,state):
    E = 0
    for ii in range(L):
        for jj in range(L):
            site = (ii,jj)
            nbrs = Neighbors(L,site)
            for nbr in nbrs:
                (k,l) = nbr
                E += -np.cos((np.deg2rad(state[ii,jj]) - np.deg2rad(state[k,l])))/3
    return E

def getMeanMag(MagArray,N):
    return(MagArray.mean()/N)

def getMeanEng(EArray,N):
    return EArray.mean()/N

def getSpecificHeat(Energy_array,Temperature,N):
    return (Energy_array.std() / Temperature)**2 / N

def getSusceptibility(MagArray,Temperature,N):
    return (MagArray.std()/Temperature)**2/N

def Quench(Tmax,Tmin,Tstep,nsteps,N,L,h,state,acceptedMoves,energy,magnetization, show = False, critical = False, run = True):
    filepath = '/home/mike/Desktop/Project/Triangular_XY_Model/'
    if run == True:
        TempList = np.arange(Tmax,Tmin,Tstep) #setup temperature list
        Cp,Chi,M,E,Temp = [], [], [], [], []
        Xi1,Xi2,Xi3,Xi4,Xi5,Xi6 = [],[],[],[],[],[]

        expand = input('Would you like to choose a different step size for a certain temperature region (y/n): ')
        if expand == 'y':
            Tmidmax = float(input('Max Temperature for region: '))
            Tmidmin = float(input('Min Temperature for region: '))
            Tmidstep = float(input('Step Size for region: '))
            TempList = np.ravel(list(np.arange(Tmax,Tmidmax,Tstep)) + list(np.arange(Tmidmax-Tmidstep,Tmidmin,Tmidstep)) + list(np.arange(Tmidmin-Tmidstep,Tmin,Tstep)))

        #TempList = np.ravel(list(np.arange(1.5,0.6,-0.02)) + list(np.arange(0.58,0.499,-0.0001)) + list(np.arange(0.45,0.2,-0.01)))

        #Start by Randomizing the state
        state,acceptedMoves,energy,magnetization = Randomize(N,L,Tmax,h,state,acceptedMoves,energy,magnetization)

        #Run Wolff Algorithim for given temperatures
        for i,T in enumerate(TempList):
            print('\nTemperature: %.4f' %T)
            MagArray,EArray = array([],int), array([],int)
            acceptedMoves = 0
            for n in range(nsteps):
                state, acceptedMoves, energy, magnetization = Stepping(N,L,T,h,state,acceptedMoves,energy,magnetization,random = True)
                MagArray = np.append(MagArray,magnetization)
                EArray = np.append(EArray,energy)
            Cp.append(getSpecificHeat(EArray,T,N))
            Chi.append(getSusceptibility(MagArray,T,N))
            M.append(getMeanMag(MagArray,N))
            E.append(getMeanEng(EArray,N))
            Temp.append(T)

            #Correlation Lengths
            Xi1.append((getCorrelation(L,state,1)))
            Xi2.append((getCorrelation(L,state,2)))
            Xi3.append((getCorrelation(L,state,3)))
            Xi4.append((getCorrelation(L,state,4)))
            Xi5.append((getCorrelation(L,state,5)))
            Xi6.append((getCorrelation(L,state,6)))

            print('Accepted Moves: ' + str(acceptedMoves))
            print('Mean Magnetization: ' + str(M[i]))
            print('Mean Energy: ' + str(E[i]))
            print('Susceptibility: ' + str(Chi[i]))
            print('Specific Heat: ' + str(Cp[i]))

        #Save Data
        np.save(filepath + 'Temp_Triangular_Wolff_10.npy', TempList)
        np.save(filepath + 'Energy_Triangular_Wolff_10.npy', E)
        np.save(filepath + 'Magnetization_Triangular_Wolff_10.npy', M)
        np.save(filepath + 'Specific_Heat_Triangular_Wolff_10.npy', Cp)
        np.save(filepath + 'Susceptibility_Triangular_Wolff_10.npy', Chi)
        np.save(filepath + 'Correlation_Function1_Wolff_10.npy', Xi1)
        np.save(filepath + 'Correlation_Function2_Wolff_10.npy', Xi2)
        np.save(filepath + 'Correlation_Function3_Wolff_10.npy', Xi3)
        np.save(filepath + 'Correlation_Function4_Wolff_10.npy', Xi4)
        np.save(filepath + 'Correlation_Function5_Wolff_10.npy', Xi5)
        np.save(filepath + 'Correlation_Function6_Wolff_10.npy', Xi6)

    elif run == False:
        print('Loading Data...')
        TempList,E,M,Cp,Chi = np.load(filepath + 'Temp_Triangular_Wolff_10.npy'), np.load(filepath + 'Energy_Triangular_Wolff_10.npy'), np.load(filepath + 'Magnetization_Triangular_Wolff_10.npy'), np.load(filepath + 'Specific_Heat_Triangular_Wolff_10.npy'), np.load(filepath + 'Susceptibility_Triangular_Wolff_10.npy')
        Xi1,Xi2,Xi3,Xi4,Xi5,Xi6 = np.load(filepath + 'Correlation_Function1_Wolff_10.npy'), np.load(filepath + 'Correlation_Function2_Wolff_10.npy'), np.load(filepath + 'Correlation_Function3_Wolff_10.npy'),np.load(filepath + 'Correlation_Function4_Wolff_10.npy'),np.load(filepath + 'Correlation_Function5_Wolff_10.npy'),np.load(filepath + 'Correlation_Function6_Wolff_10.npy')


    #Show Final Results of Simulation
    if show:
        fig1 = plt.figure(figsize = [15,10])
        gs = gridspec.GridSpec(ncols = 2, nrows = 2)
        ax1 = fig1.add_subplot(gs[0])
        ax2 = fig1.add_subplot(gs[1])
        ax3 = fig1.add_subplot(gs[2])
        ax4 = fig1.add_subplot(gs[3])
        axes = [ax1,ax2,ax3,ax4]

        fig2 = plt.figure(figsize = [15,10])
        gs2 = gridspec.GridSpec(ncols = 2, nrows = 1)
        ax1_f2 = fig2.add_subplot(gs2[0])
        ax2_f2 = fig2.add_subplot(gs2[1])

        ax1.plot(TempList,E, marker = 'o', ls = 'None', color = 'k')
        ax2.plot(TempList,M, marker = 'o', ls = 'None', color = 'b')
        ax3.plot(TempList,Cp, marker = 'o', ls = 'None', color = 'orange')
        ax4.plot(TempList,Chi, marker = 'o', ls = 'None', color = 'cyan')

        ax1_f2.plot(TempList,Xi1,marker = 'o', ls = 'None')
        ax1_f2.plot(TempList,Xi2,marker = 'o', ls = 'None')
        ax1_f2.plot(TempList,Xi3,marker = 'o', ls = 'None')
        ax1_f2.plot(TempList,Xi4,marker = 'o', ls = 'None')
        ax1_f2.plot(TempList,Xi5,marker = 'o', ls = 'None')
        ax1_f2.plot(TempList,Xi6,marker = 'o', ls = 'None')
        r_list = np.arange(1,7)

        np.array([Xi1,Xi2,Xi3,Xi4,Xi5,Xi6])
        ax2_f2.plot(r_list, )
        num = (np.where(abs(np.amax(Cp) - Cp) == 0))[0][0]
        Tc = TempList[num]

        if critical:
            #Critical Exponents
            #Specific Heat
            SpecificHeatExpPara = lambda T,alpha: ((T-Tc)/Tc)**(-alpha) - 0.25
            SpecificHeatExpFerro = lambda T,alphap: (-(T-Tc)/Tc)**(-alphap)
            popt_cp_Para,_ = curve_fit(SpecificHeatExpPara, TempList[:num], Cp[:num], [0.6455])
            popt_cp_Ferro,_ = curve_fit(SpecificHeatExpFerro, TempList[num:], Cp[num:], [-0.6])
            CpCriticalPara = SpecificHeatExpPara(TempList[:num],*popt_cp_Para)
            CpCriticalFerro = SpecificHeatExpFerro(TempList[num:],*popt_cp_Ferro)
            ax3.plot(TempList[:num],CpCriticalPara,lw = 3, label = 'Paramagnetic Phase Fit', color = 'sienna')
            #ax3.plot(TempList[num+1:], CpCriticalFerro,lw = 3, label = 'Ferromagnetic Phase Fit', color = 'darkgoldenrod')
            print(r'$\alpha$ = ' + str(popt_cp_Para[0]))
            #print(r'$\alphap$ = ' + str(popt_cp_Ferro[0]))

            #Susceptibility
            SusceptibilityExp_Para= lambda T,gamma: ((T-Tc)/Tc)**(-gamma)
            bounds = (0.1,0.5)
            popt_s,pcov = curve_fit(SusceptibilityExp_Para,TempList[:num],Chi[:num], [0.35], bounds = bounds)
            ChiCritical_Para = SusceptibilityExp_Para(TempList[:num],*popt_s)
            ax4.plot(TempList[:num],ChiCritical_Para-0.9, lw = 3, label = 'Paramagnetic Phase Fit', color = 'steelblue')
            print(r'$\Gamma$ = ' + str(popt_s[0]))

        for a in axes:
            a.grid()
            a.set_xlabel('Temperature', fontsize = 15)
            a.set_xlim([np.amin(TempList), np.amax(TempList)])
            a.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 13)
            a.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 13)
            if a == ax1:
                a.text(0.05, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
            elif a == ax2:
                a.text(0.75, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
            else:
                a.text(0.75, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
                a.legend(loc = 7, shadow  = True)
            a.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            a.xaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            a.axvline(Tc, color = 'darkred', ls = '--', zorder = 3)


        ax1.set_ylim([np.amin(E), np.amax(E)])
        ax2.set_ylim([-1.05,1.05])
        #ax3.set_ylim([np.amin(Cp), np.amax(Cp)])
        #ax4.set_ylim([np.amin(Chi), np.amax(Chi)])
        ax1.set_ylabel('Energy', fontsize = 15)
        ax2.set_ylabel('Magnetization', fontsize = 15)
        ax3.set_ylabel('Specific Heat', fontsize = 15)
        ax4.set_ylabel('Susceptibility', fontsize = 15)
        fig1.suptitle('Ferromagnetic Triangular Heisenberg Model',fontsize = 20)
        gs.tight_layout(fig1)
        gs.update(top = 0.90)
        fig1.show()
        fig2.show()

        save = input('Save Figure? (y/n): ')
        if save == 'y':
            filename = input('Filename: ')
            fig1.savefig(filepath + filename)



@jit(nopython = True)
def flip(Cluster,state,energy,magnetization):
    Cluster = np.array(Cluster)
    for s in Cluster:
        newSpin = -state[s[0],s[1]]
        state[s[0],s[1]] = newSpin
        nbrs = Neighbors(L,s)
    return state,energy,magnetization

@jit(nopython = True)
def Flip(L, h, cluster, state, energy, magnetization):
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

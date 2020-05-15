import Triangular_Heisenberg
import matplotlib as mpl
mpl.use('Qt5Agg', warn = False, force = True)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
import numpy as np

#Run Simulation
L = 20
filepath = '/home/mike/Desktop/Project/Triangular_XY_Model/'
anti,ferro = 0,1
gen = 0
if gen == 1:
    Temp,E,M,Cp,Chi = Triangular_Heisenberg.T_dep(L)
    if anti == 1:
        #Save Data
        np.save(filepath + 'Temp_Anti.npy', Temp)
        np.save(filepath + 'Energy_Anti.npy', E)
        np.save(filepath + 'Magnetization_Anti.npy', M)
        np.save(filepath + 'Specific_Heat_Anti.npy', Cp)
        np.save(filepath + 'Susceptibility_Anti.npy', Chi)
    if ferro == 1:
        #Save Data
        np.save(filepath + 'Temp_Metro_' + str(L) + '.npy', Temp)
        np.save(filepath + 'Energy_Metro_' + str(L) + '.npy', E)
        np.save(filepath + 'Magnetization_Metro_' + str(L) + '.npy', M)
        np.save(filepath + 'Specific_Heat_Metro_' + str(L) + '.npy', Cp)
        np.save(filepath + 'Susceptibility_Metro_' + str(L) + '.npy', Chi)
#Load Data
if anti == 1:
    Temp,E,M,Cp,Chi = np.load(filepath + 'Temp_Anti.npy'), np.load(filepath + 'Energy_Anti.npy'), np.load(filepath + 'Magnetization_Anti.npy'), np.load(filepath + 'Specific_Heat_Anti.npy'), np.load(filepath + 'Susceptibility_Anti.npy')
if ferro == 1:
    Temp,E,M,Cp,Chi = np.load(filepath + 'Temp_Metro_' + str(L) + '.npy'), np.load(filepath + 'Energy_Metro_' + str(L) + '.npy'), np.load(filepath + 'Magnetization_Metro_' + str(L) + '.npy'), np.load(filepath + 'Specific_Heat_Metro_' + str(L) + '.npy'), np.load(filepath + 'Susceptibility_Metro_' + str(L) + '.npy')
    FinalState = np.load(filepath + 'Final_State' + str(L) + '.npy')
#Graph
fig1 = plt.figure(figsize = [15,10])
gs = gridspec.GridSpec(ncols = 2, nrows = 2)
ax1 = fig1.add_subplot(gs[0])
ax2 = fig1.add_subplot(gs[1])
ax3 = fig1.add_subplot(gs[2])
ax4 = fig1.add_subplot(gs[3])
axes = [ax1,ax2,ax3,ax4]

ax1.plot(Temp,E, marker = 'o', ls = 'None', color = 'k')
ax2.plot(Temp,M, marker = 'o', ls = 'None', color = 'b')
ax3.plot(Temp,Cp, marker = 'o', ls = 'None', color = 'orange')
ax4.plot(Temp,Chi, marker = 'o', ls = 'None', color = 'cyan')

for a in axes:
    num = np.where(abs(np.amax(Cp) - Cp) == 0)
    Tc = Temp[num]
    a.grid()
    a.set_xlabel('Temperature', fontsize = 15)
    a.set_xlim([np.amin(Temp), np.amax(Temp)])
    a.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 13)
    a.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 13)
    if a == ax4:
        a.text(0.05, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    elif a == ax2:
        a.text(0.75, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    else:
        a.text(0.05, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    a.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    a.axvline(Tc, color = 'darkred', ls = '--', zorder = 3)

if anti == 1:
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    #ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    #ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
if ferro == 1:
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))

ax1.set_ylim([np.amin(E), np.amax(E)])
ax2.set_ylim([-1,1])
ax3.set_ylim([np.amin(Cp), np.amax(Cp)])
ax4.set_ylim([np.amin(Chi), np.amax(Chi)])
ax1.set_ylabel('Energy', fontsize = 15)
ax2.set_ylabel('Magnetization', fontsize = 15)
ax3.set_ylabel('Specific Heat', fontsize = 15)
ax4.set_ylabel('Susceptibility', fontsize = 15)
fig1.suptitle('Triangular Ferromagnetic XY Model',fontsize = 20)
gs.tight_layout(fig1)
gs.update(top = 0.90)
fig1.show()


def ShowGroundState(L):
        X, Y = np.meshgrid(np.arange(0,L),np.arange(0,L))
        U = np.cos(FinalState)
        V = np.sin(FinalState)
        plt.figure(figsize=(4,4), dpi=100)
        Q = plt.quiver(X, Y, U, V, units='width')
        qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',
                    coordinates='figure')
        plt.title('T=%.2f'%temperature+', Scale='+str(L)+'x'+str(L))
        #plt.axis('off')
        plt.show()

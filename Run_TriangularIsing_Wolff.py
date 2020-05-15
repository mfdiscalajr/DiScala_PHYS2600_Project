import TriangularIsing
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker

#Run Simulation
L = 100
filepath = '/home/mike/Desktop/Project/TriangularIsing_Data/'
anti,ferro = 0,1
gen = 1
if gen == 1:
    Temp,E,M,Cp,Chi=TriangularIsing.T_dep(L)
    if anti == 1:
        #Save Data
        np.save(filepath + 'Temp_Anti_Wolff.npy', Temp)
        np.save(filepath + 'Energy_Anti_Wolff.npy', E)
        np.save(filepath + 'Magnetization_Anti_Wolff.npy', M)
        np.save(filepath + 'Specific_Heat_Anti_Wolff.npy', Cp)
        np.save(filepath + 'Susceptibility_Anti_Wolff.npy', Chi)
    if ferro == 1:
        #Save Data
        np.save(filepath + 'Temp_Ferro_Wolff_run4.npy', Temp)
        np.save(filepath + 'Energy_Ferro_Wolff_run4.npy', E)
        np.save(filepath + 'Magnetization_Ferro_Wolff_run4.npy', M)
        np.save(filepath + 'Specific_Heat_Ferro_Wolff_run4.npy', Cp)
        np.save(filepath + 'Susceptibility_Ferro_Wolff_run4.npy', Chi)
#Load Data
if anti == 1:
    Temp,E,M,Cp,Chi = np.load(filepath + 'Temp_Anti_Wolff.npy'), np.load(filepath + 'Energy_Anti_Wolff.npy'), np.load(filepath + 'Magnetization_Anti_Wolff.npy'), np.load(filepath + 'Specific_Heat_Anti_Wolff.npy'), np.load(filepath + 'Susceptibility_Anti_Wolff.npy')
if ferro == 1:
    Temp,E,M,Cp,Chi = np.load(filepath + 'Temp_Ferro_Wolff_run4.npy'), np.load(filepath + 'Energy_Ferro_Wolff_run4.npy'), np.load(filepath + 'Magnetization_Ferro_Wolff_run4.npy'), np.load(filepath + 'Specific_Heat_Ferro_Wolff_run4.npy'), np.load(filepath + 'Susceptibility_Ferro_Wolff_run4.npy')

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

num = np.where(abs(np.amax(Cp) - Cp) == 0)
Tc = Temp[num]
#Critical Exponents
#Specific Heat
SpecificHeatExpPara = lambda T,alpha: ((T-Tc)/Tc)**(-alpha)
SpecificHeatExpFerro = lambda T,alphap: (-(T-Tc)/Tc)**(-alphap)
popt_cp_Para,_ = curve_fit(SpecificHeatExpPara, Temp[:num], Cp[:num], [0.01])
popt_cp_Ferro,_ = curve_fit(SpecificHeatExpFerro, Temp[num:], Cp[num:], [-0.6])
CpCriticalPara = SpecificHeatExpPara(Temp[:num],*popt_cp_Para)
CpCriticalFerro = SpecificHeatExpFerro(Temp[num+1:],*popt_cp_Ferro)
scaleCp = np.amax(Cp)/np.amax(CpCriticalPara[:num])
#ax3.plot(Temp[:num],(CpCriticalPara[:num]),lw = 3, label = 'Paramagnetic Phase Fit', color = 'sienna')
#ax3.plot(Temp[num+1:], CpCriticalFerro,lw = 3, label = 'Ferromagnetic Phase Fit', color = 'darkgoldenrod')
print(r'$\alpha$ = ' + str(popt_cp_Para[0]))
#print(r'$\alphap$ = ' + str(popt_cp_Ferro[0]))
CpCrit = SpecificHeatExpPara(Temp[:num],0.01)
ax3.plot(Temp[:num], CpCrit*scaleCp -2)

#Susceptibility
SusceptibilityExp_Para= lambda T,gamma: ((T-Tc)/Tc)**(-gamma)
popt_s,pcov = curve_fit(SusceptibilityExp_Para,Temp[:num],Chi[:num], [7/4])
ChiCritical_Para = SusceptibilityExp_Para(Temp[:num],*popt_s)
scaleS = np.amax(Chi)/np.amax(ChiCritical_Para[:num])
ax4.plot(Temp[:num],(ChiCritical_Para),lw = 3, label = 'Paramagnetic Phase Fit', color = 'steelblue')
print(r'$\Gamma$ = ' + str(popt_s[0]))

for a in axes:
    a.grid()
    a.set_xlabel('Temperature', fontsize = 15)
    a.set_xlim([np.amin(Temp), np.amax(Temp)])
    a.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 13)
    a.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 13)
    if a == ax4:
        a.text(0.05, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    elif a == ax2:
        a.text(0.05, 0.90, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    else:
        a.text(0.05, 0.93, r'$T_c \approx$ %.3f' %Tc, fontsize = 15, transform = a.transAxes, color = 'k' )
    a.xaxis.set_major_locator(ticker.MultipleLocator(1))
    a.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    a.axvline(Tc, color = 'darkred', ls = '--', zorder = 3)

#if anti == 1:
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    #ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    #ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    #ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    #ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    #ax4.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
if ferro == 1:
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    #ax3.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    #ax4.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax4.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

ax1.set_ylim([np.amin(E), np.amax(E)])
ax2.set_ylim([-1.05,1.05])
ax3.set_ylim([np.amin(Cp), np.amax(Cp)])
ax4.set_ylim([np.amin(Chi), np.amax(Chi)])
ax1.set_ylabel('Energy', fontsize = 15)
ax2.set_ylabel('Magnetization', fontsize = 15)
ax3.set_ylabel('Specific Heat', fontsize = 15)
ax4.set_ylabel('Susceptibility', fontsize = 15)
fig1.suptitle('Ferromagnetic Triangular Ising Model',fontsize = 20)
gs.tight_layout(fig1)
gs.update(top = 0.90)

fig1.show()

import Heisenberg
import matplotlib as mpl
mpl.use('Qt5Agg', warn = False, force = True)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit, newton
import numpy as np


choice = input('Choose Fitting Method (Diff, DiffLog, LSF, Newton, Ratio, Actual): ')
#Extrapolate Tc from Finite-Size Scaling
Llist = [10,12,15,18,20,30]
TcList = []
for L in Llist:
    filepath = '/home/mike/Desktop/Project/Triangular_XY_Model/'
    Cp = np.load(filepath + 'Specific_Heat_Metro_' + str(L) + '.npy')
    Temp = np.load(filepath + 'Temp_Metro_' + str(L) + '.npy')

    num = np.where(abs(np.amax(Cp) - Cp) == 0)[0][0]
    Tc = Temp[num]

    TcList.append(Tc)

TcList = np.array(TcList)
#TcList = np.delete(TcList,4)
#invL = np.power(np.log(Llist),-2)
invL = np.log(Llist) ** -2
#invL = np.delete(invL,4)

fig1 = plt.figure(figsize = [15,10])
gs = gridspec.GridSpec(ncols = 1, nrows = 1)
ax1 = fig1.add_subplot(gs[0])

ax1.scatter(invL,TcList, s = 50, c = 'lightslategrey', zorder = 3)

    #Fitting
#TcScaling = lambda L,c,Tinf: np.pi ** 2 / (4 * c) * np.log(L) ** -2 + Tinf
#def TcScaling(L,c,Tinf):
#    constant = np.power(np.pi,2) / (4*c)
#    num = constant * np.log(L)**-2 + Tinf
#    return num

def TcScaling(x,c,Tinf):
    constant = np.power(np.pi,2) / (4*c)
    num = constant * x + Tinf
    return num

popt_T,pcov_T = curve_fit(TcScaling, invL, TcList, [1,1])#, bounds = bounds)
ExtrapTc = TcScaling(invL,*popt)
ax1.plot(invL,ExtrapTc, lw = 3, label = 'Extrapolated $T_c$ = %.3f' %popt[1], color = 'sienna')

ax1.grid()
ax1.set_xlabel('$ln(L)^{-2}$', fontsize = 18)
ax1.set_ylabel('$T_c(L)$', fontsize = 18)
ax1.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 16)
ax1.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 16)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.004))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.004))
ax1.legend(loc = 2, prop = {'size': 25}, shadow = True)
gs.tight_layout(fig1)
gs.update(top = 0.90)
fig1.show()

#Susceptibility
Chi10 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[0]) + '.npy')
Chi12 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[1]) + '.npy')
Chi15 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[2]) + '.npy')
Chi18 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[3]) + '.npy')
Chi20 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[4]) + '.npy')
Chi30 = np.load(filepath + 'Susceptibility_Metro_' + str(Llist[5]) + '.npy')

Temp = np.load(filepath + 'Temp_Metro_' + str(L) + '.npy')
redT = (Temp-TcList[0])/TcList[0]

fig2 = plt.figure(figsize = [15,10])
gs = gridspec.GridSpec(ncols = 1, nrows = 1)
ax1_f2 = fig2.add_subplot(gs[0])


ChiScaling = lambda L,gamma,nu,Chi: L ** (-gamma/nu) * Chi

if choice == 'Diff':
    def ChiScalingFunc(x,gamma,nu):
        return (Ly/Lx) ** (gamma/nu) * x

    Lx,Ly = 10,12
    bounds = ([1,0.5],[1.4,0.67])
    popt_cp,pcov_cp = curve_fit(ChiScalingFunc, (Chi10), (Chi12), [1, 0.55], bounds = bounds, method = 'trf')

    minGam,minNu = popt_cp[0], popt_cp[1]
    print('Gamma = %.2f and Nu = %.2f' %(minGam,minNu))
    print('Covariance Matrix: \n' + str(pcov_cp) )

if choice == 'DiffLog':
    def ChiScalingLog(x,Gamma,nu):
        return (gamma/nu) * np.log(Ly/Lx) + x

    Lx,Ly = 10,12
    bounds = ([1,0.5],[1.4,0.67])
    popt_cp,pcov_cp = curve_fit(ChiScalingLog, np.log(Chi10), np.log(Chi12), [1, 0.55], bounds = bounds, method = 'trf')

    minGam,minNu = popt_cp[0], popt_cp[1]
    print('Gamma = %.2f and Nu = %.2f' %(minGam,minNu))
    print('Covariance Matrix: \n' + str(pcov_cp) )

if choice == 'Newton':
    Lx,Ly = 10,12
    def ChiDiff(alphaTilde,ratio):
        return (Lx ** (-alphaTilde))/(Ly ** -(alphaTilde)) * ratio

    loop = [newton(ChiDiff,1.8,args = (Chi12[i]/Chi10[i],)) for i in range(Chi12.size)]

if choice == 'LSF':
    nuList = np.arange(0.4,0.82,0.02)
    gammaList = np.arange(1,1.5,0.02)

    Chi20Scaling = np.array([[ChiScaling(Llist[-2],gam,n,Chi20) for n in nuList] for gam in gammaList])
    Chi30Scaling = np.array([[ChiScaling(Llist[-1],gam,n,Chi30) for n in nuList] for gam in gammaList])
    data_points = len(Chi30Scaling[0,0])

#Least Square Fit Method
#print('Minimizing Gamma and Nu using Susceptibility Data...')
    Xi2 = np.zeros([len(gammaList), len(nuList)])
    for i,gam in enumerate(gammaList):
        for j,n in enumerate(nuList):
            #print('Testing Gamma = %.2f and Nu = %.2f' %(gam,n))
            data20 = Chi20Scaling[i,j]
            data30 = Chi30Scaling[i,j]
            xi2 = 0.0
            for k in range(data_points):
                #sigma = np.sqrt(data30[k])
                sigma = 1
                residual = (data20[k] - data30[k])/sigma
                xi2 = xi2 + np.power(residual,2)
            #print('Reduced Xi2 = ' +str(xi2/(data_points-1)))
            Xi2[i,j] = xi2/(data_points-1)

    GlobalMin = np.amin(Xi2)
    MinLoc = np.where(abs(GlobalMin - Xi2) == 0)
    minGam, minNu = gammaList[MinLoc[0][0]], nuList[MinLoc[1][0]]
    print('Minimized Gamma = %.3f and Minimized Nu = %.3f' %(minGam,minNu))

#Plot Results
if choice != 'Ratio' and choice != 'Actual':
    ax1_f2.plot(Temp,ChiScaling(Llist[0],minGam,minNu,Chi10), marker = 'o', ls = 'None', label = 'L = %s' %Llist[0])
    ax1_f2.plot(Temp,ChiScaling(Llist[1],minGam,minNu,Chi12), marker = '^', ls = 'None', label = 'L = %s' %Llist[1])
    ax1_f2.plot(Temp,ChiScaling(Llist[2],minGam,minNu,Chi15), marker = 'P', ls = 'None', label = 'L = %s' %Llist[2])
    ax1_f2.plot(Temp,ChiScaling(Llist[3],minGam,minNu,Chi18), marker = 'X', ls = 'None', label = 'L = %s' %Llist[3])
    ax1_f2.plot(Temp,ChiScaling(Llist[4],minGam,minNu,Chi20), marker = 'D', ls = 'None', label = 'L = %s' %Llist[4])
    ax1_f2.plot(Temp,ChiScaling(Llist[5],minGam,minNu,Chi30), marker = 's', ls = 'None', label = 'L = %s' %Llist[5])

#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi10), marker = 'o', ls = 'None', label = 'L = %s' %Llist[0])
#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi12), marker = '^', ls = 'None', label = 'L = %s' %Llist[1])
#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi15), marker = 'P', ls = 'None', label = 'L = %s' %Llist[2])
#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi18), marker = 'X', ls = 'None', label = 'L = %s' %Llist[3])
#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi20), marker = 'D', ls = 'None', label = 'L = %s' %Llist[4])
#ax1_f2.plot(Temp,ChiScaling(1,1,1,Chi30), marker = 's', ls = 'None', label = 'L = %s' %Llist[5])

if choice == 'Ratio':
    ax1_f2.plot(Temp,Chi30/Chi10, lw = 3, label = 'L = 30 and L = 10 Ratio')
    ax1_f2.text(0.25,13,'Expected Bounds $\pm$ 8.5',fontsize = 20)

if choice == 'Actual':
    ax1_f2.plot(Temp,ChiScaling(Llist[0],1.3,0.67,Chi10), marker = 'o', ls = 'None', label = 'L = %s' %Llist[0])
    ax1_f2.plot(Temp,ChiScaling(Llist[1],1.3,0.67,Chi12), marker = '^', ls = 'None', label = 'L = %s' %Llist[1])
    ax1_f2.plot(Temp,ChiScaling(Llist[2],1.3,0.67,Chi15), marker = 'P', ls = 'None', label = 'L = %s' %Llist[2])
    ax1_f2.plot(Temp,ChiScaling(Llist[3],1.3,0.67,Chi18), marker = 'X', ls = 'None', label = 'L = %s' %Llist[3])
    ax1_f2.plot(Temp,ChiScaling(Llist[4],1.3,0.67,Chi20), marker = 'D', ls = 'None', label = 'L = %s' %Llist[4])
    ax1_f2.plot(Temp,ChiScaling(Llist[5],1.3,0.67,Chi30), marker = 's', ls = 'None', label = 'L = %s' %Llist[5])

ax1_f2.set_xlabel('Temperature', fontsize = 18)
ax1_f2.set_ylabel('Susceptibility', fontsize = 18)
ax1_f2.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 16)
ax1_f2.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 16)
ax1_f2.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax1_f2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax1_f2.grid()
ax1_f2.legend(loc = 2, prop = {'size': 25}, shadow = True)
gs.tight_layout(fig2)
gs.update(top = 0.90)
fig2.show()

#Specific Heat
Cp10 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[0]) + '.npy')
Cp12 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[1]) + '.npy')
Cp15 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[2]) + '.npy')
Cp18 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[3]) + '.npy')
Cp20 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[4]) + '.npy')
Cp30 = np.load(filepath + 'Specific_Heat_Metro_' + str(Llist[5]) + '.npy')

Temp = np.load(filepath + 'Temp_Metro_' + str(Llist[0]) + '.npy')
redT = (Temp-TcList[0])/TcList[0]

fig3 = plt.figure(figsize = [15,10])
gs = gridspec.GridSpec(ncols = 1, nrows = 1)
ax1_f3 = fig3.add_subplot(gs[0])

CpScaling = lambda L,alpha,nu,Cp: L ** (-alpha/nu) * Cp

if choice == 'Diff':
    def CpScalingFunc(x,alpha,nu):
        return (Ly/Lx) ** (alpha/nu) * x

    Lx,Ly = 10,12
    bounds = ([-0.015,0.5],[-0.01,0.67])
    popt_cp,pcov_cp = curve_fit(CpScalingFunc, (Cp10), (Cp12), [-0.01, 0.55], bounds = bounds, method = 'trf')

    minAlpha,minNu = popt_cp[0], popt_cp[1]
    print('Alpha = %.2f and Nu = %.2f' %(minAlpha,minNu))
    print('Covariance Matrix: \n' + str(pcov_cp) )

if choice == 'DiffLog':
    def CpScalingLog(x,alpha,nu):
        return (alpha/nu) * np.log(Ly/Lx) + x

    Lx,Ly = 10,12
    bounds = ([-0.015,0.5],[-0.01,0.67])
    popt_cp,pcov_cp = curve_fit(CpScalingFunc, (Cp10), (Cp12), [-0.01, 0.55], bounds = bounds, method = 'trf')

    minAlpha,minNu = popt_cp[0], popt_cp[1]
    print('Alpha = %.2f and Nu = %.2f' %(minAlpha,minNu))
    print('Covariance Matrix: \n' + str(pcov_cp) )

if choice == 'Newton':
    Lx,Ly = 10,12
    def CpDiff(alphaTilde,ratio):
        return (Lx ** (-alphaTilde))/(Ly ** -(alphaTilde)) * ratio

    loop = [newton(CpDiff,1.8,args = (Cp12[i]/Cp10[i],)) for i in range(Cp12.size)]

#Least Square Fit
#print('Minimizing Alpha and Nu using Specific Heat Data...')
if choice == 'LSF':
    nuList = np.arange(0.4,0.82,0.02)
    alphaList = np.arange(-0.001,-0.018,-0.001)

    Cp20Scaling = np.array([[CpScaling(Llist[-2],alp,n,Cp20) for n in nuList] for alp in alphaList])
    Cp30Scaling = np.array([[CpScaling(Llist[-1],alp,n,Cp30) for n in nuList] for alp in alphaList])
    data_points = len(Cp30Scaling[0,0])

    Xi2 = np.zeros([len(alphaList), len(nuList)])
    for i,gam in enumerate(alphaList):
        for j,n in enumerate(nuList):
            #print('Testing alpha = %.2f and Nu = %.2f' %(gam,n))
            data20 = Cp20Scaling[i,j]
            data30 = Cp30Scaling[i,j]
            xi2 = 0.0
            for k in range(data_points):
                sigma = np.sqrt(data30[k])
                #sigma = 1
                residual = (data20[k] - data30[k])/sigma
                xi2 = xi2 + np.power(residual,2)
            #print('Reduced Xi2 = ' +str(xi2/(data_points-1)))
            Xi2[i,j] = xi2/(data_points-1)

    GlobalMin = np.amin(Xi2)
    MinLoc = np.where(abs(GlobalMin - Xi2) == 0)
    minAlpha, minNu = alphaList[MinLoc[0][0]], nuList[MinLoc[1][0]]
    print('Minimized Alpha = %.3f and Minimized Nu = %.3f' %(minAlpha,minNu))

#Plot Results
if choice != 'Ratio' and choice != 'Actual':
    ax1_f3.plot(Temp,CpScaling(Llist[0],0,minNu,Cp10), marker = 'o', ls = 'None', label = 'L = %s' %Llist[0])
    ax1_f3.plot(Temp,CpScaling(Llist[1],0,minNu,Cp12), marker = '^', ls = 'None', label = 'L = %s' %Llist[1])
    ax1_f3.plot(Temp,CpScaling(Llist[2],0,minNu,Cp15), marker = 'P', ls = 'None', label = 'L = %s' %Llist[2])
    ax1_f3.plot(Temp,CpScaling(Llist[3],0,minNu,Cp18), marker = 'X', ls = 'None', label = 'L = %s' %Llist[3])
    ax1_f3.plot(Temp,CpScaling(Llist[4],0,minNu,Cp20), marker = 'D', ls = 'None', label = 'L = %s' %Llist[4])
    ax1_f3.plot(Temp,CpScaling(Llist[5],0,minNu,Cp30), marker = 's', ls = 'None', label = 'L = %s' %Llist[5])


if choice == 'Ratio':
    ax1_f3.plot(Temp,Cp30/Cp10, lw = 3, label = 'L = 30 and L = 10 Ratio')
    ax1_f3.text(0.25,1.25,'Expected Bounds $\pm$ 0.977', fontsize = 20)

if choice == 'Actual':
    ax1_f3.plot(Temp,CpScaling(Llist[0],-0.014,0.67,Cp10), marker = 'o', ls = 'None', label = 'L = %s' %Llist[0])
    ax1_f3.plot(Temp,CpScaling(Llist[1],-0.014,0.67,Cp12), marker = '^', ls = 'None', label = 'L = %s' %Llist[1])
    ax1_f3.plot(Temp,CpScaling(Llist[2],-0.014,0.67,Cp15), marker = 'P', ls = 'None', label = 'L = %s' %Llist[2])
    ax1_f3.plot(Temp,CpScaling(Llist[3],-0.014,0.67,Cp18), marker = 'X', ls = 'None', label = 'L = %s' %Llist[3])
    ax1_f3.plot(Temp,CpScaling(Llist[4],-0.014,0.67,Cp20), marker = 'D', ls = 'None', label = 'L = %s' %Llist[4])
    ax1_f3.plot(Temp,CpScaling(Llist[5],-0.014,0.67,Cp30), marker = 's', ls = 'None', label = 'L = %s' %Llist[5])

ax1_f3.set_xlabel('Temperature', fontsize = 18)
ax1_f3.set_ylabel('Specific Heat', fontsize = 18)
ax1_f3.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 16)
ax1_f3.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 16)
ax1_f3.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax1_f3.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax1_f3.grid()
ax1_f3.legend(loc = 2, prop = {'size': 25}, shadow = True)
gs.tight_layout(fig3)
gs.update(top = 0.90)
fig3.show()


def FitAlpha(L):
    filepath = '/home/mike/Desktop/Project/Triangular_XY_Model/'
    Cp = np.load(filepath + 'Specific_Heat_Metro_' + str(L) + '.npy')
    Temp = np.load(filepath + 'Temp_Metro_' + str(L) + '.npy')

    fig1 = plt.figure(figsize = [15,10])
    gs = gridspec.GridSpec(ncols = 1, nrows = 1)
    ax1 = fig1.add_subplot(gs[0])

    ax1.plot(Temp,Cp, marker = 'o', ls = 'None', color = 'orange')

    num = np.where(abs(np.amax(Cp) - Cp) == 0)[0][0]
    Tc = Temp[num]

    ax1.grid()
    ax1.set_xlabel('Temperature', fontsize = 15)
    ax1.set_ylabel('Specific Heat', fontsize = 15)
    fig1.suptitle('Specific Heat: %s x %s Triangular XY' %(L,L),fontsize = 20)
    ax1.set_xlim([np.amin(Temp), np.amax(Temp)])
    ax1.set_ylim([np.amin(Cp) - 0.05, np.amax(Cp) + 0.05])
    ax1.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 16)
    ax1.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 16)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax1.axvline(Tc, color = 'darkred', ls = '--', zorder = 3)


    alpha_guess = float(input('Initial Guess for Alpha: '))

    SpecificHeatExpPara = lambda T,alpha: (((T-Tc)/Tc)**(-alpha)) - 0.67
    #bounds = (0.1,0.5)
    popt_cp_Para,pcov = curve_fit(SpecificHeatExpPara, Temp[:num], Cp[:num], [alpha_guess])#, bounds = bounds)

    CpCriticalPara = SpecificHeatExpPara(Temp[:num],*popt_cp_Para)
    ax1.plot(Temp[:num],CpCriticalPara,lw = 3, label = r'$\alpha$ = %.4f' %popt_cp_Para[0], color = 'sienna')
    #ax3.plot(TempList[num+1:], CpCriticalFerro,lw = 3, label = 'Ferromagnetic Phase Fit', color = 'darkgoldenrod')
    print(r'$\alpha$ = ' + str(popt_cp_Para[0]))

    ax1.legend(loc = 1, prop = {'size': 25}, shadow = True)
    gs.tight_layout(fig1)
    gs.update(top = 0.90)
    fig1.show()

    return Tc, popt_cp_Para, pcov

def FitGamma(L):
    filepath = '/home/mike/Desktop/Project/Triangular_XY_Model/'
    Chi = np.load(filepath + 'Susceptibility_Metro_' + str(L) + '.npy')
    Cp = np.load(filepath + 'Specific_Heat_Metro_' + str(L) + '.npy')
    Temp = np.load(filepath + 'Temp_Metro_' + str(L) + '.npy')

    fig1 = plt.figure(figsize = [15,10])
    gs = gridspec.GridSpec(ncols = 1, nrows = 1)
    ax1 = fig1.add_subplot(gs[0])

    ax1.plot(Temp,Chi, marker = 'o', ls = 'None', color = 'blue')

    num = np.where(abs(np.amax(Cp) - Cp) == 0)[0][0]
    Tc = Temp[num]

    ax1.grid()
    ax1.set_xlabel('Temperature', fontsize = 15)
    ax1.set_ylabel('Specific Heat', fontsize = 15)
    fig1.suptitle('Susceptibility: %s x %s Triangular XY' %(L,L),fontsize = 20)
    ax1.set_xlim([np.amin(Temp), np.amax(Temp)])
    ax1.set_ylim([np.amin(Chi) - 0.05, np.amax(Chi) + 0.05])
    ax1.tick_params(axis = 'y', direction = 'in', which = 'both', labelsize = 16)
    ax1.tick_params(axis = 'x', direction = 'in', which = 'both', labelsize = 16)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.125))
    ax1.axvline(Tc, color = 'darkred', ls = '--', zorder = 3)


    gamma_guess = float(input('Initial Guess for Gamma: '))

    SusceptibilityExpPara = lambda T,gamma: (((T-Tc)/Tc)**(-gamma)) - 0.67
    #bounds = (0.1,0.5)
    popt_chi_Para,pcov = curve_fit(SusceptibilityExpPara, Temp[:num], Chi[:num], [gamma_guess])#, bounds = bounds)

    ChiCriticalPara = SusceptibilityExpPara(Temp[:num],*popt_chi_Para)
    ax1.plot(Temp[:num],ChiCriticalPara,lw = 3, label = r'$\gamma$ = %.4f' %popt_chi_Para[0], color = 'sienna')
    #ax3.plot(TempList[num+1:], CpCriticalFerro,lw = 3, label = 'Ferromagnetic Phase Fit', color = 'darkgoldenrod')
    print(r'$\gamma$ = ' + str(popt_chi_Para[0]))

    ax1.legend(loc = 1, prop = {'size': 25}, shadow = True)
    gs.tight_layout(fig1)
    gs.update(top = 0.90)
    fig1.show()

    return popt_chi_Para, pcov

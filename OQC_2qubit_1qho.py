import numpy as np
pi = np.arccos(-1)
import scipy as sp 
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as qt
import itertools
from Oscillator import *

import matplotlib.font_manager
import matplotlib as mpl


synch = {}

# Initialize random seed
np.random.seed(101)
# Store ifnormation for each phenomenological case
Xstore = dict([(k,{}) for k in range(4)])
disorder = dict([(k,{}) for k in range(4)])





############################################################ CASE 1: Initial qubit state is at ground #########################################################
dim_qho = 20

Nqb = 2
Nqho = Nqb - 1
wqb0 = 2*pi*10 # 1 GHz
gmat = np.zeros((Nqho,Nqb))
gmat[0,0] = 0.05*wqb0
gmat[0,1] = 0.05*wqb0

kappa_qb = np.zeros((Nqb,))
kappa_qho = np.zeros((Nqho,))
# kappa_qho[0] = wqb0*0.1




wdrive = wqb0

qb_args = {'w':[wqb0 for k in range(Nqb)]}
qho_args = {'w':[wqb0*1 for k in range(Nqho)]}
# qho_args = {'w':[wqb0*0.95*(1+k)/Nqho for k in range(Nqho)]}

sys_args = {'g':gmat}
args = {'dim_qho':dim_qho,'qb_args':qb_args,'qho_args':qho_args,'sys_args':sys_args}

OQC = OscillatorComputer(Nqb=Nqb,args=args)
OQC.Initialize_State(qb0=[0,0],qho0=[0.7])
OQC.Define_Operators()
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)

output = qt.mesolve(OQC.Hamiltonian(wdrive=wdrive), OQC.psi0, time, OQC.clist(kappa_qb=kappa_qb,kappa_qho=kappa_qho))
alpha_states = OQC.Find_Coherent_Time_State(output)
Xj = np.zeros((nsteps,4),dtype=np.complex128)
Xj = OQC.Get_normalized_psi_Xj(res=output,alpha_t=alpha_states)
# Xj[:,0] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=0)
# Xj[:,1] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=1)
# Xj[:,2] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=2)
# Xj[:,3] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=3)
n_a = qt.expect(OQC.qho_ops['a'][0].dag()*OQC.qho_ops['a'][0],output.states) # qho n
a_t = qt.expect(OQC.qho_ops['a'][0],output.states) 
a_tabs = np.abs(a_t)
phi_t = np.arctan(a_t.imag/a_t.real)

Xjlist = [Xj[:,j] for j in range(int(2**Nqb))]
Lambda = OQC.Limit_Cycle(alphastates=alpha_states,Xjlist_t=[Xj[-1,j] for j in range(int(2**OQC.Nqb))])
Dj,D = OQC.Quantum_Sync_Order(Lambda=Lambda,dt=(time[1]-time[0])/nsteps,nsteps=nsteps)

Sjk = OQC.Synchronization_degree(Xjlist=[Xj[:,j] for j in range(int(2**OQC.Nqb))])

# grid = plt.GridSpec(1,1)
# fig = plt.figure(figsize=(12,8))
# ax= fig.add_subplot(grid[0,0])

# ax.plot(wqb0*time,D)
# ax.set_xlabel(r'$\omega_{qb}$t')

## PLOTTING

grid = plt.GridSpec(3, 2)
fig = plt.figure(figsize=(30, 20))
ax00= fig.add_subplot(grid[0,0])
ax01= fig.add_subplot(grid[0,1],sharex=ax00)
ax10= fig.add_subplot(grid[1,0],sharex=ax00)
ax11= fig.add_subplot(grid[1,1],sharex=ax00)
ax2_= fig.add_subplot(grid[2,0],sharex=ax00)
ax2a= fig.add_subplot(grid[2,1],sharex=ax00)

ax00.plot(wqb0*time,a_t.real,'k-',label=r'Re{$\alpha$}')
ax00.plot(wqb0*time,a_t.imag,'k--',label=r'Im{$\alpha$}')
ax00.plot(wqb0*time,a_tabs,'r-',label=r'$|\alpha|$')
ax00.set_ylabel(r'$\alpha$')

ax00.set_xlabel(r'$\omega_{qb}$t')
y0 = 0.03
ax01.plot(wqb0*time,np.mod(phi_t,2*pi),'b',label=r'$\phi(t)$ ')
# ax.set_ylim([-y0,y0])
ax00.legend()
ax01.legend()
ax01.set_ylabel(r'$\phi$  mod 2 $\pi$')
ax01.set_xlabel(r'$\omega_{qb}$t')

x0 = 10
# ax.set_xlim([tf-10,tf])
thetaj_t = np.zeros((nsteps,4),dtype=np.float128)
for j in range(4):
    thetaj_t[:,j] = np.arctan(Xj[:,j].imag/Xj[:,j].real)#np.arctan2(Xj[:,j].imag/Xj[:,j].real,np.cos(wqb0*time % pi))  -time*wqb0 % pi + time*wqb0 #np.unwrap(, discont=np.pi)np.arctan(Xj[:,j].imag/Xj[:,j].real)#

ax11a = ax11.twinx()
cmap = ['k','r','b','g']
for j in range(4):
    # ax10.plot(wqb0*time,Xj[:,j].real,cmap[j]+'--',label='Real X_{}'.format(j))
    # ax10.plot(wqb0*time,Xj[:,j].imag,cmap[j]+':',label='Imag X_{}'.format(j))
    ax10.plot(wqb0*time,np.abs(Xj[:,j]),cmap[j]+'-',label='abs X_{}'.format(j))
    Xstore[0].update({j:np.abs(Xj[:,j])})
    ax11.plot(wqb0*time,np.mod(thetaj_t[:,j],2*pi),cmap[j]+'-',label=r'$\theta$'+str(j))
    disorder[0].update({j:Dj[j]})
    ax11a.plot(wqb0*time,D*np.ones((nsteps,)),'y--',label='Quantum Synch. Order')
    ax11a.plot(wqb0*time,Dj[j]*np.ones((nsteps,)),cmap[j]+'--',label='Quantum Synch. Order')

ax10.set_ylabel(r'$C_{X_{j}}$')
ax10.set_xlabel(r'$\omega_{qb}$t')
ax10.legend()
ax11.legend()
ax11.set_xlabel(r'$\omega_{qb}$t')
ax11.set_ylabel(r'$\theta_j$  mod 2 $\pi$')
ax11.set_ylim([0,2*pi])
ax11a.set_ylim([0,D+0.5])
ax11a.set_ylabel('Quantum Synch. Disorder',color ='y')
# ax.set_xlim([95,100])
# ax2.set_xlim([tf-10,tf])
cmap = ['k-','r-','b-','g-','k^','r--','b--','g--']
count = 0
explore =[]
theta_jk_nonoise ={}
synch.update({0:{}})
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            ax2_.plot(wqb0*time,np.mod(thetaj_t[:,j]-thetaj_t[:,k],2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            ax2a.plot(wqb0*time,Sjk[:,j,k],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            count += 1
            theta_jk_nonoise.update({(j,k):thetaj_t[:,j]-thetaj_t[:,k]})
            synch[0].update({(j,k):Sjk[:,j,k]})

# ax.plot(time,thetaj_t[:,1]-thetaj_t[:,3],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
ax2_.set_xlabel(r'$\omega_{qb}$t')
ax2_.set_ylabel(r'$(\theta_j - \theta_k)$  mod 2 $\pi$')
ax2_.legend()
ax2_.set_ylim([-0.1,2*pi])
ax2a.legend()
ax2a.set_ylabel('Quantum Synch. Degree')
# ax.set_xlim([40,50])
# ax.set_xlim([tf-20,tf])
# print(count)
# fig.savefig('OQC_2qb_1qho_NoQHOnoise.pdf', bbox_inches = 'tight',pad_inches = 0)

############################################################ CASE 2: Initial qubit state is at ground + dissipation #########################################################

dim_qho = 20
Nqb = 2
Nqho = Nqb - 1
wqb0 = 2*pi*1 # 1 GHz
gmat = np.zeros((Nqho,Nqb))
gmat[0,0] = 0.05*wqb0
gmat[0,1] = 0.05*wqb0

kappa_qb = np.zeros((Nqb,))
kappa_qho = np.zeros((Nqho,))
kappa_qho[0] = wqb0*0.1




wdrive = wqb0

qb_args = {'w':[wqb0 for k in range(Nqb)]}
qho_args = {'w':[wqb0*1 for k in range(Nqho)]}
# qho_args = {'w':[wqb0*0.95*(1+k)/Nqho for k in range(Nqho)]}
sys_args = {'g':gmat}
args = {'dim_qho':dim_qho,'qb_args':qb_args,'qho_args':qho_args,'sys_args':sys_args}

OQC = OscillatorComputer(Nqb=Nqb,args=args)
OQC.Initialize_State(qb0=[0,0],qho0=[0.7])
OQC.Define_Operators()
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)

output = qt.mesolve(OQC.Hamiltonian(wdrive=wdrive), OQC.psi0, time, OQC.clist(kappa_qb=kappa_qb,kappa_qho=kappa_qho))
alpha_states = OQC.Find_Coherent_Time_State(output)
Xj = np.zeros((nsteps,4),dtype=np.complex128)
Xj = OQC.Get_normalized_psi_Xj(res=output,alpha_t=alpha_states)
n_a = qt.expect(OQC.qho_ops['a'][0].dag()*OQC.qho_ops['a'][0],output.states) # qho n
a_t = qt.expect(OQC.qho_ops['a'][0],output.states) 
a_tabs = np.abs(a_t)
phi_t = np.arctan(a_t.imag/a_t.real)
Xjlist = [Xj[:,j] for j in range(int(2**Nqb))]

Xjlist = [Xj[:,j] for j in range(int(2**Nqb))]
Lambda = OQC.Limit_Cycle(alphastates=alpha_states,Xjlist_t=[Xj[-1,j] for j in range(int(2**OQC.Nqb))])
Dj,D = OQC.Quantum_Sync_Order(Lambda=Lambda,dt=(time[1]-time[0])/nsteps,nsteps=nsteps)
Sjk = OQC.Synchronization_degree(Xjlist=[Xj[:,j] for j in range(int(2**OQC.Nqb))])

#####


grid = plt.GridSpec(3, 2)
fig = plt.figure(figsize=(30, 20))
ax00= fig.add_subplot(grid[0,0])
ax01= fig.add_subplot(grid[0,1],sharex=ax00)
ax10= fig.add_subplot(grid[1,0],sharex=ax00)
ax11= fig.add_subplot(grid[1,1],sharex=ax00)
ax2_= fig.add_subplot(grid[2,0],sharex=ax00)
ax2a= fig.add_subplot(grid[2,1],sharex=ax00)

ax00.plot(wqb0*time,a_t.real,'k-',label=r'Re{$\alpha$}')
ax00.plot(wqb0*time,a_t.imag,'k--',label=r'Im{$\alpha$}')
ax00.plot(wqb0*time,a_tabs,'r-',label=r'$|\alpha|$')
ax00.set_ylabel(r'$\alpha$')

ax00.set_xlabel('t')
y0 = 0.03
ax01.plot(wqb0*time,np.mod(phi_t,2*pi),'b',label=r'$\phi(t)$')
# ax.set_ylim([-y0,y0])
ax00.legend()
ax01.legend()
ax01.set_ylabel(r'$\phi$  mod 2 $\pi$')
ax01.set_xlabel(r'$\omega_{qb}$t')
x0 = 10
# ax.set_xlim([tf-10,tf])
thetaj_t = np.zeros((nsteps,4),dtype=np.float128)
for j in range(4):
    thetaj_t[:,j] = np.arctan(Xj[:,j].imag/Xj[:,j].real)#np.arctan2(Xj[:,j].imag/Xj[:,j].real,np.cos(wqb0*time % pi))  -time*wqb0 % pi + time*wqb0 #np.unwrap(, discont=np.pi)np.arctan(Xj[:,j].imag/Xj[:,j].real)#

ax11a = ax11.twinx()
cmap = ['k','r','b','g']
for j in range(4):
    ax10.plot(wqb0*time,Xj[:,j].real,cmap[j]+'--',label='Real X_{}'.format(j))
    ax10.plot(wqb0*time,Xj[:,j].imag,cmap[j]+':',label='Imag X_{}'.format(j))
    ax10.plot(wqb0*time,np.abs(Xj[:,j]),cmap[j]+'-',label='abs X_{}'.format(j))
    Xstore[1].update({j:np.abs(Xj[:,j])})
    ax11.plot(wqb0*time,np.mod(thetaj_t[:,j],2*pi),cmap[j]+'-',label=r'$\theta$'+str(j))
    disorder[1].update({j:Dj[j]})
    ax11a.plot(wqb0*time,D*np.ones((nsteps,)),'y--',label='Quantum Synch. Order')
    ax11a.plot(wqb0*time,Dj[j]*np.ones((nsteps,)),cmap[j]+'--',label='Quantum Synch. Order')

ax10.set_ylabel(r'$C_{X_{j}}$')
ax10.set_xlabel(r'$\omega_{qb}$t')
ax10.legend()
ax11.legend()
ax11.set_ylim([0,2*pi])
ax11.set_xlabel(r'$\omega_{qb}$t')
ax11.set_ylabel(r'$\theta_j$  mod 2 $\pi$')
ax11.set_ylim([-0.05,6])
ax11a.set_ylim([0,D+1])
ax11a.set_ylabel('Quantum Synch. Disorder',color ='y')
# ax.set_xlim([95,100])
# ax2.set_xlim([tf-10,tf])
cmap = ['k-','r-','b-','g-','k^','r--','b--','g--']
count = 0
explore =[]
theta_jk_noise = {}
synch.update({1:{}})
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            ax2_.plot(wqb0*time,np.mod((thetaj_t[:,j]-thetaj_t[:,k]) ,2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            ax2a.plot(wqb0*time,Sjk[:,j,k],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            count += 1
            theta_jk_noise.update({(j,k):thetaj_t[:,j]-thetaj_t[:,k]})
            synch[1].update({(j,k):Sjk[:,j,k]})
            # synch.update({1:{}})
# ax.plot(time,thetaj_t[:,1]-thetaj_t[:,3],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
ax2_.set_xlabel(r'$\omega_{qb}$t ')
ax2_.set_ylabel(r'$(\theta_j - \theta_k)$ mod 2 $\pi$')
ax2_.set_ylim([0,2*pi])
ax2_.legend()
ax2a.legend()
ax2a.set_ylabel('Quantum Synch. Degree')
# ax.set_xlim([40,50])
# ax.set_xlim([tf-20,tf])
# print(count)

fig.savefig('OQC_2qb_1qho_WithQHOnoise.pdf', bbox_inches = 'tight',pad_inches = 0)


############################################################ CASE 3: Initial (normalized) random qubit state #########################################################

dim_qho = 20

Nqb = 2
Nqho = Nqb - 1
wqb0 = 2*pi*10 # 1 GHz
gmat = np.zeros((Nqho,Nqb))
gmat[0,0] = 0.05*wqb0
gmat[0,1] = 0.05*wqb0

kappa_qb = np.zeros((Nqb,))
kappa_qho = np.zeros((Nqho,))




wdrive = wqb0

qb_args = {'w':[wqb0 for k in range(Nqb)]}
qho_args = {'w':[wqb0*1 for k in range(Nqho)]}
# qho_args = {'w':[wqb0*0.95*(1+k)/Nqho for k in range(Nqho)]}
sys_args = {'g':gmat}
args = {'dim_qho':dim_qho,'qb_args':qb_args,'qho_args':qho_args,'sys_args':sys_args}

OQC = OscillatorComputer(Nqb=Nqb,args=args)
OQC.Initialize_State(qb0=[],qho0=[0.7])
OQC.Define_Operators()
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)
randpsi0 = OQC.psi0
output = qt.mesolve(OQC.Hamiltonian(wdrive=wdrive), OQC.psi0, time, OQC.clist(kappa_qb=kappa_qb,kappa_qho=kappa_qho))
alpha_states = OQC.Find_Coherent_Time_State(output)
Xj = np.zeros((nsteps,4),dtype=np.complex128)
Xj = OQC.Get_normalized_psi_Xj(res=output,alpha_t=alpha_states)
# Xj[:,0] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=0)
# Xj[:,1] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=1)
# Xj[:,2] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=2)
# Xj[:,3] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=3)
n_a = qt.expect(OQC.qho_ops['a'][0].dag()*OQC.qho_ops['a'][0],output.states) # qho n
a_t = qt.expect(OQC.qho_ops['a'][0],output.states) 
a_tabs = np.abs(a_t)
phi_t = np.arctan(a_t.imag/a_t.real)

Xjlist = [Xj[:,j] for j in range(int(2**Nqb))]
Lambda = OQC.Limit_Cycle(alphastates=alpha_states,Xjlist_t=[Xj[-1,j] for j in range(int(2**OQC.Nqb))])
Dj,D = OQC.Quantum_Sync_Order(Lambda=Lambda,dt=(time[1]-time[0])/nsteps,nsteps=nsteps)

Sjk = OQC.Synchronization_degree(Xjlist=[Xj[:,j] for j in range(int(2**OQC.Nqb))])

# grid = plt.GridSpec(1,1)
# fig = plt.figure(figsize=(12,8))
# ax= fig.add_subplot(grid[0,0])

# ax.plot(wqb0*time,D)
# ax.set_xlabel(r'$\omega_{qb}$t')

## PLOTTING

grid = plt.GridSpec(3, 2)
fig = plt.figure(figsize=(30, 20))
ax00= fig.add_subplot(grid[0,0])
ax01= fig.add_subplot(grid[0,1],sharex=ax00)
ax10= fig.add_subplot(grid[1,0],sharex=ax00)
ax11= fig.add_subplot(grid[1,1],sharex=ax00)
ax2_= fig.add_subplot(grid[2,0],sharex=ax00)
ax2a= fig.add_subplot(grid[2,1],sharex=ax00)

ax00.plot(wqb0*time,a_t.real,'k-',label=r'Re{$\alpha$}')
ax00.plot(wqb0*time,a_t.imag,'k--',label=r'Im{$\alpha$}')
ax00.plot(wqb0*time,a_tabs,'r-',label=r'$|\alpha|$')
ax00.set_ylabel(r'$\alpha$')

ax00.set_xlabel(r'$\omega_{qb}$t')
y0 = 0.03
ax01.plot(wqb0*time,np.mod(phi_t,2*pi),'b',label=r'$\phi(t)$ ')
# ax.set_ylim([-y0,y0])
ax00.legend()
ax01.legend()
ax01.set_ylabel(r'$\phi$  mod 2 $\pi$')
ax01.set_xlabel(r'$\omega_{qb}$t')

x0 = 10
# ax.set_xlim([tf-10,tf])
thetaj_t = np.zeros((nsteps,4),dtype=np.float128)
for j in range(4):
    thetaj_t[:,j] = np.arctan(Xj[:,j].imag/Xj[:,j].real)#np.arctan2(Xj[:,j].imag/Xj[:,j].real,np.cos(wqb0*time % pi))  -time*wqb0 % pi + time*wqb0 #np.unwrap(, discont=np.pi)np.arctan(Xj[:,j].imag/Xj[:,j].real)#

ax11a = ax11.twinx()
cmap = ['k','r','b','g']
for j in range(4):
    # ax10.plot(wqb0*time,Xj[:,j].real,cmap[j]+'--',label='Real X_{}'.format(j))
    # ax10.plot(wqb0*time,Xj[:,j].imag,cmap[j]+':',label='Imag X_{}'.format(j))
    ax10.plot(wqb0*time,np.abs(Xj[:,j]),cmap[j]+'-',label='abs X_{}'.format(j))
    Xstore[2].update({j:np.abs(Xj[:,j])})
    ax11.plot(wqb0*time,np.mod(thetaj_t[:,j],2*pi),cmap[j]+'-',label=r'$\theta$'+str(j))
    disorder[2].update({j:Dj[j]})
    ax11a.plot(wqb0*time,D*np.ones((nsteps,)),'y--',label='Quantum Synch. Order')
    ax11a.plot(wqb0*time,Dj[j]*np.ones((nsteps,)),cmap[j]+'--',label='Quantum Synch. Order')

ax10.set_ylabel(r'$C_{X_{j}}$')
ax10.set_xlabel(r'$\omega_{qb}$t')
ax10.legend()
ax11.legend()
ax11.set_xlabel(r'$\omega_{qb}$t')
ax11.set_ylabel(r'$\theta_j$  mod 2 $\pi$')
ax11.set_ylim([0,2*pi])
ax11a.set_ylim([0,D+0.5])
ax11a.set_ylabel('Quantum Synch. Disorder',color ='y')
# ax.set_xlim([95,100])
# ax2.set_xlim([tf-10,tf])
cmap = ['k-','r-','b-','g-','k^','r--','b--','g--']
count = 0
explore =[]
theta_jk_nonoise_rand={}
synch.update({2:{}})
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            ax2_.plot(wqb0*time,np.mod(thetaj_t[:,j]-thetaj_t[:,k],2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            ax2a.plot(wqb0*time,Sjk[:,j,k],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            theta_jk_nonoise_rand.update({(j,k):thetaj_t[:,j]-thetaj_t[:,k]})
            count += 1
            synch[2].update({(j,k):Sjk[:,j,k]})
            # synch.update({1:{}})


# ax.plot(time,thetaj_t[:,1]-thetaj_t[:,3],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
ax2_.set_xlabel(r'$\omega_{qb}$t')
ax2_.set_ylabel(r'$(\theta_j - \theta_k)$  mod 2 $\pi$')
ax2_.legend()
ax2_.set_ylim([-0.1,2*pi])
ax2a.legend()
ax2a.set_xlabel(r'$\omega_{qb}$t')
ax2a.set_ylabel('Quantum Synch. Degree')
# ax.set_xlim([40,50])
# ax.set_xlim([tf-20,tf])
# print(count)
# fig.savefig('OQC_2qb_1qho_nonoise_randpsi0.pdf', bbox_inches = 'tight',pad_inches = 0)


############################################################ CASE 4: Initial (normalized) random qubit state + Noise  #########################################################
dim_qho = 20

Nqb = 2
Nqho = Nqb - 1
wqb0 = 2*pi*10 # 1 GHz
gmat = np.zeros((Nqho,Nqb))
gmat[0,0] = 0.05*wqb0
gmat[0,1] = 0.05*wqb0

kappa_qb = np.zeros((Nqb,))
kappa_qho = np.zeros((Nqho,))
kappa_qho[0] = wqb0*0.1




wdrive = wqb0

qb_args = {'w':[wqb0 for k in range(Nqb)]}
qho_args = {'w':[wqb0*1 for k in range(Nqho)]}
sys_args = {'g':gmat}
args = {'dim_qho':dim_qho,'qb_args':qb_args,'qho_args':qho_args,'sys_args':sys_args}

OQC = OscillatorComputer(Nqb=Nqb,args=args)
OQC.Initialize_State(qb0=[],qho0=[0.7])
OQC.Define_Operators()
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)

output = qt.mesolve(OQC.Hamiltonian(wdrive=wdrive), randpsi0, time, OQC.clist(kappa_qb=kappa_qb,kappa_qho=kappa_qho))
alpha_states = OQC.Find_Coherent_Time_State(output)
Xj = np.zeros((nsteps,4),dtype=np.complex128)
Xj = OQC.Get_normalized_psi_Xj(res=output,alpha_t=alpha_states)
# Xj[:,0] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=0)
# Xj[:,1] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=1)
# Xj[:,2] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=2)
# Xj[:,3] = OQC.Get_Psi_Xj_Coefficient(output,alpha_states,j=3)
n_a = qt.expect(OQC.qho_ops['a'][0].dag()*OQC.qho_ops['a'][0],output.states) # qho n
a_t = qt.expect(OQC.qho_ops['a'][0],output.states) 
a_tabs = np.abs(a_t)
phi_t = np.arctan(a_t.imag/a_t.real)

Xjlist = [Xj[:,j] for j in range(int(2**Nqb))]
Lambda = OQC.Limit_Cycle(alphastates=alpha_states,Xjlist_t=[Xj[-1,j] for j in range(int(2**OQC.Nqb))])
Dj,D = OQC.Quantum_Sync_Order(Lambda=Lambda,dt=(time[1]-time[0])/nsteps,nsteps=nsteps)

Sjk = OQC.Synchronization_degree(Xjlist=[Xj[:,j] for j in range(int(2**OQC.Nqb))])

# grid = plt.GridSpec(1,1)
# fig = plt.figure(figsize=(12,8))
# ax= fig.add_subplot(grid[0,0])

# ax.plot(wqb0*time,D)
# ax.set_xlabel(r'$\omega_{qb}$t')

## PLOTTING


plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['text.usetex'] = True
ax = []
cm = 1/2.54  # centimeters in inches
grid = plt.GridSpec(3,2,wspace=0.5,hspace=0.1)
fig = plt.figure(figsize=(13, 6))
borderwidth = 1.75
mpl.rcParams['axes.linewidth'] = borderwidth
linesize = 1.05 
labelfontsize = 9
tickfontsize = 9
tickwidth = 1.2
axesfontsize = 14
offset_view = 0.1
offset_view1 = 0.01
yoff = 0.15
########################################
ax.append(fig.add_subplot(grid[0,0]))
ax.append(fig.add_subplot(grid[0,1]))
ax.append(fig.add_subplot(grid[1,0],sharex=ax[0]))
ax.append(fig.add_subplot(grid[1,1],sharex=ax[1]))
ax.append(fig.add_subplot(grid[2,0],sharex=ax[0]))
ax.append(fig.add_subplot(grid[2,1],sharex=ax[1]))


ax[0].plot(wqb0*time,a_t.real,'k-',label=r'Re{$\alpha$}')
ax[0].plot(wqb0*time,a_t.imag,'k--',label=r'Im{$\alpha$}')
ax[0].plot(wqb0*time,a_tabs,'r-',label=r'$|\alpha|$')
ax[0].set_ylabel(r'$\alpha$')
# ax[0].set_xlabel(r'$\omega_{qb}$t')



ax[1].plot(wqb0*time,np.mod(phi_t,2*pi),'k',label=r'$\phi(t)$ ')
ax[1].set_ylabel(r'$\phi$  mod 2 $\pi$')

x0 = 10
# ax.set_xlim([tf-10,tf])
thetaj_t = np.zeros((nsteps,4),dtype=np.float128)
for j in range(4):
    thetaj_t[:,j] = np.arctan(Xj[:,j].imag/Xj[:,j].real)#np.arctan2(Xj[:,j].imag/Xj[:,j].real,np.cos(wqb0*time % pi))  -time*wqb0 % pi + time*wqb0 #np.unwrap(, discont=np.pi)np.arctan(Xj[:,j].imag/Xj[:,j].real)#

ax11a = ax[3].twinx()
cmap = ['k','r','b','g']
labels = [r'$|00\rangle$',r'$|01\rangle$',r'$|10\rangle$',r'$|11\rangle$']
for j in range(4):
    # ax10.plot(wqb0*time,Xj[:,j].real,cmap[j]+'--',label='Real X_{}'.format(j))
    # ax10.plot(wqb0*time,Xj[:,j].imag,cmap[j]+':',label='Imag X_{}'.format(j))
    ax[2].plot(wqb0*time,np.abs(Xj[:,j]),cmap[j]+'-',label=labels[j])
    Xstore[3].update({j:np.abs(Xj[:,j])})
    ax[3].plot(wqb0*time,np.mod(thetaj_t[:,j],2*pi),cmap[j]+'-',label=r'$\theta$'+str(j))
    disorder[3].update({j:Dj[j]})
    ax11a.plot(wqb0*time,D*np.ones((nsteps,)),'y--',label='Quantum Synch. Order')
    ax11a.plot(wqb0*time,Dj[j]*np.ones((nsteps,)),cmap[j]+'--',label='Quantum Synch. Order')

yticks_list = [round(k*(np.ceil(D)/4),2)for k in range(5)]
ax11a.set_yticks(yticks_list, [str(tick) for tick in yticks_list])
ax11a.set_ylabel('Quantum Synch. Disorder')

cmap = ['k-','r-','b-','g-','k--','r--','b--','g--']
count = 0
explore =[]
theta_jk_noise_rand = {}
synch.update({3:{}})
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            ax[4].plot(wqb0*time,np.mod(thetaj_t[:,j]-thetaj_t[:,k],2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            ax[5].plot(wqb0*time,Sjk[:,j,k],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
            theta_jk_noise_rand.update({(j,k):thetaj_t[:,j]-thetaj_t[:,k]})
            count += 1
            synch[3].update({(j,k):Sjk[:,j,k]})
            # synch.update({1:{}})
yticks_list = [0,0.25,0.50,0.75,1.0]
ax[0].set_yticks(yticks_list, [str(tick) for tick in yticks_list])
ax[0].legend(loc='center', bbox_to_anchor=(1.175, 0.5),  fancybox=True, shadow=True, ncol=1)

yticks_list = [0,0.25,0.50,0.75,1.0]
ax[2].set_yticks(yticks_list, [str(tick) for tick in yticks_list])
ax[2].set_ylabel(r'$\tilde{Z}_{j}$')
ax[2].legend(loc='center', bbox_to_anchor=(1.175, 0.5),  fancybox=True, shadow=True, ncol=1)

# ax[3].legend()
ax[3].set_ylabel(r'$\theta_j$  mod 2 $\pi$')
ax[3].set_ylim([0-yoff,2*pi+yoff])

# ax.plot(time,thetaj_t[:,1]-thetaj_t[:,3],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k))
ax[4].set_xlabel(r'$\omega_{qb}$t')
ax[4].set_ylabel(r'$(\theta_j - \theta_p)$  mod 2 $\pi$')



ax[4].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])
ax[4].legend(loc='center', bbox_to_anchor=(1.175, 0.5),  fancybox=True, shadow=True, ncol=1)
# ax[4].set_ylim([-0.1,2*pi])
ax[4].set_ylim([0-yoff,2*pi+yoff])
# ax[5].set_ylim([0-yoff,2*pi+yoff])

ax[5].set_xlabel(r'$\omega_{qb}$t')
ax[5].set_ylabel('Degree of Separation, R')




ax[0].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[1].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[2].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[3].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[4].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[5].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
ax[0].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )
ax[1].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )
ax[2].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )
ax[3].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )

ax[1].set_yticks([0,pi/2,pi,3*pi/2,2*pi], ['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$'])
ax[3].set_yticks([0,pi/2,pi,3*pi/2,2*pi], ['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$'])
ax[4].set_yticks([0,pi/2,pi,3*pi/2,2*pi], ['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$'])
yticks_list = [0,0.0625,0.0625*2,0.0625*3,0.0625*4]
ax[5].set_yticks(yticks_list, [str(tick) for tick in yticks_list])
# ax.set_xlim([40,50])
# ax.set_xlim([tf-20,tf])
# print(count)
fig.savefig('FIG_Full_Analytics_CASE_D.pdf', bbox_inches = 'tight',pad_inches = 0)
fig.savefig('OQC_2qb_1qho_withnoise_randpsi0.pdf', bbox_inches = 'tight',pad_inches = 0)




############################################################ PAPER FIGURES #########################################################


############################################################ Figure 1  #########################################################


# from matplotlib import rc
# rc('text', usetex=True)
plt.rcParams['font.serif'] = ['Times New Roman']
ax = []
cm = 1/2.54  # centimeters in inches
grid = plt.GridSpec(1,1,wspace=0.075,hspace=0.05)
fig = plt.figure(figsize=(3.5, 3.25))
borderwidth = 1.35
# mpl.rcParams['axes.linewidth'] = borderwidth[(j,k)]
linesize = 1.25 
labelfontsize = 9
tickfontsize = 9
tickwidth = 1.2
axesfontsize = 10
offset_view = 0.01
offset_view1 = 0.01
########################################
ax.append(fig.add_subplot(grid[0,0]))
# ax.append(fig.add_subplot(grid[0,1],sharey=ax[0]))
# ax.append(fig.add_subplot(grid[0,2],sharey=ax[0]))
# ax.append(fig.add_subplot(grid[0,3],sharey=ax[0]))
# ax.append(fig.add_subplot(grid[1,0],sharex=ax[0]))
# ax.append(fig.add_subplot(grid[1,1],sharex=ax[1],sharey=ax[4]))
# ax.append(fig.add_subplot(grid[1,2],sharex=ax[2],sharey=ax[4]))
# ax.append(fig.add_subplot(grid[1,3],sharex=ax[3],sharey=ax[4]))

wqb0 = 2*pi*10
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)
explore = []
count = 0
cmap = ['k-','r-','b-','g-']
labels = [r'$\tilde{Z}_0$',r'$\tilde{Z}_1$',r'$\tilde{Z}_2$',r'$\tilde{Z}_3$']
for j in range(4):
    ax[0].plot(wqb0*time,Xstore[0][j],cmap[j],label=r'|'+ labels[j] +'|',linewidth = linesize)




ax[0].set_ylabel(r'$|\tilde{Z}_j|$',fontsize=axesfontsize )
ax[0].set_ylim([0-offset_view,1+offset_view])
ax[0].set_xlim([0-offset_view1,time[-1]+offset_view1])
ax[0].legend(loc='center', bbox_to_anchor=(0.5, 1.125),  fancybox=True, shadow=True, ncol=4)
ax[0].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )

ax[0].set_yticks([0,0.25,0.5,0.75,1], ['0',r'0.25',r'0.50',r'0.75',r'1.0'])
ax[0].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])
ax[0].tick_params(axis='x', labelbottom=True, labelsize=tickfontsize,width= tickwidth )

ax[0].set_xlabel(r'$\tilde{\omega}$t ',fontsize=axesfontsize )


# ax[0].set_yticks([0,pi/2,pi,3*pi/2,2*pi])



fig.savefig('FIG_1_Computational_Coefficients.pdf', dpi = 1200,bbox_inches = 'tight',pad_inches = 0.1)



############################################################ Figure 2  #########################################################



# mpl.rcParams['mathtext.default'] = 'regular'
# from matplotlib import rc
# rc('text', usetex=True)
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['text.usetex'] = True
ax = []
cm = 1/2.54  # centimeters in inches
grid = plt.GridSpec(2, 4,wspace=0.075,hspace=0.05)
fig = plt.figure(figsize=(13, 6))
borderwidth = 1.75
mpl.rcParams['axes.linewidth'] = borderwidth
linesize = 1.05 
labelfontsize = 9
tickfontsize = 9
tickwidth = 1.2
axesfontsize = 14
offset_view = 0.1
offset_view1 = 0.01
########################################
ax.append(fig.add_subplot(grid[0,0]))
ax.append(fig.add_subplot(grid[0,1],sharey=ax[0]))
ax.append(fig.add_subplot(grid[0,2],sharey=ax[0]))
ax.append(fig.add_subplot(grid[0,3],sharey=ax[0]))
ax.append(fig.add_subplot(grid[1,0],sharex=ax[0]))
ax.append(fig.add_subplot(grid[1,1],sharex=ax[1],sharey=ax[4]))
ax.append(fig.add_subplot(grid[1,2],sharex=ax[2],sharey=ax[4]))
ax.append(fig.add_subplot(grid[1,3],sharex=ax[3],sharey=ax[4]))

wqb0 = 2*pi*10
nsteps = 2000
tf = 500
time = np.linspace(0,tf/wqb0,nsteps)
explore = []
tags = []
count = 0
cmap = ['k-','r-','b-','g-','k--','r--','b--','g--']
cmap2 = ['k-','r-','b-','g-','k--','r--','b--','g--']
ax[1]
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            ax[0].plot(wqb0*time,np.mod(theta_jk_nonoise[(j,k)] ,2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[1].plot(wqb0*time,np.mod(theta_jk_noise[(j,k)] ,2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[2].plot(wqb0*time,np.mod(theta_jk_nonoise_rand[(j,k)] ,2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[3].plot(wqb0*time,np.mod(theta_jk_noise_rand[(j,k)] ,2*pi),cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[4].plot(wqb0*time,synch[0][(j,k)],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[5].plot(wqb0*time,synch[1][(j,k)],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[6].plot(wqb0*time,synch[2][(j,k)],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            ax[7].plot(wqb0*time,synch[3][(j,k)],cmap[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            count += 1
            tags.append((j,k))


ax[0].set_ylabel(r'$(\theta_j - \theta_p)$ mod 2$\pi$',fontsize=axesfontsize )
ax[0].set_ylim([0-offset_view,2*pi+offset_view])
ax[0].legend(loc='upper center', bbox_to_anchor=(2.075, 1.3),  fancybox=True, shadow=True, ncol=6)
ax[0].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )

ax[0].set_yticks([0,pi/2,pi,3*pi/2,2*pi], ['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$'])
ax[0].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])
xtext, ytext = 0, 1.085
ax[0].text(xtext, ytext, '(a)', horizontalalignment='left',verticalalignment='top',transform=ax[0].transAxes,weight='bold')
ax[0].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )
paneltext = ['a','b','c','d']
for k in range(1,4):
    ax[k].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
    ax[k].tick_params(axis='y', labelleft=False, labelsize=tickfontsize,width= tickwidth )
    ax[k].tick_params(axis='x', labelbottom=False, labelsize=tickfontsize,width= tickwidth )
    ax[k].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])
    ax[k].text(xtext, ytext, '({})'.format(paneltext[k]),horizontalalignment='left',verticalalignment='top',transform=ax[k].transAxes,weight='bold')

ax[4].set_ylim([0-offset_view1,0.25+offset_view1])
yticks_list = [0,0.0625,0.0625*2,0.0625*3,0.0625*4]
ax[4].set_yticks(yticks_list, [str(tick) for tick in yticks_list])
ax[4].set_xlabel(r'$\tilde{\omega}$t ',fontsize=axesfontsize )
ax[4].set_ylabel(r'Degree of Separation , R',fontsize=axesfontsize )
for k in range(5,8):
    ax[k].set_xlabel(r'$\tilde{\omega}$t ',fontsize=axesfontsize )
    ax[k].tick_params(axis='both', which='major', labelsize=tickfontsize,width= tickwidth )
    # ax[k].set_xlabel(r'$\omega_{qb}$t ',fontsize=axesfontsize )
    ax[k].tick_params(axis='y', labelleft=False, labelsize=tickfontsize,width= tickwidth )
    ax[k].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])

# ax[3].plot(wqb0*time,np.ones_like(time)*np.mod(delta_theta_ss[1],2*pi),':k')

# ax[0].set_yticks([0,pi/2,pi,3*pi/2,2*pi])



fig.savefig('FIG_2_Phase_Difference.pdf', dpi = 1200,bbox_inches = 'tight',pad_inches = 0.1)

plt.rcParams['font.serif'] = ['Times New Roman']
ax = []
cm = 1/2.54  # centimeters in inches
grid = plt.GridSpec(1,1,wspace=0.075,hspace=0.05)
fig = plt.figure(figsize=(4.5, 4.25))
borderwidth = 1.35
# mpl.rcParams['axes.linewidth'] = borderwidth[(j,k)]
linesize = 1.25 
labelfontsize = 6
tickfontsize = 9
tickwidth = 1.2
axesfontsize = 10
offset_view = 0.01
offset_view1 = 0.01
########################################

Fmat = np.zeros((4,4),dtype=np.complex128)
f00 =sp.integrate.simpson(y=(a_t*gmat[0,0])[1599:],x=time[1599:])/(time[-1]-time[1599])
Fmat[0,1] = f00
Fmat[0,2] = f00
Fmat[1,2] = f00
Fmat[1,3] = f00
Fmat[2,3] = f00

Fmat[1,0] = np.conj(f00)
Fmat[2,0] = np.conj(f00)
Fmat[2,1] = np.conj(f00)
Fmat[3,1] = np.conj(f00)
Fmat[3,2] = np.conj(f00)
ei,ve = np.linalg.eigh(Fmat)
theta = []
z0 = - f00/np.conj(f00)
for z in [np.mod(np.arctan(z0.imag/z0.real),2*pi),np.mod(pi/2,2*pi),np.mod(pi/2,2*pi),np.mod(0,2*pi)]:#ei[-1]*ve[:,-1]:
    # theta_ss = np.arctan(z.imag/z.real)
    # theta.append(theta_ss)
    theta.append(z)
store =[]
delta_theta_ss = []
for theta_a in range(len(theta)):
    store.append(theta_a)
    for theta_b in range(len(theta)):
        if theta_b != theta_a and theta_b not in store:
            delta_theta_ss.append(theta[theta_a] - theta[theta_b])
ax = []
cm = 1/2.54  # centimeters in inches
grid = plt.GridSpec(1, 1,wspace=0.075,hspace=0.05)
fig = plt.figure(figsize=(3.5, 3.25))
borderwidth = 1.75

mpl.rcParams['axes.linewidth'] = borderwidth
linesize = 1.05 
labelfontsize = 9
tickfontsize = 9
tickwidth = 1.2
axesfontsize = 14
offset_view = 0.1
offset_view1 = 0.01
########################################
ax.append(fig.add_subplot(grid[0,0]))
explore =[]
count = 0
cmap = ['k-','r-','b-','g-','k--','r--']
cmapss = ['k-.','r-.','b-.','g-.','k:','r:']
for j in range(4):
    explore.append(j)
    for k in range(4):
        if j != k and k not in explore:
            # print(j,k,count,theta_jk_noise_rand[(j,k)] )
            ax[0].plot(wqb0*time,np.mod(theta_jk_noise_rand[(j,k)] ,2*pi),cmap[count],linewidth = linesize)
            ax[0].plot(wqb0*time,np.ones_like(time)*np.mod(delta_theta_ss[count],2*pi),cmapss[count],label=r'$\Delta\theta$'+str(j) +str(k),linewidth = linesize)
            count += 1
ax[0].legend(loc='center', bbox_to_anchor=(1.2, 0.5),  fancybox=True, shadow=True, ncol=1)
ax[0].set_yticks([0,pi/2,pi,3*pi/2,2*pi], ['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$'])
ax[0].set_xticks([0,125,250,375,500], ['0',r'125',r'250',r'375',r'500'])
ax[0].tick_params(axis='x', labelbottom=True, labelsize=tickfontsize,width= tickwidth )

ax[0].set_ylabel(r'$(\theta_{j} - \theta_{p})$ mod $2\pi$',fontsize=axesfontsize )
ax[0].set_xlabel(r'$\tilde{\omega}$t ',fontsize=axesfontsize )

fig.savefig('FIG_3_Dynamical_Map.pdf', dpi = 1200,bbox_inches = 'tight',pad_inches = 0.1)


disorder
p = int(nsteps/int(time[-1]/(1/wqb0)))

for j in range(4):
    print(j,'-----------------------------------------------------')
    for tag in tags:
        DegreeSync = 0
        for k in range(int(time[-1]*wqb0 )):
            # print(synch[3][(0,1)][p*k:p*(k+1)],time[p*k:p*(k+1)]*wqb0)

            DegreeSync += sp.integrate.simpson(synch[j][tag][p*k:p*(k+1)],time[p*k:p*(k+1)])/(1/wqb0)
        DegreeSync = DegreeSync/int(time[-1]*wqb0 )
        print('D{} = {}; R{} = {}'.format(tag,(disorder[j][tag[0]] - disorder[j][tag[1]])/np.max([k for k in disorder[0]]),\
                                          tag,DegreeSync))
import numpy as np
pi = np.arccos(-1)
import quimb as qub
import quimb.tensor as qtn
import scipy as sp 
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as qt


import itertools





class OscillatorComputer:
    def __init__(self,Nqb,Ntsp=None,args=None) -> None:
        self.args = args
        self.Nqb = Nqb
        self.Nqho = Nqb-1
        
        if Ntsp != None:
            self.distmat = self.args['tsp_args']['distmat']
        self.Ntsp = Ntsp

            
        self.Xjstates = dict([(j,[]) for j in range(int(2**self.Nqb))])
        self.Define_Operators()
        # self.F = {(0,1):(self.qb_ops['sx'][0]+self.qb_ops['sx'][0]*self.qb_ops['sz'][1])*(self.distmat[0,1]/2),\
        #         (0,2):(self.qb_ops['sx'][1]+self.qb_ops['sz'][0]*self.qb_ops['sx'][1])*(self.distmat[0,2]/2),\
        #         (0,3):(self.qb_ops['sx'][0]*self.qb_ops['sx'][1] - self.qb_ops['sy'][0]*self.qb_ops['sy'][1] )*(self.distmat[0,3]/2),\
        #         (1,2):(self.qb_ops['sx'][0]*self.qb_ops['sx'][1] + self.qb_ops['sy'][0]*self.qb_ops['sy'][1] )*(self.distmat[1,2]/2),\
        #         (1,3):(self.qb_ops['sx'][1]-self.qb_ops['sz'][0]*self.qb_ops['sx'][1])*(self.distmat[1,3]/2),\
        #         (2,3):(self.qb_ops['sx'][0]-self.qb_ops['sx'][0]*self.qb_ops['sz'][1])*(self.distmat[2,3]/2)
        #         }
    def Find_Coherent_Time_State(self,res):
        
        a_t = [qt.expect(self.qho_ops['a'][n],res.states) for n in range(self.Nqho)]
        store = []
        for nt in range(len(a_t[0])):
            store.append(qt.tensor([qt.coherent(self.args['dim_qho'],a_t[n][nt]) for n in range(self.Nqho)]))
        return store

    def Get_Psi_Xj_Coefficient(self,res,alpha_t,j=0):
        _Xj = []
        for tn in range(len(alpha_t)):
            _states = qt.tensor(alpha_t[tn],self.qbbasis[j])
            _tmp = (_states.dag() * res.states[tn]).full()[0,0]#qt.expect(res.states[tn],_states)#
            _Xj.append(_tmp)
        return np.array(_Xj)

    def Get_normalized_psi_Xj(self,res,alpha_t):
        nsteps = len(alpha_t)
        _N = int(2**self.Nqb)
        Xj = np.zeros((nsteps,_N),dtype=np.complex128)
        for j in range(_N):
            Xj[:,j] = self.Get_Psi_Xj_Coefficient(res=res,alpha_t=alpha_t,j=j)

         
        for n in range(nsteps):
            norm = 0
            for j in range(_N):
                _tmp = Xj[n,j]*qt.tensor(alpha_t[n],self.qbbasis[j])
                norm = norm + (_tmp.dag()*_tmp).full()[0,0]
                # print(Xj[n,j])
            
            norm = 1/np.sqrt(norm)
            # print(norm) 
            for j1 in range(_N):
                Xj[n,j1] = Xj[n,j1]*norm
            for j1 in range(_N):
                self.Xjstates[j1].append(Xj[n,j1] *qt.tensor(alpha_t[n],self.qbbasis[j1]))

        return Xj

    def Initialize_State(self,qb0:list,qho0:list):
        self.qbbasis = []
        lst = list(itertools.product([0, 1], repeat=self.Nqb))
        self.binarystring = lst
        for n0 in lst :
                self.qbbasis.append(qt.tensor([qt.basis(2,n1) for n1 in n0]))
        # sys psi0 random
        if len(qb0) != 0:
            psi0_sys = qt.basis(2,qb0[0])
            for site in range(1,self.Nqb):
                psi0_sys = qt.tensor(psi0_sys,qt.basis(2,qb0[site]))
            psi0_bath = qt.coherent(self.args['dim_qho'],qho0[0])
            for site in range(1,self.Nqho):
                psi0_bath = qt.tensor(psi0_bath,qt.coherent(self.args['dim_qho'],qho0[site]))
            self.psi0 = qt.tensor(psi0_bath,psi0_sys)
            # in the order of j =0,...,2^nqb
            
        else:
            coeff = np.array(2*np.random.rand(int(2**(self.Nqb)))-1,dtype=np.complex_)
            coeff.imag = 2*np.random.rand(int(2**(self.Nqb)))-1
            coeff = coeff/np.sqrt(np.matmul(np.conj(coeff.T),coeff))
            psi0_sys = 0
            for n in range(int(2**(self.Nqb))):
                psi0_sys= psi0_sys + coeff[n]*self.qbbasis[n]
            psi0_bath = qt.coherent(self.args['dim_qho'],qho0[0])
            for site in range(1,self.Nqho):
                psi0_bath = qt.tensor(psi0_bath,qt.coherent(self.args['dim_qho'],qho0[site]))
            self.psi0 = qt.tensor(psi0_bath,psi0_sys)

    def clist(self,kappa_qb,kappa_qho):
        cops = []

        for site in range(self.Nqb):
            if kappa_qb[site] != 0:
                cops.append(kappa_qb[site]*self.qb_ops['sm'][site])

        for site in range(self.Nqho):
            if kappa_qho[site] != 0:
                cops.append(kappa_qho[site]*self.qho_ops['a'][site])
        return cops
    def Define_Operators(self):
        # H_qho otimes Hqb otimes Hqb ...
        self.qb_ops = {}
        self.qho_ops = {}

        _a = qt.destroy(self.args['dim_qho'])
        _Iqho = qt.qeye(self.args['dim_qho'])

        _Iqb = qt.qeye(2)
        _sx = qt.sigmax()
        _sy = qt.sigmay()
        _sz = qt.sigmaz()
        _sm = qt.sigmam()

        Iqho = _Iqho.copy()
   
        Iqb = _Iqb.copy()
        for k in range(self.Nqho-1):
            Iqho = qt.tensor(Iqho,_Iqho)

        for k in range(self.Nqb-1):
            
            Iqb = qt.tensor(Iqb,_Iqb)

            
        qbops_sx = dict([(k,_sx.copy()) for k in range(self.Nqb)])
        qbops_sy = dict([(k,_sy.copy()) for k in range(self.Nqb)])
        qbops_sz = dict([(k,_sz.copy()) for k in range(self.Nqb)])
        qbops_sm = dict([(k,_sm.copy()) for k in range(self.Nqb)])

        Iqbtmp = _Iqb.copy()
        for l in range(self.Nqb):
            for k in range(self.Nqb):
                if l > k:
                    qbops_sx[l] = qt.tensor(qbops_sx[l],Iqbtmp)
                    qbops_sy[l] = qt.tensor(qbops_sy[l],Iqbtmp)
                    qbops_sz[l] = qt.tensor(qbops_sz[l],Iqbtmp)
                    qbops_sm[l] = qt.tensor(qbops_sm[l],Iqbtmp)
                elif l < k:
                    qbops_sx[l] = qt.tensor(Iqbtmp,qbops_sx[l])
                    qbops_sy[l] = qt.tensor(Iqbtmp,qbops_sy[l])
                    qbops_sz[l] = qt.tensor(Iqbtmp,qbops_sz[l])
                    qbops_sm[l] = qt.tensor(Iqbtmp,qbops_sm[l])
        # Fourier = np.zeros((int(2**self.Nqb),int(2**self.Nqb)))
        # Fop = qt.Qobj(Fourier)
        # Four = Fop.full()
        # for m in range(int(2**self.Nqb)):
        #     for n in range(int(2**self.Nqb)):
        #         Four[m,n] = np.exp(1j*2*pi/int(2**self.Nqb) *m*n)


        # Four = Four/np.sqrt(int(2**self.Nqb))
        # F = qt.Qobj(Four,dims=[[2]*self.Nqb,[2]*self.Nqb])

        # F= qt.tensor(Iqho,F)
        for l in range(self.Nqb):
                qbops_sx[l] = qt.tensor(Iqho,qbops_sx[l])
                qbops_sy[l] = qt.tensor(Iqho,qbops_sy[l])
                qbops_sz[l] = qt.tensor(Iqho,qbops_sz[l])
                qbops_sm[l] = qt.tensor(Iqho,qbops_sm[l])





        qhoops_a = dict([(k,_a.copy()) for k in range(self.Nqho)])
        Iqhotmp = _Iqho.copy()
        for l in range(self.Nqho):
            for k in range(self.Nqho):
                if l > k:
                    qhoops_a[l] = qt.tensor(qhoops_a[l],Iqhotmp)
                elif l < k:
                    qhoops_a[l] = qt.tensor(Iqhotmp,qhoops_a[l])
                    
        for l in range(self.Nqho):
            qhoops_a[l] = qt.tensor(qhoops_a[l],Iqb)


        if self.Ntsp != None: 
            if self.Ntsp > 0:
                _atsp = qt.destroy(self.args['dim_tsp'])
                _Itsp = qt.qeye(self.args['dim_tsp'])
                Itsp = _Itsp.copy()
                for k in range(self.Ntsp-1):
                    Itsp = qt.tensor(Itsp,Itsp)
                for l in range(self.Nqb):
                    qbops_sx[l] = qt.tensor(Itsp,qbops_sx[l])
                    qbops_sy[l] = qt.tensor(Itsp,qbops_sy[l])
                    qbops_sz[l] = qt.tensor(Itsp,qbops_sz[l])
                    qbops_sm[l] = qt.tensor(Itsp,qbops_sm[l])

                for l in range(self.Nqho):
                    qhoops_a[l] = qt.tensor(Itsp,qhoops_a[l])

                tspops_a = dict([(k,_atsp.copy()) for k in range(self.Ntsp)])
                Itsptmp = _Itsp.copy()
                for l in range(self.Ntsp):
                    for k in range(self.Ntsp):
                        if l > k:
                            tspops_a[l] = qt.tensor(tspops_a[l],Itsptmp)
                        elif l < k:
                            tspops_a[l] = qt.tensor(Itsptmp,tspops_a[l])

                for l in range(self.Ntsp):
                    tspops_a[l] = qt.tensor(tspops_a[l],qt.tensor(Iqho,Iqb))

                F = qt.tensor(Itsp,F)
                self.qb_ops = {'1':qt.tensor(qt.tensor(Itsp,Iqho),Iqb),'sx':qbops_sx,'sy':qbops_sy,'sz':qbops_sz,'sm':qbops_sm}
                self.qho_ops = {'1':qt.tensor(qt.tensor(Itsp,Iqho),Iqb),'a':qhoops_a}
                self.tsp_ops = {'1':qt.tensor(qt.tensor(Itsp,Iqho),Iqb),'a':tspops_a}
            else:
                self.qb_ops = {'1':qt.tensor(Iqho,Iqb),'sx':qbops_sx,'sy':qbops_sy,'sz':qbops_sz,'sm':qbops_sm}
                self.qho_ops = {'1':qt.tensor(Iqho,Iqb),'a':qhoops_a}     
        else:
            self.qb_ops = {'1':Iqb,'sx':qbops_sx,'sy':qbops_sy,'sz':qbops_sz,'sm':qbops_sm}
            self.qho_ops = {'1':qt.tensor(Iqho,Iqb),'a':qhoops_a}     

    def Hamiltonian(self,wdrive):
        args_qb = self.args['qb_args'].copy()
        args_qho = self.args['qho_args'].copy()
        args_sys = self.args['sys_args'].copy()
        if self.Ntsp != None:
            args_tsp = self.args['tsp_args'].copy()
        gmat = args_sys['g']
        Hqb = 0 
        for k in range(self.Nqb):
            Hqb = Hqb - (1/2)*(args_qb['w'][k]- wdrive)*self.qb_ops['sz'][k]
        Hqho = 0
        for k in range(self.Nqho):
            Hqho = Hqho - (1/2)*(args_qho['w'][k] - wdrive)*self.qho_ops['a'][k].dag()*self.qho_ops['a'][k]


        H0 = Hqb + Hqho

        Hint = 0
        for j in range(self.Nqho):
            for k in range(self.Nqb):
                Hint = Hint +  gmat[j,k]*(self.qho_ops['a'][j]*self.qb_ops['sm'][k].dag() + self.qho_ops['a'][j].dag()*self.qb_ops['sm'][k])

        Htsp = 0
        # for k in range(1,int(2**self.Nqb)):
        #     for j in range(k):
        #         Htsp = Htsp + args_tsp['eps']*self.F[(j,k)]

        Htotal = H0 + Hint
        self.Ham = Htotal
        return self.Ham
    def Limit_Cycle(self,alphastates,Xjlist_t):
        f = 0
        for j in range(int(2**self.Nqb)):
            _state = Xjlist_t[j]*qt.tensor(alphastates[-1],self.qbbasis[j])
            _expHam = qt.expect(self.Ham,_state)
            f = f + _expHam
        return f/int(2**self.Nqb)

    def Quantum_Sync_Order(self,Lambda,dt,nsteps):
        Ncomp = int(2**self.Nqb)
        Dj = []
        for j in range(Ncomp):
            Dj.append(sum([np.linalg.norm(self.Xjstates[j][n+1] - np.exp(-1j*Lambda*dt)*self.Xjstates[j][n]) for n in range(nsteps-1)])/Ncomp)
        # _nlen = len(Xjlist) # list of Xj states
        
        # _ntime = len(Xjlist[0])
        # D_l = []
        # for j in range(_nlen):
        # # ||psij- psik||^_2 since Xj is the coefficient associated with the qubit state. 
        #     D_l.append(np.linalg.norm(Xjlist[j] ))
        D = np.max(Dj)

        return Dj,D
    def Synchronization_degree(self,Xjlist):
        _nlen = len(Xjlist) # list of Xj states
        
        _ntime = len(Xjlist[0])
        Sjk = np.zeros((_ntime,_nlen,_nlen),dtype=np.complex128)
        for j in range(_nlen):
            for k in range(_nlen):
                for n in range(_ntime):
                    if j != k:
                        Sjk[n,j,k] = ((self.Xjstates[j][n]-self.Xjstates[k][n]).dag()*(self.Xjstates[j][n]-self.Xjstates[k][n])).full()[0,0]
                    else:
                        Sjk[n,j,k] =0 
    
        return Sjk/_nlen
    

    def Correlation(self,k0,k1,Nsteps):
        corr = []
        for nt in range(Nsteps):
            corr.append((self.Xjstates[k0][nt].dag()*self.Oracle_Ops[(k1,k0)]*self.Xjstates[k1][nt]).full()[0,0])
        return np.array(corr)
    
    def Oracle(self):
        store = {}
        count_0 = 0
        for l0 in self.binarystring:
            count_1 = 0
            for l1 in self.binarystring:
                if l0 != l1:
                    tmp = [n for n in range(self.Nqb) if l0[n] != l1[n]]
                    op =1
                    for n in tmp:   
                        op =op*self.qb_ops['sx'][n]
                    store.update({(count_0,count_1):op})
                count_1 += 1 
            count_0 += 1 
        self.Oracle_Ops = store
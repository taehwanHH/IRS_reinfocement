import numpy as np
from numpy import pi, sqrt
from itertools import product
b=2
a = np.linspace(0, 2*pi*(1-1/(2**b)), 2**b, endpoint=True)
a
class IRSEnv:
    def __init__(self, N , b, T=20, Nt=4, Nr=1):
        self.done = False
        self.N = N
        self.Nt = Nt                        # # of tx antennas
        self.Nr = Nr                        # # of rx antennas
        self.b = b                          # # of quantization bits(phase)
        self.T = T                          # Terminal
        self.Nh = N                         # # of horizontal IRS elemnent size
        self.Nv = N                         # # of vertical IRS elemnent size
        self.N_IRS = self.Nh * self.Nv      # # of IRS elements


        # SNR threshold
        self.SNR_th = 23

        # Coordinate setting ([x,y,z])
        C_BS = np.array([0,0,10])
        C_IRS =np.array([250,250,5])
        C_UE = np.array([300,200,0])

        # enviroment parameter
        K_dB = 10                            # Rician K (decibel) factor
        self.K = 10**(K_dB/10)
        p_tx_dB = 30
        P_tx = 10**(p_tx_dB/10)
        fc = 2.5                             # carrier frequency [GHz]
        self.lambda_c = 3e8/(fc*1e9)         # carrier wavelength
        self.d = self.lambda_c/2             # antenna element separation
        W = 10e6                             # bandwidth 10MHz
        ts = 1/W
        self.P_noise = 10**((-174 + 10*np.log10(W))/10)

        # distance
        D_BS_IRS = np.linalg.norm(C_BS-C_IRS)
        D_IRS_UE = np.linalg.norm(C_IRS-C_UE)
        D_BS_UE = np.linalg.norm(C_BS-C_UE)

        G_Tx = 10**(5/10); G_R = 10**(0/10); G_Rx = 10**(5/10)

        # pathloss
        self.PL_f = P_tx*self.PL_LoS(fc,D_BS_IRS)*G_Tx*G_R
        self.PL_G = self.PL_LoS(fc,D_IRS_UE)*G_R*G_Rx
        self.PL_h = P_tx*self.PL_NLoS(fc,D_BS_UE)*G_Tx*G_Rx

        # IRS steering angle
        self.AoA_theta_R=pi/4
        self.AoD_theta_R=pi/4
        self.AoA_phi_R=pi/4
        self.AoD_phi_R=pi/4
        self.AoA_theta_U=pi/4


        self.state = np.ones((self.N**2,1),complex)        # state (current phase  N^2 X 1 matrix)
        self.count = 0                                      # time step count
        self.action_list = self.make_action(self.b,self.N_IRS) # action list
        self.action_list = np.array(self.action_list)
        self.action_list = np.exp(1j*self.action_list)


    def PL_LoS(self,fc,x):
        return 10**((-28-20*np.log10(fc)-22*np.log10(x))/10)

    def PL_NLoS(self,fc,x):
        return 10**((-22.7-26*np.log10(fc)-36.7*np.log10(x))/10)

    def make_action(self,b, N):
        phase_list = np.linspace(0, 2*pi*(1-1/(2**b)), 2**b, endpoint=True)
        action_list = list(product(phase_list, repeat=N))
        return action_list

    def reset(self):
        # generating LOS channel for Tx-IRS, IRS-Rx
        a_h_AoA = np.exp(-1j*(2*np.pi*self.d/self.lambda_c)*np.array([range(0,self.Nh)]).T*np.cos(self.AoA_phi_R)*np.sin(self.AoA_theta_R))
        a_h_AoD= np.exp(-1j*(2*np.pi*self.d/self.lambda_c)*np.array([range(0,self.Nh)]).T*np.cos(self.AoD_phi_R)*np.sin(self.AoD_theta_R))
        a_v_AoA = np.exp(-1j*(2*np.pi*self.d/self.lambda_c)*np.array([range(0,self.Nv)]).T*np.cos(self.AoA_theta_R)*np.cos(self.AoA_phi_R))
        a_v_AoD = np.exp(-1j*(2*np.pi*self.d/self.lambda_c)*np.array([range(0,self.Nv)]).T*np.cos(self.AoD_theta_R)*np.cos(self.AoD_phi_R)) 

        a_R_arrival = np.kron(a_h_AoA, a_v_AoA)                   # N-by-1 angle of arrival at IRS
        a_R_departure = np.kron(a_h_AoD, a_v_AoD)                 # N-by-1 angle of departure at IRS
        a_B = np.exp(-1j*(2*np.pi*self.d/self.lambda_c)*np.array([range(0,self.Nt)]).T*np.sin(self.AoA_theta_U)) # M(Nt)-by-1(Nr) angle of departure at BS

        self.F_LoS = a_R_arrival@a_B.T                      # N-by-M(Nt) matrix
        self.g_LoS = a_R_departure                          # N elements col vector

        self.state = np.ones((self.N**2,1),complex)        # state reset
        self.count = 0
        self.done = False

        return self.state
    
    def step(self,action):
        # NLOS channel for UE-to-IRS
        F_NLoS = 1/sqrt(2)*(np.random.randn(self.N_IRS,self.Nt)+1j*np.random.randn(self.N_IRS,self.Nt)) # N_IRS-by-1(Nt)col vec
        #  Rician channel for UE-to-IRS
        F = sqrt(self.PL_f)*(sqrt(self.K/(self.K+1))*self.F_LoS + sqrt(1/(self.K+1))*F_NLoS) # N_IRS-by-1(Nt)col vec
        g_NLoS = 1/sqrt(2)*(np.random.randn(self.N_IRS,self.Nr)+1j*np.random.randn(self.N_IRS,self.Nr))
        #  Rician channel for IRS-to-BS
        g = sqrt(self.PL_G)*(sqrt(self.K/(self.K+1))*self.g_LoS + sqrt(1/(self.K+1))*g_NLoS)    # N-by-Nr matrix [g1 g2]
        g = g.reshape(-1)

        #  Direct path from BS to UE
        h=sqrt(self.PL_h)*(1/sqrt(2)*(np.random.randn(self.Nt,self.Nr)+1j*np.random.randn(self.Nt,self.Nr)))
        
        #  Effective channel, Mc
        Mc_1 = np.diag(g).conjugate().T@F@F.conjugate().T@np.diag(g) 
        Mc_2 = np.diag(g).conjugate().T@F@h
        Mc_3 = h.conjugate().T@F.conjugate().T@np.diag(g)
        Mc_4 = h.conjugate().T@h
        Mc_t = np.concatenate((Mc_1,Mc_2),axis=1)
        Mc_b = np.concatenate((Mc_3,Mc_4),axis=1)
        Mc = np.concatenate((Mc_t,Mc_b),axis=0).conjugate()

        ## Phase design

        # IRS phase steering matrix [dim:NxN]
        # rand_IRS = np.exp(1j*(-np.pi + (2*np.pi)*np.random.rand(self.N_IRS))) # randomly generated
        # h_random = h.conjugate().T + g.conjugate().T@np.diag(rand_IRS)@F
        # gamma_random = np.linalg.norm(h_random)**2

        ## Phase design
        action_sel = self.action_list[action]
        self.state = self.state * action_sel.reshape(-1,1)        # phase shift
        # calculate SNR
        psi_1 = np.append(self.state.conjugate().T,1).reshape(1,-1)     # [psi 1] array
        gamma = np.abs(psi_1@Mc@(psi_1.conjugate().T))                  # [psi 1]Mc[psi 1]'
        SNR_lin = gamma/self.P_noise                                    
        SNR = 10*np.log10(SNR_lin/10)
        SNR = float(SNR)

        # get reward
        if SNR >= self.SNR_th:
            reward = SNR
        else:
            reward = SNR -100
        
        self.count += 1

        if self.count == 20:
            self.done = True
        
        info = {}

        return self.state,reward, self.done, info
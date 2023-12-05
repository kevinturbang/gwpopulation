import numpy as np

class OmegaComputerMDC(object):
    def __init__(self,hyper_params_dict, freqs=np.linspace() , N_MC_averages=20000, tmp_min=2., tmp_max=100.):
        self.hyperparameters_dict = hyper_params_dict
        self.freqs = freqs
        self.N_MC_averages = N_MC_averages
        self.tmp_min, self.tmp_max = tmp_min, tmp_max
        self.perform_initial_draws()
        
    def perform_initial_draws(self):
        self.m1s_drawn = np.random.uniform(self.tmp_min, self.tmp_max, size=self.N_MC_averages)

        c_qs = np.random.uniform(size=int(self.N_MC_averages))
        self.qs_drawn = (self.tmp_min/self.m1s_drawn)**(1.)+c_qs*(1**(1.)-(self.tmp_min/self.m1s_drawn)**(1.))
    
        self.zs_drawn = np.random.uniform(0,10,size=self.N_MC_averages)
    
        # Computing eergy spectra for original draws

        self.dEdfs = np.array([dEdf(self.m1s_drawn[ii]+self.qs_drawn[ii]*self.m1s_drawn[ii],self.freqs*(1+self.zs_drawn[ii]),eta=(self.qs_drawn[ii]*self.m1s_drawn[ii])/self.m1s_drawn[ii]/(1+(self.qs_drawn[ii]*self.m1s_drawn[ii])/self.m1s_drawn[ii])**2) for ii in range(self.N_MC_averages)])
    
        # Keeping track of old draw probabilities

        self.p_m1_old = 1/(self.tmp_max-self.tmp_min)*np.ones(self.N_MC_averages)
        self.p_z_old = 1/(10-0)*np.ones(self.N_MC_averages)
        self.p_q_old = 1/(1-self.tmp_min/self.m1s_drawn)
        
    def massModelm1(self, m1):
        Norm = (1+self.hyperparameters_dict["alpha"])/(pow(self.hyperparameters_dict["mMax"], 1+self.hyperparameters_dict["alpha"]) - pow(self.hyperparameters_dict["mMin"], 1+self.hyperparameters_dict["alpha"]));
        Beta = np.where((m1 > self.hyperparameters_dict["mMin"]) & (m1 < self.hyperparameters_dict["mMax"]),Norm * pow(m1,self.hyperparameters_dict["alpha"]),0)
        G = 1/(np.sqrt(2*np.pi)*self.hyperparameters_dict["sig_m1"])*np.exp(-(m1-self.hyperparameters_dict["mu_m1"])*(m1-self.hyperparameters_dict["mu_m1"])/(2*self.hyperparameters_dict["sig_m1"]*self.hyperparameters_dict["sig_m1"]))
        mixture = (1-self.hyperparameters_dict["f_peak"]) * Beta + self.hyperparameters_dict["f_peak"] * G
        Smoothener = np.where((m1>= self.hyperparameters_dict["mMin"]) & (m1<self.hyperparameters_dict["mMin"]+self.hyperparameters_dict["dmMin"]), 1/(1+np.exp(1.0*self.hyperparameters_dict["dmMin"]/(m1-self.hyperparameters_dict["mMin"]) + 1.0*self.hyperparameters_dict["dmMin"]/(m1-self.hyperparameters_dict["mMin"]-self.hyperparameters_dict["dmMin"]))),1)
        Smoothener = np.where(m1<self.hyperparameters_dict["mMin"],0,Smoothener)
        p = mixture * Smoothener
        return p
        
    def massModelq(self, q):
        Norm = (1.+self.hyperparameters_dict["bq"])/(1-(self.hyperparameters_dict["mMin"]/self.m1s_drawn)**(1.+self.hyperparameters_dict["bq"]))
        p_q = Norm * pow(q,self.hyperparameters_dict["bq"])
    
        Smoothener = np.where((q*self.m1s_drawn>= self.hyperparameters_dict["mMin"]) & (q*self.m1s_drawn<self.hyperparameters_dict["mMin"]+self.hyperparameters_dict["dmMin"]), 1/(1+np.exp(1.0*self.hyperparameters_dict["dmMin"]/(q*self.m1s_drawn-self.hyperparameters_dict["mMin"]) + 1.0*self.hyperparameters_dict["dmMin"]/(q*self.m1s_drawn-self.hyperparameters_dict["mMin"]-self.hyperparameters_dict["dmMin"]))),1)
        Smoothener = np.where(q*self.m1s_drawn<self.hyperparameters_dict["mMin"],0,Smoothener)
    
        p = p_q * Smoothener
        return p
    
    def R_z_model(self,z):
        return (1+z)**self.hyperparameters_dict["alpha_z"]/(1+((1+z)/(1+self.hyperparameters_dict["zp"]))**(self.hyperparameters_dict["alpha_z"]+self.hyperparameters_dict["beta_z"]))
    
    def f_z(self, z):
        rate = self.R_z_model(z)
        rate_final = rate/np.sqrt(OmgM*(1.+z)**3.+OmgL)/(1.+z)
        return rate_final
    
    def evaluate_omega(self):
        p_q_new = self.massModelq(self.qs_drawn)
        p_m1_new = self.massModelm1(self.m1s_drawn)
        p_z_new = self.f_z(self.zs_drawn)
        
        Rz_norm = self.R_z_model(0)
        
        m1_tmp = np.linspace(self.tmp_min,self.tmp_max, 1000)
        m1_norm = simps(self.massModelm1(m1_tmp), x=m1_tmp)
        
        #How to properly normalize p_q
        
        w_i = p_z_new*p_m1_new*p_q_new/(self.p_z_old*self.p_m1_old*self.p_q_old)
        
        Omega_spectrum_new = (self.freqs)*(np.einsum("if,i->if",self.dEdfs,w_i))
        Omega_spectrum_new_avged = 1/rhoC/H0*self.hyperparameters_dict["R0"]/1e9/year/Rz_norm/m1_norm*np.mean(Omega_spectrum_new, axis=0)
        
        return Omega_spectrum_new_avged
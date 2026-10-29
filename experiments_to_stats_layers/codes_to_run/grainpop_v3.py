#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 08:47:50 2023

@author: aforte
"""

from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.stats import linregress
from scipy.stats import ks_2samp
import numpy as np

class GrainPop:
    def __init__(self,pop_centers,pop_std,pop_skew,pop_percent,unc,num_grains,
                 min_age=0,max_age=4500,age_inc=1,seeds=None):
        self.pop_centers=np.array(pop_centers)
        self.pop_std=np.array(pop_std)
        self.pop_skew=np.array(pop_skew)
        self.pop_percent=np.array(pop_percent)
        self.unc=unc
        self.num_grains=num_grains
        self.seeds=seeds
        self.__consistency_check()
        self.determine_pop_grains()
        self.min_age=min_age
        self.max_age=max_age
        self.t=np.arange(min_age,max_age+age_inc,age_inc) 
        self.build_sample()
        
    def __consistency_check(self):
        # Check for number of entries
        if ((len(self.pop_centers)==len(self.pop_std)) & 
            (len(self.pop_std)==len(self.pop_skew)) & 
            (len(self.pop_skew)==len(self.pop_percent))):
            pass
        else:
            raise Exception('The number of entries for pop_centers, pop_std, pop_skew, and pop_percent must all contain the same number of entries')
        
        # Check seed entry
        if self.seeds==None:
            self.seeds=np.zeros(self.pop_centers.shape).astype(int)   
        elif len(self.seeds)==len(self.pop_centers):
            self.seeds=np.array(self.seeds).astype(int)
        else:
            raise Exception('If you provide a list of seeds, you must provide one for each population')
        
        # Check that percentage are valid
        if np.isclose(np.sum(self.pop_percent),1):
            pass
        else:
            raise Exception('Sum of entries in pop_percent must equal 1')
        
        # Force number grains to be integer value
        if type(self.num_grains)==int:
            pass
        else:
            self.num_grains=int(self.num_grains)
            
        self.num_pop=len(self.pop_centers)
        
    def determine_pop_grains(self):
        # Generate the number of grains per population
        png=self.pop_percent * self.num_grains
        self.pop_num_grains=png.astype(int)
        # Update the number of grains in case through rounding the total population has changed
        self.num_grains=np.sum(self.pop_num_grains)
        
    def build_sample(self):
        smpl=[]
        pop=[]
        for i in range(self.num_pop):
            grains=skewnorm.rvs(self.pop_skew[i],loc=self.pop_centers[i],
                                scale=self.pop_std[i],
                                size=self.pop_num_grains[i],
                                random_state=self.seeds[i])
            pop_member=np.ones(self.pop_num_grains[i])*i
            smpl.append(grains)
            pop.append(pop_member)
        ages=np.concatenate(smpl,axis=0)
        population=np.concatenate(pop,axis=0)
        # Filter any ages that are nonzero or greater than age of Earth or 
        # alternative specified range
        idx=(ages>self.min_age) & (ages<=self.max_age)
        self.ages=ages[idx]
        self.population=population[idx]
        self.one_sigma=self.ages*self.unc
        # Update number of grains again if any were flagged
        self.num_grains=len(self.ages)
        
    def __pdp_one(self,x,e):
        term1=1/(e*np.sqrt(2*np.pi))
        term2=np.exp((-(self.t-x)**2)/(2*e**2))
        return term1*term2
    
    def pdp_func(self):
        pdp=sum(list(map(self.__pdp_one,self.ages,self.one_sigma)))/self.num_grains
        self.pdp=pdp

class GrainPopReal:
    def __init__(self, ages, errors, numGrains, labels=None, min_age=0, max_age=4500, age_inc=1):
        self.ages = np.concatenate(ages)  # Flatten list of arrays into one array
        self.errors = np.concatenate(errors)
        self.num_grains = sum([int(x) for x in numGrains])  # Ensure all elements are integers
        self.labels = labels if labels is not None else []  # Handle missing labels
        self.min_age = min_age
        self.max_age = max_age
        self.t = np.arange(min_age, max_age + age_inc, age_inc)
        self.filter_ages()
        self.one_sigma = self.errors  # Assuming one_sigma refers to errors
        self.pdp_func()
        
    def filter_ages(self):
        """Removes ages outside the specified range"""
        idx = (self.ages > self.min_age) & (self.ages <= self.max_age)
        self.ages = self.ages[idx]
        self.errors = self.errors[idx]
        self.num_grains = len(self.ages)
    
    def __pdp_one(self,x, e):
        term1 = 1 / (e * np.sqrt(2 * np.pi))
        term2 = np.exp((-(self.t - x) ** 2) / (2 * e ** 2))
        return term1 * term2
    
    def pdp_func(self):
        pdp=sum(list(map(self.__pdp_one,self.ages,self.one_sigma)))/self.num_grains
        self.pdp=pdp
        
class ZirconGenerator:
    def __init__(self,total_area,fractions,zr_ppms,mean_grain_radii,
                 std_grain_radii,e_rates,size_cutoff=25,rho_s=2650):
        self.total_area=total_area # meters squared
        self.fractions=np.array(fractions) # must sum to 1
        self.zr_ppms=np.array(zr_ppms)
        self.mean_grain_radii=np.array(mean_grain_radii) # in microns
        self.std_grain_radii=np.array(std_grain_radii) # in microns
        self.e_rates=np.array(e_rates) # meters per million years
        self.size_cutoff=size_cutoff # cutoff radius for grains in microns
        self.Zr_mm=91.224
        self.Si_mm=28.085
        self.O_mm=15.999
        self.Zir_mm=self.Zr_mm + self.Si_mm + 4*self.O_mm
        self.rho_Zir=4650 # in kg/m3
        self.rho_s=rho_s # in kg/m3
        self.__consistency_check()
        self.__calc_sediment_mass()
        self.__calc_zircon_mass()
        self.__calc_num_zircons()
        
    def __consistency_check(self):
        # Check number of inputs are consistent
        if ((len(self.fractions)==len(self.zr_ppms)) & 
            (len(self.zr_ppms)==len(self.mean_grain_radii)) & 
            (len(self.mean_grain_radii)==len(self.std_grain_radii)) & 
            (len(self.std_grain_radii)==len(self.e_rates))):
            pass
        else:
            raise Exception('The number of entries for fractions, zr_ppms, grain_radii, and e_rates must all contain the same number of entries')
            
        # Check that fractions sum to 1 approximately
        if np.isclose(np.sum(self.fractions),1):
            pass
        else:
            raise Exception('Sum of entries in fractions must equal 1')
            
        self.num_units=len(self.fractions)
    
    def __calc_sediment_mass(self):
        self.sed_masses=self.fractions*self.total_area*self.e_rates*self.rho_s
    
    def __calc_zircon_mass(self):
        # Assume all zirconium is partitioned into zircon and calculated
        # effectively the ppm of zircon in sediment
        zirc_ppms=self.zr_ppms*(self.Zir_mm/self.Zr_mm)
        # calculate zircon mass
        self.zirc_masses=(self.sed_masses*zirc_ppms)/1e6
        self.total_zirc_mass=np.sum(self.zirc_masses)
        self.zirc_mass_fractions=self.zirc_masses/self.total_zirc_mass
        
    def __grain_mass(self,R):
        R=R/1e6
        V=(4/3)*np.pi*R**3
        return V*self.rho_Zir
    
    def __calc_num_zircons(self):
        x=np.arange(0.05,1.05,0.1)
        unfums=np.zeros(self.num_units)
        for i in range(self.num_units):
            ppfs=norm.ppf(x,loc=self.mean_grain_radii[i],scale=self.std_grain_radii[i])
            grain_masses=self.__grain_mass(ppfs)
            unfums[i]=(self.zirc_masses[i]/np.sum(0.1*grain_masses)).astype(int)
        self.zirc_unfiltered_num=unfums
        cutoff_cdfs=norm.cdf(self.size_cutoff,loc=self.mean_grain_radii,scale=self.std_grain_radii)
        self.zirc_filtered_num=(self.zirc_unfiltered_num*(1-cutoff_cdfs)).astype(int)
        self.total_num_zirc=np.sum(self.zirc_filtered_num)
        self.zirc_num_fractions=self.zirc_filtered_num/self.total_num_zirc

class ZirconGeneratorLL:
    def __init__(self,fractions,zirc_masses,mean_grain_radii,std_grain_radii,size_cutoff=25,rep_volume=1):
        self.fractions=np.array(fractions) # must sum to 1
        self.zirc_masses=np.array(zirc_masses) # zircon concentration in kg/m3 as output from ConcentrationTracker for each unit
        self.mean_grain_radii=np.array(mean_grain_radii) # in microns
        self.std_grain_radii=np.array(std_grain_radii) # in microns
        self.size_cutoff=size_cutoff # cutoff radius for grains in microns
        self.rep_volume=rep_volume # representative volume of sediment in m3 for doing calculations, if total number of zircons becomes low because of low concentration, might need to increase this value
        self.rho_Zir=4650 # in kg/m3
        self.__consistency_check()
        self.__calc_zircon_mass()
        self.__calc_num_zircons()

    def __consistency_check(self):
        # Check number of inputs are consistent
        if ((len(self.fractions)==len(self.zirc_masses)) & 
            (len(self.zirc_masses)==len(self.mean_grain_radii)) & 
            (len(self.mean_grain_radii)==len(self.std_grain_radii))):
            pass
        else:
            raise Exception('The number of entries for fractions, zirc_masses, and grain_radii must all contain the same number of entries')
            
        # Check that fractions sum to 1 approximately
        if np.isclose(np.sum(self.fractions),1):
            pass
        else:
            raise Exception('Sum of entries in fractions must equal 1')
            
        self.num_units=len(self.fractions)

    def __calc_zircon_mass(self):
        # Convert the zircon concentration in kg/m3 to a fraction of zircon by mass, assuming
        # a representative volume of sediment
        self.zirc_masses=self.zirc_masses*self.rep_volume
        self.total_zirc_mass=np.sum(self.zirc_masses)
        self.zirc_mass_fractions=self.zirc_masses/self.total_zirc_mass

    def __grain_mass(self,R):
        R=R/1e6
        V=(4/3)*np.pi*R**3
        return V*self.rho_Zir
    
    def __calc_num_zircons(self):
        x=np.arange(0.05,1.05,0.1)
        unfums=np.zeros(self.num_units)
        for i in range(self.num_units):
            ppfs=norm.ppf(x,loc=self.mean_grain_radii[i],scale=self.std_grain_radii[i])
            grain_masses=self.__grain_mass(ppfs)
            unfums[i]=(self.zirc_masses[i]/np.sum(0.1*grain_masses)).astype(int)
        self.zirc_unfiltered_num=unfums
        cutoff_cdfs=norm.cdf(self.size_cutoff,loc=self.mean_grain_radii,scale=self.std_grain_radii)
        self.zirc_filtered_num=(self.zirc_unfiltered_num*(1-cutoff_cdfs)).astype(int)
        self.total_num_zirc=np.sum(self.zirc_filtered_num)
        self.zirc_num_fractions=self.zirc_filtered_num/self.total_num_zirc
        if self.total_num_zirc>100000:
            pass
        else:
            raise Exception('Total number of zircons is less than 100,000. Increase representative volume of sediment with "rep_volume" argument.')
            


class Mixer:
    def __init__(self,pops,splits,num_grains,num_iterations,min_age=0,
                 max_age=4500,age_inc=1,start_seed=0):
        self.pops=pops
        self.splits=splits
        self.num_grains=num_grains
        self.num_iterations=num_iterations
        self.min_age=min_age
        self.max_age=max_age
        self.t=np.arange(min_age,max_age+age_inc,age_inc) 
        self.start_seed=start_seed
        self.__consistency_check()
        self.__gen()
        
    def __consistency_check(self):
        if len(self.pops)==len(self.splits):
            pass
        else:
            raise Exception('The number of entries for pops and splits must all contain the same number of entries')
            
        
    def __gen(self):
        self.grains_per_parent=(self.num_grains*np.array(self.splits)).astype(int)
        self.num_parents=len(self.pops)
        
    def __pdp_one(self,x,e):
        term1=1/(e*np.sqrt(2*np.pi))
        term2=np.exp((-(self.t-x)**2)/(2*e**2))
        return term1*term2
    
    def pdp_func(self,ages,unc):
        pdp=sum(list(map(self.__pdp_one,ages,unc)))/len(ages)
        return pdp
    
    def generate_single_sample(self,seed):
        rng=np.random.default_rng(seed)
        ages=[]
        ages_u=[]
        for i in range(self.num_parents):
            pop=self.pops[i]
            idx=rng.choice(pop.num_grains,self.grains_per_parent[i])
            ages.append(pop.ages[idx])
            ages_u.append(pop.one_sigma[idx])
        return np.concatenate((ages),axis=0),np.concatenate((ages_u),axis=0)
    
    def sampler(self):
        ages=[]
        ages_u=[]
        pdp=[]
        for i in range(self.num_iterations):
            [a,au]=self.generate_single_sample(self.start_seed+i)
            p=self.pdp_func(a,au)
            ages.append(a)
            ages_u.append(au)
            pdp.append(p)
        return ages,ages_u,pdp
    
class Comparer:
    def __init__(self,ages1,ages2,pdps1,pdps2):
        self.ages1=ages1
        self.ages2=ages2
        self.pdps1=pdps1
        self.pdps2=pdps2
        self.pop1_numtrials=self.ages1.shape[0]
        self.pop2_numtrials=self.ages2.shape[0]
            
    def ks_compare(self):
        ks=np.zeros((self.pop1_numtrials,self.pop2_numtrials))
        for i in range(self.pop1_numtrials):
            for j in range(self.pop2_numtrials):
                res=ks_2samp(self.ages1[i,:],self.ages2[j,:])
                ks[i,j]=res.statistic
        ks_mn=np.mean(ks.ravel())
        ks_std=np.std(ks.ravel())
        ks_25=np.percentile(ks.ravel(),25)
        ks_75=np.percentile(ks.ravel(),75)
        ks_list=[ks_mn,ks_std,ks_25,ks_75]
        return ks_list
    
    def pdpcross_compare(self):
        R2=np.zeros((self.pop1_numtrials,self.pop2_numtrials))
        for i in range(self.pop1_numtrials):
            for j in range(self.pop2_numtrials):
                [_,_,R,_,_]=linregress(self.pdps1[i,:],self.pdps2[j,:])
                R2[i,j]=R**2
        R2_mn=np.mean(R2.ravel())
        R2_std=np.std(R2.ravel())
        R2_25=np.percentile(R2.ravel(),25)
        R2_75=np.percentile(R2.ravel(),75)
        R2_list=[R2_mn,R2_std,R2_25,R2_75]
        return R2_list
            
    def both_compare(self):
        ks=np.zeros((self.pop1_numtrials,self.pop2_numtrials))
        R2=np.zeros((self.pop1_numtrials,self.pop2_numtrials))
        for i in range(self.pop1_numtrials):
            for j in range(self.pop2_numtrials):
                res=ks_2samp(self.ages1[i,:],self.ages2[j,:])
                ks[i,j]=res.statistic
                [_,_,R,_,_]=linregress(self.pdps1[i,:],self.pdps2[j,:])
                R2[i,j]=R**2
        ks_mn=np.mean(ks.ravel())
        ks_std=np.std(ks.ravel())
        ks_25=np.percentile(ks.ravel(),25)
        ks_75=np.percentile(ks.ravel(),75)
        ks_list=[ks_mn,ks_std,ks_25,ks_75]
        R2_mn=np.mean(R2.ravel())
        R2_std=np.std(R2.ravel())
        R2_25=np.percentile(R2.ravel(),25)
        R2_75=np.percentile(R2.ravel(),75)
        R2_list=[R2_mn,R2_std,R2_25,R2_75]
        return ks_list,R2_list
            
        
    
        



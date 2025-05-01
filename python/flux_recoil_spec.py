import numpy as np
import pandas as pd
import scipy.constants as co
import scipy.stats as ss
import itertools
import pickle
from scipy import signal
import ENDF6el as endfel
import masses as ms
import scipy.constants as co
import periodictable as pt
from pyteomics import mass as pyteo_mass
import re
import h5py
import os 
from pathlib import Path
import json
import time

#############
# Constants #
#############

#constants
NA = co.physical_constants['Avogadro constant'][0]
s2day = 1/(60*60*24) #seconds to days
#constants for calcs, first in SI units
gn = co.physical_constants['neutron gyromag. ratio'][0] #default is s^-1 T^-1; CGS is s^-1 Gauss^-1
mub = co.physical_constants['Bohr magneton'][0] #default is J T^-1
hbar = co.physical_constants['reduced Planck constant'][0] #default in J s

#convert to CGS
#see https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units
m_n_CGS = co.physical_constants['neutron mass'][0]*1e3 #convert to grams
gn_CGS = gn/1e4
mub_CGS = mub*1e3
hbar_CGS = hbar*1e7

# extrapolate line from lower-energy fast neutrons
E_thresh = 2e-2 # upper bound of linear region
E_therm = 0.15e-6 # near boundary of where thermal distribution has peak

#############
# Functions #
#############

def integrate_df(df):
    # (left-sided rectangular integral)
    dE = -df['E'].diff(periods = -1)
    dE.iat[-1] = dE.iat[-2]
    A = df['spec']*dE
    return A.sum()

def fast_extrapolation_line(E,fitted_line):
    # return height of fitted loglog line, if energy is larger than thermal threshold
    return np.exp(fitted_line.intercept + fitted_line.slope*np.log(E))*(E > E_therm)

def Emax(En): #En in keV; returns maximum recoil energy for neutron energy
    return (4*ms.m_e*ms.m_n*En)/(ms.m_e+ms.m_n)**2

def Enmin(Er): #recoil energy in keV; returns minimum neutron energy to give that recoil energy
    return (Er*(ms.m_e+ms.m_n)**2)/(4*ms.m_e*ms.m_n)

def dsigdErNE(En,Er):
    if(Er<Emax(En)):
      return 8*np.pi*m_n_CGS**2*gn_CGS**2*mub_CGS**2/hbar_CGS**2/Emax(En)
    else: 
      return 0

def SNOLAB_flux(Enmin=1e-3):

  # read in fast neutron flux spectrum (from reading_n_spectra.ipynb)
  fast_flux_df = pd.read_pickle('../data_files/FDF.txt') # 'E' in MeV, 'spec' in neutrons cm^-2 sec^-1 MeV^-1

  #use numpy arrays
  ff = np.asarray(fast_flux_df['E']);
  ffspec = np.asarray(fast_flux_df['spec']);

  # calculate flux level of fast neutrons
  fast_flux = integrate_df(fast_flux_df)
  print('fast flux: {} n/m^2/day'.format((fast_flux*10000)*24*60*60))

  #smooth the data
  ffspec_smooth = signal.savgol_filter(ffspec, 501, 3) # window size 501, polynomial order 3

  cutoff=0.3

  ffhe = ff[ff>cutoff]
  ffhespec = ffspec[ff>cutoff]
  
  #smooth the data
  ffhespec_smooth = signal.savgol_filter(ffhespec, 2001, 3) # window size 1001, polynomial order 3
  
  ffle = ff[ff<=cutoff]
  fflespec = ffspec[ff<=cutoff]
  print(np.size(ffle))
  
  #smooth the data
  fflespec_smooth = signal.savgol_filter(fflespec, 75, 3) # window size 1001, polynomial order 3

  etot = np.concatenate((ffle,ffhe))
  etot = np.unique(etot)
  print('shape of etot: {}'.format(np.shape(etot)))
  etotspec = np.zeros((np.size(etot),),dtype=np.float64)
  print(np.size(etot),np.size(etotspec))

  etotspec[(etot<=ffle[-1])] = fflespec_smooth
  etotspec[(etot>ffle[-1])] = ffhespec_smooth

  
  fast_lin_df = ffle[ffle < E_thresh]
  fast_lin_df_spec = fflespec_smooth[ffle< E_thresh]
  
  fitted_line = ss.linregress(np.log(fast_lin_df), np.log(fast_lin_df_spec))
  print(fitted_line)
  
  EE = np.geomspace(Enmin, 2e-2, 10_000)

  #ax1.plot(ff, ffspec,label='simulated flux')
  #ax1.plot(etot, etotspec,color='orange',label="smoothed flux")
  #plt.plot(EE, fast_extrapolation_line(EE), color='orange', linestyle = 'dashed',label="extrapolated flux")

  #trim the extrapolated part
  upperE=np.min(etot)
  EE=EE[EE<upperE]
  EF=fast_extrapolation_line(EE,fitted_line)

  #cat them together
  E=np.append(EE,etot)
  F=np.append(EF,etotspec)

  print(np.max(EE),np.min(etot))

  return E,F,ff,ffspec

def dRdEr(Er,En,F,N=100,Z=14,A=28):


  #get min neutron energy
  mass = ms.getMass(Z,A)
  Enmin = Er/(4*mass*ms.m_n/(mass+ms.m_n)**2)
  #print(Enmin)

  #cut down the density of points
  idx=np.arange(0,len(En),1)
  cidx=idx%N==0
  En=En[cidx]
  F=F[cidx]

  #trim the flux energies for ones that can actually contribute
  cEn=En>=Enmin
  En=En[cEn]
  F=F[cEn]

  if(np.shape(En)[0]<2):
    return 0.0


  #get the filenames
  symbol=pt.elements[Z].symbol
  symbol_lower=symbol.lower()
  sigtotfile='../data_files/xn_data/{0:}{1:}_el.txt'.format(symbol_lower,A)
  endffile='../data_files/xn_data/n-{0:03d}_{1:}_{2:03d}.endf'.format(Z,symbol,A)
  print(sigtotfile,endffile)


  #print(np.shape(En))
  dsig=np.zeros(np.shape(En))
  for i,E in enumerate(En):
    E*=1e6
    dsder = endfel.fetch_der_xn(En=E,M=mass,pts=1000,eps=1e-5,sigtotfile=sigtotfile,endffile=endffile)
    val = dsder(Er)
    if val>0:
      dsig[i] = val 
    #print(E,dsig[i])


  dsig*=(endfel.barns2cm2*endfel.keV2MeV)
  d = {'E':En,'spec':F*dsig}
  df = pd.DataFrame(data=d)
  #data=pd.DataFrame(np.array([En, F*dsig]), columns=['E', 'spec'])
  integral = integrate_df(df)

  integral*=(1/s2day) 
  integral*=(1/(ms.getAMU(Z,A)*ms.amu2g*1e-3))  
  return integral

def dRdErfast(Er,En,F,N=100,Z=14,A=28):

  #vectorize Er
  if isinstance(Er, float):
        Er=[Er]
  Er = np.asarray(Er)
  #print(Er)

  #get min neutron energy
  mass = ms.getMass(Z,A)
  Enmin = Er/(4*mass*ms.m_n/(mass+ms.m_n)**2)
  #print(Enmin)

  #cut down the density of points
  idx=np.arange(0,len(En),1)
  cidx=idx%N==0
  En=En[cidx]
  F=F[cidx]

  if(np.shape(En)[0]<2):
    return 0.0

  #get the filenames
  #sigtotfile='../data_files/xn_data/si{}_el.txt'.format(A)
  #endffile='../data_files/xn_data/n-{0:03d}_Si_{1:03d}.endf'.format(Z,A)
  #print(sigtotfile,endffile)

  #get the filenames
  symbol=pt.elements[Z].symbol
  symbol_lower=symbol.lower()
  sigtotfile='../data_files/xn_data/{0:}{1:}_el.txt'.format(symbol_lower,A)
  endffile='../data_files/xn_data/n-{0:03d}_{1:}_{2:03d}.endf'.format(Z,symbol,A)
  print(sigtotfile,endffile)

  #make big ole matrix
  dsig=np.zeros((np.shape(En)[0],np.shape(Er)[0]))
  for i,E in enumerate(En):
    E*=1e6
    dsder = endfel.fetch_der_xn(En=E,M=mass,pts=1000,eps=1e-5,sigtotfile=sigtotfile,endffile=endffile)
    dsig[i,:] = dsder(Er)
    #print(dsig[i,:])


  #remove negatives 
  dsig[dsig<0]=0


  #integrate
  integral=np.zeros(np.shape(Er))
  for i,E in enumerate(Er): 

    #trim the flux energies for ones that can actually contribute
    enidx=np.arange(0,len(En),1)
    cEn=En>=Enmin[i]
    Entemp=En[cEn]
    Ftemp=F[cEn]
    enidx=enidx[cEn]
    xn=dsig[enidx,i]
    xn*=(endfel.barns2cm2*endfel.keV2MeV)
    if(np.shape(enidx)[0]<2):
      integral[i]=-999999999
    else:
      #print(np.shape(enidx))
      d = {'E':Entemp,'spec':Ftemp*xn}
      df = pd.DataFrame(data=d)
      #data=pd.DataFrame(np.array([En, F*dsig]), columns=['E', 'spec'])
      integral[i] = integrate_df(df)


  integral*=(1/s2day) 
  integral*=(1/(ms.getAMU(Z,A)*ms.amu2g*1e-3))  
  return integral,dsig

def dRdErCompoundSave(Er,En,F,*,N=1,Comp='Si',perc_cut=0.0,name='SNOLAB',path="saved_data/",save=True,force=False,debug=False):

 #find my location and top location
 dir_path = os.path.dirname(os.path.realpath(__file__))
 wd=os.getcwd()
 path=os.path.normpath(dir_path+'/../'+path)

 #make the data path if doesn't exist
 Path(path).mkdir(parents=False, exist_ok=True)

 #get string name for file
 filename='dRdEr-{}.h5'.format(name)
 filename=path+'/'+filename
 if debug: print(filename)

 boolGotFile=os.path.isfile(filename)
 if(boolGotFile):  
   f = h5py.File(filename,'r')
   path='{}/{}/'.format(Comp,perc_cut)
   Exvars = (path+'Er' in f)&(path+'En' in f)&(path+'F' in f)
   f.close()
   if(not Exvars):
     print('you dont have variables')
     boolGotFile=False 

 fileresF=0
 fileresEr=0
 #read the file if it exists
 if(not force):
   if(boolGotFile):  
     f = h5py.File(filename,'r')
     path='{}/{}/'.format(Comp,perc_cut)
     Er_read = np.asarray(f[path+'Er'])
     En_read = np.asarray(f[path+'En'])
     F_read = np.asarray(f[path+'F'])
     fileresF=np.shape(F_read)[0]
     fileresEr=np.shape(Er_read)[0]

     boolEr=False
     boolEn=False
     boolF=False
     if(np.shape(Er_read)==np.shape(Er)):
       boolEr=(Er==Er_read).all()
     if(np.shape(En_read)==np.shape(En)):
       boolEn=(np.isclose(En,En_read,atol=1e-15)).all()
       if debug and not boolEn:
           print(En[En!=En_read],En_read[En!=En_read])
           print(En[En!=En_read]-En_read[En!=En_read])
     if(np.shape(F_read)==np.shape(F)):
       boolF=np.isclose(F,F_read,atol=1e-15).all()

     Exiso = path+'iso' in f
     Exisodict = path+'isodict' in f

     #if the file has it at exactly the resolution requested, then return it
     if(boolEr&boolEn&boolF&Exiso&Exisodict):
       iso = np.asarray(f[path+'iso'])
       isodict_str=f[path+'isodict'].asstr()[0]
       #have to replace ' with " for real JSON
       p = re.compile('(?<!\\\\)\'')
       isodict_str = p.sub('\"', isodict_str)
       isodict = json.loads(isodict_str)
       f.close()
       return iso,isodict

     print('closing file')
     f.close() 

     if(debug): print("Calculating. Er succeeded?",boolEr,"En succeeded?",boolEn,"F succeeded?",boolF)

 #otherwise, compute it and save it 
 start = time.time()
 iso,isodict=dRdErCompound(Er,En,F,N=N,Comp=Comp,perc_cut=perc_cut)
 end = time.time()
 print('Evaluation Time: {:1.5f} sec.'.format(end-start))


 #do this if save is requested, and if it is equal or better resolution, or if force is set

 betterRes=False
 print(fileresF,fileresEr)
 print(np.shape(F)[0],np.shape(Er)[0])
 if(fileresF<=np.shape(F)[0])&(fileresEr<=np.shape(Er)[0]):
   betterRes=True

 if(save&(betterRes|force)): 
   f = h5py.File(filename,'a')
   path='{}/{}/'.format(Comp,perc_cut)
   
   #remove vars
   exEr = path+'Er' in f
   exEn = path+'En' in f
   exF = path+'F' in f
   exiso = path+'iso' in f
   exisodict = path+'isodict' in f
   
   
   if exEr:
     del f[path+'Er']
   if exEn:
     del f[path+'En']
   if exF:
     del f[path+'F']
   if exiso:
     del f[path+'iso']
   if exisodict:
     del f[path+'isodict']
       
   dset = f.create_dataset(path+'Er',np.shape(Er),dtype=np.dtype('float64').type)
   dset[...] = Er
   dset = f.create_dataset(path+'En',np.shape(En),dtype=np.dtype('float64').type)
   dset[...] = En
   dset = f.create_dataset(path+'F',np.shape(F),dtype=np.dtype('float64').type)
   dset[...] = F
   dset = f.create_dataset(path+'iso',np.shape(iso),dtype=np.dtype('float64').type)
   dset[...] = iso
   ds = f.create_dataset(path+'isodict', shape=4, dtype=h5py.string_dtype())
   ds[:] = str(isodict)
       
       
   f.close()
 
 if(save&(not betterRes)&(not force)):
   print('use force=True to save a spectrum with worse resolution than previous') 

 return iso,isodict

def dRdErCompound(Er,En,F,N=100,Comp='Si',perc_cut=0.0):

  iso=organizeCompound(Comp)
  print(iso) 

  #get the proper normalization divisor
  newiso={}
  norm=0
  count=0
  for itope in iso:
    eta=iso[itope]['a'] 
    A=int(iso[itope]['A']) 
    Z=int(iso[itope]['Z']) 
    norm+=eta*(ms.getAMU(Z,A)*ms.amu2g*1e-3) #correct normalization for hetero mixture of isotopes 
    newiso[itope]=eta
    count+=1

  print(norm)

  #fetch all the drde's and renormalize them
  for itope in iso:
    A=int(iso[itope]['A']) 
    Z=int(iso[itope]['Z']) 
    Sy=iso[itope]['Symbol'] 
    print(A,Z,Sy)
    unnorm=(ms.getAMU(Z,A)*ms.amu2g*1e-3) #undo the standard normalization for one isotope  
    drde,dsig=dRdErfast(Er,En,F,N=N,Z=Z,A=A)
    drde[drde<0]=0 #trim the negatives.
    iso[itope]['Er']=Er
    iso[itope]['dRdEr']=drde*unnorm/norm
    #m = re.search(r'(^[A-Z][a-z]?)\[([1-9][0-9]?[0-9]?)\]', j)
    #print(m.group(1),m.group(2),pt.elements.symbol(m.group(1)).number)

  #print(iso)

  #make a data structure for the output
  isout=np.zeros((np.shape(Er)[0],count+1))
  count1=0
  for itope in iso:
    isout[:,count1+1] = iso[itope]['dRdEr']
    count1+=1

  #sum up the full drde
  #isout[:,0]=np.sum(isout,1)
  count2=0
  for itope in iso:
    eta=newiso[itope]
    if(count2<count): 
      isout[:,0] += eta*isout[:,count2+1] 
    count2+=1

  return isout, newiso

def organizeCompound(Comp='Si'):

  #first we want to get the stats for the compound and find each needed isotope
  num=0
  isotope_struct={}
  isotope_abund={}
  for m in pyteo_mass.isotopologues(Comp):
    #print(mass.isotopic_composition_abundance(m))
    #print(m)
    isotope_struct[num]=m
    isotope_abund[num]=pyteo_mass.isotopic_composition_abundance(m)
    num=num+1

  sorted_isotopes = sorted(isotope_abund.items(),key=lambda x:x[1],reverse=True)

  #print(sorted_isotopes)
  isotopes_used={}
  for i in sorted_isotopes:
      #print(isotope_struct[i[0]])
      #print(i[1])
      isotope_abundance=i[1]
      for j in isotope_struct[i[0]]:
        #print(j)
        #print(compPerc(isotope_struct[i[0]],j))
        atom_abundance=compPerc(isotope_struct[i[0]],j)
        m = re.search(r'(^[A-Z][a-z]?)\[([1-9][0-9]?[0-9]?)\]', j)
        #print(m.group(1),m.group(2),pt.elements.symbol(m.group(1)).number)
        if(j in isotopes_used):
          ab=isotopes_used[j]['a']+(isotope_abundance*atom_abundance)
        else:
          ab=(isotope_abundance*atom_abundance)
        isotopes_used[j]={'a':ab,'Z':pt.elements.symbol(m.group(1)).number,'A':m.group(2),'Symbol':m.group(1)}


  #normalization check.
  s=0.0
  for i in isotopes_used:
    s+=isotopes_used[i]['a']

  print('Compound Isotope Breakdown Sum (should be 1.0):{}'.format(s))
 
  #for i in isotopes_used.items():
  #  print(i[1])
 
  sorted_isotopes_used = dict(sorted(isotopes_used.items(),key=lambda x:x[1]['a'],reverse=True))

  return sorted_isotopes_used

def compPerc(comp,element='Ge[70]'):
  s=0
  for i in comp:
    s+=comp[i]

  if(s==0):
    return 0;

  for i in comp:
    if(i==element):
      return comp[i]/s
  
  return 0.0

def dRdErNE(Er,En,F,N=1,Z=14,A=28,eta=None): #for neutron scattering off electrons eta is the electrons/atom

  #vectorize Er
  if isinstance(Er, float):
        Er=[Er]
  Er = np.asarray(Er)
  #print(Er)

  #get min neutron energy
  Enmin = Er/(4*ms.m_e*ms.m_n/(ms.m_e+ms.m_n)**2)
  #print(Enmin)

  #get the number of electrons eta=Z if eta is None...
  if (eta==None):
    eta=Z

  print(eta)

  #cut down the density of points
  idx=np.arange(0,len(En),1)
  cidx=idx%N==0
  En=En[cidx]
  F=F[cidx]

  if(np.shape(En)[0]<2):
    return 0.0

  #make big ole matrix
  dsig=np.zeros((np.shape(En)[0],np.shape(Er)[0]))
  for i,E in enumerate(En):
    E*=1e3 #E in keV
    dsder = lambda x: dsigdErNE(E,x)
    dsderv = np.vectorize(dsder)
    dsig[i,:] = dsderv(Er*1e3)
    #print(E,dsig[i,:])


  #remove negatives 
  dsig[dsig<0]=0


  #integrate
  integral=np.zeros(np.shape(Er))
  for i,E in enumerate(Er): 

    #trim the flux energies for ones that can actually contribute
    enidx=np.arange(0,len(En),1)
    cEn=En>=Enmin[i]
    Entemp=En[cEn]
    Ftemp=F[cEn]
    enidx=enidx[cEn]
    xn=dsig[enidx,i]
    #xn*=(endfel.barns2cm2*endfel.keV2MeV)
    #xn*=(endfel.barns2cm2)
    if(np.shape(enidx)[0]<2):
      integral[i]=-999999999
    else:
      #print(np.shape(enidx))
      d = {'E':Entemp,'spec':Ftemp*xn}
      df = pd.DataFrame(data=d)
      #data=pd.DataFrame(np.array([En, F*dsig]), columns=['E', 'spec'])
      integral[i] = integrate_df(df)


  integral*=(1/s2day) 
  integral*=(eta/(ms.getAMU(Z,A)*ms.amu2g*1e-3))  
  return integral,dsig

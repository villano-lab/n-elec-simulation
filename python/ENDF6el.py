import ENDF6
import pandas as pds
import scipy, scipy.interpolate
import numpy as np
import masses as ms

barns2cm2 = 1e-24
keV2MeV = 1e-3

directory='../data_files/xn_data/'

def set_dir(d='../data_files/xn_data/'):
  global directory
  directory=d
  return

def print_dir():
  global directory
  print(directory)
  return

def fetch_elastic(filename='../data_files/xn_data/si28_el.txt'):
  el = pds.read_csv(filename, skiprows=11,skipfooter=2, \
          names=['neutE', 'xn'],sep='\s+',engine='python')

  neute = np.asarray(el["neutE"],dtype=float)
  xn = np.asarray(el["xn"],dtype=float)

  #make sure we are strictly increasing
  d = np.diff(neute)
  d=np.append(d,0)
  neute = neute[d>0]
  xn = xn[d>0]

  #get an interpolant function
  f=scipy.interpolate.UnivariateSpline(
        neute,
        xn,
        k=3,
        s=0,
        check_finite=True)

  return f

def fetch_elastic_angular(filename='../data_files/xn_data/n-014_Si_028.endf'):

  #check if it's an ENDF file somehow?

  #get the section of the ENDF file that has ang dists
  f = open(filename)
  lines = f.readlines()
  sec = ENDF6.find_section(lines, MF=4, MT=2)

  #check if it has a legendre coeffs section?--I'm assuming it does right now..

  #get number of data points
  arr=np.str_.split(sec[3])
  num=int(arr[0])
  #print(num)
  al = np.zeros((num,64)) #assume no more than 64 legendre coeffs
  en = np.zeros((num,))

  #get the coeffs into the data structure
  readl=True
  linecnt=0
  ecnt=0
  tote=num
  mpoles=0
  for ln in sec[4:-1]:
      #print(ln)
      #print(ecnt)
      #break away if you're done reading e points
      if ecnt==(tote-1):
        break
      #read the number of multipoles
      if readl:
        arr=np.str_.split(ln)
        mpoles=int(arr[4])
        en[ecnt]=float(arr[1].replace("-", "e-").replace("+", "e+").lstrip("e"))
        #print(mpoles)
        readl=False
      else:
        #arr=np.str_.split(ln)
        a = ENDF6.read_line(ln)
        #print(a)
        i1=linecnt*6
        i2=i1+6
        #print(i1,i2)
        #print(mpoles,en[ecnt])
        #print(np.shape(al),np.shape(a))
        al[ecnt,i1:i2] = a
        #ecnt+=1
        linecnt+=1
        if linecnt==np.ceil(mpoles/6.0):
           linecnt=0
           ecnt+=1
           readl=True

      arr = np.char.split(ln)
      #print(np.shape(arr))
      if(np.shape(arr)==()): continue
      #print([np.fromstring(x) for x in arr])

  #close file
  f.close()

  return (en,al)

def al(lterms=[0],endffile='../data_files/xn_data/n-014_Si_028.endf'): #En in eV

  global directory

  #get the data
  (en,al)=fetch_elastic_angular(endffile)

  #get the zeroth coefficient which is understood to be 1
  a0 = np.ones(np.shape(en))

  #default
  err = np.zeros(np.shape(en))

  #set up
  #make sure we are strictly increasing
  d = np.diff(en)
  d=np.append(d,0)
  en = en[d>0]


  f={}
  for l in lterms:
    #return the correct coefficient as a function of neutron energy
    if l==0:
      cdat = a0[d>0]
    elif l<np.shape(al)[1]:
      cdat = al[d>0,l-1]
    else:
      cdat = err[d>0]

    #get an interpolant function
    tempf=scipy.interpolate.UnivariateSpline(
          en,
          cdat,
          k=3,
          s=0,
          check_finite=True)

    label='a{}'.format(l)
    f[label] = tempf

  return f

def fetch_diff_xn(En=1e6,*,f=None,a=None,sigtotfile='../data_files/xn_data/si28_el.txt',endffile='../data_files/xn_data/n-014_Si_028.endf',NL=64):

  global directory

  #fetch the necessary info
  l = np.arange(0,NL)
  if(a==None):
    a = al(l,endffile=endffile)

  if(f==None):
    f = fetch_elastic(filename=sigtotfile)


  prel = (2*l+1)/2

  #loop through and get coeffs
  c=np.zeros(np.shape(l))
  for i,obj in enumerate(a):
    #print(a[obj](En))
    c[i] = prel[i]*a[obj](En)*f(En/1e6)*(1/(2*np.pi))

  fout = np.polynomial.legendre.Legendre(c)
  return fout

def fetch_der_xn(En=1e6,*,M=ms.getMass(14,28),pts=100,eps=1e-5,f=None,a=None,sigtotfile='../data_files/xn_data/si28_el.txt',endffile='../data_files/xn_data/n-014_Si_028.endf'):

  global directory

  #units change
  En=En/1e6

  #get the angular cross section in CM
  dsdomeg=fetch_diff_xn(En=En*1e6,f=f,a=a,sigtotfile=sigtotfile,endffile=endffile)
  #dsdomegv=np.vectorize(dsdomeg)

  #get jacobian and stuff
  fac = M*ms.m_n/(M+ms.m_n)**2
  jac = (1/(2*fac))*2*np.pi

  #get a pointwise description of cross section dsig/dEr (1/MeV)
  ct = np.linspace(-1.0,1.0,pts)
  escale = 2*fac*(1-np.linspace(-1.0,1.0,pts))
  X=En*escale
  Y=(jac/En)*dsdomeg(ct)

  #add a point just past the end
  X = np.append([np.max(X)+eps],X)
  Y = np.append([0],Y)

  #get an interpolant function
  f=scipy.interpolate.UnivariateSpline(
          np.flip(X),
          np.flip(Y),
          k=1,
          s=0,
          check_finite=True)

  return f

def calc_der_xn(Er,*,En=1,M=ms.getMass(14,28),fd=None):

  global directory

  if(fd==None):
    return -1


  #get jacobian and stuff
  fac = M*ms.m_n/(M+ms.m_n)**2
  jac = (1/(2*fac))*2*np.pi

  ct = 1-(Er/(2*fac))


  return jac*fd(ct)

import os
import math
import numpy as np
import numpy.matlib
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from scipy import interpolate
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy.signal import filtfilt
from scipy.stats import kurtosis
from statistics import mode
from sklearn.externals import joblib


from numpy.fft import fft,ifftshift,ifft
from numpy import exp,pi,arange,zeros_like,isreal


def startprocess(uploadfilepath, ll):
    LL = int(ll)
    fs = 300
    Fs1 = 1000

    # load 1.mat
    x1 = loadmat(uploadfilepath)
    x = x1['data']

    xx1 = x[:,3] #ECG signal in column 4
    xx1 = xx1.transpose()

    xx2 = x[:,4] #SCG signal in column 5
    xx2 = xx2.transpose()

    # create data.txt file ~ ECG data for plotting in JS
    jp=0
    with open('./data.txt','w') as f:
        for fx in xx1:
            f.write("%s\n" %fx)
            jp+=1

    t = np.linspace(0,LL,LL*fs)

    tnew = np.arange(0,(len(xx1)/Fs1)+(1/fs), 1/fs)
    ii = len(tnew)

    #delete duplicate pngs
    for file in os.listdir('./'):
        if file.endswith('.png'):
            os.remove(file)

    #resampling  //alternative x_ecg1 = resampy.resample(xx1,fs,Fs1)
    x_ecg1 = signal.resample(xx1,ii)
    x_ecg = x_ecg1[:LL*fs]#[40*fs:(40+LL)*fs]
    x_ecg = x_ecg-statistics.mean(x_ecg)
    x_ecg = np.divide(x_ecg,max(abs(x_ecg)))

    #resampling  //alternative x_scg1 = resampy.resample(xx2,fs,Fs1)
    x_scg1 = signal.resample(xx2,ii)
    x_scg = x_scg1[:LL*fs]#[40*fs:(40+LL)*fs]
    x_scg = x_scg-statistics.mean(x_scg)
    x_scg = np.divide(x_scg,max(abs(x_scg)))

    Ae = np.array([x_ecg , delayseq(x_ecg,1,1), delayseq(x_ecg,2,1), delayseq(x_ecg,3,1), delayseq(x_ecg,4,1), delayseq(x_ecg,5,1), delayseq(x_ecg,6,1), delayseq(x_ecg,7,1), delayseq(x_ecg,8,1), delayseq(x_ecg,9,1), delayseq(x_ecg,10,1)])
    A1 = np.transpose(Ae)
    matmul= np.matmul(np.transpose(A1),A1)
    c = np.linalg.inv(matmul)
    d = np.transpose(A1)
    P1 = np.matmul(A1,np.matmul(c,d))

    Xnew1 = np.matmul(np.transpose(P1), x_scg)

    tio = []
    tio = msm_rpeakdetector_filtering(np.transpose(Xnew1),fs)
    [y,  see, z1, AOpk] = [tio[0], tio[1], tio[2], tio[3]]

    AOpk = list(AOpk)

    #if first AO pk is less than 50 ms, than reject it
    if AOpk[0] <= math.floor(0.050*fs):
        AOpk.remove(AOpk[0])


    #%if last AO pk is 50 ms nearer to the last sample, than reject it
    if AOpk[len(AOpk)-1] >= len(x_scg)-(math.floor(0.050*fs)+1):
        AOpk.remove(AOpk[len(AOpk)-1])

    AOpk=list(np.add(AOpk,1))#Since this python code has deviation of 1

    c = []

    tAOpk = returnval(t,AOpk)
    c = []
    Xnew1AOpk = returnval(Xnew1,AOpk)
    c = []
    x_scgAOpk = returnval(x_scg,AOpk)
    
    # Plotting of SCG over ECG subspace
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, x_ecg) #we can plot(x_ecg)
    axs[0].grid(True)
    axs[1].plot(t, x_scg) #we can plot(x_scg)
    axs[1].grid(True)
    axs[2].plot(t, Xnew1)
    axs[2].plot(tAOpk,Xnew1AOpk, "r*")
    axs[2].grid(True)
    axs[3].plot(t, x_scg)
    axs[3].plot(tAOpk,x_scgAOpk, "r*")
    axs[3].grid(True)
    fig.tight_layout()
    plt.savefig('ecgscg.png',dpi=1000)
    
    Npeaks=len(AOpk)
    dAO=np.diff(AOpk)/fs
    HR1=(np.divide(60,dAO)).transpose()
    
    
    col = ['blue','green','red','cyan','magenta','yellow','black','white','0.75','0.95','0.82','0.71','0.85','0.86','0.88','0.77']
    
    
    Beat_enAvg = 0
    dias_enAvg = 0
    
    BEnr = []
    BEnt = []
    BSC = []
    MSA = []
    MSF = []
    ix = []
    ACF = []
    IA = []
    Kurt = []
    dias = []
    dias_enAvg = []
    DEnr = []
    BSEnt = []
    DEnt = []

    i = 0
    fig, axs = plt.subplots(2, 1)
    while i < Npeaks-1:
        xdata = x_scg[AOpk[i]:AOpk[i+1]+1]

        lxdata = len(xdata)
        xidx = np.arange(0,lxdata)
        ff = interpolate.interp1d(xidx,xdata)
        xidxn= np.linspace(0,lxdata-1,500)
        xdata1 = ff(xidxn)

        BEnr.append(sum(np.power(xdata1,2)) / len(xdata1))

        xdata1N = xdata1-min(xdata1)
        xdata1N = (xdata1N/sum(xdata1N)) + 1e-10
        BEnt.append(sum(np.multiply(-xdata1N,logi2(xdata1N))))

        xdata2 = xdata1N-statistics.mean(xdata1N)
        xdata2 = np.divide(xdata2,max(abs(xdata2)))
        Beat_enAvg = Beat_enAvg + (xdata2/(Npeaks-1))
        axs[0].plot(xdata2,color=col[i], linewidth=0.7)
        axs[0].grid(True)

        #ACF - new feature
        acf1= np.correlate(xdata2, xdata2, mode='full')
        acf2 = acf1[len(xdata2):len(acf1)+1]
        ACF.append(acf2[int(fs * 20e-3)])

        f1=abs(fft(xdata2,512))
        end=len(f1)
        df = int((end/2))
        f2=f1[0:df]

        ff=np.linspace(0,fs/2,len(f2))
        f9 = f2.transpose()
        BSC.append(np.matmul(ff,f9)/sum(f2))
        MSA.append(max(f2))

        f5 = list(f2)
        gt = f5.index(max(f5))
        ix.append(gt)
        MSF.append(ff[ix])

        #BSEnt - new feature
        f77 = f2/sum(f2) + 1e-10
        BSEnt.append(sum(np.multiply(-f77,logi2(f77))))


        IA.append(min(xdata2))
        Kurt.append(kurtosis(xdata2,fisher=False))


        M1 = math.floor((1+len(xdata2))/2)
        M2 = math.floor((1+M1)/2)
        M3 = math.floor((M1+len(xdata2))/2)
        dias = xdata2[M2-1:M3]
        dias_enAvg = np.zeros(len(dias))
        dias_enAvg = np.add(dias_enAvg,(dias/(Npeaks-1)))
        DEnr.append((sum(np.power(dias,2)))/len(dias))

        #DEnt - new feature
        diasN = dias - min(dias)
        diasN = (diasN/sum(diasN)) + 1e-10
        DEnt.append(sum(np.multiply(-diasN,logi2(diasN))))

        axs[1].plot(dias, color=col[i], linewidth=0.7)
        axs[1].grid(True)
        fig.tight_layout()

        i+=1

    axs[0].plot(Beat_enAvg,'k', linewidth=1.2)
    axs[0].grid(True)
    axs[1].plot(dias_enAvg,'k', linewidth=1.2)
    axs[1].grid(True)
    fig.tight_layout()
    plt.savefig('diastolic.png',dpi=1000)

    # Circular differences
    end2 = len(BEnr)-1
    end3 = len(BEnt)-1
    end4 = len(HR1)-1


    BEnrC = [BEnr[end2-2], BEnr[end2-1], BEnr[end2]]
    for p in BEnr:
        BEnrC.append(p)
    BEntC = [BEnt[end3-2], BEnt[end3-1], BEnt[end3]]
    for o in BEnt:
        BEntC.append(o)
    HR1C = [HR1[end4-2], HR1[end4-1], HR1[end4]]
    for y in HR1:
        HR1C.append(y)

    DBEnr = []
    DBEnt = []
    DHR = []

    j = 3

    while j<len(BEnrC):
        DBEnr.append(abs(BEnrC[j]-BEnrC[j-3]))
        DBEnt.append(abs(BEntC[j]-BEntC[j-3]))
        DHR.append(abs(HR1C[j]-HR1C[j-3]))
        j+=1

    MSF_correct = MSF[len(MSF)-1]

    #DEnt ACF BSEnt = new features added

    features = np.column_stack((  Kurt, IA, DEnr, DEnt, BEnr, BEnt, HR1, DBEnr, DBEnt, DHR, ACF, BSEnt, BSC, MSA, MSF_correct  ))

    # Loading of saved model
    clf = joblib.load('modelSVM.pkl')

    mV = [6.56043554080732,-0.910699352510899,0.0358424392589364,7.80090350979337,0.0378061002667829,8.89165422694508,75.8466232514644,0.00773739427986052,0.0210119985919791,4.89811519877420,22.6559558595196,6.20550374085353,23.7984618950680,38.5373856525196,5.60746704952832]
    sV = [2.91461483772910,0.138325556588010,0.0270663081956779,0.0644412165538784,0.0147259384024373,0.0414045632233591,13.5260852638283,0.00760001516497325,0.0239120254960552,6.03953269810211,12.9025432248733,0.410515390184659,5.10818899094412,12.6435633618736,2.58011273024057]
    mX = np.matlib.repmat(mV,np.shape(features)[0],1)
    sX = np.matlib.repmat(sV,np.shape(features)[0],1)
    featZ = np.divide((features-mX),sX)



    pred_levels = clf.predict(featZ)

    classID = mode(pred_levels)

    classes = ["Breathlessness/Dyspnea","Normal Respiration","Long and Labored Breathing Pattern"] #%corresponding to SB, NB and LB

    decision = classes[classID-1]
    print("The subject has:", decision)

    return decision



''' Delay Sequence Starts here  '''

def delayseq(x, delay_sec:float, fs:int):
    """
    x: input 1-D signal
    delay_sec: amount to shift signal [seconds]
    fs: sampling frequency [Hz]
    xs: time-shifted signal
    """

    assert x.ndim == 1, 'only 1-D signals for now'

    delay_samples = delay_sec*fs
    delay_int = round(delay_samples)

    nfft = nextpow2(x.size+delay_int)

    fbins = 2*pi*ifftshift((arange(nfft)-nfft//2))/nfft

    X = fft(x,nfft)
    Xs = ifft(X*exp(-1j*delay_samples*fbins))

    if isreal(x[0]):
        Xs = Xs.real

    xs = zeros_like(x)
    xs[delay_int:] = Xs[delay_int:x.size]

    return xs

def nextpow2(n:int) -> int:
    return 2**(int(n)-1).bit_length()

''' Delay Sequence Ends here  '''


''' MSM family starts here '''

def logaru(x): #correct
    ff = []
    for i in x:
        ff.append(math.log(i))
    return ff


def msm_envelope(y,thrcoeff,windowSize): #correct
    yns=np.power(y,2)
    th=thrcoeff*np.std(yns)
    eps = pow(2,-52)
    zns = np.array(yns)
    kk = []
    ran = len(yns)
    for i in range(ran):
        if yns[i]>th:
            kk.append(True)
        else:
            kk.append(False)
    ynst = np.multiply(kk,yns) + eps
    ynst=ynst/(max(abs(ynst))+0.1)
    se=np.multiply(-ynst, logaru(ynst))

    h=np.ones(windowSize)/windowSize
    senvelope = filtfilt(h,1,se)

    return senvelope

def gausswin(L,a): #correct
    N = L-1
    oo = np.arange(0,N+1,1)
    n = oo.transpose() - N/2
    w = exp(-(1/2) * np.power(( a * n/(N/2) ),2))
    return w

def msm_filtering(x,L1,sigma1): #correct
  w = gausswin(L1, sigma1)
  b = np.diff(w,1)
  a = 1
  y = filtfilt(b,a,x)
  return y

def returnval(x,y): #correct
    c = []
    for i in y:
        c.append(x[i])
    return c


def msm_rpeakdetector_filtering(x,Fs): #correct
  L1 = math.floor(0.12*(Fs/2))
  sigma1=2
  y=msm_filtering(x,L1,sigma1)
  thrcoeff=1
  windowSize=math.floor(0.1*Fs)
  d=[0]
  x99 = list(np.diff(y,1))
  for p in x99:
    d.append(p)

  see = msm_envelope(d,thrcoeff,windowSize)
  L2=math.floor(2.5*Fs)
  sigma2=math.floor(0.05*Fs)

  [rlap, s, z1]=msm_peakfindinglogic(L2,sigma2,see)
  NW = math.floor(0.0833*Fs)


  Rpeak = msm_truepeaklogic(rlap,NW,x) # testing here
  dummy = [i for i in range(len(x))]

  t = np.divide(dummy, Fs)
  return y,see,z1,Rpeak



def msm_truepeaklogic(rlap,NW,sig1):
    nq=1
    sig2 = []
    cp = list(np.zeros(NW))
    #print(len(cp))
    #print(len(sig1))

    #cp = list(cp[0])
    aa = np.transpose(cp)
    bb = np.transpose(sig1)
    cc = np.transpose(cp)
    for i in aa:
        sig2.append(i)
    for i in bb:
        sig2.append(i)
    for i in cc:
        sig2.append(i)

    #print(len(sig2))
    rlap=np.add(rlap,NW)
    k = 0
    rlap =list(rlap)
    R_W = []
    while k < len(rlap):
        WI = rlap[k] - NW
        WE = rlap[k] + NW
        wsig = []
        wsig = sig2[WI-1 : WE] #len=49, mean=0.0251

        loc1 = max(np.absolute(wsig))
        wsigabs = []
        wsigabs = list(np.absolute(wsig))
        max1 = wsigabs.index(max(np.absolute(wsig)))

        loc2 = max(wsig)
        wsigabs2 = []
        wsigabs2 = list(wsig)
        max2 = wsigabs2.index(max(wsig))

        if max1 < 0.2*max2:
            R_L1 = max1
            R_VS1 = loc1
        else:
            R_L1 = max2
            R_VS1 = loc2

        R_W.append((R_VS1, R_L1+WI-1))
        nq+=1
        k+=1

    gh = []
    for p in R_W:
        (f,d) = p
        gh.append(d)

    B_L = np.subtract(gh,NW)
    return B_L

def msm_zerocros(z): #correct
  p = 1
  i = 0
  f = []
  while i < len(z)-1:
    if(np.sign(z[i])>0) and (np.sign(z[i+1])<0):
      f.append(i)
      p=p+1
    i+=1
  z=list(z)
  s = np.subtract(returnval(z,np.add(f,1)),returnval(z,f))
  rlap = f
  return rlap,s

def msm_peakfindinglogic(L2,sigma2,senvelope): #correct
  L=L2
  s=L/(2*sigma2)
  n=np.arange(-L/2,L/2,1)
  G=np.exp(np.multiply(np.power(np.divide(n,s),2),(-0.5)))
  h = np.diff(G)
  len(h)
  len(senvelope)
  z=np.convolve(senvelope,h)
  LG=len(h)
  ss=LG/2
  end=len(z)-1
  z1=z[int(ss-1):int(end-ss+1)]
  [rlap,s] = msm_zerocros(z1)
  return rlap,s,z1

def logi2(x): #returns log2 of a list
    logit = []
    for i in x:
        logit.append(math.log2(i))
    return logit

def autocorr1(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)/2:]

def autocorr2(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))
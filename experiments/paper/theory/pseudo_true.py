import numpy as np
from math import comb
from scipy.stats import poisson
from scipy.special import betaln
from scipy.optimize import minimize_scalar
def M(at,w,A,B): return (w*at**A*(1-at)**B).sum()
def kl(p,q): return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
def bb(k,n,mu,s): return comb(n,k)*np.exp(betaln(mu*s+k,(1-mu)*s+n-k)-betaln(mu*s,(1-mu)*s))
def Delta(at,w,n,mu,s):
    return sum(comb(n,k)*M(at,w,k,n-k)*kl(M(at,w,k+1,n-k)/M(at,w,k,n-k),(mu*s+k)/(s+n)) for k in range(n+1))
def Rorc(at,w,n):
    tot=0
    for k in range(n+1):
        D=M(at,w,k,n-k); p=M(at,w,k+1,n-k)/D
        tot+=comb(n,k)*(w*at**k*(1-at)**(n-k)*kl(at,p)).sum()
    return tot
PRIORS=[(np.array([.1,.5,.9]),np.array([.25,.5,.25])),(np.array([.1,.5,.9]),np.array([.4,.2,.4])),(np.array([.05,.95]),np.array([.5,.5])),
        (np.array([.2,.3,.85]),np.array([.3,.4,.3])),(np.array([.1,.9]),np.array([.8,.2]))]
# gap_EB_n01: the part of the pseudo-true fit's gap on levels seen at most once (zero for the moment-matched beta)
for at,w in PRIORS:
    mu=M(at,w,1,0); v=M(at,w,2,0)-mu**2; smm=mu*(1-mu)/v-1
    print(at,w,"s_mm=%.3f"%smm)
    for nb in [0.5,1,2,5,10]:
        ns=np.arange(0,80); pn=poisson.pmf(ns,nb); wn=pn  # level-size dist of a level; test-weight ~ Poisson too for equal q_x
        def nll(ls):
            s=np.exp(ls); return -sum(pn[n]*sum(comb(n,k)*M(at,w,k,n-k)*np.log(bb(k,n,mu,s)) for k in range(n+1)) for n in range(2,60))
        spt=np.exp(minimize_scalar(nll,bounds=(-6,8),method='bounded').x)
        g_mm=sum(wn[n]*Delta(at,w,n,mu,smm) for n in range(60)); g_pt=sum(wn[n]*Delta(at,w,n,mu,spt) for n in range(60))
        ro=sum(wn[n]*Rorc(at,w,n) for n in range(60))
        g01=sum(wn[n]*Delta(at,w,n,mu,spt) for n in range(2))
        print("  nbar=%4.1f s_pseudo=%.3f oracle=%.4f gap_mm=%.5f gap_EB=%.5f gap_EB_n01=%.6f"%(nb,spt,ro,g_mm,g_pt,g01))

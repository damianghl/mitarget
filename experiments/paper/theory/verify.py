import numpy as np, sympy as sp
from math import comb
from scipy.stats import beta as Beta
rng=np.random.default_rng(0)
# 1. Beta third central moment formula
mu,s=sp.symbols('mu s',positive=True); a,b=mu*s,(1-mu)*s
m3c=2*(b-a)*a*b/((a+b)**3*(a+b+1)*(a+b+2)); v=mu*(1-mu)/(s+1)
print("beta k3 formula ok:", sp.simplify(m3c-2*v*(1-2*mu)/(s+2))==0)
# helpers on a discrete prior
def M(atoms,w,A,B): return (w*atoms**A*(1-atoms)**B).sum()
def Mb(mu,s,A,B): # beta mixed moment
    return np.exp(sp.N(0)*0) if False else float(sp.N(sp.beta(mu*s+A,(1-mu)*s+B)/sp.beta(mu*s,(1-mu)*s)))
def kl(p,q): return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
def Delta(atoms,w,n,mu,s):
    tot=0
    for k in range(n+1):
        D=M(atoms,w,k,n-k); po=M(atoms,w,k+1,n-k)/D; pb=(mu*s+k)/(s+n)
        tot+=comb(n,k)*D*kl(po,pb)
    return tot
def lead(atoms,w,n,mu,s):
    tot=0
    for k in range(n+1):
        A,Dd,C=M(atoms,w,k+1,n-k),M(atoms,w,k,n-k),M(atoms,w,k,n+1-k)
        dA=A-Mb(mu,s,k+1,n-k); dD=Dd-Mb(mu,s,k,n-k)
        tot+=comb(n,k)*(dA*Dd-A*dD)**2/(2*Dd*A*C)
    return tot
for atoms,w in [(np.array([.1,.5,.9]),np.array([.25,.5,.25])),(np.array([.2,.3,.85]),np.array([.3,.4,.3])),
                (np.array([.4,.45,.6,.7]),np.array([.2,.3,.3,.2]))]:
    m1=M(atoms,w,1,0); m2=M(atoms,w,2,0); var=m2-m1**2; s_=m1*(1-m1)/var-1
    k3=M(atoms,w,0,0)*0+(w*(atoms-m1)**3).sum(); k3b=2*var*(1-2*m1)/(s_+2)
    print("prior",atoms,"s*=%.3f k3=%.5f k3_beta=%.5f"%(s_,k3,k3b))
    for n in range(0,6): print("   n=%d Delta=%.3e  leading=%.3e"%(n,Delta(atoms,w,n,m1,s_),lead(atoms,w,n,m1,s_)))
# 2. n=2 marginal under t=1/(s+1): P2 = mu^2+mu(1-mu)t
t=1/(s+1); P2=sp.simplify(sp.beta(a+2,b)/sp.beta(a,b)); print("P2 linear in t:",sp.simplify(P2-(mu**2+mu*(1-mu)*t))==0)
# 3. Good-Turing identity w_n=(n+1)/(N+1) E F_{n+1}(N+1)
qx=rng.dirichlet(np.ones(50)*0.3); N=40
from scipy.stats import binom
for n in range(4):
    wn=(qx*binom.pmf(n,N,qx)).sum(); gt=(n+1)/(N+1)*binom.pmf(n+1,N+1,qx).sum()
    print("GT n=%d %.6f %.6f"%(n,wn,gt))
# 4. type-II ML consistency when all n_x<=2 (n=1 and n=2 levels), mu fixed = m1
atoms,w=np.array([.1,.5,.9]),np.array([.25,.5,.25]); m1=.5; var=(w*(atoms-.5)**2).sum(); t_true=var/(m1*(1-m1))
from scipy.optimize import minimize_scalar
for K in [200,2000,20000,200000]:
    q=rng.choice(atoms,size=K,p=w); k=rng.binomial(2,q); f=np.bincount(k,minlength=3)
    P=lambda t:np.array([(1-m1)**2+m1*(1-m1)*t,2*m1*(1-m1)*(1-t),m1**2+m1*(1-m1)*t])
    r=minimize_scalar(lambda t:-(f*np.log(P(t))).sum(),bounds=(1e-9,1-1e-9),method='bounded')
    print("K=%6d t_hat=%.4f t_true=%.4f"%(K,r.x,t_true))

################################## ---    --- ########################################################################
#####--- Czesc eksperymentalna pracy magisterskiej pod tytulem                                          # 
#####--- "Hamiltonowskie Sieci Neuronowe"                                                               #
#####--- autor: Agata Machula                                                                           #
#####--- opiekun: dr Magdalena Chmara                                                                   #
#####--- marzec 2023                                                 

# Rozwazac bedziemy hamiltonowska siec neuronowa
# przyk³ad rozwi¹zanego rr - rownanie wahadla.
# p'=-k*sin(q) q'=2p

#install.packages("torch")
library(torch)


# DEFINIUJEMY FUNKCJE

#funkcja autograd_grad liczy i zwraca sume gradientow
dfx <-function(x,f){
  return(autograd_grad(f,x,grad_outputs = torch_ones(x$size(),dtype=torch_float()),create_graph = TRUE)[[1]])
}

#rozwiazanie parametryczne: nn - siec neuronowa
q_param <-function(t,nn,X0){
  t0=X0[1]
  q0=X0[2]
  p0=X0[3]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  f=1-exp(-t)
  q_hat = q0 + f*N1
  p_hat = p0 + f*N2
  return(q_hat)
}


p_param <-function(t,nn,X0){
  t0=X0[1]
  q0=X0[2]
  p0=X0[3]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  f=1-exp(-t)
  q_hat = q0 + f*N1
  p_hat = p0 + f*N2
  return(p_hat)
}


# funkcja bledu 
funkcja_bledu <- function(t,q,p){
  qd = dfx(t,q)
  pd = dfx(t,p)
  k=2.4
  fq = qd - 2*p
  fp = pd + k*sin(q)
  L = mean((fq)^2 + fp^2)
  return(L)
}

#definiujemy funkcje aktywacji
sinus <- nn_module(
  forward = function(input) return(torch_sin(input))
)

#definiujemy hamiltonian
#l=2.083
#m=0.3
#g=10
hamiltonian <- function(q,p){
  k = 2.4
  K= p^2
  V = k*(1-cos(q))
  ham = K + V
  return(ham)
}

pertPunktow <- function(siatka, t0, tf, sigma =0.5){
  delta_t = siatka[2]-siatka[1]
  szum = delta_t*torch_rand_like(siatka)*sigma
  t = siatka + szum
  t[3]=torch_ones(1,1)*(-1)
  t[t<t0]=t0 - t[t<t0] #zamiana punktów z ujemnych na dodatnie, bo t0=0
  t[t>tf]=2*tf - t[t>tf]
  t$requires_grad = FALSE
  return(t)
}


#funkcja energii
Energia <- function(q,p){
  k=2.4
  q = torch_reshape(q,list(-1,1))
  p = torch_reshape(p,list(-1,1))
  E = k*(1-cos(q)) + p^2
  E = torch_reshape(E,list(-1,1))
  return(E)
}

#######################--- BUDOWA SIECI ---####################################
#dwuwarstwowa siec

neural_network <- nn_module(
  classname = "neural_ode",
  #definiowanie warstw i neuronow
  initialize = function(W_ukryta){
    
    self$actF = sinus()
    
    #warstwy
    self$Lin_1 = nn_linear(1,W_ukryta)
    self$Lin_2 = nn_linear(W_ukryta,W_ukryta)
    self$Lin_out = nn_linear(W_ukryta,2)
  },
  
  #funkcja forward wywolywana jest, gdy do sieci 'wrzucimy' dane wejsciowe x
  forward = function(t){
    t %>%
      #warstwa 1
      self$Lin_1() %>%
      self$actF() %>%
      #warstwa 2
      self$Lin_2() %>%
      self$actF() %>%
      #wyjscie
      self$Lin_out()
    
  }
)

#TEST SIECI
#t = torch_linspace(0,1,20, requires_grad = T)
#t=torch_reshape(t,list(-1,1))
#model=neural_network(10)
#z=model(t)[,1]$unsqueeze(dim=2)
#z



#########################--- TRENOWANIE SIECI ---#####################################
# TRENING

#funkcja uczaca
run_odeNet <- function(X0,tf,neurons,epochs,n_train,lr){
  NN = neural_network(neurons) # model z losowymi wagami
  optimizer = optim_adam(NN$parameters,lr=lr, betas = c(0.999,0.9999)) # stosujemy optymalizacje - Adam algorithm
  Loss_history = {} # wektor kolejnych bledow
  t0=X0[1]
  q0=X0[2]
  px0=X0[3]
  
  #liczymy hamiltonian, ktory ma pozostac staly
  ham0 = hamiltonian(q0,px0)
  
  siatka = torch_linspace(t0,tf,n_train)
  siatka = torch_reshape(siatka,list(-1,1))
  #petla uczaca
  for (e in 1:epochs) {
    #dane wejsciowe
    #NN$parameters
    t = pertPunktow(siatka,t0,tf,0.03*tf)
    t$requires_grad = TRUE  # potrzebne by liczyc  gradient po x
    NN$train()  
    
    #wyliczamy predykcje
    q = q_param(t,NN,X0)
    p = p_param(t,NN,X0)
    
    #liczymy blad
    Ltot = funkcja_bledu(t,q,p)
    # uwzgledniamy stalosc energii
    ham = hamiltonian(q,p)
    Ltot = Ltot + mean((ham-ham0)^2)
    loss=Ltot$item()
    
    # liczymy gradient funkcji bledu
    autograd_backward(tensors = Ltot,retain_graph = FALSE)
    # robimy krok przeciwny do kierunku wskazanego prrzez  gradient
    optimizer$step()
    #NN$parameters
    # zerujemy gradient opptymalizacji
    optimizer$zero_grad()
    #zapisujemy kolejne bledy
    Loss_history[e]=loss
    
  }
  print(Loss_history[seq(1,epochs,10)])
  return(NN)
}

################--- ROZWIAZANIE ZAGADNIENIA POCZATKOWEGO ---########################

#sprawdzenie dzialania
#N=200
t0=0
t_max=20
#dt=t_max/N
q0=0
p0=0.75
X0=c(t0,q0,p0)
n_train = 200 # dlugosc podzialu w dyskretyzacji dziedziny - liczba el. uczacych
neurons = 50 # liczba neuronow w warstwie ukrytej
lr = 8e-3 # wspolczynnik uczenia
epochs = 10000 # liczba iteracji
tf=t_max
# tworzymy i trenujemy model
model <- run_odeNet(X0,t_max,neurons,epochs,n_train,lr)

# zapisanie modelu
model$eval()
torch_save(model,"HNNwahadlo.rt")

#zaladowanie modelu
model = torch_load("HNNwahadlo.rt")
model2 = torch_load("klasNNwahadlo.rt")

#test

n_test=10*n_train
t = torch_linspace(t0,t_max,n_test) #10
t=torch_reshape(t,list(-1,1))
t$requires_grad=TRUE #nowa l


# rozwiazanie numeryczne
q = q_param(t,model,X0)
p = p_param(t,model,X0)

q_kl = q_param(t,model2,X0)
p_kl = p_param(t,model2,X0)


#rozwiazanie dokladne przy pomocy pakietu
library(deSolve)

wahadlo <- function(t, x, parms) {
  with(as.list(c(parms, x)), {
    dP <- -k*sin(Q)
    dQ <- 2*P
    res <- c(dQ, dP)
    list(res)
  })
}

parms <- c(k=2.4)

times <- seq(t0, t_max, length.out=n_test)

xstart <- c(Q = q0, P = p0)
#rozwizanie
out <- lsoda(xstart, times, wahadlo, parms)
t_p=out[,1]
q_p=out[,2]
p_p=out[,3]



#wyliczenia bledow 
dq_p=q_p-q_p
dp_p=p_p-p_p

dq=q_p - q[,1]
dp=p_p - p[,1]

dq_kl=q_p - q_kl[,1]
dp_kl=p_p - p_kl[,1]


###################---- ANALIZA MODELI ----##################

# porownanie na wykresie
par(mfrow=c(1,2))
plot(t_p,q_p, type = 'l',lty=1,lwd=3,ylim=c(-1.5,2.5),xlab = 't',ylab = 'q',main = "K¹t wychylenia w czasie")
lines(t,q, type = 'l',lty=2,lwd=3,col='red')
lines(t,q_kl, type = 'l',lty=2,lwd=3,col='blue')
legend('topleft',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)


plot(t_p,p_p, type = 'l',lty=1,lwd=3,ylim=c(-2,2),xlab = 't',ylab = 'p',main = "Wykres pêdu od czasu")
lines(t,p, type = 'l',lty=3,lwd=3,col='red')
lines(t,p_kl, type = 'l',lty=2,lwd=3,col='blue')
legend('topleft',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)


plot(q_p,p_p, type = 'l',lty=1,lwd=3,xlab = 'q',ylab = 'p',main='Portret fazowy')
lines(q,p, type = 'l',lty=2,lwd=3,col='red')
lines(q_kl,p_kl, type = 'l',lty=2,lwd=3,col='blue')
legend('center',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)



plot(t,Energia(q,p),type ='l',lwd=3,lty=1,col="red",ylim=c(0.45,0.57),main='Energia w czasie',ylab = 'Energia')
lines(t_p,Energia(q_p,p_p), type ='l',lwd=3)
lines(t,Energia(q_kl,p_kl), type ='l',lwd=3,lty=1,col='blue')
legend('bottomleft',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

###################---- ANALIZA BLEDOW ----##################
par(mfrow=c(1,1))

plot(t_p,dq_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlab = 't',ylab = 'delta q',main = 'B³¹d w czasie')
lines(t,dq,type='l',lty=1,lwd=2,col='red')
lines(t,dq_kl,type='l',lty=1,lwd=2,col='blue')
legend('topright',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

plot(t_p,dp_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlab = 't',ylab = 'delta p',main = 'B³¹d w czasie')
lines(t,dp,type='l',lty=1,lwd=2,col='red')
lines(t,dp_kl,type='l',lty=1,lwd=2,col='blue')
legend('topright',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

plot(dq_p,dp_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlim=c(-0.1,0.1),xlab = 'delta q',ylab = 'delta p',main = 'B³¹d')
lines(dq,dp,type='l',lty=1,lwd=3,col='red')
lines(dq_kl,dp_kl,type='l',lty=1,lwd=2,col='blue')
legend('topleft',legend = c('prawdziwe', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)



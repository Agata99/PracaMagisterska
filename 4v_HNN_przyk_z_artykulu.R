################################## ---    --- ########################################################################
#####--- Czesc eksperymentalna pracy magisterskiej pod tytulem                                          # 
#####--- "Hamiltonowskie Sieci Neuronowe"                                                               #
#####--- autor: Agata Machula                                                                           #
#####--- opiekun: dr Magdalena Chmara                                                                   #
#####--- luty 2023                                                                                      #
#####################################################################################################################
# Napisany kod jest czêœci¹ pracy magisterskiej i zosta³ napisany na podstawie                                      #
# opisu sieci zawartego w artykule "Hamiltonian Neural Network for solving equations of motion"                     #
# dostêpnego pod linkiem                                                                                            #
# https://arxiv.org/abs/2001.11107                                                                                  #
# (dostêp 19.02.2023)                                                                                               #
# oraz na podstawie kodu pochodzacego z tegoz https://github.com/mariosmat/hamiltonianNNetODEs (dostep 19.02.2023)  #

#####################################################################################################################
# Rozwazac bedziemy hamiltonowska siec neuronowa, na koniec porownammy z 
# klasyczna siecia rozwazana w pliku .r (wystarczy wczytac model "klasNNprz1.rt")
# przyk³ad rozwi¹zanego rr pochodzi z artyku³u Hamiltonian NN for solving  equations of motion: 
# dx/dt =p , dp//dt = -(x+x^3).

#install.packages("torch")
library(torch)


# DEFINIUJEMY FUNKCJE

#funkcja autograd_grad liczy i zwraca sume gradientow
dfx <-function(x,f){
  return(autograd_grad(f,x,grad_outputs = torch_ones(x$size(),dtype=torch_float()),create_graph = TRUE)[[1]])
}

#rozwiazanie parametryczne: nn - siec neuronowa
x_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  px0=X0[3]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  px_hat = px0 + f*N2
  return(x_hat)
}


px_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  px0=X0[3]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  px_hat = px0 + f*N2
  return(px_hat)
}


# funkcja bledu 
funkcja_bledu <- function(t,x,px){
  xd = dfx(t,x)
  pxd = dfx(t,px)
  fx = xd - px
  fpx = pxd + x + x^3
  L = mean((fx)^2 + fpx^2)
  return(L)
}

#definiujemy funkcje aktywacji
sinus <- nn_module(
  forward = function(input) return(torch_sin(input))
)

#definiujemy hamiltonian

hamiltonian <- function(x,px){
  K = 0.5*px^2
  V = 0.5*x^2 + (x^4)/4
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
Energia <- function(x,px){
  dl_x = length(x)
  x = torch_reshape(x,list(-1,1))
  px = torch_reshape(px,list(-1,1))
  E = 0.5*px^2 + 0.5*x^2 + 0.25*x^4
  E = torch_reshape(E,list(-1,1))
  return(E)
}
#######################--- BUDOWA SIECI ---####################################
#trzywarstwowa siec

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
  x0=X0[2]
  px0=X0[3]
  
  #liczymy hamiltonian, ktory ma pozostac staly
  ham0 = hamiltonian(x0,px0)
  
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
    x = x_param(t,NN,X0)
    px = px_param(t,NN,X0)
    
    #liczymy blad
    Ltot = funkcja_bledu(t,x,px)
    # uwzgledniamy stalosc energii
    ham = hamiltonian(x,px)
    Ltot = Ltot + mean((ham-ham0)^2)
    loss=Ltot$item()
    
    # liczymy gradient funkcji bledu
    autograd_backward(tensors = Ltot,retain_graph = FALSE)
    # robimy krok przeciwny do kierunku wskazanego przez  'gradient'
    optimizer$step()
    #NN$parameters
    # zerujemy gradient opptymalizacji
    optimizer$zero_grad()
    #zapisujemy kolejne bledy
    Loss_history[e]=loss
    
  }
  #print(Loss_history[seq(1,epochs,10)])
  plot(log(Loss_history),type = 'l',xlab='iteracja',ylab='b³¹d',main='Funkcja b³êdu w skali log')
  return(NN)
}

################--- ROZWIAZANIE ZAGADNIENIA POCZATKOWEGO ---########################

#sprawdzenie dzialania
#N=200
t0=0
t_max=4*pi
#dt=t_max/N
x0=1.3
px0=1
X0=c(t0,x0,px0)
n_train = 200 # dlugosc podzialu w dyskretyzacji dziedziny - liczba el. uczacych
neurons = 50 # liczba neuronow w warstwie ukrytej
lr = 8e-3 # wspolczynnik uczenia
epochs = 10000 # liczba iteracji
tf=t_max
# tworzymy i trenujemy model
model <- run_odeNet(X0,t_max,neurons,epochs,n_train,lr)

# zapisanie modelu
model$eval()
torch_save(model,"HNNprz1.rt")

#zaladowanie modelu
model = torch_load("HNNprz1.rt")
model2 = torch_load("klasNNprz1.rt")


#test

n_test=10*n_train
t = torch_linspace(t0,t_max,n_test) #10
t=torch_reshape(t,list(-1,1))
t$requires_grad=TRUE #nowa l

# rozwiazanie numeryczne
# HNN
x = x_param(t,model,X0)
px = px_param(t,model,X0)
#  klas NN
x_kl = x_param(t,model2,X0)
px_kl = px_param(t,model2,X0)



#rozwiazanie dokladne przy pomocy pakietu
library(deSolve)

oscylator <- function(t, x, parms) {
  with(as.list(c(parms, x)), {
    dX <- P
    dP <- -(X+X^3)
    res <- c(dX, dP)
    list(res)
  })
}

parms <- c()

times <- seq(t0, t_max, length.out=n_test)

xstart <- c(X = x0, P = px0)
#rozwizanie
out <- lsoda(xstart, times, oscylator, parms)
t_p=out[,1]
x_p=out[,2]
px_p=out[,3]


#wyliczenia bledow 
dx_p=x_p-x_p
dpx_p=px_p-px_p

dx=x_p - x[,1]
dpx=px_p - px[,1]

#max(dx)
#min(dx)

dx_kl=x_p - x_kl[,1]
dpx_kl=px_p - px_kl[,1]

#max(dpx)
#min(dpx)

###################---- ANALIZA MODELI ----##################

# porownanie na wykresie
par(mfrow=c(1,2))
plot(t_p,x_p, type = 'l',lty=1,lwd=4,ylim=c(-1.5,3.2),xlab = 't',ylab = 'x',main = "Wykres k¹ta wychylenia w czasie")
lines(t,x, type = 'l',lty=2,lwd=4,col='red')
lines(t,x_kl, type = 'l',lty=2,lwd=4,col='blue')
legend('topleft',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)


plot(t_p,px_p, type = 'l',lty=1,lwd=4,ylim=c(-2,5),xlab = 't',ylab = 'p',main = "Wykres pêdu od czasu")
lines(t,px, type = 'l',lty=2,lwd=4,col='red')
lines(t,px_kl, type = 'l',lty=2,lwd=4,col='blue')
legend('topleft',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)


plot(x_p,px_p, type = 'l',lty=1,lwd=4,xlab = 'x',ylab = 'p',main='Portret fazowy')
lines(x,px, type = 'l',lty=2,lwd=4,col='red')
lines(x_kl,px_kl, type = 'l',lty=2,lwd=4,col='blue')
legend('center',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,2,2),lwd=3)



plot(t,Energia(x,px),type ='l',lwd=4,lty=1,col="red",ylim = c(1.7,2.2),main='Energia w czasie',ylab = 'Energia')
lines(t_p,Energia(x_p,px_p), type ='l',lwd=4)
lines(t,Energia(x_kl,px_kl), type ='l',lwd=4,lty=1,col='blue')
legend('bottomleft',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

###################---- ANALIZA BLEDOW ----##################
par(mfrow=c(1,1))

plot(t_p,dx_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlab = 't',ylab = 'delta x',main = 'B³¹d w czasie')
lines(t,dx,type='l',lty=1,lwd=2,col='red')
lines(t,dx_kl,type='l',lty=1,lwd=2,col='blue')
legend('topright',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

plot(t_p,dpx_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlab = 't',ylab = 'delta p',main = 'B³¹d w czasie')
lines(t,dpx,type='l',lty=1,lwd=2,col='red')
lines(t,dpx_kl,type='l',lty=1,lwd=2,col='blue')
legend('topright',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

plot(dx_p,dpx_p,type='l',lty=1,lwd=3,col='black',ylim=c(-0.1,0.1),xlim=c(-0.1,0.1),xlab = 'delta x',ylab = 'delta p',main = 'B³¹d')
lines(dx,dpx,type='l',lty=1,lwd=3,col='red')
lines(dx_kl,dpx_kl,type='l',lty=1,lwd=2,col='blue')
legend('topleft',legend = c('numeryczne', 'HSN', 'SN'),col=c('black','red','blue'),lty=c(1,1,1),lwd=3)

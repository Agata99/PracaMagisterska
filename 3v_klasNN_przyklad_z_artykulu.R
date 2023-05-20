################################## ---    --- ########################################################################
####--- Czesc eksperymentalna pracy magisterskiej pod tytulem                                          # 
####--- "Hamiltonowskie Sieci Neuronowe"                                                               #
####--- autor: Agata Machula                                                                           #
####--- opiekun: dr Magdalena Chmara                                                                   #
####--- luty 2023                                                                                      #
#####################################################################################################################
# Napisany kod jest czêœci¹ pracy magisterskiej i zosta³ napisany na podstawie                                      #
# opisu sieci zawartego w artykule "Hamiltonian Neural Network for solving equations of motion"                     #
# dostêpnego pod linkiem                                                                                            #
# https://arxiv.org/abs/2001.11107                                                                                  #
# (dostêp 19.02.2023)                                                                                               #
# oraz na podstawie kodu pochodzacego z tegoz https://github.com/mariosmat/hamiltonianNNetODEs (dostep 19.02.2023)  #

#####################################################################################################################

# model klasycznej sieci neuronowej do rozwi¹zania prostego równanie ró¿niczkowego 
# przyk³ad rozwi¹zanego rr pochodzi z artyku³u Hamiltonian NN for solving  equations of motion: 
# dx/dt =p , dp//dt = -(x+x^3).
#model wytrenowany na podobnych parametrach, co w artykule, nie zastosowano natomiast 
#hamiltonianu w modelu

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


#######################--- BUDOWA SIECI ---####################################
# siec neuronowa - klasyczna wersja
#trzywarstwowa klasyczna siec

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


#########################--- TRENOWANIE SIECI ---######################################
#funkcja uczaca
run_odeNet <- function(X0,tf,neurons,epochs,n_train,lr){
  NN = neural_network(neurons) # model z losowymi wagami
  optimizer = optim_adam(NN$parameters,lr=lr) # stosujemy optymalizacje - stochastyczny spadek gradientu
  Loss_history = {} # wektor kolejnych bledow
  t0=X0[1]
  x0=X0[2]
  px0=X0[3]
  #petla uczaca
  for (e in 1:epochs) {
    #dane wejsciowe
    #NN$parameters
    t = torch_linspace(t0,tf,n_train)
    t=torch_reshape(t,list(-1,1))
    t$requires_grad = TRUE  # potrzebne by liczyc  gradient po x
    NN$train()  
    
    #wyliczamy predykcje
    x = x_param(t,NN,X0)
    px = px_param(t,NN,X0)
    
    #liczymy blad
    Ltot = funkcja_bledu(t,x,px)
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
torch_save(model,"klasNNprz1.rt")

#zaladowanie modelu
model = torch_load("klasNNprz1.rt")


################--- ROZWIAZANIE ZAGADNIENIA POCZATKOWEGO ---########################

t = torch_linspace(t0,t_max,n_train)
t=torch_reshape(t,list(-1,1))

# rozwiazanie numeryczne
x = x_param(t,model,X0)
px = px_param(t,model,X0)


#rozwiazanie 'dokladne' przy pomocy pakietu
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

times <- seq(t0, t_max, length.out=n_train)

xstart <- c(X = x0, P = px0)
#rozwizanie
out <- lsoda(xstart, times, oscylator, parms)


# porownanie na wykresie
plot(out[,1],out[,2], type = 'l',lwd=3,xlab = 't',ylab = 'x',main = "K¹t wychylenia w czasie")
lines(t,x, type = 'l',col='blue',lwd=3,lty=2)
legend('topleft',legend = c('prawdziwe', 'SN'),col=c('black','blue'),lty=c(1,2),lwd=3)

plot(out[,1],out[,3], type = 'l',lwd=3,xlab = 't',ylab = 'px',main = "Wykres pedu od czasu")
lines(t,px, type = 'l',col='blue',lwd=3,lty=2)
legend('topleft',legend = c('prawdziwe', 'SN'),col=c('black','blue'),lty=c(1,2),lwd=3)

plot(out[,2],out[,3], type = 'l',lwd=3,xlab = 'x',ylab = 'px',main = "Portret fazowy")
lines(x,px, type = 'l',col='blue',lwd=3,lty=2)
legend('center',legend = c('prawdziwe', 'SN'),col=c('black','blue'),lty=c(1,2),lwd=3)



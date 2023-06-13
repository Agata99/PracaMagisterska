################################## ---    --- ########################################################################
#####--- Czesc eksperymentalna pracy magisterskiej pod tytulem                                          # 
#####--- "Hamiltonowskie Sieci Neuronowe"                                                               #
#####--- autor: Agata Machula                                                                           #
#####--- opiekun: dr Magdalena Chmara                                                                   #
#####--- luty 2023                             

# model klasycznej sieci neuronowej do rozwi¹zania prostego równanie ró¿niczkowego 
# Rozwazac bedziemy klasyczna siec neuronowa
# przyk³ad rozwi¹zanego rr - rownanie wahadla
# p'=-k*sin(q) q'=2p

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


#######################--- BUDOWA SIECI ---####################################
# siec neuronowa - klasyczna wersja
#dwuwarstwowa klasyczna siec

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
torch_save(model,"klasNNwahadlo.rt")

#zaladowanie modelu
model = torch_load("klasNNwahadlo.rt")

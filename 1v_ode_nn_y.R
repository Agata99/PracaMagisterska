# Agata Machula 
# praca magisterska - probny model NN
# 10.02.2022

# model klasycznej sieci neuronowej do rozwi¹zania prostego
# równanie ró¿niczkowego zwyczajnego


#install.packages("torch")
library(torch)

# rozwiazujemy rownanie dy/dx=-gamma*y

#trzywarstwowa siec z sigmoidalna funkcja aktywacji
neural_network <- nn_module(
  classname = "neural_ode",
  #definiowanie warstw i neuronow
  initialize = function(W_ukryta){
    
    self$actF = nn_sigmoid()
    
    #warstwy
    self$Lin_1 = nn_linear(1,W_ukryta)
    self$Lin_2 = nn_linear(W_ukryta,W_ukryta)
    self$Lin_out = nn_linear(W_ukryta,1)
  },
  
  #funkcja forward wywolywana jest, gdy do sieci 'wrzucimy' dane wejsciowe x
  forward = function(x){
    x %>%
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
#x = torch_linspace(0,1,20, requires_grad = T)
#x=torch_reshape(x,list(-1,1))
#model=neural_network(10)
#model(x)

# DEFINIUJEMY FUNKCJE

#funkcja autograd_grad liczy i zwraca sume gradientow
dfx <-function(x,f){
  return(autograd_grad(f,x,grad_outputs = torch_ones(x$size(),dtype=torch_float()),create_graph = TRUE)[[1]])
}

#rozwiazanie parametryczne: nn - siec neuronowa
g_param <-function(x,nn,g0=10){
  N=nn(x)
  y=g0 + x*N
  return(y)
}

#prawa strona rownania
g <- function(x,g_p,gamma=2){
  y=-gamma*g_p
  return(y)
}

# funkcja bledu = mean((  d(g_param(x,nn))/dx - (-gamma*g_param(x,nn))  )^2)
funkcja_bledu <- function(x,g_p,nn){
  nn_d = dfx(x,nn(x))
  g_d = dfx(x,g_p)

  Lx = mean((g_d-g(x,g_p))^2)
  return(Lx)
}

# rozwiazanie dokladne
g_analytic <- function(x,gamma=2,g0=10){
  y=g0*exp(-gamma*x)
  return(y)
}

# TRENING

 
x0=0 
n_train = 100 # dlugosc podzialu w dyskretyzacji dziedziny - liczba el. uczacych
neurons = 30 # liczba neuronow w warstwie ukrytej
lr = 8e-3 # wspolczynnik uczenia
epochs = 1000 # liczba iteracji

#funkcja uczaca
run_odeNet <- function(x0,neurons,epochs,n_train,lr){
  NN = neural_network(neurons) # model z losowymi wagami
  optimizer = optim_sgd(NN$parameters,lr=lr) # stosujemy optymalizacje - stochastyczny spadek gradientu
  Loss_history = {} # wektor kolejnych bledow
  
  #petla uczaca
  for (xx in 1:epochs) {
    #dane wejsciowe
    x = torch_linspace(x0,1,n_train)
    x=torch_reshape(x,list(-1,1))
    x$requires_grad = TRUE  # potrzebne by liczyc  gradient po x
    NN$train()
    #wyliczamy predykcje
    y_p = g_param(x,NN)
    #liczymy blad
    Ltot = funkcja_bledu(x,y_p,NN)
    loss=Ltot$item()
    # liczymy gradient funkcji bledu
    autograd_backward(tensors = Ltot,retain_graph = FALSE)
    # robimy krok przeciwny do kierunku wsskazanego prrzez  gradient
    optimizer$step()
    # zerujemy gradient opptymalizacji
    optimizer$zero_grad()
    #zapisujemy kolejne bledy
    Loss_history[xx]=loss
    
  }
  print(Loss_history)
  return(NN)
}

# tworzymy i trenujemy model
model <- run_odeNet(x0,neurons,epochs,n_train,lr)

x = torch_linspace(x0,1,n_train)
x=torch_reshape(x,list(-1,1))

# rozwiazanie numeryczne
y = g_param(x,model)
#rozwiazanie dokladne
y_dokl = g_analytic(x)

# porownanie na wykresie
plot(x,y_dokl,type = 'l', main = "Porownanie rozwiazania dokladnego i uzyskanego przez NN",ylab="y(x)")
lines(x,y, type = 'l',col='red')
legend(x=0.4,y=10,legend = c("Rozw. dokladne", "NN"),lty=1,col=c("black","red"))




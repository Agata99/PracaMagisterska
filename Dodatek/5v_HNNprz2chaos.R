# Agata Machula 
# praca magisterska - model HNN
# 01.03.2022

# przyk³ad rozwi¹zanego rr pochodzi z artyku³u Hamiltonian NN for solving  equations of motion: 
# dx/dt =p , dp//dt = 


#install.packages("torch")
library(torch)


#definiujemy funkcje aktywacji
sinus <- nn_module(
  forward = function(input) return(torch_sin(input))
)

#definiujemy hamiltonian

hamiltonian <- function(x,y,px,py){
  K = 0.5*(px^2 + py^2)
  V = 0.5*(x^2+y^2) + (x^2*y-y^3/3)
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

#trzywarstwowa siec

neural_network <- nn_module(
  classname = "neural_ode",
  #definiowanie warstw i neuronow
  initialize = function(W_ukryta){
    
    self$actF = sinus()
    
    #warstwy
    self$Lin_1 = nn_linear(1,W_ukryta)
    self$Lin_2 = nn_linear(W_ukryta,W_ukryta)
    self$Lin_out = nn_linear(W_ukryta,4)
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


# DEFINIUJEMY FUNKCJE

#funkcja autograd_grad liczy i zwraca sume gradientow
dfx <-function(x,f){
  return(autograd_grad(f,x,grad_outputs = torch_ones(x$size(),dtype=torch_float()),create_graph = TRUE)[[1]])
}

#rozwiazanie parametryczne: nn - siec neuronowa
x_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  y0=X0[3]
  px0=X0[4]
  py0=X0[5]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  N3=nn(t)[,3]$unsqueeze(dim=2)
  N4=nn(t)[,4]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  y_hat = y0 + f*N2
  px_hat = px0 + f*N3
  py_hat = py0 + f*N4
  return(x_hat)
}

y_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  y0=X0[3]
  px0=X0[4]
  py0=X0[5]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  N3=nn(t)[,3]$unsqueeze(dim=2)
  N4=nn(t)[,4]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  y_hat = y0 + f*N2
  px_hat = px0 + f*N3
  py_hat = py0 + f*N4
  return(y_hat)
}

px_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  y0=X0[3]
  px0=X0[4]
  py0=X0[5]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  N3=nn(t)[,3]$unsqueeze(dim=2)
  N4=nn(t)[,4]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  y_hat = y0 + f*N2
  px_hat = px0 + f*N3
  py_hat = py0 + f*N4
  return(px_hat)
}

py_param <-function(t,nn,X0){
  t0=X0[1]
  x0=X0[2]
  y0=X0[3]
  px0=X0[4]
  py0=X0[5]
  N1=nn(t)[,1]$unsqueeze(dim=2)
  N2=nn(t)[,2]$unsqueeze(dim=2)
  N3=nn(t)[,3]$unsqueeze(dim=2)
  N4=nn(t)[,4]$unsqueeze(dim=2)
  f=1-exp(-t)
  x_hat = x0 + f*N1
  y_hat = y0 + f*N2
  px_hat = px0 + f*N3
  py_hat = py0 + f*N4
  return(py_hat)
}

# funkcja bledu 
funkcja_bledu <- function(t,x,y,px,py){
  xd = dfx(t,x)
  yd = dfx(t,y)
  pxd = dfx(t,px)
  pyd = dfx(t,py)
  
  Lx=mean((xd-px)^2)
  Ly=mean((yd-py)^2)
  Lpx=mean((pxd+x+2*x*y)^2)
  Lpy=mean((pyd+y+x^2-y^2)^2)
  L = Lx+Ly+Lpx+Lpy
  return(L)
}


# TRENING

#funkcja uczaca
run_odeNet <- function(X0,tf,neurons,epochs,n_train,lr){
  NN = neural_network(neurons) # model z losowymi wagami
  optimizer = optim_adam(NN$parameters,lr=lr, betas = c(0.999,0.9999)) # stosujemy optymalizacje - Adam algorithm
  Loss_history = {} # wektor kolejnych bledow
  t0=X0[1]
  x0=X0[2]
  y0=X0[3]
  px0=X0[4]
  py0=X0[5]
  #liczymy hamiltonian, ktory ma pozostac staly
  ham0 = hamiltonian(x0,y0,px0,py0)
  
  siatka = torch_linspace(t0,tf,n_train)
  siatka = torch_reshape(siatka,list(-1,1))
  #petla uczaca
  for (e in 1:epochs) {
    #dane wejsciowe
    #NN$parameters
    t = pertPunktow(siatka,t0,tf,0.3*tf)
    t$requires_grad = TRUE  # potrzebne by liczyc  gradient po x
    NN$train()  
    
    #wyliczamy predykcje
    x = x_param(t,NN,X0)
    y = y_param(t,NN,X0)
    px = px_param(t,NN,X0)
    py = py_param(t,NN,X0)
    
    #liczymy blad
    Ltot = funkcja_bledu(t,x,y,px,py)
    # uwzgledniamy stalosc energii
    ham = hamiltonian(x,y,px,py)
    Ltot = Ltot + 0.5*mean((ham-ham0)^2)
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
t_max=6*pi
#dt=t_max/N
x0=0.3
y0=-0.3
px0=0.3
py0=0.15
X0=c(t0,x0,y0,px0,py0)
n_train = 500 # dlugosc podzialu w dyskretyzacji dziedziny - liczba el. uczacych
neurons = 50 # liczba neuronow w warstwie ukrytej
lr = 8e-3 # wspolczynnik uczenia
epochs = 10000 # liczba iteracji
tf=t_max
# tworzymy i trenujemy model
model <- run_odeNet(X0,t_max,neurons,epochs,n_train,lr)

# zapisanie modelu
model$eval()
torch_save(model,"HNNprz2.rt")

#zaladowanie modelu
model = torch_load("HNNprz2.rt")

t = torch_linspace(t0,t_max,n_train)
t=torch_reshape(t,list(-1,1))

# rozwiazanie numeryczne
x = x_param(t,model,X0)
y = y_param(t,model,X0)
px = px_param(t,model,X0)
py = py_param(t,model,X0)


#rozwiazanie dokladne przy pomocy pakietu
library(deSolve)

oscylator <- function(t, x, parms) {
  with(as.list(c(parms, x)), {
    dX <- Px
    dPx <- -(X+2*X*Y)
    dY <- Py
    dPy<- -(Y+X^2 - Y^2)
    res <- c(dX,dY, dPx,dPy)
    list(res)
  })
}

parms <- c()

times <- seq(t0, t_max, length.out=n_train)

xstart <- c(X = x0,Y=y0, Px = px0,Py=py0)
#rozwizanie
out <- lsoda(xstart, times, oscylator, parms)
t_p=out[,1]
x_p=out[,2]
y_p=out[,3]
Px_p=out[,4]
Py_p=out[,5]

# porownanie na wykresie
plot(t_p,x_p, type = 'l',xlab = 't',ylab = 'x')
lines(t,x, type = 'l',col='red')

plot(t_p,y_p, type = 'l',xlab = 't',ylab = 'y')
lines(t,y, type = 'l',col='red')

plot(t_p,Px_p, type = 'l',xlab = 't',ylab = 'px')
lines(t,px, type = 'l',col='red')

plot(x_p,y_p, type = 'l',xlab = 'x',ylab = 'y')
lines(x,y, type = 'l',col='red')

#funkcja energii
Energia <- function(x,y,px,py){
  x = torch_reshape(x,list(-1,1))
  y = torch_reshape(y,list(-1,1))
  px = torch_reshape(px,list(-1,1))
  py = torch_reshape(py,list(-1,1))
  E = 0.5*(px^2+py^2) + 0.5*(x^2+y^2) + (x^2*y-y^3/3)
  E = torch_reshape(E,list(-1,1))
  return(E)
}
plot(t,Energia(x,y,px,py),type = 'l',col='red')
lines(t_p,Energia(x_p,y_p,Px_p,Py_p), type ='l')

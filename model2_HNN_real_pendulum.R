################################## ---    --- ########################################################################
#################--- Czesc eksperymentalna pracy magisterskiej pod tytulem                                          # 
#################--- "Hamiltonowskie Sieci Neuronowe"                                                               #
#################--- autor: Agata Machula                                                                           #
#################--- opiekun: dr Magdalena Chmara                                                                   #
#################--- kwiecien 2023      ostatnia modyfikacja maj 2023                                                                            #
#####################################################################################################################
# Napisany kod jest czêœci¹ pracy magisterskiej i zosta³ napisany na podstawie                                      #
# opisu sieci zawartego w artykule Greydanus et. all "Hamiltonian Neural Network"                                   #
# dostêpnego pod linkiem                                                                                            #
# https://proceedings.neurips.cc/paper_files/paper/2019/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf             #
# (dostêp 4.04.2023)                                                                                                #
# oraz na podstawie kodu pochodzacego z tegoz https://github.com/greydanus/hamiltonian-nn (dostep 4.04.2023)        #

#####################################################################################################################
#install.packages('torch')

library(torch)
library(deSolve)

# zdefiniowanie funkcji pomocniczych
# funkcja rozwiazujaca  zagadnienie poczatkowe (jako argumenty przyjmuje: model 
#wytrenowanej sieci, zakres czasu calkowania i warunek poczatkowy)

integrate_model<-function(model, t_eval, y0,...){
  
  fun <- function(t, np_x,parm){
    with(as.list(c(np_x, parm)),{
      x=torch_tensor(np_x, requires_grad=TRUE,dtype = torch_float32())$unsqueeze(1)
      dx=model$pochodna(x)
      dx=as.array(torch_reshape(dx,-1))#$reshape(-1)
      list(dx)
    })}
  return(ode(y=y0,times=t_eval,func=fun,parm=c(...),method='ode45'))
}

#funkcja bledu

f_bledu_L2<- function(a,b){
  return(mean((a-b)^2))
}

#funkcje Hamiltona z wyliczonym na podstawie danych wspolczynnikiem k 

hamiltonian <- function(q,p){
  k=2.4  # wielkosc odczytana z kodu autorow
  H=k*(1-cos(q))+ p^2
  return(H)
}


############### --- POBRANIE I PRZYGOTOWANIE DANYCH --- ####################################################

dane=read.csv("./real_pend_h_1.txt",sep='', dec='.', header = FALSE)
colnames(dane)<-c('trial','t','o','v') # nadanie nazw kolumnom
dane=dane[-1,] # usuniecie 1. wiersza

wekt_x=cbind(dane$o,dane$v) # wybranie wektora [p,q]
wekt_t=as.numeric(dane$t) # wybranie wektora czasu
# obliczenie zmian wektora x w czasie
wekt_dx=(wekt_x[2:nrow(wekt_x),]-wekt_x[1:nrow(wekt_x)-1,])/(wekt_t[2:length(wekt_t)]-wekt_t[1:length(wekt_t)-1])
wekt_x=wekt_x[1:nrow(wekt_x)-1,]
wekt_t=wekt_t[1:length(wekt_t)-1]

# podzial na dane testowe i treningowe
test_split=0.8
train_set_size = as.numeric(nrow(wekt_x) * test_split)
test_set_size = as.numeric(nrow(wekt_x) * (1-test_split))
test_x=wekt_x[(train_set_size+1):(train_set_size+test_set_size),] #wekt_x[444:555,]
test_dx=wekt_dx[(train_set_size+1):(train_set_size+test_set_size),] #wekt_dx[444:555,]
test_t=wekt_t[(train_set_size+1):(train_set_size+test_set_size)] #wekt_t[444:555]
x=wekt_x[1:train_set_size,] # treningowy wekt [1:444]
dx=wekt_dx[1:train_set_size,]

#wykres danych
plot(x[,1],x[,2], col='steelblue',xlab = 'theta',ylab='theta\'',main = 'Wizualizacja danych eksperymentalnych',pch=16)
points(test_x[,1],test_x[,2], col='magenta',pch=16)
legend('center',legend = c('dane trening', 'dane test'),col=c('steelblue','magenta'),pch=16)

#######################--- BUDOWA SIECI ---##################################################################
# siec neuronowa - klasyczna wersja, tzw multilayer perceptron MLP
klasyczna_siec <- nn_module(
  classname = "MLP",
  #definiowanie warstw i neuronow
  initialize = function(W_ukryta){
    
    self$actF = nn_tanh() #tangens hiperboliczny
    
    #warstwy
    self$Lin_1 = nn_linear(2,W_ukryta)
    self$Lin_2 = nn_linear(W_ukryta,W_ukryta)
    self$Lin_out = nn_linear(W_ukryta,2, bias = FALSE)
    
    for(warstwa in c(self$Lin_1,self$Lin_2,self$Lin_out)) {
      nn_init_orthogonal_(warstwa$weight)}    
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

#hamiltonowska siec neuronowa
hamiltonowska_siec <- nn_module(
  classname = "HNN",
  #definiowanie
  initialize = function(model_d,baseline){ #podstawa hnn jest zwykla siec MLP
    self$baseline = baseline # T or F
    self$model_d = model_d #MLP
    self$J= self$tensor_przeksz(2)  # 2 - wymiar danych wejciowych do sieci
                                    # J to w tym przypadku 'macierz' przeksztalcenia hamiltonowskich rr
  },
  
  #funkcja forward wywolywana jest, gdy do sieci 'wrzucimy' dane wejsciowe x
  forward = function(x){
    if (self$baseline==T){  #baseline = T oznacza klasyczna siec
      return(self$model_d(x))
    }
    y=self$model_d(x)
    return(torch_split(y,1,2))  # zwraca dwa wyjscia hnn
  },
  
  pochodna = function(x){
    if (self$baseline==T){
      return(self$model_d(x))
    } 
    
    N1=self$forward(x)[[1]]$unsqueeze(dim=2)
    N2=self$forward(x)[[2]]$unsqueeze(dim=2)
    
    z = torch_zeros_like(x)
    
    dN2 = autograd_grad(sum(N2), x, create_graph = TRUE)[[1]]
    z=torch_matmul(dN2 , torch_t(self$J)) # mnozenie wektora dN2 razy macierz J

    return(z)
  },
  
  tensor_przeksz = function(n){ 
    J = torch_eye(n)
    J = torch_cat(list(J[(floor(n/2)+1):nrow(J)],-J[1:floor(n/2)]),1)
    return(J)
  }
)


#########################--- TRENOWANIE SIECI ---############################################################

train <- function(seed, baseline,W_ukryta,learn_rate, epochs,ile,test_x,test_dx,x,dx){
  torch_manual_seed(seed) #ustalenie ziarna
  set.seed(seed)
  
  if (baseline==T) {
    print("Training baseline model:")
  } else{
    print("Training HNN model:")
  }
  nn_model=klasyczna_siec(W_ukryta) # klasyczna siec o w_ukryta liczbie neuronow
  model = hamiltonowska_siec(model_d=nn_model,baseline=baseline)
  optimizer = optim_adam(model$parameters, learn_rate, weight_decay = 1e-5) # stosujemy optymalizacje - Adam algorithm

  x=torch_tensor(x,requires_grad = TRUE, dtype = torch_float32())
  test_x=torch_tensor(test_x,requires_grad = TRUE, dtype = torch_float32())
  dxdt=torch_tensor(dx)
  test_dxdt = torch_tensor(test_dx)
  
  #petla uczaca
  for (step in 1:epochs) {
    #krok treningowy
    dxdt_hat=model$pochodna(x) #pochodna z modelu
    # obliczamy blad
    loss = f_bledu_L2(dxdt,dxdt_hat)
    loss$backward()
    # robimy krok przeciwny do kierunku wskazanego przez  'gradient'
    optimizer$step()
    #NN$parameters
    # zerujemy gradient opptymalizacji
    optimizer$zero_grad()
    
    #walidacja
    test_dxdt_hat=model$pochodna(test_x)
    test_loss = f_bledu_L2(test_dxdt,test_dxdt_hat)
    
    if (step %% ile==0) { #wyswietlamy blad w co ktores iteracji
      print(paste0('train_loss ', loss$item(), ', test_loss ',test_loss$item()))
    }
  }
  return(model)
  }
########################---- MODEL HAMILTONOWSKI ----##########################################################

seed=0
baseline=FALSE
W_ukryta=200
learn_rate=1e-3
epochs=3000
ile=200
test_x=test_x
test_dx=test_dx
x=x
dx=dx
ile=200

#trening sieci
hnn_model = train(seed, baseline,W_ukryta,learn_rate, epochs,ile,test_x,test_dx,x,dx)

# zapisanie modelu
#hnn_model$eval()
#torch_save(hnn_model,"HNN_realpendulum.rt")

#zaladowanie modelu
#hnn_model = torch_load("HNN_realpendulum.rt")

#######################---- MODEL PODSTAWOWY ----###########################################################

seed=0
baseline=TRUE
W_ukryta=200
learn_rate=1e-3
epochs=3000
ile=200
test_x=test_x
test_dx=test_dx
x=x
dx=dx
ile=200

#trening sieci
base_model = train(seed, baseline,W_ukryta,learn_rate, epochs,ile,test_x,test_dx,x,dx)

# zapisanie modelu
#base_model$eval()
#torch_save(base_model,"HNN_realpendulum2.rt")

#zaladowanie modelu
#base_model = torch_load("HNN_realpendulum2.rt")


################--- ROZWIAZANIE ZAGADNIENIA POCZATKOWEGO ---###############################################

#warunek poczatkowy
y0=array(c(0.75,0))
#krok czasowy
t_eval = torch_linspace(0,20,1000)
t_eval=as.array(torch_reshape(t_eval,list(-1,1)))

# rozwiazanie w przypadku rownania wyuczonego klasyczna siecia i hamiltonowska
base_ivp = integrate_model(base_model, t_eval, y0)
hnn_ivp = integrate_model(hnn_model, t_eval,y0)

#wykresy rozwiazan
plot(hnn_ivp[,2],hnn_ivp[,3], col='red',type ='l',lwd=3,xlab = 'q',ylab = 'p',
     main = 'Portret fazowy rozwi¹zañ')
lines(base_ivp[,2],base_ivp[,3],col='blue',type ='l',lwd=3)
legend('bottomleft',legend = c('HSN', 'SN'),col=c('red','blue'),lty=c(1,1),lwd=3)


plot(hnn_ivp[,1],hnn_ivp[,2], type = 'l', col='red',lwd=3,xlab = 't',ylab = 'q',
     main = 'K¹t wychylenia w czasie')
lines(base_ivp[,1],base_ivp[,2],col='blue',type ='l',lwd=3)
legend('bottomleft',legend = c('HSN', 'SN'),col=c('red','blue'),lty=c(1,1),lwd=3)

plot(hnn_ivp[,1],hnn_ivp[,3], type = 'l', col='red',lwd=3,xlab = 't',ylab = 'p',
     main = 'Pêd w czasie')
lines(base_ivp[,1],base_ivp[,3],col='blue',type ='l',lwd=3)
legend('bottomleft',legend = c('HSN', 'SN'),col=c('red','blue'),lty=c(1,1),lwd=3)

###################---- ANALIZA MODELI ----###########################################################
# porownanie z danymi eksperymentalnymi

#
t_eval = test_t - min(test_t)
t_span = c(min(t_eval), max(t_eval))
x0=test_x[1,]
true_x=test_x
# rozwiazanie modelu na realnych danych 
base_path = integrate_model(base_model,t_eval,x0)
base_x=base_path[,c(2,3)]
hnn_path = integrate_model(hnn_model,t_eval,x0)
hnn_x=hnn_path[,c(2,3)]


par(mfrow=c(2,2))
#wykresy
plot(true_x[,1],true_x[,2],type = 'l', col='black',lwd=3,xlab = 'q',ylab = 'p',
     main = 'Portret fazowy rozwi¹zañ')
lines(base_x[,1],base_x[,2],type = 'l', col='blue',lwd=3)
lines(hnn_x[,1],hnn_x[,2],type = 'l', col='red',lwd=3)
legend('bottomleft',legend = c('dane', 'SN','HSN'),col=c('black','blue','red'),lty=c(1,1,1),lwd=3)

#wykres sredniej roznicy kwadratow miedzy wspolrzednymi
plot(t_eval,apply((true_x-base_x)^2,1,mean) ,type = 'l', col='blue',lwd=3,xlab = 't',
     ylab = 'MSE', main="MSE miêdzy wspó³rzêdnymi")
lines(t_eval,apply((true_x-hnn_x)^2,1,mean),type = 'l', col='red',lwd=3)
legend('topleft',legend = c('SN','HSN' ),col=c('blue','red'),lty=c(1,1),lwd=3)

#wielkosc, ktora rzeczywiscie zachowuje hnn
true_hq=hnn_model(torch_tensor(true_x))[[2]]
base_hq=hnn_model(torch_tensor(base_x))[[2]]
hnn_hq=hnn_model(torch_tensor(hnn_x))[[2]]

plot(t_eval,true_hq,type = 'l',lwd=3, col='black', main="Wielkoœæ zachowana przez HSN",ylim=c(-11,-10.5),xlab='t')
lines(t_eval,base_hq,type = 'l', col='blue',lwd=3)
lines(t_eval,hnn_hq,type = 'l', col='red',lwd=3)
legend('bottomleft',legend = c('dane', 'SN','HSN'),col=c('black','blue','red'),lty=c(1,1,1),lwd=3)

#obliczanie energii calkowitej z hamiltonianu ze wzoru
true_e = hamiltonian(true_x[,1],true_x[,2])
base_e = hamiltonian(base_x[,1],base_x[,2])
hnn_e = hamiltonian(hnn_x[,1],hnn_x[,2])

plot(t_eval,true_e,type = 'l', col='black', main="Energia ca³kowita",ylim = c(0,1),
     lwd=3,xlab = 't', ylab = 'Energia')
lines(t_eval,base_e,type = 'l', col='blue',lwd=3)
lines(t_eval,hnn_e,type = 'l', col='red',lwd=3)
legend('bottomleft',legend = c('dane', 'SN','HSN'),col=c('black','blue','red'),lty=c(1,1,1),lwd=3)


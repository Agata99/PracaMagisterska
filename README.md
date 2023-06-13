# PracaMagisterska
Repozytorium zawiera kody programów napisane w ramach pracy magisterskiej pod tytułem "Hamiltonowskie sieci neuronowe". Do implementacji sieci zastosowano język R, opierając się na bibliotece torch.

## Model I

Plik *model1_HNN_nielin_oscyl.R* zawiera implementację hamiltonowskiej sieci neuronowej rozwiązującej równanie różniczkowe opisujące ruch układu fizycznego. W programie przeprowadzana jest analiza i porównanie działania HSN z klasyczną siecią neuronową. Przed sprawdzeniem wyniku działania tego programu, należy otworzyć plik *model1_klasNN_nielin_oscyl.R* i wytrenować klasyczną sieć, i zapisać dany model do pliku
*klasNNprz1.rt* (bądź skorzystać z załączonego pliku). Model ten jest niezbędny do działania programu *model1_HNN_nielin_oscyl.R*.

## Model II

Plik *model2_HNN_real_pendulum.R* zawiera implementacje hamiltonowskiej sieci neuronowej, która napędzana jest przez dane eksperymentalne. Niezbędne do działania modelu dane dołączone są pod nazwą *real_pend_h_1.txt*. Sieć ta daje w wyniku wektory pochodnych, determinujące ruch układu. Program zawiera wykresy porównujące działania modelu z danymi eksperymentalnymi.

## Dodatek

Pozostałe pliki zawierają programy, które powstały na potrzeby zapoznania się z biblioteką torch i działaniem sieci neuronowych. Znajdują się tu programy rozwiązujące proste równania różniczkowe (dwa przykłady) oraz przykłady, które nie weszły do pracy magisterskiej.

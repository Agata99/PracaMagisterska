# PracaMagisterska
Repozytorium zawiera kody programów napisane w ramach pracy magisterskiej pod tytułem "Hamiltonowskie sieci neuronowe".

## Model I

Plik *4v_HNN_przyk_z_artykulu.R* zawiera implementację hamiltonowskiej sieci neuronowej rozwiązującej równanie różniczkowe opisujące ruch układu fizycznego. W programie przeprowadzana jest analiza i porównanie działania HSN z klasyczną siecią neuronową. Przed sprawdzeniem wyniku działania tego programu, należy otworzyć plik *3v_klasNN_przyklad_z_artykulu.R* i wytrenować klasyczną sieć, i zapisać dany model do pliku
*klasNNprz1.rt* (bądź skorzystać z załączonego pliku). Model ten jest niezbędny do działania programu *HNN_przyk_z_artykulu.R* .

## Model II

Plik *HNN_real_pendulum.R* zawiera implementacje hamiltonowskiej sieci neuronowej, która napędzana jest przez dane eksperymentalne. Niezbędne do działania modelu dane dołączone są pod nazwą *real_pend_h_1.txt*. Sieć ta daje w wyniku wektory pochodnych, determinujące ruch układu. Program zawiera wykresy porównujące działania modelu z danymi eksperymentalnymi.

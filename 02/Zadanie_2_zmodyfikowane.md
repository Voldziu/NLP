# Zadanie 2: Modele językowe 

Opracowanie: dr inż. Arkadiusz Janz, mgr inż. Piotr Miłkowski 

Zmodyfikowane: Konrad Kiełczyński 

Czas wykonania: 2 tygodnie (oddanie lab 5) 

## Część I: Modelowanie języka (6 pkt) 

- A. Należy zapoznać się z implementacją następujących klas w bibliotece _Transformers_ <u>(htps://huggingface.co/docs/transformers/indext</u> ) **[0,25 pkt]** : 

- _BertModel,_ 

- _BertForMaskedLM,_ 

   - _BertForSequenceClassification,_ 

- _BertForTokenClassification._ 

- B. Następnie, należy zapoznać się z biblioteką _PEFT_ oraz przykładami z repozytorium **[0,25 pkt]** : 

- <u>https://huggingface.co/docs/peft/quicktour</u> 

- <u>https://github.com/huggingface/peft/tree/main/examples</u> 

- C. Proszę również zapoznać się z tutorialem z następującego źródła i powtórzyć ten tutorial we własnym środowisku uruchomieniowym **[0,25 pkt]** : 

- <u>https://medium.com/@nubyra/parameter-efficient-fine-tuning-peft-of-bertbase-model-to-predict-medical-diagnosis-5086a1828f4b</u> 

- D. Na koniec proszę zapoznać się również z następującymi klasami w bibliotece _Transformers_ **[0,25 pkt]** : 

- _GPT2Model,_ 

- _GPT2LMHeadModel,_ 

   - _GPT2ForSequenceClassification,_ 

- _GPT2ForTokenClassification._ 

## Plan realizacji ćwiczenia 

1. Należy wykonać strojenie modelu BERT-base na danych pozyskanych w ramach Zadania 1. Można również wykorzystać inne dane treningowo-testowe. Proszę wykorzystać model polskojęzyczny (np. `allegro/herbert-base-cased` ). Uwaga: proszę zastosować również adapter PEFT!!! [1,5pkt] 

   - a. Wykonać strojenie dla zadania klasyfikacji tekstu ( _BertForSequenceClassification_ ) 

   - b. Wykonać strojenie dla zadania klasyfikacji tokenów ( _BertForTokenClassification_ ) 

   - c. Wydzielić niewielki zbiór testowy i ocenić jakość predykcji. 

2. Zwizualizować przestrzeń wektorową dla przykładów testowych ( wystarczy jedynie klasyfikacja tekstu) dla modelu przed dostrojeniem (model bazowy) oraz po dostrojeniu i zrobić interaktywne wykresy przedstawiające to, w jaki sposób przypadki testowe organizują się w przestrzeni wektorowej względem przypisanych etykiet. Porównać oba wykresy i opisać, jak dostrajanie zmienia organizację przestrzeni. [1pkt] 

- W przypadku wykorzystania adaptera PEFT proszę wykorzystać reprezentacje tokenu [CLS] jako reprezentację tekstu. 

3. Powtórzyć podpunkty 1 i 2 z wykorzystaniem klas _GPT2ForSequenceClassification i GPT2ForTokenClassification._ Proszę wykorzystać model polskojęzyczny (np. `sdadas/polish-gpt2-medium` ) Uwaga: proszę zastosować również adapter PEFT!!! [1pkt] 

- W przypadku wykorzystania adaptera PEFT proszę wykorzystać reprezentacje ostatniego tokenu sekwencji tekstowej jako reprezentację tekstu. 

- Proszę również zamaskować tokeny paddingu i znacznik <eos> za pomocą etykietki -100. 

4. Wykorzystać token [MASK] w modelu BERT do powiększenia zbiorów treningowych (ang. data augmentation). [0,5pkt] 

5. Wykorzystać generatywne zdolności GPT-2 i różne hiperparametry generacji, np. temperatura, top-p, top-k, powiększenia zbiorów treningowych (ang. data augmentation). [1pkt] 

- Uwaga! W podpunktach 4 i 5 można również połączyć wykorzystanie modelu BERT i GPT-2 w ramach zagadnienia powiększania zbiorów danych. Wystarczy wytrenować jeden model na powiększonym zbiorze danych. 

## Część II Analiza własności modeli językowych (4 pkt) 

- A. Na wstępie należy zapoznać się z następującym zbiorem danych 

- <u>https://huggingface.co/datasets/clarin-knext/wsd_polish_datasets</u> 

- B. Proszę zapoznać się również z podanym artykułem [0,25pkt] : 

- <u>http://ai.stanford.edu/blog/contextual/</u> 

- <u>https://aclanthology.org/D19-1006.pdf</u> 

- C. Na koniec proszę zapoznać się z techniką Logit Lens przedstawioną w poniższych materiałach [0,25pkt] : 

- <u>https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gptthe-logit-lens</u> 

- <u>https://arxiv.org/abs/2303.08112</u> 

## Plan realizacji ćwiczenia 

1. Należy wykorzystać dołączony do zadania zbiór danych z podpunktu A i odtworzyć badanie z podpunktu B dla modeli BERT i GPT-2 . [2pkt] 

   - a. wykonać analizę anizotropii (Anisotropy) 

   - b. wykonać analizę zależności od kontekstu (Context-Specificity) 

2. Proszę wykorzystać technikę Logit Lens omówioną w materiałach z podpunktu C niniejszej instrukcji, następnie dokonać analizy modelu GPT-2 [1,5pkt] 

   - a. do analizy należy wybrać model polskojęzyczny `sdadas/polishgpt2-medium` , 

   - b. należy przygotować zestaw co najmniej 20 promptów, np. zdania faktograficzne („Stolicą Francji jest”) oraz zdania sprawdzające wiedzę gramatyczną (np. zgodność form fleksyjnych), 

   - c. dla każdej warstwy należy przepuścić stan ukryty ostatniego tokenu przez końcową normalizację (LayerNorm) i macierz wyjściową modelu, uzyskując rozkład prawdopodobieństwa po słowniku, 

   - d. należy przygotować wykres (heatmapę) najbardziej prawdopodobnych tokenów w kolejnych warstwach dla wybranych promptów, 

   - e. należy wyliczyć miarę ilościową: rangę oraz prawdopodobieństwo tokenu przewidywanego przez model na wyjściu w zależności od warstwy, uśrednione po wszystkich promptach, 

   - f. należy przeanalizować uzyskane wyniki i podsumować, od której warstwy model „zna” odpowiedź oraz czy różni się to dla zdań faktograficznych i gramatycznych. 


# Zadanie 1: Korpusy anotowane oraz osadzenia słów i tekstów 

Opracowanie: dr inż. Arkadiusz Janz, dr hab. inż. Maciej Piasecki prof. PWr 

Zmodyfikowane: Konrad Kiełczyński 

#### Czas wykonania: 2 tygodnie (oddanie lab 3) 

Korpus językowy jest zbiorem tekstów, który stanowi podstawowy zasób praktycznie w każdym zadaniu związanym z przetwarzaniem języka naturalnego. Przy pomocy wielkich korpusów można budować ogólne reprezentacje językowe, szukać kontekstów występowania słów, a także analizować specyficzne konstrukcje składniowe i semantyczne w danym języku (lub językach - w przypadku korpusu wielojęzycznego). Mniejsze korpusy często są także ręcznie wzbogacane o dodatkowe metadane, dzięki czemu możliwe jest wykorzystanie takich danych w uczeniu maszynowym do zadań klasyfikacji całych dokumentów, zdań, a także fraz, czy też pojedynczych słów. Elektroniczne wersje korpusów tekstowych służą do prowadzenia badań nad językiem, tworzenia słowników, wyszukiwarek, a także systemów tłumaczenia maszynowego, określania stylu literackiego, rozpoznawania wydźwięku i emocji, ujednoznaczniania znaczeń słów, wydobywania nazw własnych i wielu innych zastosowań. 

Często wyzwaniem dla badaczy jest określenie składu korpusu, czyli takiego doboru tekstów, by były one reprezentatywne z perspektywy konkretnego problemu do rozwiązania. W kontekście budowania wielkich, ogólnych modeli językowych, istotne jest zebranie jak największej liczby tekstów z różnych źródeł, dziedzin oraz stylów wypowiedzi (książki, artykuły, blogi, komentarze, recenzje, itp.). Często jednak problemy są bardzo specyficzne i ukierunkowane dziedzinowo, np.: 

- detekcja fałszywych wiadomości dotyczących COVID-19, 

- rozpoznawanie wydźwięku opinii pacjentów o lekarzach, 

- określanie autorstwa tekstów literackich, 

- wydobywanie nazw hoteli w ofertach turystycznych, 

- predykcja kursów giełdowych na podstawie tekstów ekonomicznych, 

- budowa systemu dialogowego dla użytkowników operatorów telefonicznych, 

- konstrukcja chatbota dla petentów Urzędu Miejskiego Wrocławia, 

- utworzenie wirtualnej sekretarki lekarza pierwszego kontaktu, 

- wykrywanie mowy obraźliwej w komentarzach na Twitterze, 

- streszczanie artykułów prasowych. 

W każdym z wyżej wymienionych przypadków należy utworzyć tzw. korpus reprezentatywny dla danej dziedziny. Modele językowe powstałe z wielkich i ogólnych korpusów językowych można stosować do tego typu zadań, lecz często wcześniej dostraja się je (ang. fine-tuning) na korpusach wzorcowych. Z założenia korpusy wzorcowe mają pomóc w realizacji zadania, gdyż zawierają teksty z konkretnej dziedziny, zawierają specyficzne dla danego problemu słownictwo, określone konstrukcje leksykalno-składniowo-semantyczne i dzięki temu spodziewamy się poprawy jakości działania metody dostrojonej na takim korpusie. 

## Część I 

### Anotacja (5 pkt) 

Należy wykonać anotację wybranego zbioru danych zgodnie z procedurą 2+1. W trakcie anotacji należy monitorować poziom zgodności (2 iteracje) oraz aktualizować wytyczne (po iteracji 1). W anotacji należy posłużyć się zbiorami dostarczonymi w ramach instrukcji lub własnym zbiorem danych (np. powiązanym z projektem naukowowdrożeniowym). 

### Plan realizacji ćwiczenia 

1. Wczytać wybrany korpus tekstów i przygotować do anotacji [1pkt] 

1. Ustalić wytyczne anotacji i wykonać anotację w wybranym narzędziu [1pkt] 

   - a. Sugerowane narzędzie: Label Studio (https://labelstud.io), instalacja poleceniem pip install label-studio. Narzędzie obsługuje anotację zarówno całych tekstów, jak i fragmentów tekstu. 

   - b. Alternatywnie: Doccano (https://github.com/doccano/doccano) lub Google Spreadsheets (tylko anotacja na poziomie całego tekstu). 

   - c. Zapisać wytyczne w formie krótkiej notatki dla anotatorów 

   - d. Anotacja przeprowadzona na poziomie całego tekstu 

   - e. Anotacja na poziomie poszczególnych fragmentów tekstu 

   - f. Należy pamiętać o dobrych praktykach anotacji (niezależność anotacji poszczególnych anotatorów, próbka dobrana losowo, itd.) 

2. Wykonać eksport oetykietowanych danych i wyliczyć zgodność [0.5pkt] 

   - a. Wyliczyć wartość Kappy Cohena 

   - b. Wyliczyć wartość Kappy Fleissa 

3. Ustalić nowe wytyczne i powtórzyć anotację na nowej próbce danych [1pkt] 

4. Ponownie wyliczyć zgodność i porównać z wart. z pierwszej iteracji [0.5pkt] 

   - a. Wyliczyć wartość Kappy Cohena 

   - b. Wyliczyć wartość Kappy Fleissa 

5. Jeśli osiągnięto odpowiedni poziom zgodności, należy zaanotować pozostałą część zbioru (każda osoba anotuje po N niezależnych przykładów) 

6. Zwięźle podsumować pozyskany zbiór za pomocą statystyk opisowych [ 1 pkt ] 

### Sugerowane zbiory danych 

### 1. Zbiór PolEmo 

<u>https://clarin-pl.eu/dspace/handle/11321/710</u> 

- Własną anotację można porównać z anotacją dostarczoną w ramach korpusu 

- W ramach anotacji fragmentów tekstu sugeruje się wykonanie anotacji aspektów (ang. Aspect-based Sentiment Analysis) lub modyfikatorów wydźwięku na poziomie fraz (wzmacnianie, osłabianie, odwracanie polaryzacji) 

### 2.Detekcja mowy nienawiści 

<u>https://huggingface.co/datasets/poleval/poleval2019_cyberbullying</u> 

- Należy posłużyć się biblioteką `huggingface datasets`, aby wczytać zbiór. 

- W ramach anotacji fragmentów tekstu sugeruje się wykonanie anotacji aspektów lub modyfikatorów opinii na poziomie fraz (wzmacnianie, osłabianie, odwracanie) 

### 3. Inne, własne zadania: 

- emocje w komentarzach filmowych 

- fake newsy i dezinformacja 

   - styl i autorstwo 

   - opinie pacjentów o lekarzach 

- … 

### Sugerowana literatura 

Handbook of Linguistic Annotation, Nancy Ide, James Pustejovsky, Springer, 2017; sugerowany artykuł: ,,Inter-Annotator Agreement'' (s. 297) 

Artykuł o anotacji wydźwięku (Sentiment): 

<u>https://medium.com/besedo-engineering/sentiment-analysis-part-3-data-annotationd0845d1a8a1b</u> 

### Artykuł o anotacji jednostek nazewniczych (NER) 

<u>https://medium.com/@rongqianhui/named-entity-recognition-annotation-schemese684f9cd5a56</u> 

DiaBiz.Kom – Towards a Polish Dialogue Act Corpus Based on ISO 24617-2 Standard, <u>https://aclanthology.org/2022.coling-1.320.pdf</u> 

Dokumentacja Label Studio (szablony do klasyfikacji tekstu i NER) <u>https://labelstud.io/templates</u> 

### Artykuł o Docanno 

<u>https://towardsdatascience.com/doccano-a-tool-to-annotate-text-data-to-traincustom-nlp-models-f4e34ad139c3</u> 

## Część II 

Pre-trenowane modele przestrzeni wektorowych służą do modelowania różnorodnych zjawisk językowych występujących w korpusach tekstów. W niniejszej części należy zapoznać się z podstawowymi metodami tworzenia modeli przestrzeni wektorowych (model worka słów (ang. Bag-of-Words) i TF-IDF, Word2Vec, Fasttext) oraz ze współczesnymi enkoderami zdań opartymi na modelach transformerowych, a następnie zastosować je do analizy tekstów oraz ich anotacji. Należy zestawić rezultaty otrzymane za pomocą wybranych modeli osadzania i algorytmów redukcji wymiarowości danych z uzyskanymi w Części I anotacjami. Celem ćwiczenia jest wstępna analiza oetykietowanych danych oraz weryfikacja ich przydatności do automatycznego przetwarzania za pomocą prostych klasyfikatorów. 

### Plan realizacji ćwiczenia 

### Word Embeddings (2 pkt) 

1. Wykonać osadzenia słów w anotowanym korpusie (co najmniej 2 różne modele) [0.5pkt] 

   - a. Word2vec 

      - i. dla języka polskiego: 

<u>https://dsmodels.nlp.ipipan.waw.pl</u> 

   - b. Fasttext 

      - i. dla języka polskiego: <u>https://huggingface.co/clarin-pl/fastText-kgr10</u> 

2. Wizualizacja danych za pomocą t-SNE (lub UMAP) z odniesieniem do etykiet w ramach anotacji. Wykonać wykres interaktywny, w którym po najechaniu na punkt można odczytać słowo i / lub anotację. [0.5pkt] 

3. Wygenerować listy k-najbardziej podobnych dla słów zaanotowanych w korpusie i porównać otrzymane listy dla co najmniej 2 różnych modeli przestrzeni wektorowych [0.5pkt] 

4. Opracować krótką dyskusję otrzymanych rezultatów [0.5pkt] 

### Pytania pomocnicze do dyskusji: 

1. Czy osadzenia pre-trenowane na dużych korpusach dobrze odwzorowują znaczenie słów w utworzonym korpusie dziedzinowym? 

2. Czy w zebranym korpusie tekstowym występuje grupowanie słów związanych z anotacją? 

   - Przykładowo, w kontekście wydźwięku, czy występuje grupowanie na słowa pozytywne i negatywne? Jeśli nie, to dlaczego osadzenia mogą nie oddawać emocjonalnego znaczenia słów? 

3. Na czym polegają różnice między wygenerowanymi za pomocą różnych modeli listami podobieństwa i z czego one wynikają? 

4. Jak moglibyśmy ulepszyć osadzenia dla rozwiązywanego zadania? 

### Text Embeddings: modele statyczne a kontekstowe (1.5 pkt) 

1. Wygenerować osadzenia zdań lub całych tekstów za pomocą dwóch modeli: [0.5pkt] 

   - a. model statyczny: uśrednione wektory Word2Vec lub wektory zdań Fasttext (można wykorzystać modele z punktu Word Embeddings), 

   - b. współczesny enkoder zdań z biblioteki SentenceTransformers <u>(https://sbert.net), np.:</u> 

      - i. sdadas/mmlw-roberta-base: <u>https://huggingface.co/sdadas/mmlw-roberta-base</u> 

      - ii. intfloat/multilingual-e5-base: <u>https://huggingface.co/intfloat/multilingual-e5-base</u> 

      - iii. należy zapoznać się z kartą modelu, ponieważ część modeli wymaga dodania prefiksu do tekstu wejściowego. 

2. Dla każdego z modeli, wizualizacja danych za pomocą t-SNE (lub UMAP) z odniesieniem do etykiet w ramach anotacji. Wykonać wykresy interaktywne, w których po najechaniu na punkt można odczytać zdanie (tekst) i / lub anotację. [0.5pkt] 

3. Proszę przeprowadzić analizę otrzymanych osadzeń, porównać model statyczny z enkoderem kontekstowym i odpowiedzieć na zadane pytania pomocnicze (patrz poniżej). [0.5pkt] 

### Pytania pomocnicze do dyskusji 

1. Czy są zdania, które model uznał za podobne, ale w rzeczywistości różnią się znaczeniem (np. tylko negacją)? 

2. Czy są zdania, które są semantycznie podobne, ale model uznał je za odległe, bo użyto innych słów? 

3. Czy zdania o różnej długości, ale podobnym sensie, są traktowane przez model podobnie? Czy długość tekstu wpływa na „wyrazistość” jego wektora (szczególnie przy uśrednianiu wektorów słów)? 

4. Który z modeli lepiej separuje klasy z anotacji i dlaczego? 

Proszę uzasadnić swoje odpowiedzi. 

### Klasyfikacja: przydatność zbioru do automatycznego przetwarzania (1.5 pkt) 

Na zaanotowanym zbiorze (etykiety na poziomie całego tekstu) należy wytrenować proste klasyfikatory i zweryfikować, czy zebrane dane umożliwiają automatyczne rozpoznawanie etykiet. Uwaga: w tym zadaniu nie należy dostrajać modeli transformerowych , dostrajanie jest przedmiotem Zadania 2. 

1. Wytrenować i porównać klasyfikatory (np. regresja logistyczna) na różnych reprezentacjach tekstu: [0.5pkt] 

   - a. TF-IDF (Bag-of-Words), 

   - b. osadzenia z modelu statycznego, 

   - c. osadzenia z enkodera zdań. 

2. Porównać wyniki z lematyzacją i bez lematyzacji tekstu: [0.5pkt] 

   - a. porównanie należy wykonać co najmniej dla TF-IDF oraz modelu statycznego (Word2Vec / Fasttext), 

   - b. do lematyzacji można wykorzystać np. spaCy (pl_core_news_lg), Morfeusz2 lub Stanza, 

   - c. należy zwrócić uwagę, czy wybrany model Word2Vec został wytrenowany na formach słów, czy na lematach. 

3. Opracować wyniki i dyskusję: [0.5pkt] 

   - a. ewaluacja z użyciem walidacji krzyżowej (lub wydzielonego zbioru testowego) z zachowaniem rozkładu klas (stratyfikacja), 

   - b. miary: accuracy oraz macro-F1, macierz pomyłek, 

   - c. analiza przykładowych błędów klasyfikacji w odniesieniu do przypadków spornych z anotacji (Część I). 

### Pytania pomocnicze do dyskusji 

1. Jak lematyzacja wpływa na rozmiar słownika i na skuteczność TF-IDF? Dlaczego w języku polskim efekt ten może być silniejszy niż w języku angielskim? 

2. Czy lematyzacja pomaga modelom statycznym? Czy Fasttext (n-gramy znakowe) jest na nią mniej wrażliwy niż Word2Vec? 

3. Dlaczego lematyzacja nie jest zwykle stosowana przy enkoderach transformerowych? 

4. Czy prosta reprezentacja TF-IDF jest konkurencyjna wobec enkodera zdań? Z czego to wynika? 

5. Czy przykłady, na których myli się klasyfikator, pokrywają się z przykładami, przy których anotatorzy byli niezgodni? 

Proszę uzasadnić swoje odpowiedzi. 

### Sugerowana literatura 

1. Wprowadzenie do Word Embeddings 

Fragment książki D. Jurafsky o modelach przestrzeni wektorowych <u>https://web.stanford.edu/~jurafsky/slp3/6.pdf</u> 

Notatki do wykładu NLP, Stanford CS224N <u>https://web.stanford.edu/class/cs224n/readings/ cs224n_winter2023_lecture1_notes_draft.pdf</u> 

Efficient Estimation of Word Representations in Vector Space <u>https://arxiv.org/pdf/1301.3781</u> 

Distributed Representations of Words and Phrases and their Compositionality <u>https://arxiv.org/abs/1310.4546</u> 

Bag of Tricks for Efficient Text Classification <u>https://arxiv.org/pdf/1607.01759</u> 

2. Eksploracja modeli przestrzeni wektorowych 

Tutorial do ćwiczeń z NLP, Stanford <u>https://web.stanford.edu/class/cs224n/assignments/a1_preview/ exploring_word_vectors.html</u> 

Algorytm klasteryzacji HDBSCAN <u>https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html</u> 

Redukcja wymiarowości za pomocą t-SNE <u>https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochasticneighbor-embedding-0e365ee898ea</u> 

Redukcja wymiarowości za pomocą UMAP <u>https://umap-learn.readthedocs.io/en/latest/how_umap_works.html</u> 

### 3. Enkodery zdań 

Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <u>https://arxiv.org/abs/1908.10084</u> 

Dokumentacja SentenceTransformers <u>https://sbert.net</u> 

Polskie modele MMLW i benchmark PL-MTEB <u>https://arxiv.org/abs/2402.13350</u> 

### 4. Lematyzacja dla języka polskiego 

spaCy: modele dla języka polskiego <u>https://spacy.io/models/pl</u> 

### Morfeusz2 

<u>http://morfeusz.sgjp.pl</u> 


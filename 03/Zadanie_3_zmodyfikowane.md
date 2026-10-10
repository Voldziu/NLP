# Zadanie 3: Wyszukiwanie semantyczne 

Opracowanie: Arkadiusz Janz, Konrad Wojtasik 

## Zmodyfikowane: Konrad Kiełczyński 

Czas wykonania: 3 tygodnie (oddanie lab 8) 

Celem niniejszego ćwiczenia jest zapoznanie się z technologią semantycznego wyszukiwania opartą na modelach językowych. Zadanie wyszukiwania polega na zwróceniu dokumentów, które są relewantne do zapytania użytkownika. Semantyczne wyszukiwanie odnosi się do procesu przetwarzania języka naturalnego (NLP), w którym kluczowym krokiem jest opisanie znaczenia zapytania i znaczenia analizowanych dokumentów, a następnie wykorzystanie uzyskanych opisów semantycznych w dopasowaniu dokumentów do zapytania. 



<!-- Start of picture text -->
Relevant<br>document<br>Query<br><!-- End of picture text -->

Źródło: https://huggingface.co/learn/nlp-course/chapter5/6 

#### **Przykład 1: Zapytanie "Co to jest sztuka?"** 

- Dokument 1: "Sztuka to forma wyrazu artystycznego." 

- Dokument 2: "Sztuka to sposób bycia." 

W semantycznym wyszukiwaniu system identyfikuje znaczenie słowa "sztuka" w obu dokumentach i ocenia, że są one relewantne do zapytania. 

#### **Przykład 2: Zapytanie "Jaka jest cena produktu Y?"** 

- Dokument 1: **"Produkt X kosztuje 100 złotych, zaś Produkt Z - 200 złotych."** 

- Dokument 2: **"Produkt Y kosztuje 20 złotych."** 

W semantycznym wyszukiwaniu system rozpoznaje znaczenie zapytania i potrafi wyodrębnić poprawny dokument, który odpowiada na zapytanie użytkownika. 

Współczesne systemy wyszukiwania nie posługują się słowami kluczowymi, gdzie uwzględnia się tylko dosłowne wystąpienia słów czy fraz, a wykorzystują modele przestrzeni wektorowych (Word Embeddings, TF-IDF, LSA) i modele językowe (BERT, RoBERTa, LaBSE i inne) w celu uchwycenia semantyki tekstu. W niniejszym ćwiczeniu rozważone zostaną różnorodne metody osadzania tekstów na potrzeby wyszukiwania semantycznego. 

W praktyce stosuje się również podejścia hybrydowe, łączące klasyczne wyszukiwanie leksykalne (np. BM25) z wyszukiwaniem gęstym opartym na osadzeniach, ponieważ oba podejścia popełniają inne błędy i dobrze się uzupełniają. 

### Część I: Polyencoders (7 pkt) 

- A. Należy zapoznać się z biblioteką SentenceTransformers(htps://sbert.net/t <u>) oraz</u> poniższym opisem **[0.25 pkt]** : 

<u>https://sbert.net/examples/applications/retrieve_rerank/README.html</u> 

- B. Należy zapoznać się z podanym artykułem **[0.25 pkt]** : 

<u>https://arxiv.org/pdf/1905.01969</u> 

- C. Należy zapoznać się z metrykami oceny opisanymi w podanym artykule [0.25pkt]: 

<u>https://www.pinecone.io/learn/offline-evaluation/</u> 

Guideline dotyczący doboru indeksu wyszukiwania: <u>https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index</u> 

### Plan realizacji ćwiczenia 

Celem niniejszej części jest weryfikacja działania pre-trenowanych modeli bi-enkoder i cross-enkoder oraz klasycznego wyszukiwania leksykalnego (BM25) na zadanym zbiorze danych. Należy zaimplementować typowy potok wyszukiwania oparty na mechanizmie retrievera (w tym retrievera hybrydowego) i re-rankera. 

1. Należy zapoznać się z opisem typowego potoku wyszukiwania przedstawionym w następującym artykule [0.25 pkt] : <u>https://www.pinecone.io/learn/series/rag/rerankers/</u> 

2. Należy zapoznać się z niniejszym tutorialem i wykonać go samodzielnie w ramach własnego środowiska uruchomieniowego **[0.5 pkt]** : <u>https://huggingface.co/learn/nlp-course/chapter5/6</u> 

   - a. należy wykonać kilka przykładowych wyszukiwań posługując się kodem załączonym w tutorialu 

   - b. przeanalizować uzyskane rezultaty i podsumować je; przeanalizować błędy wyszukiwania 

3. Analiza działania modelu retriever (bi-enkoder) na wybranym zbiorze danych (np. `sentence-transformers/squad` dostępny w ramach platformy HuggingFace [2.5 pkt] <u>https://huggingface.co/datasets/sentence-transformers/squad)</u> 

   - a. sugeruje się wykorzystanie modelu `multi-qa-mpnet-base-dot-v1 (https://sbert.net/examples/applications/retrieve_re` <u>`rank/README.html)`</u> 

   - b. wykonać osadzenia zapytań i dokumentów (kolumny `question` i `answer` ) oraz przeprowadzić analizę przykładowych wyszukiwań, tym razem posługując się innym tutorialem: <u>https://sbert.net/examples/applications/semantic-search/README.html</u> 

   - c. przed indeksowaniem należy usunąć powtarzające się dokumenty (ten sam akapit występuje w zbiorze przy wielu pytaniach), 

   - d. należy dodać indeks wyszukiwania FAISS w celu dopasowania dokumentów do zapytań za pomocą osadzeń i ponownie wykonać wyszukiwanie (ustalić top_k = 5) 

   - e. zaimplementować wybraną metrykę oceny skuteczności wyszukiwania 

   - f. porównać wyniki uzyskane bez wykorzystania indeksu i wraz z wykorzystaniem indeksu FAISS. 

4. Wyszukiwanie hybrydowe [1.5 pkt] 

   - a. zaimplementować wyszukiwanie leksykalne BM25 na tym samym zbiorze dokumentów (np. biblioteka `bm25s` : https://github.com/xhluca/bm25s lub `rank_bm25` ) i ocenić je za pomocą ustalonych metryk oceny, 

   - b. połączyć wyniki BM25 i bi-enkodera za pomocą metody Reciprocal Rank Fusion (RRF): RRF(d) = Σ 1 / (k + rank_i(d)), gdzie rank_i(d) to pozycja dokumentu d na liście wyników i-tego systemu, a k = 60 <u>(https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf),</u> 

   - c. porównać skuteczność wyszukiwania BM25, bi-enkodera oraz wyszukiwania hybrydowego (top_k = 5) za pomocą ustalonych metryk oceny, 

   - d. przeanalizować przykładowe zapytania, dla których lepiej działa wyszukiwanie leksykalne (np. nazwy własne, liczby, rzadkie terminy), oraz takie, dla których lepiej działa wyszukiwanie gęste (np. parafrazy, synonimy), 

   - e. wyjaśnić, dlaczego RRF wykorzystuje pozycje dokumentów na listach wyników (rangi), a nie bezpośrednio wartości podobieństwa zwracane przez oba systemy. 

5. Analiza działania modelu re-ranker (cross-enkoder) [1.5 pkt] 

   - a. do potoku wyszukiwania hybrydowego zaimplementowanego w podpunkcie 4 należy dodać moduł re-rankingu 

   - b. ponownie wykonać wyszukiwania (top_k = 5) dla wybranych zapytań ze zbioru i dokonać re-rankingu za pomocą cross-enkodera (proszę wykorzystać jeden z pre-trenowanych modeli: <u>https://sbert.net/docs/pretrained-models/ce-msmarco.html)</u> 

   - c. ponownie ocenić skuteczność wyszukiwania za pomocą ustalonych metryk oceny i porównać z poprzednio uzyskanymi wynikami (BM25, bi-enkoder, wyszukiwanie hybrydowe). 

### Część II RAG (3 pkt) 

- A. Należy zapoznać się z podanym artykułem: <u>https://medium.com/@dinabavli/rag-basics-basic-implementation-of-retrievalaugmented-generation-rag-e80e0791159d</u> 

### Plan realizacji ćwiczenia 

1. Wykorzystać API REST z dostępem do modeli językowych CLARIN 

   - a. należy zarejestrować się, a następnie zalogować na platformie usług <u>https://services.clarin-pl.eu/login</u> 

   - b. endpoint z listą modeli: `/api/v1/oapi/models` 

   - c. endpoint do uruchomienia modelu: `/api/v1/oapi/chat/completions` 

2. Należy wybrać sobie jakiś zbiór danych (np. dane IMDB z recenzjami filmów <u>https://huggingface.co/datasets/stanfordnlp/imdb), zaindeksować dane i wykorzystać</u> w potoku przygotowanym w poprzedniej części ćwiczenia. 

   - a. Można również użyć gotowych indeksów FAISS do Wikipedii. Sugerowany indeks: 

<u>https://www.kaggle.com/datasets/jjinho/wikipedia-2023-07-faiss-index</u> 

   - b. można wykorzystać retriever hybrydowy przygotowany w Części I. 

3. Dodać do potoku moduł generacji odpowiedzi (RAG) wykorzystując podane API <u>https://python.langchain.com/docs/tutorials/rag/</u> 

   - a. przygotować odpowiedni prompt do modelu, który przekształci wyszukaną przez potok dokument na odpowiedź w języku naturalnym 

   - b. wykonać własne, przykładowe zapytania i przekierować wyniki wyszukiwania na zaimplementowany moduł RAG 

   - c. przeanalizować rezultaty wyszukiwania po zastosowaniu RAG 

d. podsumować uzyskane wyniki 


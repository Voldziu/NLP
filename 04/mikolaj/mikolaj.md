# Ściąga do obrony: Causal Tracing na gpt2-xl (Część I, pkt 1–2)

Kod: `04/utils/tracing/` (pętla, szum, grupy, analiza, wykresy), eksperymenty: `04/mikolaj/mikolaj.ipynb`, wykresy: `04/mikolaj/figures/`, surowe wyniki: `04/artifacts/mikolaj/*.csv` (czyta je też Dominik).

## 1. Causal tracing w trzech zdaniach

Causal tracing to analiza mediacji na sieci: szukamy, przez które stany ukryte płynie wpływ podmiotu na odpowiedź. Psujemy osadzenia podmiotu szumem ($\nu = 0{,}1346 \approx 3\sigma$), a potem w jednym miejscu (warstwa × pozycja × komponent) wstawiamy stan z przebiegu czystego i patrzymy, ile $P(\text{odpowiedź})$ wraca. $\text{TE} = P_{\text{clean}} - P_{\text{corr}}$ (ile podmiot w ogóle znaczy), $\text{IE} = P_{\text{przywr.}} - P_{\text{corr}}$ (ile odzyskuje ten jeden stan).

Wzory, nazwy komponentów i okna: sekcja „Notatki” na końcu.

## 2. Wyniki: Space Needle (odtworzenie notebooka #2 / Rys. 1 ROME)

clean 0,976 · corrupted 0,033 · TE 0,943

| strumień | szczyt | gdzie | znaczenie |
|---|---|---|---|
| `block_output` | 0,97 na `le` (warstwa 12–28), 0,97 na `downtown` (od ok. 32) | oba miejsca | cały stan: widać całą ścieżkę |
| `mlp_activation` | **0,90**, `le`, środek okna 17 | **wczesne miejsce** | MLP na ost. tokenie podmiotu wydobywa fakt |
| `attention_output` | **0,79**, `downtown`, środek 32 | **późne miejsce** | atencja przenosi fakt na ostatni token |

Przejście: po ok. 30 warstwie `le` traci znaczenie, a `downtown` zyskuje. `The`, `Space`, `Need` mają IE ≈ 0.

## 3. Wyniki: 5 własnych faktów

Wybór: odpowiedź to pierwszy token, model zna fakt (top-1), **TE ≥ 0,6**. Odrzucone: *Steve Jobs → Apple* (corrupted 0,79, model zgaduje z „co-founder of” bez podmiotu), *Microsoft → Gates* (TE 0,17), *Toyota → Japan* (TE < 0).

| fakt | clean / corr. | MLP na ost. tokenie podmiotu | atencja na ost. tokenie |
|---|---|---|---|
| Barack Obama → Hawaii | 0,976 / 0,014 | `Obama` 0,86, środek **0** | `of` 0,49, środek 31 |
| Brazil → Portuguese | 0,943 / 0,155 | `Brazil` 0,94, środek **4** | `is` 0,61, środek 31 |
| Mount Fuji → Japan | 0,727 / 0,100 | `Fuji` 0,56, środek 9 | `in` 0,58, środek 31 |
| Albert Einstein → relativity | 0,657 / 0,033 | `Einstein` 0,84, środek 11 | `of` 0,67, środek 27 |
| Eiffel Tower → Paris | 0,934 / 0,002 | **brak** (max 0,12 na `el`) | `of` 0,24, środek 27 |

- **Późne miejsce uniwersalne**: atencja na ostatnim tokenie, środek okna 27–32 w każdym fakcie.
- **Wczesne miejsce zależy od podmiotu**: częste jednotokenowe nazwy (Obama, Brazil) mają je w pierwszych warstwach. Nazwy wielotokenowe trzeba najpierw złożyć, więc fakt pojawia się głębiej (Fuji 9, Einstein 11, Space Needle 17). Przy Einsteinie widać składanie: atencja na `Einstein` w warstwie ok. 1 daje 0,52.
- **Pułapka jednotokenowego podmiotu**: `Brazil`, blok w warstwie 0 daje 0,94 = clean, bo przywrócenie jedynego zaszumionego tokenu cofa całe zaszumienie. Informacja leży w tym, **do kiedy** `Brazil` wystarcza (do ok. 20–28 warstwy).
- **Brazil, wysokie corrupted (0,155)**: „official language of … is” samo zawęża odpowiedź do języków.

## 4. Eiffel Tower: przywracanie grup tokenów (`trace_groups`)

| grupa | blok | MLP | atencja |
|---|---|---|---|
| cały podmiot | **0,925** | **0,913** | 0,003 |
| `el + Tower` | 0,41 | 0,51 | 0,003 |
| `E iff el` | 0,31 | 0,31 | 0,003 |
| `Tower` | 0,004 | 0,005 | 0,002 |

Kontrola, Space Needle: cały podmiot 0,983, sam `le` 0,970.

Informacja o Eiffel Tower jest **rozłożona na kilka tokenów** („Eiffel” i „Tower” potrzebne razem), więc tracing jednej pozycji jej nie znajduje. Fakt opuszcza podmiot wcześnie (ok. warstwy 13–20), a `of` przejmuje go dopiero od ok. 30. Pomiędzy jest rozproszony po kilku tokenach „is located in the city”. Wniosek: standardowy tracing zakłada jeden token niosący fakt, co jest prawdą średnio, ale nie zawsze.

## 5. Decyzje i uzasadnienia

| decyzja | powód |
|---|---|
| szum 0,1346 (stała z notebooka #2) | porównywalność z oryginałem; 3σ osadzeń `wte` gpt2-xl = 0,1447, prawie to samo |
| szum na `block_input` warstwy 0 | psujemy przyczynę (podmiot), zanim model coś z niej policzy, czyli „zmieniamy X” |
| szum zamiast podmiany podmiotu | podmiana wprowadza inny fakt (przełączenie), szum tylko usuwa informację |
| okno 10 warstw dla MLP i atencji | pojedynczy wkład podmodułu jest mały, a fakt rozłożony na warstwy |
| najpierw gpt2 small | sprawdzenie API w sekundach. Wynik: brak wczesnego miejsca, P(Seattle) = 0,08, mały model nie zna faktu |
| kryterium TE przy wyborze faktów | tracing mierzy odzyskanie utraconego P; bez TE nie ma czego lokalizować |
| cache wyników w CSV | gpt2-xl liczy kilka minut na fakt; Dominik czyta te same pliki |

## 6. Survey KE (I.1) w skrócie

- **Zadanie**: $(s, r, o) \to (s, r, o^*)$. Model po edycji ma dawać $o^*$ dla wszystkich sformułowań $\mathcal{X}_e$, a poza zakresem zachowywać się jak przed edycją: $\min \mathbb{E}\,\mathcal{L}(f^*(x), y^*)$ przy $f^*(x) = f(x)$ dla $x \notin \mathcal{X}_\mathcal{E}$.
- **6 metryk**: **Accuracy** (działa na promptach edycji), **Locality** (nie zmienia nic poza zakresem), **Generality** (działa na parafrazach), **Portability** (przenosi się na wnioski, np. relację odwrotną), **Retainability** (kolejne edycje nie kasują poprzednich), **Scalability** (wiele edycji naraz).
- **Taksonomia**: External Memorization (SERAC, GRACE, T-Patcher), Global Optimization (constrained fine-tuning, hipersieci), **Local Modification (ROME, MEMIT)**: zlokalizuj i edytuj wagi MLP.
- Zbiory: zsRE, CounterFact (z ROME).
- Causal tracing to krok „zlokalizuj” w Local Modification. ROME edytuje MLP z wczesnego miejsca jako pamięć klucz → wartość (aktualizacja rzędu 1).

## 7. Pytania pomocnicze (instrukcja)

**Na czym polega Causal Tracing?** Trzy przebiegi (czysty, zaszumiony, zaszumiony z przywróceniem jednego stanu), TE i IE, mapa IE warstwa × pozycja dla bloku, MLP i atencji. Lokalizuje, gdzie model przechowuje i przenosi fakt.

**Które tokeny podmiotu mają największy wpływ?** **Ostatni token podmiotu** (`le`, `Obama`, `Einstein`, `Fuji`). Atencja jest przyczynowa, więc dopiero on „widzi” całą nazwę. Wyjątki: `ack` w „Barack” (0,49) i Eiffel Tower (potrzebny cały podmiot).

**Czemu blok, MLP i atencja dają różne wzorce?** Blok to cały stan, suma wszystkiego, więc widać oba miejsca. MLP działa na jednym tokenie i jest pamięcią, więc jasne pole jest na podmiocie w środkowych warstwach. Atencja to jedyny transport między pozycjami, więc jasne pole jest na ostatnim tokenie w późnych warstwach. Razem: **MLP wydobywa fakt na podmiocie, atencja przenosi go na koniec.**

## 8. Pytania prowadzącego

**Czemu szum, a nie podmiana podmiotu?** Podmiana („Eiffel Tower” zamiast „Space Needle”) wprowadza inny fakt i mierzy przełączenie odpowiedzi. Szum tylko niszczy informację, zachowując długość i pozycje.

**Przywrócenie ostatniego tokenu w ostatniej warstwie daje clean: odkrycie?** Nie. Ten wektor wprost wyznacza rozkład następnego tokenu. Liczy się wczesne miejsce na podmiocie.

**Co jeśli TE ≈ 0?** Nie ma czego lokalizować: model nie zna faktu albo odpowiada z kontekstu (Steve Jobs). Wykres byłby szumem.

**Fakt jest „w dwóch miejscach”?** To jeden fakt na dwóch etapach przepływu: w warstwie 15 jest na `le`, w 40 już na `downtown`.

**Czemu okno 10 warstw?** Przywrócenie jednego wkładu MLP lub atencji przy reszcie zaszumionej daje prawie 0, bo pojedynczy wkład jest mały, a fakt powstaje w kilku warstwach. Na osi X jest środek okna $[l-5, l+4]$. Na brzegach okno jest krótsze.

**Czemu szum na warstwie 0, nie wyżej?** Chcemy zepsuć przyczynę przed jakimkolwiek przetwarzaniem. Szum wyżej zostawiłby czyste warstwy poniżej.

**Czemu atencja na podmiocie nic nie daje?** Na podmiocie fakt robi MLP. Atencja tam co najwyżej składa nazwę. Przywrócona sama atencja nie naprawi zaszumionego wejścia i MLP.

**Interchange intervention (notebook #1) a causal tracing?** Ten sam mechanizm (`VanillaIntervention`). Interchange podmienia stan z innego zdania (Spain ↔ Italy) i sprawdza przełączenie odpowiedzi. Tracing przywraca stan z tego samego zdania bez szumu i sprawdza odzyskanie.

**Czemu szum to `ConstantSourceIntervention`, a przywrócenie `VanillaIntervention`?** Szum nie potrzebuje źródła (dodaje wektor do bazy). Przywrócenie kopiuje wartość ze źródła, czyli przebiegu czystego.

**Czemu gpt2 small nie pokazuje wczesnego miejsca?** Słabo zna fakt (P = 0,08). Odpowiedź powstaje dopiero na ostatnim tokenie jako zgadywanie „dużego miasta w USA” z kontekstu.

**Czy stała 0,1346 pasuje do każdego modelu?** Nie, to 3σ osadzeń gpt2-xl. Dla gpt2 small 3σ ≈ 0,43. Przy 0,1346 szum jest tam za słaby.

**Czemu Obama ma wczesne miejsce w warstwach 0–4, a Space Needle w 12–21?** „ Obama” to jeden częsty token, który od razu wskazuje encję. „Space Need le” trzeba najpierw złożyć atencją.

**Czy Eiffel podważa ROME?** Nie. ROME pokazuje średnie po setkach faktów. Eiffel pokazuje ograniczenie tracingu jednej pozycji i to, że edycja tylko na ostatnim tokenie podmiotu może nie wystarczyć, gdy informacja jest rozproszona.

**Czym causal tracing różni się od ROME?** Tracing tylko **lokalizuje** (nic nie zmienia). ROME na tej podstawie **edytuje** wagi $W_{out}$ MLP z wczesnego miejsca.

**Jak ocenić, czy edycja się udała?** Metryki z surveyu: accuracy, generality (parafrazy), locality (inne fakty bez zmian), portability (wnioski), retainability i scalability (wiele edycji).

## 9. Kod

- `utils/tracing/core.py`: `trace_stream`, `trace_all_streams`, `TraceResult` (zapis i odczyt CSV/JSON), `run_clean`, `run_corrupted`.
- `utils/tracing/interventions.py`: `make_noise_intervention`, `restore_config`, okna.
- `utils/tracing/groups.py`: `trace_groups`, `groups_summary` (przywracanie zbioru pozycji naraz).
- `utils/tracing/analysis.py`: `position_summary`, `streams_summary`, `peak`.
- `utils/tracing/plot.py`: `plot_trace_streams`, `plot_trace_grid`, `plot_groups`.

# Notatki

## Causal tracing krok po kroku (terminy transformerowe)

### 1. Stan ukryty i strumień rezydualny

Dla każdej pozycji tokenu $i$ model trzyma wektor $h_i^{(l)} \in \mathbb{R}^{1600}$ (gpt2-xl): **stan ukryty** (*hidden state*) po warstwie $l$. Ciąg $h_i^{(0)}, h_i^{(1)}, \dots, h_i^{(48)}$ to **strumień rezydualny** (*residual stream*) pozycji $i$.

Wejście do warstwy 0 to osadzenie tokenu plus osadzenie pozycji:

$$h_i^{(0)} = W_E[x_i] + W_P[i]$$

### 2. Równania jednej warstwy (bloku) GPT-2

Blok $l$ ma dwie podwarstwy z połączeniami rezydualnymi:

$$a_i^{(l)} = \mathrm{Attn}^{(l)}\big(\mathrm{LN}(h_1^{(l-1)}), \dots, \mathrm{LN}(h_i^{(l-1)})\big)$$
$$\tilde h_i^{(l)} = h_i^{(l-1)} + a_i^{(l)}$$
$$m_i^{(l)} = W_{out}^{(l)}\, \sigma\big(W_{in}^{(l)}\, \mathrm{LN}(\tilde h_i^{(l)})\big)$$
$$h_i^{(l)} = \tilde h_i^{(l)} + m_i^{(l)}$$

- $a_i^{(l)}$ to wyjście atencji. Zależy od **wszystkich pozycji $\le i$** (atencja przyczynowa) i jest jedynym mechanizmem przenoszenia informacji między pozycjami.
- $m_i^{(l)}$ to wyjście MLP. Zależy **tylko od pozycji $i$**, bo MLP działa na każdym tokenie osobno.
- Oba wkłady są **dodawane** do strumienia rezydualnego.

Po 48 warstwach model liczy logity następnego tokenu z $h_{\text{last}}^{(48)}$ (LN + unembedding).

### 3. Nazwy w pyvene

| nazwa w pyvene | symbol | wymiar (gpt2-xl) |
|---|---|---|
| `block_input` warstwy $l$ | $h_i^{(l-1)}$ (wejście bloku); dla warstwy 0 to osadzenia $h_i^{(0)}$ | 1600 |
| `attention_output` warstwy $l$ | $a_i^{(l)}$ | 1600 |
| `mlp_activation` warstwy $l$ | $\sigma(W_{in}\,\mathrm{LN}(\tilde h_i^{(l)}))$: aktywacje neuronów MLP **przed** $W_{out}$ | 6400 |
| `block_output` warstwy $l$ | $h_i^{(l)}$ (wyjście bloku) | 1600 |

`mlp_activation` i $m_i^{(l)}$ niosą tę samą informację: $m = W_{out} \cdot \text{aktywacje}$, a $W_{out}$ jest stałe. Przywrócenie aktywacji jest więc równoważne przywróceniu $m_i^{(l)}$.

### 4. Zaszumienie (`block_input`, warstwa 0)

Dla pozycji podmiotu $i \in S$ (`The`, ` Space`, ` Need`, `le`):

$$h_i^{(0)} \leftarrow h_i^{(0)} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \nu^2 I),\ \nu = 0{,}1346$$

To jedyna interwencja na `block_input` i służy wyłącznie do zepsucia wejścia, nie do pomiaru. Wszystkie dalsze wartości liczą się z zaszumionych osadzeń. Oznaczamy je gwiazdką: $h_i^{*(l)}$, $a_i^{*(l)}$, $m_i^{*(l)}$.

### 5. Trzy rodzaje przywrócenia

W każdym przypadku robimy przebieg zaszumiony, w którym jedną wielkość zastępujemy jej wartością z przebiegu czystego, liczymy dalej normalnie i mierzymy $P(\text{Seattle})$.

**(a) `block_output`, pozycja $i$, warstwa $l$:** $h_i^{*(l)} \leftarrow h_i^{(l)}$

Podmieniamy **cały stan ukryty**. $h_i^{(l)}$ jest sumą osadzenia i wszystkich wkładów $a$ i $m$ z warstw $1..l$, więc przywracamy wszystko, co pozycja $i$ zawierała po warstwie $l$. Pytanie: **czy w $h_i^{(l)}$ jest informacja wystarczająca do odtworzenia faktu?**

**(b) `mlp_activation`, pozycja $i$, warstwy $l \in W$:** $m_i^{*(l)} \leftarrow m_i^{(l)}$ dla $l \in W$

Podmieniamy **tylko wkład MLP**. Reszta strumienia ($h^{(0)}$ i wkłady atencji) dalej jest zaszumiona. Pytanie: **czy wkład MLP na tej pozycji w tych warstwach wystarcza do odtworzenia faktu?** Jeśli tak, to MLP **wytwarza** tę informację.

**(c) `attention_output`, pozycja $i$, warstwy $l \in W$:** $a_i^{*(l)} \leftarrow a_i^{(l)}$ dla $l \in W$

Podmieniamy **tylko wkład atencji**. Ponieważ $a_i$ zależy od innych pozycji, przywrócenie go na pozycji $i$ dostarcza do $i$ poprawną informację **przeniesioną** z innych pozycji. Pytanie: **czy tu odbywa się przeniesienie faktu między pozycjami?**

### 6. Okno $W$

Dla `block_output` przywracamy **jedną** warstwę: $W = \{l\}$. To wystarcza, bo $h_i^{(l)}$ zawiera skumulowaną sumę wszystkich wcześniejszych wkładów.

Dla MLP i atencji pojedynczy wkład $m_i^{(l)}$ lub $a_i^{(l)}$ to tylko jeden składnik tej sumy, mały w porównaniu z całym $h_i^{(l)}$. Przywrócenie jednego składnika przy reszcie zaszumionej daje zwykle efekt bliski zera. Dlatego ROME przywraca **10 kolejnych warstw jednocześnie**:

$$W(l) = \{\,l-5, \dots, l+4\,\} \cap \{0, \dots, 47\}$$

Na osi X wykresów MLP i atencji wartość $l$ to **środek** tego zbioru. Każda kolumna to inny zbiór $W(l)$, przesuwany o jedną warstwę. Zbiory sąsiednich kolumn nachodzą na siebie, a przy brzegach są krótsze ($W(0) = \{0..4\}$, $W(47) = \{42..47\}$).

Przykład: `mlp_activation`, wiersz `le`, kolumna 17. Przywrócono $m_{\text{le}}^{(l)}$ dla $l = 12, \dots, 21$ i wyszło $P(\text{Seattle}) = 0{,}86$.

### 7. Miara

$$\mathrm{TE} = P_{\text{czyste}}(o) - P_{\text{zaszumione}}(o) \qquad \mathrm{IE} = P_{\text{przywrócone}}(o) - P_{\text{zaszumione}}(o)$$

Heatmapa pokazuje $P_{\text{przywrócone}}$ (albo IE) dla każdej pary (pozycja, warstwa lub środek okna).

### 8. Wynik dla Space Needle (gpt2-xl)

- **(a) `block_output`**: wysokie dla $h_{\text{le}}^{(l)}$ przy $l \approx 12\text{–}30$ i dla $h_{\text{downtown}}^{(l)}$ przy $l \gtrsim 30$. Informacja jest najpierw w stanie ostatniego tokenu podmiotu, potem w stanie ostatniego tokenu.
- **(b) MLP**: wysokie tylko dla `le`, środek okna $\approx 12\text{–}20$. Wkłady MLP na ostatnim tokenie podmiotu w środkowych warstwach **wytwarzają** informację o fakcie. To *early site* z ROME.
- **(c) atencja**: wysokie tylko dla `downtown`, środek okna $\approx 27\text{–}36$. Atencja w późnych warstwach **przenosi** informację z `le` na ostatni token. To *late site*.

ROME edytuje potem wagi $W_{out}^{(l)}$ MLP z *early site*. Sam causal tracing niczego nie edytuje, tylko lokalizuje.

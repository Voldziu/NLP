# Wytyczne anotacji v1 (PolEmo – opinie)

## Zasady ogólne
- Anotujesz **samodzielnie**. Nie konsultuj etykiet z innymi anotatorami do końca iteracji.
- Czytaj cały tekst, zanim cokolwiek zaznaczysz.
- Oceniasz **opinię autora**, nie własne zdanie o produkcie, lekarzu czy hotelu.
- Przy wątpliwości wybierz etykietę, która najlepiej oddaje **ogólne wrażenie** autora. Zapisz numer tekstu i krótką uwagę do omówienia po iteracji.
- Nie szukaj w internecie ani nie sprawdzaj oryginalnych etykiet korpusu.

## Zadanie 1: wydźwięk całego tekstu

| Etykieta | Kiedy |
|---|---|
| `plus_m` | Autor ogólnie chwali lub poleca. Brak istotnej krytyki. |
| `minus_m` | Autor ogólnie krytykuje lub odradza. Brak istotnej pochwały. |
| `zero` | Tekst opisuje fakty lub nie wyraża oceny. |
| `amb` | Są wyraźne pozytywy **i** negatywy, żadne nie dominuje. |

Przykłady:
- „Pokój czysty, obsługa miła, na pewno wrócimy." → `plus_m`
- „Brud, hałas, nie polecam nikomu." → `minus_m`
- „Hotel ma 50 pokoi i basen." → `zero`
- „Jedzenie świetne, ale obsługa fatalna." → `amb`

Przypadki graniczne:
- Jedna drobna uwaga w pochwalnej opinii („trochę drogo, ale wszystko super") → `plus_m`.
- Ironia: oceniaj intencję autora, nie dosłowne znaczenie.
- Pytania i ogólniki bez oceny → `zero`.

## Zadanie 2: fragmenty tekstu

Zaznaczaj **całe słowa**, bez interpunkcji. Zaznaczaj najkrótszy sensowny fragment. Zaznaczenia się nie nakładają.

### Aspekty
Aspekt to nazwa ocenianego elementu (np. obsługa, pokój, doktor, cena, jedzenie). Zaznacz **tylko nazwę**, nie przymiotnik ani całe zdanie.

| Etykieta | Kiedy |
|---|---|
| `aspekt_plus` | Element oceniony pozytywnie. |
| `aspekt_minus` | Element oceniony negatywnie. |
| `aspekt_neutralny` | Element wspomniany bez oceny. |

### Modyfikatory
Słowo zmieniające siłę lub kierunek oceny. Zaznacz je **osobno** od aspektu.

| Etykieta | Kiedy | Przykłady |
|---|---|---|
| `wzmocnienie` | Zwiększa siłę oceny. | bardzo, wyjątkowo, totalnie |
| `oslabienie` | Zmniejsza siłę oceny. | trochę, raczej, dość |
| `negacja` | Odwraca polaryzację. | nie, bez, żadnej |

Przykład: „**Obsługa** była **bardzo** miła, a **pokój** **nie** był czysty."
- Obsługa → `aspekt_plus`
- bardzo → `wzmocnienie`
- pokój → `aspekt_minus`
- nie → `negacja`

Zasady:
- Aspekt zaznaczaj tylko wtedy, gdy autor go jawnie ocenia lub wymienia. Nie domyślaj się.
- Polaryzację aspektu ustalaj **po uwzględnieniu negacji** („nie był czysty" → `aspekt_minus`).
- Jeśli ten sam aspekt występuje wielokrotnie, zaznacz każde wystąpienie.
- Zaimków („on", „to") nie zaznaczamy jako aspektów.

## Procedura
1. Otwórz swój projekt w Label Studio i anotuj teksty po kolei.
2. Dla każdego tekstu: najpierw fragmenty, potem etykieta całego tekstu.
3. Zapisz uwagi o trudnych przypadkach.
4. Po zakończeniu wyeksportuj dane (JSON) i przekaż je do liczenia zgodności.
5. Po obliczeniu zgodności spotykamy się, omawiamy rozbieżności i tworzymy wytyczne v2.

# Wytyczne anotacji v1 (PolEmo – opinie)

## Zasady ogólne
- Anotujesz **samodzielnie**. Nie konsultuj etykiet z innymi anotatorami do końca iteracji.
- Czytaj cały tekst, zanim cokolwiek zaznaczysz.
- Oceniasz **opinię autora**, nie własne zdanie o produkcie, lekarzu czy hotelu.
- Przy wątpliwości wybierz etykietę, która najlepiej oddaje **ogólne wrażenie** autora. Zapisz numer tekstu i krótką uwagę do omówienia po iteracji.
- Nie szukaj w internecie ani nie sprawdzaj oryginalnych etykiet korpusu.
- Nie staraj się na siłę wybierać `plus_m` lub `minus_m`, jeśli tekst nie wyraża wyraźnej opinii.

## Zadanie 1: wydźwięk całego tekstu

| Etykieta | Kiedy |
|---|---|
| `plus_m` | Autor ogólnie chwali lub poleca. Brak istotnej krytyki. |
| `minus_m` | Autor ogólnie krytykuje lub odradza. Brak istotnej pochwały. |
| `zero` | Tekst opisuje fakty lub nie wyraża oceny. |
| `amb` | Są wyraźne pozytywy **i** negatywy, żaden nie dominuje. |

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

### Modyfikatory
Słowo zmieniające siłę lub kierunek oceny. Zaznacz je **osobno** od aspektu.

| Etykieta | Kiedy | Przykłady |
|---|---|---|
| `wzmocnienie` | Zwiększa siłę oceny. | bardzo, wyjątkowo, totalnie |
| `oslabienie` | Zmniejsza siłę oceny. | trochę, raczej, dość |
| `negacja` | Odwraca polaryzację. | nie, bez, żadnej |

Przykład: „Obsługa była **bardzo** miła, jedzenie **nawet** dobre, ale pokój  **nie** był czysty."
- bardzo → `wzmocnienie`
- nie → `negacja`
- nawet → `oslabienie`



## Procedura
1. Otwórz swój projekt w Label Studio i anotuj teksty po kolei.
2. Dla każdego tekstu: najpierw fragmenty, potem etykieta całego tekstu.
3. Zapisz uwagi o trudnych przypadkach.
4. Po zakończeniu wyeksportuj dane (JSON) i przekaż je do liczenia zgodności.
5. Po obliczeniu zgodności spotykamy się, omawiamy rozbieżności i tworzymy wytyczne v2.

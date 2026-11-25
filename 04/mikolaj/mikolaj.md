# Mikołaj: Causal Tracing na gpt2-xl (Część I, pkt 1–2)

Zakres (3 pkt): Część I.1–3 (lektury) + plan realizacji pkt 1–2 z [instrukcji](../Zadanie_4.md).
**Nie zależysz od nikogo.** Cały kod tracingu jest w [`../utils/tracing/`](../utils/tracing/), a notebook tylko go wywołuje.

Po skończeniu ten plik zamieniamy w ściągę do obrony (jak `03/mikolaj/mikolaj.md`).

## Do zrobienia

1. **I.1: survey KE** ([arXiv 2310.16218](https://arxiv.org/pdf/2310.16218)) **[0.5 pkt]**. Notatka w notebooku: definicja zadania edycji wiedzy (krotka *(s, r, o) → (s, r, o\*)*), metryki: **accuracy / edit success**, **locality** (specificity), **generality** (generalization, parafrazy), **portability**, **retainability**, **scalability**.
2. **I.2: ROME** ([arXiv 2202.05262](https://arxiv.org/abs/2202.05262)) **[0.5 pkt]**. Notatka: causal tracing (clean / corrupted / corrupted-with-restoration), efekt pośredni (IE), obserwacja „MLP w środkowych warstwach na ostatnim tokenie podmiotu” i „atencja na ostatnim tokenie”, oraz jak ROME edytuje MLP jako pamięć klucz–wartość (rank-one update).
3. **I.3: pyvene 101** **[0.5 pkt]**. Notatka: `IntervenableConfig`, `RepresentationConfig(layer, component)`, `IntervenableModel(config, model)`, wywołanie `intervenable(base, sources, unit_locations)`, typy interwencji (`VanillaIntervention`, `ConstantSourceIntervention`, `CollectIntervention`). Notebook #1 (`Basic_Intervention.ipynb`): przejrzeć i opisać komponenty.
4. **Plan pkt 1: odtworzenie notebooka #2** (Space Needle → Seattle) na gpt2-xl. Wywołanie `trace_all_streams(...)`, porównanie wykresów z Rys. 1 w ROME.
5. **Plan pkt 2: 5 własnych faktów** **[1.5 pkt]**. Faktograficzne, różne relacje, odpowiedź jako **pierwszy token** po prompcie, zaszumiony **tylko podmiot**. Najpierw `top_next_tokens(...)`: model musi znać fakt (odpowiedź w top-3, sensowne P), inaczej wykres nic nie pokaże.
6. **Pytania pomocnicze** w notebooku:
   - Na czym polega Causal Tracing?
   - Które tokeny podmiotu mają największy wpływ na predykcję? (`position_summary`)
   - Dlaczego wzorce dla `attention_output`, `block_output` i `mlp_activation` się różnią? (`streams_summary`)
7. Zapis wyników: CSV/JSON automatycznie do `artifacts/mikolaj/`, końcowe PNG do `mikolaj/figures/`. **Dominik czyta Twoje CSV** do porównania z BoolQ, więc nie zmieniaj nazw po zapisaniu (`<fakt>__<stream>`).

## Jak używać `utils/tracing`

```python
model, tok = load_gpt2_xl()
text = "The Space Needle is in downtown"
subject = find_token_positions(tok, text, "The Space Needle")   # [0, 1, 2, 3]
results = trace_all_streams(model, tok, text, subject, next_token_prob(tok, " Seattle"),
                            out_dir=artifacts_dir("mikolaj"), name="space_needle")
plot_trace_streams(results, save_path="figures/space_needle.png")
streams_summary(results)                    # tabela: clean, corrupted, szczyt IE, warstwa
position_summary(results["mlp_activation"]) # które tokeny niosą fakt
```

- Wynik jest cache'owany: drugie uruchomienie wczytuje CSV (`overwrite=True` liczy od nowa).
- Koszt: ok. 48 warstw × liczba tokenów × 3 strumienie przebiegów gpt2-xl. Dla 7–10 tokenów to kilka minut na fakt (MPS).
- Poziom szumu: `compute_noise_level(model)` (3 × std osadzeń) albo stała z notebooka `NOTEBOOK_NOISE_LEVEL_GPT2_XL = 0.1346` (3 × std osadzeń podmiotów, policzona przez autorów). Do reprodukcji użyj stałej, a w notatce zapisz, ile wychodzi z `compute_noise_level`.

## Propozycje faktów (do wyboru / zmiany)

| Prompt | Podmiot | Odpowiedź |
|---|---|---|
| `The Eiffel Tower is located in the city of` | `The Eiffel Tower` | ` Paris` |
| `Barack Obama was born in the state of` | `Barack Obama` | ` Hawaii` |
| `The official language of Brazil is` | `Brazil` | ` Portuguese` |
| `Steve Jobs was the co-founder of` | `Steve Jobs` | ` Apple` |
| `Mount Everest is part of the mountain range called the` | `Mount Everest` | ` Himal` (pierwszy token!) |
| `Lionel Messi plays the sport of` | `Lionel Messi` | ` football` / ` soccer` |

## Kod

- `utils/tracing/core.py`: pętla tracingu, `TraceResult`.
- `utils/tracing/interventions.py`: szum i konfiguracje pyvene (opis metody w docstringu).
- `utils/tracing/analysis.py`: `position_summary`, `streams_summary`, `peak`.
- `utils/tracing/plot.py`: wykresy w stylu ROME.

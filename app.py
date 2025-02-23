import streamlit as st
import pandas as pd
import numpy as np
from analysis_IND_COL import (
    klasyfikuj_tekst,
    eng_ind_embeddings, eng_col_embeddings,
    pol_ind_embeddings, pol_col_embeddings,
    jap_ind_embeddings, jap_col_embeddings,
    generate_interactive_pca_2d,
    generate_interactive_tsne_2d,
    generate_interactive_pca_3d,
    generate_interactive_tsne_3d,
    get_embedding,
    EMBEDDING_MODEL,
    train_ml_classifier, ml_klasyfikuj_tekst, get_ml_classifier,
    generate_statistical_report,
    generate_interactive_distribution_charts,
    INTEGRATED_REPORT
)

st.set_page_config(
    page_title="Analiza reprezentacji wektorowych emocji",
    layout="centered"
)


def run_streamlit_app():
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")
    st.markdown(INTEGRATED_REPORT)

    st.markdown("""
    ## Przykłady zdań użytych do tworzenia embeddingów

    ### English – Individualistic
    - *I confidently choose my own path in life.*
    - *I trust my instincts to guide my decisions.*

    ### English – Collectivistic
    - *We work together to overcome challenges.*
    - *Our strength lies in our unity.*

    ### Polish – Individualistic
    - *Z pewnością wybieram własną ścieżkę w życiu.*
    - *Ufam swoim instynktom przy podejmowaniu decyzji.*

    ### Polish – Collectivistic
    - *Razem pokonujemy wyzwania, wspierając się nawzajem.*
    - *Nasza siła tkwi w jedności i wzajemnym zaufaniu.*

    ### Japanese – Individualistic
    - *私は自信を持って自分の人生の道を選びます。*
    - *私は自分の本能を信じ, 決断の指針とします。*

    ### Japanese – Collectivistic
    - *私たちは協力して困難を乗り越えます。*
    - *私たちの強さは団結にあります。*
    """)

    # Łączenie wszystkich embeddingów i etykiet w jedną listę/array
    all_emb = np.concatenate([
        eng_ind_embeddings, eng_col_embeddings,
        pol_ind_embeddings, pol_col_embeddings,
        jap_ind_embeddings, jap_col_embeddings
    ], axis=0)
    all_lbl = (
            ["ENG_IND"] * len(eng_ind_embeddings) +
            ["ENG_COL"] * len(eng_col_embeddings) +
            ["POL_IND"] * len(pol_ind_embeddings) +
            ["POL_COL"] * len(pol_col_embeddings) +
            ["JAP_IND"] * len(jap_ind_embeddings) +
            ["JAP_COL"] * len(jap_col_embeddings)
    )

    # --- Wizualizacje 2D (PCA i t-SNE) ---
    st.subheader("Interaktywne wizualizacje 2D")
    st.write("Poniżej przedstawiono dwa wykresy 2D: PCA oraz t-SNE.")

    fig_2d_pca = generate_interactive_pca_2d(all_emb, all_lbl)
    fig_2d_tsne = generate_interactive_tsne_2d(all_emb, all_lbl)

    st.write("**Wizualizacja PCA (2D)**")
    st.plotly_chart(fig_2d_pca, use_container_width=True)

    st.write("**Wizualizacja t-SNE (2D)**")
    st.plotly_chart(fig_2d_tsne, use_container_width=True)

    # --- Wizualizacje 3D (PCA i t-SNE) ---
    st.subheader("Interaktywne wizualizacje 3D")
    st.write("""
    Poniżej znajdują się interaktywne wykresy 3D, które pozwalają lepiej zbadać strukturę danych.
    **Uwaga:** Filtrowanie klastrów odbywa się przez interaktywną legendę – kliknij na niej, aby ukrywać lub pokazywać grupy.
    """)
    fig_pca_3d = generate_interactive_pca_3d(all_emb, all_lbl)
    fig_tsne_3d = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_3d, use_container_width=True)
    st.plotly_chart(fig_tsne_3d, use_container_width=True)

    # --- Klasyfikacja nowego tekstu (metoda centroidów) ---
    st.subheader("Klasyfikacja nowego tekstu (metoda centroidów)")
    st.write(
        "Metoda centroidów polega na porównaniu embeddingu nowego tekstu z uśrednionymi wektorami każdej kategorii.")
    user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.")
    if st.button("Klasyfikuj (centroidy)"):
        results = klasyfikuj_tekst(user_text)
        st.write("**Ranking podobieństwa (im wyżej, tym bliżej):**")
        for cat, val in results:
            st.write(f"- {cat}: {val:.4f}")

    # --- Klasyfikacja nowego tekstu (uczenie maszynowe) ---
    st.subheader("Klasyfikacja nowego tekstu (uczenie maszynowe)")
    st.write(
        "Regresja logistyczna trenuje model na wszystkich embeddingach, aby automatycznie rozpoznawać kategorię nowego tekstu.")
    if st.button("Trenuj/Wczytaj klasyfikator ML i przetestuj"):
        clf = get_ml_classifier(all_emb, all_lbl)
        pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
        prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
        st.table(prob_df[["Etykieta", "Prawdopodobieństwo (%)"]])

    # --- Raport statystyczny ---
    st.subheader("Raport statystyczny")
    st.write(
        "Poniżej znajdują się wyniki testów statystycznych sprawdzających istotność różnic między językami oraz wewnątrz jednego języka (hipoteza H₁: różnicowanie kategorii IND vs. COL)."
    )
    report_text = generate_statistical_report()
    st.text_area("Raport statystyczny", report_text, height=300)

    # --- Wykresy rozkładu dystansów ---
    st.subheader("Wykresy rozkładu dystansów")
    st.write(
        "Wybierz metrykę oraz język, aby zobaczyć interaktywny wykres rozkładu dystansów między zdaniami IND i COL."
    )
    metric_options = ["Euklides", "Kosinus", "Manhattan"]
    language_options = ["ENG", "POL", "JAP"]
    selected_metric = st.selectbox("Wybierz metrykę", metric_options)
    selected_language = st.selectbox("Wybierz język", language_options)
    distribution_figures = generate_interactive_distribution_charts()
    fig_distribution = distribution_figures[selected_metric][selected_language]
    st.plotly_chart(fig_distribution, use_container_width=True)

    st.markdown("""
# Wnioski

1. **Skuteczne odróżnienie kategorii (H₁)**  
   Z przeprowadzonych analiz wynika, że model text-embedding-3-large potrafi w wyraźny sposób odróżnić zdania o zabarwieniu indywidualistycznym (IND) od tych o charakterze kolektywistycznym (COL). Co więcej, dzieje się to w trzech różnych językach – polskim, angielskim i japońskim.  
   Wysoka trafność rozróżniania potwierdzona została przez testy statystyczne wykorzystujące kilka miar odległości (Euklides, Kosinus oraz Manhattan). Ich rezultaty (p < 0.01) wskazują, że istnieje istotna różnica pomiędzy tymi kategoriami we wszystkich trzech językach.  

2. **Zróżnicowanie między językami (H₂/H₃)**  
   - **Język angielski**  
     W języku angielskim obserwujemy stosunkowo duże wartości mediany dla wszystkich miar (m.in. Euklides ~1.2480, Kosinus ~0.7788), co sugeruje mocne rozdzielenie zdań indywidualistycznych od kolektywistycznych. Taki wynik może odzwierciedlać wyraźne różnice kulturowe obecne w społeczeństwach anglosaskich.  

   - **Język japoński**  
     W kontekście języka japońskiego wartości mediany są nieco mniejsze (Euklides ~1.1835, Kosinus ~0.7004), ale nadal pokazują czytelny podział między zdaniami IND a COL. Jest to interesujące o tyle, że język japoński silnie osadzony jest w tradycji kolektywistycznej. Mimo tego, model wychwytuje subtelne różnice w zależności od postawy indywidualistycznej lub kolektywistycznej.  

   - **Język polski**  
     Polski odznacza się tutaj jeszcze mniejszymi różnicami (Euklides ~1.0858, Kosinus ~0.5894), co może wskazywać na większe podobieństwo pomiędzy zdaniami IND i COL. Niemniej jednak, różnice te wciąż pozostają istotne statystycznie, co świadczy o tym, że model skutecznie rozpoznaje rozmaite niuanse językowe także w tym obszarze.

# Dyskusja

## Matematyczno-algorytmiczne podejście do badania emocji kulturowych
Opisana metoda bazuje na wektorowych reprezentacjach tekstu (tzw. embeddingach), które można traktować jako punkty w wielowymiarowej przestrzeni. Takie rozwiązanie umożliwia:
- **Precyzyjne porównanie** różnych wypowiedzi lub zdań na podstawie ich „podobieństwa” geometrycznego bądź kierunkowego.  
- **Standaryzację wyników** w postaci liczb, co pozwala na obiektywną analizę i statystyczną ocenę różnic.  
- **Elastyczność** – w razie potrzeby łatwo rozbudować analizę o kolejne języki lub zastosować inne metody obliczeniowe.

Wykorzystanie embeddingów to podejście dalekie od typowo subiektywnych ocen eksperckich, dzięki czemu istnieje możliwość uzyskania bardziej spójnych, powtarzalnych i skalowalnych wyników. Z perspektywy interdyscyplinarnej stanowi to obiecujące narzędzie do badania zjawisk kulturowych i językowych w sposób możliwie wolny od uprzedzeń, wynikających z indywidualnych interpretacji.

## Szczegółowa analiza metryk

- **Euklides**  
  Metryka ta ujmuje dystans jako „rzeczywistą” odległość geometryczną w wielowymiarowej przestrzeni. Niższa mediana (np. 1.0858 dla polskiego) oznacza, że zdania IND i COL są bliżej siebie, zaś wyższa (jak w przypadku angielskiego, 1.2480) wskazuje na wyraźniejsze rozdzielenie tych kategorii.  

- **Kosinus (1 - cosinus)**  
  Ten wskaźnik skupia się na kącie pomiędzy wektorami. Jeśli dwie reprezentacje mają zbliżony kierunek, wartość metryki będzie niższa (jak w polskim, ~0.5894). Wyższe wyniki, jak dla języka angielskiego (0.7788), potwierdzają większe różnice w interpretacji zdań przez model.  

- **Manhattan**  
  W przeciwieństwie do Euklidesa, metryka ta sumuje różnice w każdej współrzędnej (tzw. taksówkowa odległość). Zbliżone wartości w polskim (46.7398) pokazują, że różnice między zdaniami nie są tak duże jak w języku angielskim (53.7099) czy japońskim (50.8755).

## Czynniki wpływające na wyniki

- **Kontekst kulturowy i cechy języka**  
  Każdy z badanych języków wyraża postawy indywidualistyczne i kolektywistyczne w nieco inny sposób. Angielski może bardziej jednoznacznie rozgraniczać te dwa nurty niż polski, którego konstrukcja językowa z bogatą fleksją bywa mniej kategoryczna.  
- **Wpływ tłumaczeń i data setów treningowych**  
  Model text-embedding-3-large został w dużej mierze wytrenowany na materiałach anglojęzycznych, co może przekładać się na lepszą (lub inaczej ukierunkowaną) detekcję różnic w języku angielskim niż w pozostałych.  
- **Semantyczna jednorodność zdań**  
  W wypadku polskich zdań, podobieństwo słownictwa i konstrukcji gramatycznych może skutkować mniejszym dystansem w przestrzeni embeddingów, nawet jeśli przekaz emocjonalny odbiega od siebie.

# Pozytywne aspekty i implikacje

- **Uniwersalny charakter analizy**  
  Wykorzystanie metod embeddingowych daje nadzieję na standaryzację i ujednolicenie badań nad językami o odmiennych strukturach gramatycznych i różnym podłożu kulturowym.  
- **Możliwość wielopoziomowej eksploracji**  
  Zastosowanie takich technik pozwala badaczom na porównywanie wyników za pomocą rozmaitych metryk (Euklides, Kosinus, Manhattan), co pomaga uchwycić różne aspekty podobieństwa między zdaniami.  
- **Stabilność statystyczna**  
  Bardzo niskie wartości p (p < 0.01) wskazują, że obserwowane różnice nie są przypadkowe. Dzięki temu wyniki są pewniejsze, a dalsze rozszerzenia badania mogą być prowadzone na mocnym fundamencie metodologicznym.

# Podsumowanie

- **Efektywność modelu**  
  Model text-embedding-3-large wyraźnie różnicuje zdania o charakterze indywidualistycznym i kolektywistycznym. Uzyskane wizualizacje (PCA, t-SNE) oraz wyniki kluczowych metryk potwierdzają, że takie matematyczno-algorytmiczne podejście skutecznie wychwytuje i odwzorowuje różnice w przekazie emocjonalnym.  

- **Różnice między językami**  
  Najmocniej odseparowane kategorie zaobserwowano w języku angielskim, co może wynikać z dominującego charakteru kultury indywidualistycznej w regionach anglosaskich. W japońskim i polskim, choć różnice również są widoczne, to jednak o nieco mniejszej skali, co może odzwierciedlać bardziej skomplikowany lub mniej jednoznaczny sposób wyrażania tych postaw.  

- **Znaczenie naukowe i praktyczne**  
  Opisana metoda pozwala na rzetelną, ilościową analizę postaw wyrażanych w różnych językach. Daje to narzędzia do nowatorskich badań w dziedzinach takich jak socjologia, lingwistyka, psychologia kulturowa czy komunikacja międzykulturowa. Ponadto otwiera drogę do dalszych pogłębionych badań, które mogłyby obejmować dodatkowe języki bądź rozszerzone zasoby tekstów – wszystko w oparciu o powtarzalne, obiektywne metody obliczeniowe.
    """)

    # Zapis raportu do pliku
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")


if __name__ == "__main__":
    run_streamlit_app()
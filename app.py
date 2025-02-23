import streamlit as st
import pandas as pd
import os
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
    generate_interactive_distribution_charts,  # nowa funkcja
    INTEGRATED_REPORT
)

st.set_page_config(
    page_title="Analiza reprezentacji wektorowych emocji",
    layout="centered"
)

def run_streamlit_app():
    """
    Główna funkcja uruchamiająca aplikację w Streamlit.
    Zawiera:
      1) Raport teoretyczny wyjaśniający kontekst i używane metody (INTEGRATED_REPORT).
      2) Przykładowe zdania z trzech języków: angielski, polski, japoński.
      3) Wizualizacje 2D i 3D (PCA, t-SNE) pozwalające zrozumieć, jak zdania o różnych cechach
         wyglądają w przestrzeni wektorowej. **Uwaga:** Aby filtrować klastry, kliknij na legendzie wykresu.
      4) Klasyfikację nowego tekstu:
         - Metoda centroidów (porównanie z uśrednionym wektorem każdej kategorii).
         - Model ML (regresja logistyczna), który "uczy się" rozpoznawać kategorie.
      5) Raport statystyczny oraz wnioski.
      6) Zapis raportu do pliku.
    """

    # 1. Tytuł aplikacji
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")

    # 2. Raport teoretyczny
    st.markdown(INTEGRATED_REPORT)

    # 3. Przykładowe zdania użyte do tworzenia embeddingów
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

    st.subheader("Interaktywne wizualizacje 2D")
    st.write("""
    **PCA** – metoda liniowa, przedstawia dane w 2D na podstawie głównych kierunków wariancji.
    
    **t-SNE** – metoda nieliniowa, zachowuje lokalne podobieństwa oryginalnych wektorów.
    
    **Uwaga:** Aby filtrować klastry, kliknij na legendzie wykresu.
    """)
    fig_pca_2d = generate_interactive_pca_2d(all_emb, all_lbl)
    fig_tsne_2d = generate_interactive_tsne_2d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_2d, use_container_width=True)
    st.plotly_chart(fig_tsne_2d, use_container_width=True)

    st.subheader("Interaktywne wizualizacje 3D")
    st.write("""
    Poniżej znajdują się interaktywne wykresy 3D, które pozwalają lepiej zbadać strukturę danych.
    **Uwaga:** Filtrowanie klastrów odbywa się przez interaktywną legendę – kliknij na niej, aby ukrywać lub pokazywać grupy.
    """)
    fig_pca_3d = generate_interactive_pca_3d(all_emb, all_lbl)
    fig_tsne_3d = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_3d, use_container_width=True)
    st.plotly_chart(fig_tsne_3d, use_container_width=True)

    st.subheader("Klasyfikacja nowego tekstu (metoda centroidów)")
    st.write("""
    Metoda centroidów polega na porównaniu embeddingu nowego tekstu z uśrednionymi wektorami (centroidami) każdej kategorii.
    """)
    user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.")
    if st.button("Klasyfikuj (centroidy)"):
        results = klasyfikuj_tekst(user_text)
        st.write("**Ranking podobieństwa (im wyżej, tym bliżej):**")
        for cat, val in results:
            st.write(f"- {cat}: {val:.4f}")

    st.subheader("Klasyfikacja nowego tekstu (uczenie maszynowe)")
    st.write("""
    Regresja logistyczna trenuje model na wszystkich embeddingach, aby automatycznie rozpoznawać kategorię nowego tekstu.
    """)
    if st.button("Trenuj/Wczytaj klasyfikator ML i przetestuj"):
        clf = get_ml_classifier(all_emb, all_lbl)
        pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
        prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
        st.table(prob_df[["Etykieta", "Prawdopodobieństwo (%)"]])

    st.subheader("Raport statystyczny")
    st.write("""
    Poniżej znajdują się wyniki testów statystycznych sprawdzających istotność różnic między językami oraz
    wewnątrz jednego języka (hipoteza H₁: różnicowanie kategorii IND vs. COL).
    """)
    report_text = generate_statistical_report()
    st.text_area("Raport statystyczny", report_text, height=300)
    
    st.subheader("Wykresy rozkładu dystansów")
    st.write("Wybierz metrykę oraz język, aby zobaczyć interaktywny wykres rozkładu dystansów między zdaniami IND i COL.")
    metric_options = ["Euklides", "Kosinus", "Manhattan"]
    language_options = ["ENG", "POL", "JAP"]
    selected_metric = st.selectbox("Wybierz metrykę", metric_options)
    selected_language = st.selectbox("Wybierz język", language_options)
    distribution_figures = generate_interactive_distribution_charts()
    fig_distribution = distribution_figures[selected_metric][selected_language]
    st.plotly_chart(fig_distribution, use_container_width=True)

    st.markdown("""
# Wnioski

1. **Skuteczne rozróżnienie kategorii (H₁):**  
   - Model `text-embedding-3-large` z powodzeniem oddziela zdania indywidualistyczne (IND) od kolektywistycznych (COL) w trzech analizowanych językach: polskim, angielskim i japońskim.  
   - Wyniki testów statystycznych, opierające się na metrykach takich jak **Euklides**, **Kosinus (1 - cosinus)** i **Manhattan**, wykazały, że różnice między embeddingami są statystycznie istotne (p < 0.01) dla wszystkich języków.

2. **Odkrycia dotyczące różnic między językami (H₂/H₃):**  
   - **Język angielski:**  
     - *Euklides:* median = 1.2480  
     - *Kosinus:* median = 0.7788  
     - *Manhattan:* median = 53.7099  
     
     Wyniki te wskazują na wyraźny kontrast między zdaniami IND a COL, co odpowiada silnemu rozróżnieniu kulturowemu obserwowanemu w społeczeństwach anglosaskich.
     
   - **Język japoński:**  
     - *Euklides:* median = 1.1835  
     - *Kosinus:* median = 0.7004  
     - *Manhattan:* median = 50.8755  
     
     Japońskie zdania, mimo tradycyjnie kolektywistycznego kontekstu, wykazują istotne różnice między kategoriami, co potwierdza zdolność modelu do wychwytywania niuansów kulturowych.
     
   - **Język polski:**  
     - *Euklides:* median = 1.0858  
     - *Kosinus:* median = 0.5894  
     - *Manhattan:* median = 46.7398  
     
     W języku polskim zdania IND i COL są reprezentowane jako najbardziej zbliżone, co wskazuje na większą spójność semantyczną między tymi kategoriami.

# Dyskusja

## Szczegółowa analiza metryk

- **Metryka Euklidesowa:**  
  Ta metryka mierzy bezpośrednią odległość geometryczną między dwoma punktami w przestrzeni wektorowej. Mniejsze wartości mediany, jak w przypadku języka polskiego (1.0858), sugerują, że zdania IND i COL są bardziej skoncentrowane i mniej rozproszone.

- **Metryka Kosinusowa (1 - cosinus):**  
  Metryka ta analizuje kąt między wektorami, co odzwierciedla podobieństwo kierunkowe. Niższe wartości (np. 0.5894 dla polskiego) oznaczają, że zdania są bardziej podobne semantycznie. Wyższe wartości, obserwowane w języku angielskim (0.7788), wskazują na większą dywergencję między kategoriami.

- **Metryka Manhattan:**  
  Mierzy sumę bezwzględnych różnic pomiędzy współrzędnymi wektorów. Niższa mediana, jak 46.7398 dla języka polskiego, potwierdza, że dystanse między zdaniami są mniejsze w porównaniu z językiem angielskim (53.7099) i japońskim (50.8755).

## Potencjalne przyczyny obserwowanych wyników

- **Różnice kulturowe i lingwistyczne:**  
  - W kulturze anglosaskiej postawy indywidualistyczne i kolektywistyczne są wyraźnie rozdzielane, co znajduje odzwierciedlenie w większych dystansach między embeddingami.  
  - Japoński kojarzony z kolektywizmem, posiada złożony system wyrażeń oraz form grzecznościowych, które mogą wprowadzać większą różnorodność semantyczną.
  - Polski, charakteryzujący się bogatą fleksją, może wyrażać te postawy w bardziej ujednolicony sposób, co skutkuje mniejszymi różnicami w reprezentacjach wektorowych.

- **Specyfika korpusu i danych treningowych:**  
  Model `text-embedding-3-large` został prawdopodobnie wytrenowany na dużym zbiorze danych anglojęzycznych, co może wpływać na sposób, w jaki reprezentuje teksty w innych językach. Różnice w jakości oraz ilości danych dostępnych w języku japońskim i polskim mogą dodatkowo modyfikować wyniki.

- **Efekty tłumaczenia i adaptacji:**  
  Proces tłumaczenia lub adaptacji zdań z jednego języka na inny może wprowadzać subtelne zmiany, które wpływają na spójność semantyczną. W efekcie zdania w języku polskim mogą być reprezentowane jako bardziej jednorodne.

## Pozytywne odkrycia i implikacje

- **Precyzyjne rozróżnienie semantyczne:**  
  Wyniki jednoznacznie pokazują, że metody oparte na embeddingach są bardzo skuteczne w wychwytywaniu różnic między postawami indywidualistycznymi a kolektywistycznymi. To otwiera nowe perspektywy w badaniach nad językiem i kulturą.

- **Wieloaspektowa analiza dzięki różnym metrykom:**  
  Zastosowanie trzech różnych metryk (Euklides, Kosinus, Manhattan) pozwala na kompleksową analizę, która potwierdza spójność wyników niezależnie od przyjętej miary. Każda z metryk uwypukla inny aspekt różnic:
  - **Euklides** – geograficzna separacja danych,
  - **Kosinus** – kątowe podobieństwo semantyczne,
  - **Manhattan** – sumaryczne różnice we współrzędnych.
  
- **Solidność wyników statystycznych:**  
  Niskie wartości p (p < 0.01) uzyskane w testach, takich jak Manna–Whitneya, potwierdzają, że zaobserwowane różnice między grupami nie są przypadkowe, co daje silne podstawy do dalszych badań.

- **Zastosowania praktyczne:**  
  Wyniki te mogą znaleźć zastosowanie w różnych dziedzinach, takich jak analiza treści, marketing, badania społeczne czy lingwistyka, umożliwiając dokładne badanie wpływu kultury na sposób wyrażania postaw.

# Podsumowanie

- **Efektywność modelu:**  
  Model `text-embedding-3-large` skutecznie różnicuje zdania indywidualistyczne i kolektywistyczne, co potwierdzają zarówno wizualizacje (PCA, t-SNE), jak i wyniki trzech różnych metryk odległościowych.

- **Różnice między językami:**  
  Największe różnice obserwujemy w języku angielskim, co odzwierciedla wyraźny kontrast między indywidualizmem a kolektywizmem. Japoński i polski prezentują mniejsze dystanse, przy czym język polski wykazuje największą spójność semantyczną między zdaniami IND i COL.

- **Implikacje badawcze:**  
  Badanie potwierdza, że reprezentacje wektorowe stanowią użyteczne narzędzie do ilościowej analizy różnic kulturowych. Precyzyjne wyodrębnienie i wizualizacja różnic semantycznych otwiera nowe możliwości w badaniach nad językiem i kulturą, a wyniki te mogą stanowić fundament do dalszych, pogłębionych analiz.
    """)

    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")

if __name__ == "__main__":
    run_streamlit_app()

######################################
# app.py
######################################
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
    INTEGRATED_REPORT
)

# Ustawienia układu strony Streamlit (wyśrodkowany widok)
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
    3) Wizualizacje 2D i 3D (PCA, t-SNE) pozwalające zrozumieć,
       jak zdania o różnych cechach wyglądają w przestrzeni wektorowej.
    4) Klasyfikację nowego tekstu:
       - Metoda centroidów (porównanie z uśrednionym wektorem każdej kategorii).
       - Model ML (regresja logistyczna), który "uczy się" rozpoznawać kategorie.
    5) Raport statystyczny oraz wnioski.
    6) Zapis raportu do pliku.
    """

    # 1. Tytuł aplikacji
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")

    # 2. Raport teoretyczny (obszerny tekst opisujący kontekst i teorię)
    st.markdown(INTEGRATED_REPORT)

    # 3. Przykładowe zdania użyte do tworzenia embeddingów (dla lepszego wglądu w dane)
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

    # 4. Łączenie wszystkich embeddingów w jeden zbiór
    #    (eng_ind_embeddings, eng_col_embeddings, pol_ind_embeddings, ...)
    all_emb = np.concatenate([
        eng_ind_embeddings, eng_col_embeddings,
        pol_ind_embeddings, pol_col_embeddings,
        jap_ind_embeddings, jap_col_embeddings
    ], axis=0)

    # Dla każdego embeddingu (wektora) mamy przypisaną etykietę (np. ENG_IND)
    all_lbl = (
        ["ENG_IND"] * len(eng_ind_embeddings) +
        ["ENG_COL"] * len(eng_col_embeddings) +
        ["POL_IND"] * len(pol_ind_embeddings) +
        ["POL_COL"] * len(pol_col_embeddings) +
        ["JAP_IND"] * len(jap_ind_embeddings) +
        ["JAP_COL"] * len(jap_col_embeddings)
    )

    # 5. Wizualizacje 2D (PCA, t-SNE)
    # Pozwalają "spłaszczyć" 3072 wymiary do 2, co pomaga zobaczyć ogólny układ zdań.
    st.subheader("Interaktywne wizualizacje 2D")
    st.write("""
    **PCA (Principal Component Analysis)** – szuka głównych kierunków
    największej wariancji w danych i przedstawia je w 2D.

    **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – dba o to,
    żeby zdania, które były blisko w oryginalnej (wysokowymiarowej) przestrzeni,
    również znajdowały się blisko w 2D.
    """)

    # Generowanie i wyświetlenie wykresów 2D
    fig_pca_2d = generate_interactive_pca_2d(all_emb, all_lbl)
    fig_tsne_2d = generate_interactive_tsne_2d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_2d, use_container_width=True)
    st.plotly_chart(fig_tsne_2d, use_container_width=True)

    # 6. Wizualizacje 3D (PCA, t-SNE)
    # Wersja trójwymiarowa, którą można obracać interaktywnie.
    st.subheader("Interaktywna wizualizacja 3D PCA")
    st.write("""
    Tutaj widzimy trzy wymiary PCA jednocześnie. 
    Możesz obracać wykres, aby lepiej dostrzec, czy zdania IND i COL w danym języku
    układają się blisko czy daleko.
    """)
    fig_pca_3d = generate_interactive_pca_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_3d, use_container_width=True)

    st.subheader("Interaktywna wizualizacja 3D t-SNE")
    st.write("""
    W t-SNE 3D także można dostrzec grupowanie zdań. 
    Jeżeli zdania w danej kategorii (np. ENG_IND) są w tym samym zakątku wykresu,
    oznacza to, że model uznaje je za podobne.
    """)
    fig_tsne_3d = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_tsne_3d, use_container_width=True)

    # 7. Klasyfikacja metodą centroidów
    st.subheader("Klasyfikacja nowego tekstu (metoda centroidów)")
    st.write("""
    **Jak działa metoda centroidów?**  
    1. Obliczamy uśredniony wektor (centroid) dla każdej kategorii (ENG_IND, ENG_COL itd.).  
    2. Nowy tekst również zamieniamy na wektor.  
    3. Sprawdzamy, do którego centroidu jest najbliżej (np. w sensie kosinusowym).  
    4. Kategoria z najwyższym podobieństwem to odpowiedź.
    """)

    # Pole tekstowe do wprowadzenia nowego zdania
    user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.")
    if st.button("Klasyfikuj (centroidy)"):
        results = klasyfikuj_tekst(user_text)
        st.write("**Ranking podobieństwa (im wyżej, tym bliżej):**")
        for cat, val in results:
            st.write(f"- {cat}: {val:.4f}")

    # 8. Klasyfikacja przy użyciu modelu ML (Regresja Logistyczna)
    st.subheader("Klasyfikacja nowego tekstu (uczenie maszynowe)")
    st.write("""
    **Regresja logistyczna**:
    - Jest trenowana na wszystkich embeddingach i wie, które zdania należą
      do której kategorii.
    - Dla nowego tekstu model przewiduje jedną z sześciu kategorii
      oraz podaje prawdopodobieństwo przypisania do każdej z nich.
    """)

    # Przycisk do wytrenowania/wczytania modelu ML i sklasyfikowania tekstu
    if st.button("Trenuj/Wczytaj klasyfikator ML i przetestuj"):
        clf = get_ml_classifier(all_emb, all_lbl)
        pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        
        # Wyświetlenie procentowego prawdopodobieństwa
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
        prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
        st.table(prob_df[["Etykieta", "Prawdopodobieństwo (%)"]])

    # 9. Raport statystyczny
    st.subheader("Raport statystyczny")
    st.write("""
    Poniżej prezentujemy wyniki testów statystycznych, które sprawdzają,
    czy różnice między językami (np. polski i angielski) są istotne (p < 0.01).
    Używamy różnych miar odległości (Euklides, Kosinus, Manhattan) i testów
    (Shapiro-Wilka, K-S do normalności, Manna–Whitneya / t-Studenta do porównań).
    """)
    report_text = generate_statistical_report()
    st.text_area("Raport statystyczny", report_text, height=300)

    # 10. Wnioski i interpretacja
    st.subheader("Wnioski")
    st.markdown("""
**Główne spostrzeżenia**:

1. **Polski vs. Angielski**  
   - Polskie zdania indywidualistyczne i kolektywistyczne są zwykle bliżej siebie
     niż ich angielskie odpowiedniki, co może wynikać z cech językowych i kulturowych.

2. **Język japoński**  
   - Wiele wskazuje na to, że japońskie zdania IND i COL również są bliżej siebie 
     niż w angielskim, lecz niektóre wyniki plasują się między polskim a angielskim.
     Może to mieć związek z wielkością i zawartością korpusu użytego do trenowania modelu.

3. **Testy statystyczne**  
   - Jeśli p < 0.01, różnice raczej nie są przypadkowe. Wyraźnie widać,
     że polskie i japońskie zdania IND–COL różnią się od angielskich.
    """)

    # 11. Podsumowanie
    st.subheader("Podsumowanie")
    st.write("""
Analiza sugeruje, że model `text-embedding-3-large` rozróżnia zdania
indywidualistyczne i kolektywistyczne w różnych językach w odmiennym stopniu.
Polskie zdania IND i COL są w przestrzeni wektorowej bliżej siebie
niż w angielskim, a japońskie wypadają pośrednio.
Te rezultaty – przy wsparciu testów statystycznych – wskazują na
różnice kulturowe, które da się wykazać przy użyciu metod matematycznych
i algorytmicznych. 
    """)

    # 12. Zapis raportu statystycznego do pliku
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")

# Uruchomienie aplikacji
if __name__ == "__main__":
    run_streamlit_app()

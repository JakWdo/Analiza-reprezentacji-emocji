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

    st.subheader("Wnioski")
    st.markdown("""
**Główne spostrzeżenia**:

1. **Polski vs. Angielski**  
   - Polskie zdania IND i COL są zazwyczaj bliżej siebie niż angielskie, co może wynikać z cech językowych.

2. **Język japoński**  
   - Wyniki wskazują, że japońskie zdania IND i COL plasują się pośrednio między polskimi a angielskimi.

3. **Testy statystyczne**  
   - Jeśli p < 0.01, różnice są statystycznie istotne.
    """)
    st.subheader("Podsumowanie")
    st.write("""
Analiza sugeruje, że model `text-embedding-3-large` różnicuje zdania indywidualistyczne i kolektywistyczne w różnych językach.
Polskie zdania IND i COL są w przestrzeni wektorowej bliżej siebie niż ich angielskie odpowiedniki,
a japońskie zdania plasują się pośrednio. Testy statystyczne (w tym dodatkowe testy H₁) potwierdzają, że te różnice nie są przypadkowe.
    """)
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")

if __name__ == "__main__":
    run_streamlit_app()

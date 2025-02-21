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
    generate_interactive_pca_3d,
    generate_interactive_tsne_3d,
    get_embedding,
    EMBEDDING_MODEL,
    train_ml_classifier, ml_klasyfikuj_tekst,
    generate_statistical_report,
    INTEGRATED_REPORT
)

# Ustawiam layout na "centered", aby aplikacja była wyśrodkowana
st.set_page_config(page_title="Analiza reprezentacji wektorowych emocji", layout="centered")


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

    st.subheader("Wizualizacje 2D")
    st.write("Prezentuję wykresy PCA i t-SNE, które pokazują relacje między zdaniami w przestrzeni 2D.")
    if os.path.exists("all_pca_centroids.png"):
        st.image("all_pca_centroids.png", caption="PCA 2D (text-embedding-3-large)")
    else:
        st.warning(
            "Plik 'all_pca_centroids.png' nie został znaleziony. Uruchom analysis_IND_COL.py, aby wygenerować wizualizację.")
    if os.path.exists("all_tsne_centroids.png"):
        st.image("all_tsne_centroids.png", caption="t-SNE 2D (text-embedding-3-large)")
    else:
        st.warning(
            "Plik 'all_tsne_centroids.png' nie został znaleziony. Uruchom analysis_IND_COL.py, aby wygenerować wizualizację.")

    st.subheader("Interaktywna wizualizacja 3D PCA")
    st.write(
        "Tutaj mogę obracać wykres 3D PCA oraz filtrować dane według klastrów, co ułatwia analizę struktury semantycznej.")
    all_emb = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                              pol_ind_embeddings, pol_col_embeddings,
                              jap_ind_embeddings, jap_col_embeddings], axis=0)
    all_lbl = (["ENG_IND"] * len(eng_ind_embeddings) +
               ["ENG_COL"] * len(eng_col_embeddings) +
               ["POL_IND"] * len(pol_ind_embeddings) +
               ["POL_COL"] * len(pol_col_embeddings) +
               ["JAP_IND"] * len(jap_ind_embeddings) +
               ["JAP_COL"] * len(jap_col_embeddings))
    fig_pca = generate_interactive_pca_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Interaktywna wizualizacja 3D t-SNE")
    st.write("Prezentuję interaktywny wykres t-SNE 3D, który pozwala odkryć nieliniowe zależności i ukryte klastery.")
    fig_tsne = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_tsne, use_container_width=True)

    st.subheader("Klasyfikacja nowego tekstu (metoda centroidów)")
    st.write(
        "Wprowadź tekst (angielski, polski lub japoński), a zamienię go na wektor i porównam z centroidami grup. Wynik klasyfikacji pokaże, do której grupy tekst jest najbliższy semantycznie.")
    user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.")
    if st.button("Klasyfikuj (centroidy)"):
        results = klasyfikuj_tekst(user_text)
        st.write("**Ranking podobieństwa (kosinus, 3072D):**")
        for cat, val in results:
            st.write(f"- {cat}: {val:.4f}")

    st.subheader("Klasyfikacja nowego tekstu (uczenie maszynowe)")
    st.write(
        "Tutaj korzystam z klasyfikatora wytrenowanego na podstawie embeddingów. Model ML (regresja logistyczna) lepiej uchwyca złożone zależności w danych.")
    if st.button("Trenuj klasyfikator ML i przetestuj"):
        all_embeddings = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                                         pol_ind_embeddings, pol_col_embeddings,
                                         jap_ind_embeddings, jap_col_embeddings], axis=0)
        all_labels = (["ENG_IND"] * len(eng_ind_embeddings) +
                      ["ENG_COL"] * len(eng_col_embeddings) +
                      ["POL_IND"] * len(pol_ind_embeddings) +
                      ["POL_COL"] * len(pol_col_embeddings) +
                      ["JAP_IND"] * len(jap_ind_embeddings) +
                      ["JAP_COL"] * len(jap_col_embeddings))
        clf = train_ml_classifier(all_embeddings, all_labels)
        pred_label, proba = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        st.write(" Rozkład prawdopodobieństwa:", proba)

    st.subheader("Raport statystyczny")
    st.write(
        "Poniżej przedstawiam wyniki analizy statystycznej dystansów między embeddingami, obliczonych dla trzech miar (euklidesowej, kosinusowej, Manhattan) z poziomem istotności 0.01.")
    report_text = generate_statistical_report()
    st.text(report_text)

    st.subheader("Wnioski")
    st.markdown("""
**Wnioski**

1. **Porównanie języka polskiego z angielskim**  
   Wyniki analizy wykazały, że mediana dystansu między zdaniami IND i COL w języku polskim jest niższa niż w języku angielskim, niezależnie od zastosowanej metryki (euklidesowej, kosinusowej czy Manhattan). Na przykład, dla metryki kosinusowej mediana wynosi 0.5894 dla języka polskiego, podczas gdy dla języka angielskiego 0.7788.

   **Co to oznacza?**  
   Niższe wartości dystansu sugerują, że reprezentacje semantyczne zdań indywidualistycznych i kolektywistycznych w języku polskim są bardziej zbliżone do siebie. Może to wynikać z charakterystycznych cech języka polskiego, specyfiki danych treningowych lub sposobu, w jaki model interpretuje niuanse semantyczne.

2. **Analiza języka japońskiego**  
   Dla języka japońskiego mediana dystansu plasuje się pomiędzy wartościami uzyskanymi dla języka polskiego i angielskiego. Na przykład, dla metryki euklidesowej mediana wynosi 1.1835, co jest mniejsze niż dla angielskiego (1.2480), ale nieco wyższe niż dla polskiego (1.0858).

   **Co to może oznaczać?**  
   Wartość pośrednia dla języka japońskiego sugeruje, że choć kultura japońska jest silnie kolektywistyczna, model wykrywa pewne podobieństwo między reprezentacjami semantycznymi japońskimi a angielskimi. Może to wynikać z danych treningowych lub struktury językowej, co powoduje, że dystanse dla japońskiego nie są tak niskie jak dla polskiego.

3. **Znaczenie testów statystycznych**  
   Testy statystyczne (t-test lub Mann–Whitney) przy poziomie istotności 0.01 potwierdzają, że obserwowane różnice nie są wynikiem przypadku – prawdopodobieństwo losowego wystąpienia takich różnic jest mniejsze niż 1%.

   **Implikacje:**  
   Istotność statystyczna potwierdza, że różnice w dystansach między grupami mają solidne podstawy. Mniejsze dystanse w języku polskim oraz wartości pośrednie dla japońskiego są wynikiem realnych różnic w reprezentacjach semantycznych, co dodatkowo wzmacnia wnioski dotyczące specyfiki wyrażania emocji w poszczególnych językach.

**Podsumowanie:**  
- **Język polski:** Zdania IND i COL są reprezentowane przez wektory, które są do siebie znacznie bliższe, co może świadczyć o mniejszych różnicach semantycznych i bardziej zbitym wyrażaniu emocji.
- **Język japoński:** Reprezentacje semantyczne są pośrednie między językiem polskim a angielskim, co sugeruje, że model wykrywa pewne podobieństwo między japońskim a angielskim, mimo kolektywistycznych cech kulturowych.
- **Język angielski:** Reprezentacje są bardziej oddalone, co wskazuje na większe różnice semantyczne.
- Testy statystyczne (p < 0.01) potwierdzają, że różnice są istotne, co daje solidne podstawy do dalszych interpretacji.
    """)

    st.subheader("Podsumowanie")
    st.write(
        "Model OpenAI `text-embedding-3-large` generuje precyzyjne reprezentacje, które pozwalają mi wykryć subtelne różnice semantyczne między emocjami w różnych językach. Obserwacje wskazują, że język polski wykazuje mniejsze różnice między zdaniami IND i COL, natomiast japoński plasuje się pośrednio między polskim a angielskim. Wyniki te są potwierdzone analizą statystyczną oraz interaktywnymi wizualizacjami 3D.")

    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")


if __name__ == "__main__":
    run_streamlit_app()
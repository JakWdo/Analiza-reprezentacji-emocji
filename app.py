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
    generate_interactive_pca_2d,  # interaktywne wykresy 2D
    generate_interactive_tsne_2d,  # interaktywne wykresy 2D
    generate_interactive_pca_3d,
    generate_interactive_tsne_3d,
    get_embedding,
    EMBEDDING_MODEL,
    train_ml_classifier, ml_klasyfikuj_tekst, get_ml_classifier,
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

    # Przygotowanie wspólnego zbioru embeddingów i etykiet
    all_emb = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                              pol_ind_embeddings, pol_col_embeddings,
                              jap_ind_embeddings, jap_col_embeddings], axis=0)
    all_lbl = (["ENG_IND"] * len(eng_ind_embeddings) +
               ["ENG_COL"] * len(eng_col_embeddings) +
               ["POL_IND"] * len(pol_ind_embeddings) +
               ["POL_COL"] * len(pol_col_embeddings) +
               ["JAP_IND"] * len(jap_ind_embeddings) +
               ["JAP_COL"] * len(jap_col_embeddings))

    st.subheader("Interaktywne wizualizacje 2D")
    st.write("Prezentuję interaktywne wykresy PCA i t-SNE, które pokazują relacje między zdaniami w przestrzeni 2D.")
    fig_pca_2d = generate_interactive_pca_2d(all_emb, all_lbl)
    fig_tsne_2d = generate_interactive_tsne_2d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_2d, use_container_width=True)
    st.plotly_chart(fig_tsne_2d, use_container_width=True)

    st.subheader("Interaktywna wizualizacja 3D PCA")
    st.write(
        "Tutaj można obracać wykres 3D PCA oraz filtrować dane według klastrów, co ułatwia analizę struktury semantycznej.")
    fig_pca_3d = generate_interactive_pca_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_3d, use_container_width=True)

    st.subheader("Interaktywna wizualizacja 3D t-SNE")
    st.write("Prezentuję interaktywny wykres t-SNE 3D, który pozwala odkryć nieliniowe zależności i ukryte klastery.")
    fig_tsne_3d = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_tsne_3d, use_container_width=True)

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
    if st.button("Trenuj/Wczytaj klasyfikator ML i przetestuj"):
        # Używamy funkcji get_ml_classifier, która wczytuje zapisany model lub trenuje nowy
        clf = get_ml_classifier(all_emb, all_lbl)
        pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        # Prezentacja rozkładu prawdopodobieństwa w formie tabelarycznej
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
        prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
        st.table(prob_df[["Etykieta", "Prawdopodobieństwo (%)"]])

    st.subheader("Raport statystyczny")
    st.write(
        "Poniżej przedstawiam wyniki analizy statystycznej dystansów między embeddingami, obliczonych dla trzech miar (euklidesowej, kosinusowej, Manhattan) z poziomem istotności 0.01.")
    report_text = generate_statistical_report()
    st.text_area("Raport statystyczny", report_text, height=300)

    st.subheader("Wnioski")
    st.markdown("""
**Wnioski i interpretacja wyników**

1. **Porównanie języka polskiego z angielskim**  
   Analiza dystansów między wektorami zdań wskazuje, że mediana dystansu dla języka polskiego jest niższa niż dla języka angielskiego (np. mediana kosinusowa: 0.5894 vs. 0.7788). Oznacza to, że reprezentacje semantyczne zdań IND i COL w języku polskim są bardziej zbliżone, co może wynikać z cech językowych lub specyfiki danych treningowych.

2. **Analiza języka japońskiego**  
   Dla języka japońskiego uzyskano wartości pośrednie – zarówno w metryce euklidesowej, kosinusowej, jak i Manhattan. Wynik pośredni sugeruje, że chociaż kultura japońska jest silnie kolektywistyczna, model wykrywa pewne podobieństwo między reprezentacjami japońskimi a angielskimi, ale nadal zdania IND i COL są statystycznie bliżej siebie niż w angielskim.

3. **Znaczenie testów statystycznych**  
   Testy normalności (Shapiro-Wilka i Kolmogorova-Smirnova) wskazują, że rozkłady dystansów nie są normalne. W związku z tym, zastosowano test nieparametryczny Mann–Whitney. Bardzo niskie wartości p (p < 0.01) dla obu testów potwierdzają, że obserwowane różnice między grupami nie są wynikiem przypadku.

**Podsumowanie ogólne:**  
- **Metoda centroidów:** Nowy tekst jest zamieniany na wektor, a następnie porównywany z centroidami dla każdej kategorii. Ranking kosinusowych podobieństw pozwala szybko określić, która grupa jest najbardziej zbliżona semantycznie.
- **Klasyfikacja ML:** Wytrenowany model (regresja logistyczna) umożliwia przypisanie nowego tekstu do jednej z kategorii wraz z rozkładem prawdopodobieństwa, co daje dodatkową informację o pewności predykcji.
- **Analiza statystyczna:** Wyniki testów potwierdzają, że różnice między językami (m.in. mniejsze dystanse dla języka polskiego) są statystycznie istotne. Oznacza to, że reprezentacje semantyczne zdań różnią się w zależności od języka, co ma potencjalne implikacje dla badań nad modelami językowymi.

Te wyniki stanowią solidną podstawę do dalszych badań nad reprezentacjami wektorowymi i ich zastosowaniami w analizie semantycznej tekstu.
    """)

    st.subheader("Podsumowanie")
    st.write(
        "Model OpenAI text-embedding-3-large generuje precyzyjne reprezentacje, które pozwalają wykryć subtelne różnice semantyczne między emocjami wyrażanymi w różnych językach. Analiza wskazuje, że język polski charakteryzuje się mniejszymi różnicami między zdaniami indywidualistycznymi a kolektywistycznymi, natomiast wyniki dla języka japońskiego plasują się pośrednio między polskim a angielskim. Wyniki te, potwierdzone analizą statystyczną, mogą pomóc w lepszym zrozumieniu, jak modele językowe interpretują różnice semantyczne wynikające ze specyfiki językowej.")

    # Zapis raportu statystycznego do pliku
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")


if __name__ == "__main__":
    run_streamlit_app()

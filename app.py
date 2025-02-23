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

# Ustawiamy layout na "centered", co powoduje wyśrodkowanie głównej sekcji w aplikacji Streamlit
st.set_page_config(
    page_title="Analiza reprezentacji wektorowych emocji",
    layout="centered"
)

def run_streamlit_app():
    """
    Główna funkcja uruchamiająca aplikację Streamlit.
    W tym miejscu definiujemy wszystkie elementy interfejsu:
    
    1. Wprowadzenie teoretyczne (INTEGRATED_REPORT).
    2. Przykładowe zdania z trzech języków (ENG, POL, JAP).
    3. Wizualizacje 2D i 3D (PCA, t-SNE) do zrozumienia, jak zdania 
       rozkładają się w przestrzeni wektorowej.
    4. Klasyfikacja tekstu przy użyciu:
       - Metody centroidów (szybkie porównanie z uśrednionym wektorem kategorii).
       - Modelu ML (regresja logistyczna, która "uczy się" rozróżniać kategorie).
    5. Raport statystyczny (testy istotności różnic między językami).
    6. Wnioski i zapis raportu do pliku.
    """

    # 1. Tytuł aplikacji
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")

    # 2. Wprowadzenie teoretyczne: obszerny raport z analysis_IND_COL.py (INTEGRATED_REPORT)
    st.markdown(INTEGRATED_REPORT)

    # 3. Przykłady zdań użytych do treningu embeddingów (można w ten sposób zobaczyć sample tekstu)
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

    # 4. Zbieramy wszystkie embeddingi (wektory) razem i tworzymy listę ich etykiet
    #   (ENG_IND, ENG_COL, POL_IND, POL_COL, JAP_IND, JAP_COL).
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

    # 5. Wizualizacje 2D: PCA i t-SNE
    st.subheader("Interaktywne wizualizacje 2D")
    st.write("""
    **Wyjaśnienie dla osób nietechnicznych**:
    - **PCA** (Principal Component Analysis) stara się uchwycić w 2 (lub 3) wymiarach 
      to, co w oryginalnej przestrzeni (3072D) powoduje największą "różnorodność" danych.
    - **t-SNE** (t-Distributed Stochastic Neighbor Embedding) koncentruje się na tym, 
      żeby punkty, które są blisko siebie w oryginalnej przestrzeni, pozostały blisko 
      także w przestrzeni 2D. 
      
    W obu przypadkach otrzymujemy "spłaszczoną" mapę, na której widać, czy zdania 
    IND i COL (z różnych języków) są razem czy osobno.
    """)

    fig_pca_2d = generate_interactive_pca_2d(all_emb, all_lbl)
    fig_tsne_2d = generate_interactive_tsne_2d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_2d, use_container_width=True)
    st.plotly_chart(fig_tsne_2d, use_container_width=True)

    # 6. Wizualizacje 3D: PCA i t-SNE
    st.subheader("Interaktywna wizualizacja 3D PCA")
    st.write("""
    Tutaj możesz zobaczyć zdania w **trzech wymiarach** PCA. 
    Możesz obracać wykres i odfiltrowywać niektóre grupy, aby lepiej zobaczyć relacje 
    między zdaniami z różnych kategorii.
    """)
    fig_pca_3d = generate_interactive_pca_3d(all_emb, all_lbl)
    st.plotly_chart(fig_pca_3d, use_container_width=True)

    st.subheader("Interaktywna wizualizacja 3D t-SNE")
    st.write("""
    Podobnie jak przy PCA 3D, ale używamy **t-SNE**. 
    Ten wykres też można obracać, żeby zobaczyć, jak w 3D rozkładają się zdania 
    indywidualistyczne i kolektywistyczne w różnych językach.
    """)
    fig_tsne_3d = generate_interactive_tsne_3d(all_emb, all_lbl)
    st.plotly_chart(fig_tsne_3d, use_container_width=True)

    # 7. Klasyfikacja metodą centroidów
    st.subheader("Klasyfikacja nowego tekstu (metoda centroidów)")
    st.write("""
    **Na czym polega metoda centroidów?**  
    - Dla każdej grupy (ENG_IND, ENG_COL, POL_IND, POL_COL, JAP_IND, JAP_COL) 
      liczymy **centroid**, czyli uśredniony wektor wszystkich zdań treningowych 
      z danej kategorii.  
    - Jeśli wprowadzisz nowy tekst, zamieniamy go również na wektor 
      (embedding 3072D).  
    - Mierzymy, do którego centroidu Twój tekst jest najbliższy (np. 
      obliczając podobieństwo kosinusowe).  
    - Największe podobieństwo oznacza przewidywaną kategorię.  

    Ta metoda jest prosta i szybka, ale nie zawsze najbardziej precyzyjna, 
    bo zakłada, że jeden "średni" wektor dobrze reprezentuje całą kategorię.
    """)

    # Pole tekstowe i przycisk w aplikacji
    user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.")
    if st.button("Klasyfikuj (centroidy)"):
        results = klasyfikuj_tekst(user_text)
        st.write("**Ranking podobieństwa (kosinus, 3072D):**")
        for cat, val in results:
            st.write(f"- {cat}: {val:.4f}")

    # 8. Klasyfikacja przy użyciu modelu ML (regresji logistycznej)
    st.subheader("Klasyfikacja nowego tekstu (uczenie maszynowe)")
    st.write("""
    **Na czym polega model ML (regresja logistyczna)?**  
    - Najpierw bierzemy wszystkie zdania (ENG_IND, ENG_COL, POL_IND, 
      POL_COL, JAP_IND, JAP_COL) i ich embeddingi.  
    - Budujemy model (regresję logistyczną) – to narzędzie statystyczne, które 
      "uczy się" przypisywać nowe zdania do odpowiedniej kategorii.  
    - Gdy wprowadzisz nowy tekst, obliczamy jego embedding, a następnie 
      model przewiduje, do której klasy (z sześciu dostępnych) należy 
      ten tekst, **podając też prawdopodobieństwo** przynależności 
      do każdej kategorii.  

    Ta metoda może być dokładniejsza od centroidów, bo uwzględnia 
    różnorodność zdań w każdej kategorii, a nie tylko "średnią".
    """)

    # Przyciski w aplikacji do trenowania / wczytania modelu i predykcji
    if st.button("Trenuj/Wczytaj klasyfikator ML i przetestuj"):
        clf = get_ml_classifier(all_emb, all_lbl)  # wczytujemy lub trenujemy
        pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
        st.write("**Wynik klasyfikacji ML:**")
        st.write(" Przewidywana etykieta:", pred_label)
        
        # Prezentacja rozkładu prawdopodobieństwa w formie tabeli
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
        prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
        st.table(prob_df[["Etykieta", "Prawdopodobieństwo (%)"]])

    # 9. Raport statystyczny
    st.subheader("Raport statystyczny")
    st.write("""
    Poniżej przedstawiam wyniki analizy statystycznej dystansów między embeddingami, 
    obliczonych dla trzech różnych miar (euklidesowej, kosinusowej, Manhattan). 
    Sprawdzamy, czy różnice między językami (np. polski vs. angielski) są istotne 
    (p < 0.01), korzystając m.in. z testu Manna–Whitneya przy nienormalnych rozkładach danych.
    """)

    report_text = generate_statistical_report()
    st.text_area("Raport statystyczny", report_text, height=300)

    # 10. Wnioski i interpretacja
    st.subheader("Wnioski")
    st.markdown("""
**Wnioski i interpretacja wyników**

1. **Porównanie języka polskiego z angielskim**  
   - W wynikach widać, że polskie zdania IND i COL są bliżej siebie (mniejszy dystans) 
     niż analogiczne zdania w języku angielskim. 
     To może wynikać z charakterystyki języka, kultury lub danych, na których 
     trenowano model. 

2. **Analiza języka japońskiego**  
   - Japońskie zdania IND i COL są także bliżej siebie niż w angielskim, 
     choć często wypadają pośrednio między polskim a angielskim. 
     Może to odzwierciedlać fakt, że kultura japońska (bardzo kolektywistyczna) 
     przejawia się w zdaniach, ale model nie oddaje tego w 100%.

3. **Znaczenie testów statystycznych**  
   - Ze względu na nienormalne rozkłady danych zastosowaliśmy test Manna–Whitneya, 
     który potwierdza, że różnice między językami są istotne (p < 0.01). 

**Metoda centroidów** pozwala szybko oszacować, do której kategorii 
zbadany tekst najbliżej pasuje.  
**Klasyfikator ML** (regresja logistyczna) daje bogatszą analizę, w tym 
procentowe prawdopodobieństwa, co bywa cenną informacją w praktyce.
    """)

    # 11. Podsumowanie końcowe
    st.subheader("Podsumowanie")
    st.write("""
Model OpenAI `text-embedding-3-large` potrafi wychwycić subtelne różnice 
semantyczne między zdaniami indywidualistycznymi a kolektywistycznymi 
w różnych językach. 
Polski wydaje się mieć mniejsze różnice (embeddingowo) między IND i COL 
niż w angielskim, a japoński ma wynik pośredni. 
Wyniki te, potwierdzone statystycznie (p < 0.01), sugerują, że modele językowe 
mogą w różny sposób odwzorowywać cechy wynikające z kultury i struktury języka.
    """)

    # 12. Zapis raportu do pliku
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")

# Uruchomienie aplikacji
if __name__ == "__main__":
    run_streamlit_app()

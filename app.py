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
   Wyniki testów statystycznych dla trzech metryk (Euklides, Kosinus, Manhattan) potwierdzają, że model `text-embedding-3-large` potrafi wyraźnie rozróżnić zdania indywidualistyczne (IND) od kolektywistycznych (COL) w każdym z badanych języków (polskim, angielskim, japońskim). Testy normalności wykazały, że rozkłady odległości nie są normalne, co uzasadniło użycie testu nieparametrycznego Manna–Whitneya, dającego p < 0.01 dla wszystkich języków.

2. **Różnice między językami (H₂/H₃):**  
   - **Metryka Euklides:**  
     - median(Pol)=1.0858  
     - median(Jap)=1.1835  
     - median(Eng)=1.2480  
   - **Metryka Kosinus:**  
     - median(Pol)=0.5894  
     - median(Jap)=0.7004  
     - median(Eng)=0.7788  
   - **Metryka Manhattan:**  
     - median(Pol)=46.7398  
     - median(Jap)=50.8755  
     - median(Eng)=53.7099  
     
   We wszystkich metrykach obserwujemy, że zdania w języku polskim wykazują najmniejsze dystanse między IND a COL, a angielskie – największe. Zaskakujące jest jednak, że mimo teoretycznego przyporządkowania japońskiego jako kultury o najbardziej kolektywistycznym charakterze, polskie zdania wykazują jeszcze mniejsze różnice (czyli są bardziej zbliżone semantycznie) niż japońskie.

# Dyskusja

## Niezgodność oczekiwań teoretycznych z wynikami empirycznymi

**Teoretyczne założenie:**  
W literaturze kulturowej japoński model komunikacji jest często opisywany jako najbardziej kolektywistyczny, co sugerowałoby, że zdania wyrażające zarówno postawy indywidualistyczne, jak i kolektywistyczne powinny być bardziej jednorodne (czyli ich reprezentacje wektorowe – embeddingi – powinny być bardzo zbliżone).  

**Obserwacja:**  
Wyniki analizy pokazują, że choć zarówno japońskie, jak i polskie zdania IND i COL są bliżej siebie niż ich angielskie odpowiedniki, to jednak dystanse między zdaniami w języku polskim są najmniejsze. To zaskakujące rozbieżność, ponieważ można by oczekiwać, że najbardziej kolektywistyczna kultura (japońska) wykazywałaby najmniejsze różnice między kategoriami.

## Możliwe przyczyny tej anomalii

- **Specyfika korpusu i wyboru danych:**  
  - Zdania w języku polskim mogły zostać wyselekcjonowane lub skonstruowane w taki sposób, że cechy indywidualizmu i kolektywizmu są wyrażone w sposób bardzo jednoznaczny i skoncentrowany, co prowadzi do bardziej zwartych klastrów.
  - W przypadku języka japońskiego, pomimo teoretycznego kolektywizmu, wyrażenia mogą być bardziej subtelne i wielowarstwowe (np. dzięki użyciu honoryfikatywnych form i kontekstowych niuansów), co skutkuje nieco większą rozpiętością reprezentacji.

- **Cechy językowe i struktura gramatyczna:**  
  - Język polski, będący językiem fleksyjnym, może wykazywać mniejszą różnorodność w zakresie wyrażania postaw społecznych, co powoduje, że zdania IND i COL są bardziej podobne semantycznie.
  - Japoński, mimo swojej kolektywistycznej natury kulturowej, posiada bogaty system wyrażeń i form grzecznościowych, co może wpływać na większą rozbieżność w reprezentacjach, nawet przy wyraźnie kolektywistycznych wartościach.

- **Wpływ modelu i danych treningowych:**  
  - Model `text-embedding-3-large` mógł być trenowany głównie na danych anglojęzycznych, co mogło wpłynąć na sposób reprezentowania cech kulturowych w innych językach.
  - Jakość i ilość dostępnych danych w języku japońskim oraz polskim mogą znacząco różnić się od angielskich, co wpływa na precyzję i spójność wyodrębnionych wektorów.

- **Efekt tłumaczenia:**  
  - W przypadku badań porównawczych często stosuje się tłumaczenia zdań. Bezpośrednie tłumaczenia z języka angielskiego na polski mogą nie oddawać pełni niuansów kulturowych, a w rezultacie generować bardziej jednorodne reprezentacje.
  
- **Różnice w interpretacji pojęć kulturowych:**  
  - Pojęcia indywidualizmu i kolektywizmu mogą być rozumiane i wyrażane inaczej w różnych językach. W Polsce może istnieć tendencja do wyrażania tych postaw w sposób bardziej dychotomiczny, co skutkuje mniejszymi odległościami między reprezentacjami IND i COL.

# Podsumowanie

Badanie wskazuje, że:
- **Model embeddingowy** skutecznie rozróżnia zdania indywidualistyczne i kolektywistyczne w trzech językach, co potwierdzają istotne statystycznie różnice (p < 0.01).
- **Różnice semantyczne** (mierzone dystansami wektorowymi) są największe w języku angielskim, natomiast zarówno polskie, jak i japońskie zdania wykazują mniejsze rozbieżności.
- **Zaskakujący wynik:** Mimo że japoński jest teoretycznie najbardziej kolektywistyczny, to zdania w języku polskim są reprezentowane jako jeszcze bardziej zbliżone.  
- **Potencjalne przyczyny** tej anomalii mogą obejmować specyfikę korpusu, cechy językowe, wpływ danych treningowych oraz ewentualne efekty tłumaczenia i interpretacji kulturowych.  

Ogólnie rzecz biorąc, wyniki te sugerują, że choć metody oparte na reprezentacjach wektorowych są użyteczne w badaniach kulturowych, wyniki należy interpretować w kontekście specyfiki danych oraz z uwzględnieniem subtelności, jakie niesie ze sobą język i kultura.

    """)

    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    st.success("Raport statystyczny został zapisany w pliku 'raport_statystyczny.txt'.")

if __name__ == "__main__":
    run_streamlit_app()

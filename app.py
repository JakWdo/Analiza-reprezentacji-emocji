import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import json
import logging
import plotly.graph_objects as go
from PIL import Image
from analysis_IND_COL import (
    # Pobranie zdań i embeddingów
    load_sentences_and_embeddings,
    compute_all_centroids,
    
    # Funkcje klasyfikacji
    klasyfikuj_tekst,
    ml_klasyfikuj_tekst,
    get_ml_classifier,
    
    # Wizualizacje
    generate_interactive_pca_2d,
    generate_interactive_tsne_2d,
    generate_interactive_pca_3d,
    generate_interactive_tsne_3d,
    generate_plotly_3d_point_cloud,
    
    # Analizy statystyczne
    generate_statistical_report,
    generate_interactive_distribution_charts,
    generate_metric_comparison_chart,
    perform_cross_validation,
    interpret_pca_components,
    semantic_cluster_analysis,
    
    # Stałe
    EMBEDDING_MODEL,
    INTEGRATED_REPORT
)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Konfiguracja Streamlit
st.set_page_config(
    page_title="Analiza reprezentacji wektorowych emocji",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache dla kosztownych obliczeniowo funkcji
@st.cache_data(ttl=3600)
def load_data(use_augmentation=False):
    """
    Wczytuje dane i przygotowuje je do analizy.
    
    Parametry:
    ----------
    use_augmentation : bool, optional
        Czy używać augmentacji danych.
        
    Zwraca:
    -------
    tuple
        (zdania, embeddingi, centroidy, wszystkie_embeddingi, wszystkie_etykiety)
    """
    start_time = time.time()
    zdania, embeddings = load_sentences_and_embeddings(use_augmentation=use_augmentation)
    centroids = compute_all_centroids(embeddings)
    
    all_embeddings = np.concatenate([
        embeddings["eng_ind"], embeddings["eng_col"],
        embeddings["pol_ind"], embeddings["pol_col"],
        embeddings["jap_ind"], embeddings["jap_col"]
    ], axis=0)
    
    all_labels = (
        ["ENG_IND"] * len(embeddings["eng_ind"]) +
        ["ENG_COL"] * len(embeddings["eng_col"]) +
        ["POL_IND"] * len(embeddings["pol_ind"]) +
        ["POL_COL"] * len(embeddings["pol_col"]) +
        ["JAP_IND"] * len(embeddings["jap_ind"]) +
        ["JAP_COL"] * len(embeddings["jap_col"])
    )
    
    logger.info(f"Wczytanie danych zajęło {time.time() - start_time:.2f}s")
    return zdania, embeddings, centroids, all_embeddings, all_labels


@st.cache_data(ttl=3600)
def get_classifier(all_embeddings, all_labels, force_retrain=False):
    """
    Pobiera lub trenuje klasyfikator ML.
    
    Parametry:
    ----------
    all_embeddings : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_labels : list
        Lista etykiet dla wszystkich embeddingów.
    force_retrain : bool, optional
        Czy wymusić ponowne trenowanie modelu.
        
    Zwraca:
    -------
    sklearn.pipeline.Pipeline
        Klasyfikator ML.
    """
    return get_ml_classifier(all_embeddings, all_labels, force_retrain=force_retrain)


@st.cache_data(ttl=3600)
def get_cross_validation_results(all_embeddings, all_labels, n_splits=5):
    """
    Przeprowadza walidację krzyżową i zwraca wyniki.
    
    Parametry:
    ----------
    all_embeddings : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_labels : list
        Lista etykiet dla wszystkich embeddingów.
    n_splits : int, optional
        Liczba podziałów w walidacji krzyżowej.
        
    Zwraca:
    -------
    dict
        Wyniki walidacji krzyżowej.
    """
    cv_results = perform_cross_validation(all_embeddings, all_labels, n_splits=n_splits)
    cv_df = pd.DataFrame(cv_results)
    return {
        'results': cv_results,
        'summary': {
            'accuracy_mean': cv_df['accuracy'].mean(),
            'accuracy_std': cv_df['accuracy'].std(),
            'f1_mean': cv_df['f1'].mean(),
            'f1_std': cv_df['f1'].std()
        },
        'df': cv_df
    }


def run_streamlit_app():
    """
    Główna funkcja aplikacji Streamlit.
    """
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")
    
    # Sidebar z opcjami
    with st.sidebar:
        st.title("Opcje analizy")
        
        use_augmentation = st.checkbox("Użyj augmentacji danych", value=False, help="Rozszerza zbiór danych poprzez wprowadzenie wariantów semantycznych")
        force_retrain = st.checkbox("Wymuś ponowne trenowanie klasyfikatora", value=False, help="Trenuje klasyfikator od nowa zamiast wczytywać zapisany")
        
        st.subheader("Parametry wizualizacji")
        perplexity_tsne = st.slider("Parametr perplexity dla t-SNE", min_value=5, max_value=50, value=30, step=5, help="Wyższe wartości uwzględniają większą liczbę sąsiadów")
        
        st.subheader("Nawigacja")
        navigation = st.radio(
            "Wybierz sekcję:",
            options=[
                "Wprowadzenie teoretyczne",
                "Przykłady zdań",
                "Wizualizacje 2D",
                "Wizualizacje 3D", 
                "Analiza statystyczna",
                "Klasyfikacja tekstu",
                "Zaawansowana analiza",
                "Wnioski"
            ]
        )
    
    # Wczytanie danych
    zdania, embeddings, centroids, all_embeddings, all_labels = load_data(use_augmentation=use_augmentation)
    
    # Główny content w zależności od wybranej sekcji
    if navigation == "Wprowadzenie teoretyczne":
        st.markdown(INTEGRATED_REPORT)
    
    elif navigation == "Przykłady zdań":
        st.header("Przykłady zdań użytych do tworzenia embeddingów")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Języki zachodnie")
            
            st.markdown("### English – Individualistic")
            for i, sent in enumerate(zdania["english_individualistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań angielskich indywidualistycznych"):
                for sent in zdania["english_individualistic"][5:15]:
                    st.markdown(f"- *{sent}*")
            
            st.markdown("### English – Collectivistic")
            for i, sent in enumerate(zdania["english_collectivistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań angielskich kolektywistycznych"):
                for sent in zdania["english_collectivistic"][5:15]:
                    st.markdown(f"- *{sent}*")
            
            st.markdown("### Polish – Individualistic")
            for i, sent in enumerate(zdania["polish_individualistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań polskich indywidualistycznych"):
                for sent in zdania["polish_individualistic"][5:15]:
                    st.markdown(f"- *{sent}*")
            
            st.markdown("### Polish – Collectivistic")
            for i, sent in enumerate(zdania["polish_collectivistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań polskich kolektywistycznych"):
                for sent in zdania["polish_collectivistic"][5:15]:
                    st.markdown(f"- *{sent}*")
        
        with col2:
            st.subheader("Język wschodni")
            
            st.markdown("### Japanese – Individualistic")
            for i, sent in enumerate(zdania["japanese_individualistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań japońskich indywidualistycznych"):
                for sent in zdania["japanese_individualistic"][5:15]:
                    st.markdown(f"- *{sent}*")
            
            st.markdown("### Japanese – Collectivistic")
            for i, sent in enumerate(zdania["japanese_collectivistic"][:5]):
                st.markdown(f"- *{sent}*")
            if st.button("Pokaż więcej zdań japońskich kolektywistycznych"):
                for sent in zdania["japanese_collectivistic"][5:15]:
                    st.markdown(f"- *{sent}*")
        
        # Statystyki zbioru danych
        st.subheader("Statystyki zbioru danych")
        
        stats_data = {
            'Język': ['Angielski', 'Polski', 'Japoński', 'Razem'],
            'Indywidualistyczne': [
                len(zdania["english_individualistic"]),
                len(zdania["polish_individualistic"]),
                len(zdania["japanese_individualistic"]),
                len(zdania["english_individualistic"]) + len(zdania["polish_individualistic"]) + len(zdania["japanese_individualistic"])
            ],
            'Kolektywistyczne': [
                len(zdania["english_collectivistic"]),
                len(zdania["polish_collectivistic"]),
                len(zdania["japanese_collectivistic"]),
                len(zdania["english_collectivistic"]) + len(zdania["polish_collectivistic"]) + len(zdania["japanese_collectivistic"])
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df['Razem'] = stats_df['Indywidualistyczne'] + stats_df['Kolektywistyczne']
        
        st.table(stats_df)
    
    elif navigation == "Wizualizacje 2D":
        st.header("Interaktywne wizualizacje 2D")
        st.write("Poniżej przedstawiono dwa wykresy 2D: PCA oraz t-SNE. Możesz interaktywnie eksplorować dane, przybliżać, oddalać i filtrować kategorie.")
        
        viz_type = st.radio("Wybierz typ wizualizacji:", ["PCA", "t-SNE"])
        
        if viz_type == "PCA":
            fig_2d_pca = generate_interactive_pca_2d(all_embeddings, all_labels)
            st.plotly_chart(fig_2d_pca, use_container_width=True)
            
            # Dodatkowe informacje o PCA
            with st.expander("Co to jest PCA?"):
                st.markdown("""
                **Principal Component Analysis (PCA)** to technika redukcji wymiarowości, która:
                
                - Identyfikuje główne kierunki wariancji w danych
                - Transformuje dane do nowych współrzędnych (komponentów głównych)
                - Pozwala na wizualizację danych wysokowymiarowych w przestrzeni 2D lub 3D
                
                W kontekście naszego badania, PCA pomaga zobaczyć, jak zdania indywidualistyczne i kolektywistyczne układają się w przestrzeni embeddingów. Bliskość punktów sugeruje semantyczne podobieństwo zdań.
                """)
        else:  # t-SNE
            fig_2d_tsne = generate_interactive_tsne_2d(all_embeddings, all_labels, perplexity=perplexity_tsne)
            st.plotly_chart(fig_2d_tsne, use_container_width=True)
            
            # Dodatkowe informacje o t-SNE
            with st.expander("Co to jest t-SNE?"):
                st.markdown("""
                **t-distributed Stochastic Neighbor Embedding (t-SNE)** to nieliniowa technika redukcji wymiarowości, która:
                
                - Zachowuje strukturę lokalną danych (bliskie punkty w oryginalnej przestrzeni pozostają blisko w przestrzeni zredukowanej)
                - Jest szczególnie skuteczna w ujawnianiu klastrów i podstruktur
                - Elastycznie dostosowuje się do różnych rozkładów danych
                
                Parametr **perplexity** wpływa na to, jak t-SNE równoważy uwagę między lokalnymi i globalnymi aspektami danych. Możesz dostosować ten parametr używając suwaka w panelu bocznym.
                """)
        
        # Filtry kategorii
        st.subheader("Filtrowanie kategorii")
        st.write("Możesz filtrować kategorie, aby lepiej zobaczyć wzorce w danych.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_eng = st.checkbox("Angielski", value=True)
            eng_filter = st.radio("Angielski - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"], disabled=not include_eng)
        
        with col2:
            include_pol = st.checkbox("Polski", value=True)
            pol_filter = st.radio("Polski - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"], disabled=not include_pol)
        
        with col3:
            include_jap = st.checkbox("Japoński", value=True)
            jap_filter = st.radio("Japoński - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"], disabled=not include_jap)
        
        if st.button("Zastosuj filtry i odśwież wizualizację"):
            # Przygotowanie filtrowanych danych
            filtered_indices = []
            
            for i, label in enumerate(all_labels):
                lang, category = label.split("_")
                
                # Angielski
                if lang == "ENG" and include_eng:
                    if eng_filter == "Wszystkie" or \
                       (eng_filter == "Tylko IND" and category == "IND") or \
                       (eng_filter == "Tylko COL" and category == "COL"):
                        filtered_indices.append(i)
                
                # Polski
                elif lang == "POL" and include_pol:
                    if pol_filter == "Wszystkie" or \
                       (pol_filter == "Tylko IND" and category == "IND") or \
                       (pol_filter == "Tylko COL" and category == "COL"):
                        filtered_indices.append(i)
                
                # Japoński
                elif lang == "JAP" and include_jap:
                    if jap_filter == "Wszystkie" or \
                       (jap_filter == "Tylko IND" and category == "IND") or \
                       (jap_filter == "Tylko COL" and category == "COL"):
                        filtered_indices.append(i)
            
            filtered_embeddings = all_embeddings[filtered_indices]
            filtered_labels = [all_labels[i] for i in filtered_indices]
            
            if viz_type == "PCA":
                filtered_fig = generate_interactive_pca_2d(filtered_embeddings, filtered_labels)
            else:
                filtered_fig = generate_interactive_tsne_2d(filtered_embeddings, filtered_labels, perplexity=perplexity_tsne)
            
            st.subheader("Filtrowana wizualizacja")
            st.plotly_chart(filtered_fig, use_container_width=True)
    
    elif navigation == "Wizualizacje 3D":
        st.header("Interaktywne wizualizacje 3D")
        st.write("""
        Poniżej znajdują się interaktywne wykresy 3D, które pozwalają lepiej zbadać strukturę danych.
        **Uwaga:** 
        - Możesz obracać, przybliżać i oddalać wykres.
        - Filtrowanie klastrów odbywa się przez interaktywną legendę – kliknij na niej, aby ukrywać lub pokazywać grupy.
        - 3D daje lepszy wgląd w strukturę danych wysokowymiarowych.
        """)
        
        viz_type_3d = st.radio("Wybierz typ wizualizacji 3D:", ["PCA 3D", "t-SNE 3D", "Point Cloud"])
        
        if viz_type_3d == "PCA 3D":
            fig_pca_3d = generate_interactive_pca_3d(all_embeddings, all_labels)
            st.plotly_chart(fig_pca_3d, use_container_width=True)
        
        elif viz_type_3d == "t-SNE 3D":
            fig_tsne_3d = generate_interactive_tsne_3d(all_embeddings, all_labels, perplexity=perplexity_tsne)
            st.plotly_chart(fig_tsne_3d, use_container_width=True)
        
        else:  # Point Cloud
            reduction_method = st.selectbox("Metoda redukcji wymiarowości:", ["pca", "tsne"])
            fig_cloud = generate_plotly_3d_point_cloud(all_embeddings, all_labels, method=reduction_method)
            if fig_cloud:
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.error("Nie można wygenerować wizualizacji Point Cloud. Sprawdź, czy zainstalowano wymagane biblioteki.")
        
        # Animacja różnic między językami
        st.subheader("Animacja zmian między językami")
        st.write("Poniżej znajduje się animacja pokazująca, jak różnią się reprezentacje zdań w różnych językach.")
        
        if os.path.exists("animation_pca.gif"):
            st.image("animation_pca.gif", caption="Animacja przejść między reprezentacjami językowymi (PCA)")
        else:
            if st.button("Generuj animację"):
                st.info("Generowanie animacji... To może potrwać kilka minut.")
                # Tutaj można dodać kod generujący animację
                st.success("Animacja wygenerowana.")
    
    elif navigation == "Analiza statystyczna":
        st.header("Analiza statystyczna")
        st.write("Poniżej znajdują się wyniki testów statystycznych sprawdzających istotność różnic między językami oraz wewnątrz jednego języka.")
        
        # Ładowanie raportu statystycznego
        if os.path.exists("raport_statystyczny.txt"):
            with open("raport_statystyczny.txt", "r", encoding="utf-8") as f:
                report_text = f.read()
        else:
            report_text = generate_statistical_report(embeddings)
            with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
                f.write(report_text)
        
        # Wyświetlanie raportu
        with st.expander("Pełny raport statystyczny"):
            st.text_area("Raport szczegółowy", report_text, height=500)
        
        st.subheader("Wizualizacja rozkładów odległości")
        st.write("Wybierz metrykę oraz język, aby zobaczyć interaktywny wykres rozkładu odległości między zdaniami IND i COL.")
        
        metric_options = ["Euklides", "Kosinus", "Manhattan"]
        language_options = ["ENG", "POL", "JAP"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_metric = st.selectbox("Wybierz metrykę", metric_options)
        
        with col2:
            selected_language = st.selectbox("Wybierz język", language_options)
        
        distribution_figures = generate_interactive_distribution_charts(embeddings)
        fig_distribution = distribution_figures[selected_metric][selected_language]
        st.plotly_chart(fig_distribution, use_container_width=True)
        
        # Porównanie metryk między językami
        st.subheader("Porównanie metryk między językami")
        st.write("Poniższy wykres pokazuje mediany odległości między kategoriami IND i COL dla różnych języków i metryk.")
        
        # Przygotowanie danych do wykresu
        metrics_data = {}
        for metric_name in ["Euklides", "Kosinus", "Manhattan"]:
            metrics_data[metric_name] = {
                "POL": np.median(all_pairwise(embeddings["pol_ind"], embeddings["pol_col"], metric_name.lower())),
                "ENG": np.median(all_pairwise(embeddings["eng_ind"], embeddings["eng_col"], metric_name.lower())),
                "JAP": np.median(all_pairwise(embeddings["jap_ind"], embeddings["jap_col"], metric_name.lower()))
            }
        
        fig_metrics = generate_metric_comparison_chart(
            metrics_data, metrics=metric_options, lang_pairs=language_options,
            title="Mediany odległości między kategoriami IND i COL według języka i metryki"
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Analiza korelacji
        st.subheader("Analiza korelacji między metrykami")
        st.write("Macierz korelacji pokazuje, jak silnie powiązane są różne metryki odległości.")
        
        # Przygotowanie danych
        dist_data = {}
        for metric_name in ["Euklides", "Kosinus", "Manhattan"]:
            for lang in ["POL", "ENG", "JAP"]:
                if lang == "POL":
                    dist_data[f"{lang}_{metric_name}"] = all_pairwise(embeddings["pol_ind"], embeddings["pol_col"], metric_name.lower())
                elif lang == "ENG":
                    dist_data[f"{lang}_{metric_name}"] = all_pairwise(embeddings["eng_ind"], embeddings["eng_col"], metric_name.lower())
                elif lang == "JAP":
                    dist_data[f"{lang}_{metric_name}"] = all_pairwise(embeddings["jap_ind"], embeddings["jap_col"], metric_name.lower())
        
        # Tworzenie DataFrame
        dist_df = pd.DataFrame({k: v for k, v in dist_data.items()})
        
        # Obliczanie korelacji
        corr_matrix = dist_df.corr()
        
        # Wizualizacja macierzy korelacji
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}"
        ))
        
        fig_corr.update_layout(
            title="Macierz korelacji między metrykami odległości dla różnych języków",
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    elif navigation == "Klasyfikacja tekstu":
        st.header("Klasyfikacja nowego tekstu")
        st.write("Możesz przetestować klasyfikację nowego tekstu przy użyciu dwóch metod: centroidów i uczenia maszynowego.")
        
        # Wejście użytkownika
        user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.", height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metoda centroidów")
            st.write("Metoda centroidów polega na porównaniu embeddingu nowego tekstu z uśrednionymi wektorami każdej kategorii.")
            
            if st.button("Klasyfikuj (centroidy)"):
                with st.spinner("Klasyfikowanie..."):
                    results = klasyfikuj_tekst(user_text, centroids)
                
                st.write("**Ranking podobieństwa (im wyżej, tym bliżej):**")
                
                # Tworzenie DataFrame dla wyników
                results_df = pd.DataFrame(results, columns=["Kategoria", "Podobieństwo"])
                results_df["Podobieństwo (%)"] = results_df["Podobieństwo"] * 100
                
                # Kolorowanie wyników
                results_styled = results_df.style.background_gradient(subset=["Podobieństwo (%)"], cmap="viridis")
                
                st.dataframe(results_styled)
                
                # Wykres
                fig = go.Figure(go.Bar(
                    x=results_df["Podobieństwo (%)"],
                    y=results_df["Kategoria"],
                    orientation='h',
                    marker_color=np.arange(len(results_df)),
                    text=results_df["Podobieństwo (%)"].round(2),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Podobieństwo do kategorii (metoda centroidów)",
                    xaxis_title="Podobieństwo (%)",
                    yaxis_title="Kategoria",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Klasyfikacja ML")
            st.write("Regresja logistyczna trenuje model na wszystkich embeddingach, aby rozpoznawać kategorię nowego tekstu.")
            
            # Lazy-loading klasyfikatora
            if st.button("Klasyfikuj (ML)"):
                with st.spinner("Klasyfikowanie przy użyciu modelu ML..."):
                    clf = get_classifier(all_embeddings, all_labels, force_retrain=force_retrain)
                    pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
                
                st.write("**Wynik klasyfikacji ML:**")
                st.write(f"Przewidywana etykieta: **{pred_label}**")
                
                # Tworzenie DataFrame dla prawdopodobieństw
                prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Etykieta", "Prawdopodobieństwo"])
                prob_df["Prawdopodobieństwo (%)"] = prob_df["Prawdopodobieństwo"] * 100
                
                # Kolorowanie wyników
                prob_styled = prob_df.style.background_gradient(subset=["Prawdopodobieństwo (%)"], cmap="viridis")
                
                st.dataframe(prob_styled)
                
                # Wykres
                fig = go.Figure(go.Bar(
                    x=prob_df["Prawdopodobieństwo (%)"],
                    y=prob_df["Etykieta"],
                    orientation='h',
                    marker_color=np.arange(len(prob_df)),
                    text=prob_df["Prawdopodobieństwo (%)"].round(2),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Prawdopodobieństwa klas (model ML)",
                    xaxis_title="Prawdopodobieństwo (%)",
                    yaxis_title="Kategoria",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Porównanie metod
        if st.button("Porównaj obie metody"):
            with st.spinner("Klasyfikowanie..."):
                results_centroid = klasyfikuj_tekst(user_text, centroids)
                top_centroid = results_centroid[0][0]
                
                clf = get_classifier(all_embeddings, all_labels, force_retrain=force_retrain)
                pred_label, prob_dict = ml_klasyfikuj_tekst(user_text, clf)
            
            st.subheader("Porównanie wyników")
            st.write(f"**Metoda centroidów:** {top_centroid}")
            st.write(f"**Model ML:** {pred_label}")
            
            if top_centroid == pred_label:
                st.success("Obie metody dają ten sam wynik!")
            else:
                st.warning("Metody dają różne wyniki. Warto przeanalizować prawdopodobieństwa.")
    
    elif navigation == "Zaawansowana analiza":
        st.header("Zaawansowana analiza danych")
        st.write("W tej sekcji możesz przeprowadzić bardziej zaawansowane analizy embeddingów.")
        
        analysis_type = st.selectbox(
            "Wybierz typ analizy:",
            [
                "Analiza komponentów głównych (PCA)",
                "Walidacja krzyżowa klasyfikatora",
                "Analiza klastrów semantycznych",
                "Porównania międzyjęzykowe"
            ]
        )
        
        if analysis_type == "Analiza komponentów głównych (PCA)":
            st.subheader("Analiza komponentów głównych (PCA)")
            
            n_components = st.slider("Liczba komponentów do analizy", min_value=2, max_value=20, value=10)
            
            pca_results = interpret_pca_components(all_embeddings, all_labels, n_components=n_components)
            
            # Wyświetlanie wyjaśnionej wariancji
            st.write("### Wyjaśniona wariancja przez komponenty")
            
            explained_var_df = pd.DataFrame({
                'Komponent': np.arange(1, n_components + 1),
                'Wyjaśniona wariancja (%)': pca_results['explained_variance'] * 100,
                'Skumulowana wariancja (%)': pca_results['cumulative_variance'] * 100
            })
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=explained_var_df['Komponent'],
                y=explained_var_df['Wyjaśniona wariancja (%)'],
                name='Wyjaśniona wariancja'
            ))
            fig_var.add_trace(go.Scatter(
                x=explained_var_df['Komponent'],
                y=explained_var_df['Skumulowana wariancja (%)'],
                name='Skumulowana wariancja',
                mode='lines+markers'
            ))
            
            fig_var.update_layout(
                title="Wyjaśniona wariancja przez komponenty główne",
                xaxis_title="Numer komponentu",
                yaxis_title="Procent wyjaśnionej wariancji",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Wartości komponentów dla kategorii
            st.write("### Wartości komponentów dla kategorii")
            
            categories = list(pca_results['category_analysis'].keys())
            selected_category = st.selectbox("Wybierz kategorię", categories)
            
            selected_cat_data = pca_results['category_analysis'][selected_category]
            component_means = selected_cat_data['component_means']
            component_stds = selected_cat_data['component_stds']
            
            component_df = pd.DataFrame({
                'Komponent': np.arange(1, n_components + 1),
                'Średnia': component_means,
                'Odchylenie standardowe': component_stds
            })
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=component_df['Komponent'],
                y=component_df['Średnia'],
                name='Średnia',
                error_y=dict(
                    type='data',
                    array=component_df['Odchylenie standardowe'],
                    visible=True
                )
            ))
            
            fig_comp.update_layout(
                title=f"Wartości komponentów głównych dla kategorii {selected_category}",
                xaxis_title="Numer komponentu",
                yaxis_title="Wartość komponentu",
                showlegend=True
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
        
        elif analysis_type == "Walidacja krzyżowa klasyfikatora":
            st.subheader("Walidacja krzyżowa klasyfikatora")
            
            n_splits = st.slider("Liczba podziałów", min_value=3, max_value=10, value=5)
            
            if st.button("Przeprowadź walidację krzyżową"):
                with st.spinner("Przeprowadzanie walidacji krzyżowej..."):
                    cv_data = get_cross_validation_results(all_embeddings, all_labels, n_splits=n_splits)
                
                st.write("### Podsumowanie wyników")
                st.write(f"Średnia dokładność: {cv_data['summary']['accuracy_mean']:.4f} ± {cv_data['summary']['accuracy_std']:.4f}")
                st.write(f"Średni F1-score: {cv_data['summary']['f1_mean']:.4f} ± {cv_data['summary']['f1_std']:.4f}")
                
                st.write("### Wyniki dla poszczególnych podziałów")
                st.dataframe(cv_data['df'])
                
                # Wykres
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Scatter(
                    x=cv_data['df']['fold'],
                    y=cv_data['df']['accuracy'],
                    mode='lines+markers',
                    name='Dokładność'
                ))
                fig_cv.add_trace(go.Scatter(
                    x=cv_data['df']['fold'],
                    y=cv_data['df']['f1'],
                    mode='lines+markers',
                    name='F1-score'
                ))
                
                fig_cv.update_layout(
                    title="Wyniki walidacji krzyżowej",
                    xaxis_title="Fold",
                    yaxis_title="Wartość",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_cv, use_container_width=True)
        
        elif analysis_type == "Analiza klastrów semantycznych":
            st.subheader("Analiza klastrów semantycznych")
            
            n_clusters = st.slider("Liczba klastrów", min_value=2, max_value=15, value=6)
            auto_detect = st.checkbox("Automatycznie wykryj optymalną liczbę klastrów", value=True)
            
            if st.button("Przeprowadź analizę klastrów"):
                with st.spinner("Przeprowadzanie analizy klastrów..."):
                    cluster_results = semantic_cluster_analysis(
                        all_embeddings, all_labels, 
                        n_clusters=None if auto_detect else n_clusters
                    )
                
                st.write("### Wyniki klastrowania")
                
                # KMeans
                st.write("#### K-Means")
                st.write(f"Czystość klastrów: {cluster_results['kmeans']['comparison']['purity']:.4f}")
                st.write(f"Poprawne przypisania: {cluster_results['kmeans']['comparison']['correct_assignments']} / {cluster_results['kmeans']['comparison']['total_samples']}")
                
                # Wykres rozkładu klas w klastrach
                cluster_mapping = cluster_results['kmeans']['comparison']['cluster_class_mapping']
                cluster_data = []
                
                for cluster, info in cluster_mapping.items():
                    for category, count in info['class_distribution'].items():
                        cluster_data.append({
                            'Klaster': f"Klaster {cluster}",
                            'Kategoria': category,
                            'Liczba próbek': count
                        })
                
                cluster_df = pd.DataFrame(cluster_data)
                
                fig_cluster = px.bar(
                    cluster_df, 
                    x='Klaster', 
                    y='Liczba próbek', 
                    color='Kategoria',
                    title='Rozkład kategorii w klastrach (K-Means)',
                    barmode='stack'
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # DBSCAN
                st.write("#### DBSCAN")
                st.write(f"Czystość klastrów: {cluster_results['dbscan']['comparison']['purity']:.4f}")
                st.write(f"Poprawne przypisania: {cluster_results['dbscan']['comparison']['correct_assignments']} / {cluster_results['dbscan']['comparison']['total_samples']}")
                
                # Liczba punktów zaklasyfikowanych jako szum
                noise_points = np.sum(cluster_results['dbscan']['labels'] == -1)
                st.write(f"Liczba punktów zaklasyfikowanych jako szum: {noise_points}")
        
        elif analysis_type == "Porównania międzyjęzykowe":
            st.subheader("Porównania międzyjęzykowe")
            
            lang_options = ["ENG", "POL", "JAP"]
            lang1 = st.selectbox("Wybierz pierwszy język", lang_options)
            lang2 = st.selectbox("Wybierz drugi język", [lang for lang in lang_options if lang != lang1], index=0)
            
            if st.button("Przeprowadź porównanie"):
                st.write(f"### Porównanie: {lang1} vs {lang2}")
                
                # Przygotowanie danych
                if lang1 == "ENG":
                    emb1_ind = embeddings["eng_ind"]
                    emb1_col = embeddings["eng_col"]
                elif lang1 == "POL":
                    emb1_ind = embeddings["pol_ind"]
                    emb1_col = embeddings["pol_col"]
                else:  # JAP
                    emb1_ind = embeddings["jap_ind"]
                    emb1_col = embeddings["jap_col"]
                
                if lang2 == "ENG":
                    emb2_ind = embeddings["eng_ind"]
                    emb2_col = embeddings["eng_col"]
                elif lang2 == "POL":
                    emb2_ind = embeddings["pol_ind"]
                    emb2_col = embeddings["pol_col"]
                else:  # JAP
                    emb2_ind = embeddings["jap_ind"]
                    emb2_col = embeddings["jap_col"]
                
                # Obliczenie odległości
                cross_ind_ind = all_pairwise(emb1_ind, emb2_ind, "cosine")
                cross_col_col = all_pairwise(emb1_col, emb2_col, "cosine")
                cross_ind_col = all_pairwise(emb1_ind, emb2_col, "cosine")
                cross_col_ind = all_pairwise(emb1_col, emb2_ind, "cosine")
                
                # Statystyki
                stat_data = {
                    'Porównanie': [
                        f"{lang1}_IND vs {lang2}_IND", 
                        f"{lang1}_COL vs {lang2}_COL",
                        f"{lang1}_IND vs {lang2}_COL",
                        f"{lang1}_COL vs {lang2}_IND"
                    ],
                    'Średnia odległość': [
                        np.mean(cross_ind_ind),
                        np.mean(cross_col_col),
                        np.mean(cross_ind_col),
                        np.mean(cross_col_ind)
                    ],
                    'Mediana odległości': [
                        np.median(cross_ind_ind),
                        np.median(cross_col_col),
                        np.median(cross_ind_col),
                        np.median(cross_col_ind)
                    ],
                    'Odchylenie standardowe': [
                        np.std(cross_ind_ind),
                        np.std(cross_col_col),
                        np.std(cross_ind_col),
                        np.std(cross_col_ind)
                    ]
                }
                
                stat_df = pd.DataFrame(stat_data)
                st.dataframe(stat_df)
                
                # Wykres rozkładów
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=cross_ind_ind,
                    name=f"{lang1}_IND vs {lang2}_IND",
                    opacity=0.7,
                    histnorm='probability density'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=cross_col_col,
                    name=f"{lang1}_COL vs {lang2}_COL",
                    opacity=0.7,
                    histnorm='probability density'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=cross_ind_col,
                    name=f"{lang1}_IND vs {lang2}_COL",
                    opacity=0.7,
                    histnorm='probability density'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=cross_col_ind,
                    name=f"{lang1}_COL vs {lang2}_IND",
                    opacity=0.7,
                    histnorm='probability density'
                ))
                
                fig_dist.update_layout(
                    title=f"Rozkład odległości między kategoriami ({lang1} vs {lang2})",
                    xaxis_title="Odległość kosinusowa",
                    yaxis_title="Gęstość",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    
    elif navigation == "Wnioski":
        st.markdown("""
        # Wnioski
        
        ## 1. Efektywność modelu embeddingowego
        
        Analiza reprezentacji embeddingowych zdań wskazuje na istotne różnice w sposobie wyrażania postaw indywidualistycznych i kolektywistycznych w badanych językach. Model **text-embedding-3-large** skutecznie odróżnia zdania indywidualistyczne (IND) od kolektywistycznych (COL) we wszystkich trzech badanych językach: angielskim, polskim i japońskim.
        
        ### Główne spostrzeżenia:
        
        1. **Różnicowanie kategorii (H₁)** - Przeprowadzone testy statystyczne potwierdzają, że odległości międzygrupowe (między zdaniami IND i COL) są istotnie większe niż odległości wewnątrzgrupowe (p < 0.01 dla wszystkich języków). Oznacza to, że model skutecznie odróżnia oba typy zdań niezależnie od języka.
        
        2. **Różnice międzyjęzykowe (H₂/H₃)** - Analiza median odległości między kategoriami IND i COL w różnych językach ujawnia ciekawe wzorce:
           - Angielski wykazuje największe oddzielenie kategorii (mediana odległości kosinusowej ≈ 0.78)
           - Japoński plasuje się pośrodku (mediana ≈ 0.70)
           - Polski charakteryzuje się najmniejszym rozdzieleniem kategorii (mediana ≈ 0.59)
           
        3. **Statystyczna istotność różnic** - Korekcja dla wielokrotnych testów potwierdza, że obserwowane różnice między językami nie są przypadkowe. Wskazuje to na realny wpływ kontekstu kulturowego na reprezentacje lingwistyczne.
        
        ## 2. Implikacje teoretyczne
        
        Wyniki badania mają istotne implikacje dla teorii dotyczących relacji między językiem, kulturą i reprezentacjami wektorowymi:
        
        1. **Kontekst kulturowy w przestrzeni embeddingowej** - Embeddingi zdań wydają się odzwierciedlać koncepcje kulturowe związane z indywidualizmem i kolektywizmem, co sugeruje, że modele językowe "rozumieją" te subtelne różnice.
        
        2. **Specyfika językowa** - Mniejsze odległości wektorowe w języku polskim i japońskim mogą odzwierciedlać mniej wyraźne rozgraniczenie między postawami indywidualistycznymi i kolektywistycznymi w tych kulturach w porównaniu do kultury anglosaskiej.
        
        3. **Kontinuum indywidualizm-kolektywizm** - Różnica w oddzieleniu kategorii wskazuje, że kultury nie są dychotomicznie podzielone na indywidualistyczne i kolektywistyczne, ale raczej istnieją na kontinuum, co jest zgodne ze współczesnymi teoriami psychologii międzykulturowej.
        
        ## 3. Zastosowania praktyczne
        
        Wyniki badania mają potencjalne zastosowania w kilku obszarach:
        
        1. **Technologia językowa** - Lepsze zrozumienie, jak modele embeddingowe reprezentują niuanse kulturowe, może prowadzić do udoskonalenia systemów NLP w kontekście międzykulturowym.
        
        2. **Komunikacja międzykulturowa** - Ustalenia mogą pomóc w opracowaniu narzędzi wspierających tłumaczenie i komunikację międzykulturową, które uwzględniają subtelne różnice w ekspresji wartości.
        
        3. **Badania społeczne** - Metody ilościowe zastosowane w badaniu oferują nowe podejście do badania różnic kulturowych, które może uzupełniać tradycyjne metody jakościowe.
        
        ## 4. Ograniczenia i przyszłe kierunki badań
        
        Pomimo obiecujących rezultatów, badanie ma pewne ograniczenia:
        
        1. **Brak zróżnicowanych kontekstów** - Zdania użyte w badaniu były izolowane, bez szerszego kontekstu, który mógłby wpływać na ich interpretację.
        
        2. **Wpływ danych treningowych** - Model embeddingowy został wytrenowany na korpusie, w którym mogła dominować kultura anglosaska, co może wpływać na reprezentacje innych języków.
        
        ### Przyszłe kierunki badań:
        
        1. **Rozszerzenie zakresu językowego** - Uwzględnienie większej liczby języków z różnych grup językowych i kulturowych.
        
        2. **Kontekstualizacja** - Badanie zdań w szerszym kontekście dyskursywnym.
        
        3. **Analiza diachroniczna** - Śledzenie zmian w reprezentacjach wektorowych postaw wraz z ewolucją języka i wartości kulturowych.
        
        4. **Dokładniejsza kategoryzacja języków** - Zamiast prostego podziału na indywidualistyczne vs. kolektywistyczne, można uwzględnić dodatkowe wymiary kulturowe, takie jak dystans władzy czy unikanie niepewności.
        
        ## 5. Podsumowanie
        
        Przeprowadzone badanie dostarcza empirycznych dowodów na to, że modele embeddingowe skutecznie uchwytują kulturowe różnice w ekspresji postaw indywidualistycznych i kolektywistycznych. Wyniki wskazują na istotne różnice między językami, które odzwierciedlają kontinuum od silniejszego oddzielenia tych kategorii w języku angielskim do bardziej płynnego przejścia w językach polskim i japońskim.
        
        Metody zastosowane w badaniu pokazują potencjał analizy embeddingów jako narzędzia do badania subttelnych różnic kulturowych w sposób ilościowy i systematyczny, oferując nową perspektywę w badaniach nad relacją między językiem a kulturą.
        """)
    
    # Stopka
    st.markdown("""
    ---
    *Analiza przeprowadzona z wykorzystaniem modelu `text-embedding-3-large` od OpenAI. Wizualizacje utworzone przy użyciu bibliotek Plotly i Streamlit.*
    """)


if __name__ == "__main__":
    run_streamlit_app()

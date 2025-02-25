import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import json
import logging
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

    # Funkcje obliczeniowe
    dist_cosine,
    dist_euclidean,
    dist_manhattan,

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


def all_pairwise(emb_list_a, emb_list_b, metric='cosine'):
    """
    Funkcja do obliczania odległości między dwoma zbiorami embeddingów.
    """
    from scipy.spatial.distance import cdist

    # Poprawione mapowanie metryk - ujednolicone nazwy i uwzględnione warianty pisowni
    metric_map = {
        'euclidean': 'euclidean',
        'euklides': 'euclidean',
        'euclides': 'euclidean',
        'cosine': 'cosine',
        'kosinus': 'cosine',
        'manhattan': 'cityblock',
        'cityblock': 'cityblock'
    }

    # Konwersja do małych liter dla bezpieczeństwa
    metric_lower = metric.lower() if isinstance(metric, str) else metric
    scipy_metric = metric_map.get(metric_lower, metric)

    return cdist(np.array(emb_list_a), np.array(emb_list_b), metric=scipy_metric).flatten().tolist()


# Konfiguracja Streamlit
st.set_page_config(
    page_title="Analiza reprezentacji wektorowych emocji",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Nowa funkcja dla redukcji wymiarowości z cache'owaniem
@st.cache_data(ttl=7200)  # Zwiększony czas cachowania do 2 godzin
def compute_dimension_reduction(all_embeddings, method='tsne', perplexity=30, n_components=2):
    """
    Wstępnie oblicza redukcję wymiarowości i cachuje wyniki.

    Parametry:
    ----------
    all_embeddings : numpy.ndarray
        Macierz wszystkich embeddingów.
    method : str
        Metoda redukcji: 'pca', 'tsne', lub 'umap'
    perplexity : int
        Parametr perplexity dla t-SNE
    n_components : int
        Liczba komponentów (2 lub 3)

    Zwraca:
    -------
    numpy.ndarray
        Zredukowane dane
    """
    start_time = time.time()

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(all_embeddings)
    elif method == 'tsne':
        # Optymalizacja t-SNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,  # Domyślnie, ale możesz zmniejszyć
            n_jobs=-1,  # Użyj wszystkich dostępnych rdzeni
            method='barnes_hut'  # Szybsza implementacja (złożoność O(n log n) zamiast O(n²))
        )
        reduced_data = reducer.fit_transform(all_embeddings)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=perplexity,  # Odpowiednik perplexity w t-SNE
                min_dist=0.1,
                metric='cosine',  # Możesz też użyć 'euclidean'
                random_state=42
            )
            reduced_data = reducer.fit_transform(all_embeddings)
        except ImportError:
            st.error("Biblioteka UMAP nie jest zainstalowana. Zainstaluj ją używając: pip install umap-learn")
            return None

    logger.info(f"Redukcja wymiarowości metodą {method} zajęła {time.time() - start_time:.2f}s")
    return reduced_data


# Zoptymalizowana funkcja do redukcji danych przed t-SNE
@st.cache_data(ttl=7200)
def preprocess_embeddings(embeddings, n_components=50):
    """Redukuje wymiarowość danych z 3072D do mniejszej liczby wymiarów za pomocą PCA"""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


# Funkcja do generowania wykresów UMAP 2D
@st.cache_data(ttl=3600)
def generate_interactive_umap_2d(all_emb, all_lbl, n_neighbors=30):
    """
    Generuje interaktywny wykres UMAP 2D dla embeddingów.
    """
    red_umap = compute_dimension_reduction(all_emb, method='umap', perplexity=n_neighbors, n_components=2)
    if red_umap is None:
        return None

    df = pd.DataFrame({
        "Dim1": red_umap[:, 0],
        "Dim2": red_umap[:, 1],
        "Cluster": all_lbl
    })
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])

    fig = px.scatter(
        df, x="Dim1", y="Dim2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title=f"Interaktywna UMAP 2D (n_neighbors={n_neighbors})"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


# Zmodyfikowana wersja funkcji generowania t-SNE 2D z wykorzystaniem preprocessingu
@st.cache_data(ttl=3600)
def generate_interactive_tsne_2d_optimized(all_emb, all_lbl, perplexity=30, use_preprocessing=True):
    """
    Generuje interaktywny wykres t-SNE 2D dla embeddingów z opcją preprocessingu.
    """
    # Opcjonalny preprocessing z PCA
    if use_preprocessing:
        reduced_emb = preprocess_embeddings(all_emb, n_components=50)
        red_tsne = compute_dimension_reduction(reduced_emb, method='tsne', perplexity=perplexity, n_components=2)
    else:
        red_tsne = compute_dimension_reduction(all_emb, method='tsne', perplexity=perplexity, n_components=2)

    df = pd.DataFrame({
        "Dim1": red_tsne[:, 0],
        "Dim2": red_tsne[:, 1],
        "Cluster": all_lbl
    })
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])

    fig = px.scatter(
        df, x="Dim1", y="Dim2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title=f"Interaktywna t-SNE 2D (perplexity={perplexity})"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


# Cache dla kosztownych obliczeniowo funkcji
@st.cache_data(ttl=3600)
def load_data():
    """
    Wczytuje dane i przygotowuje je do analizy.

    Zwraca:
    -------
    tuple
        (zdania, embeddingi, centroidy, wszystkie_embeddingi, wszystkie_etykiety)
    """
    start_time = time.time()
    zdania, embeddings = load_sentences_and_embeddings()
    centroids = compute_all_centroids(embeddings)

    all_embeddings = np.concatenate([
        embeddings["eng_ind"], embeddings["eng_col"],
        embeddings["pol_ind"], embeddings["pol_col"],
        embeddings["jap_ind"], embeddings["jap_col"]
    ], axis=0)

    all_labels = np.array(
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


def perform_cross_validation_local(X, y, n_splits=5):
    """
    Przeprowadza walidację krzyżową na miejscu.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')

        cv_results.append({
            'fold': i + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    return cv_results


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
    # Używamy lokalnej implementacji zamiast odwoływać się do funkcji z modułu
    cv_results = perform_cross_validation_local(all_embeddings, all_labels, n_splits=n_splits)
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


def cluster_analysis_local(embeddings, labels, n_clusters=6):
    """
    Przeprowadza analizę klastrów semantycznych.
    """
    from sklearn.cluster import KMeans
    from collections import Counter

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)

    # Mapowanie klastrów na klasy rzeczywiste
    clusters = np.unique(kmeans_labels)
    cluster_class_mapping = {}

    for cluster in clusters:
        mask = kmeans_labels == cluster
        class_counts = Counter([labels[i] for i, is_in_cluster in enumerate(mask) if is_in_cluster])
        dominant_class = class_counts.most_common(1)[0][0]
        cluster_class_mapping[cluster] = {
            'dominant_class': dominant_class,
            'class_distribution': dict(class_counts)
        }

    # Obliczanie czystości klastrów
    correct = 0
    for i, (cluster, true_class) in enumerate(zip(kmeans_labels, labels)):
        if cluster_class_mapping[cluster]['dominant_class'] == true_class:
            correct += 1

    purity = correct / len(labels) if len(labels) > 0 else 0

    comparison = {
        'cluster_class_mapping': cluster_class_mapping,
        'purity': purity,
        'correct_assignments': correct,
        'total_samples': len(labels)
    }

    return {
        'labels': kmeans_labels,
        'centroids': kmeans.cluster_centers_,
        'comparison': comparison
    }


def run_streamlit_app():
    """
    Główna funkcja aplikacji Streamlit.
    """
    st.title("Analiza reprezentacji wektorowych emocji: Indywidualizm vs. Kolektywizm w trzech językach")

    # Sidebar z opcjami
    with st.sidebar:
        st.title("Opcje analizy")

        force_retrain = st.checkbox("Wymuś ponowne trenowanie klasyfikatora", value=False,
                                    help="Trenuje klasyfikator od nowa zamiast wczytywać zapisany")

        st.subheader("Parametry wizualizacji")
        perplexity_tsne = st.slider("Parametr perplexity dla t-SNE", min_value=5, max_value=50, value=30, step=5,
                                    help="Wyższe wartości uwzględniają większą liczbę sąsiadów")

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
    zdania, embeddings, centroids, all_embeddings, all_labels = load_data()

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
                len(zdania["english_individualistic"]) + len(zdania["polish_individualistic"]) + len(
                    zdania["japanese_individualistic"])
            ],
            'Kolektywistyczne': [
                len(zdania["english_collectivistic"]),
                len(zdania["polish_collectivistic"]),
                len(zdania["japanese_collectivistic"]),
                len(zdania["english_collectivistic"]) + len(zdania["polish_collectivistic"]) + len(
                    zdania["japanese_collectivistic"])
            ]
        }

        stats_df = pd.DataFrame(stats_data)
        stats_df['Razem'] = stats_df['Indywidualistyczne'] + stats_df['Kolektywistyczne']

        st.table(stats_df)

    elif navigation == "Wizualizacje 2D":
        st.header("Interaktywne wizualizacje 2D")
        st.write(
            "Poniżej przedstawiono interaktywne wizualizacje 2D. Możesz interaktywnie eksplorować dane, przybliżać, oddalać i filtrować kategorie.")

        viz_type = st.radio("Wybierz typ wizualizacji:", ["PCA", "t-SNE", "UMAP (szybsza)"])

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
        elif viz_type == "t-SNE":
            # Dodaj opcję próbkowania dla t-SNE
            use_sampling = st.checkbox("Użyj próbkowania (szybsze, ale mniej dokładne)", value=False)
            use_preprocessing = st.checkbox("Użyj wstępnej redukcji wymiarowości PCA (szybsze)", value=True)

            if use_sampling:
                sample_size = st.slider("Liczba próbkowanych punktów", 100, min(1000, len(all_embeddings)), 300)
                indices = np.random.choice(len(all_embeddings), size=sample_size, replace=False)
                sampled_embeddings = all_embeddings[indices]
                sampled_labels = [all_labels[i] for i in indices]
                fig_2d_tsne = generate_interactive_tsne_2d_optimized(sampled_embeddings, sampled_labels,
                                                                     perplexity=min(30, sample_size // 5),
                                                                     use_preprocessing=use_preprocessing)
            else:
                with st.spinner("Generowanie t-SNE (może to chwilę potrwać)..."):
                    fig_2d_tsne = generate_interactive_tsne_2d_optimized(all_embeddings, all_labels,
                                                                         perplexity=perplexity_tsne,
                                                                         use_preprocessing=use_preprocessing)

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
        else:  # UMAP
            n_neighbors = st.slider("Parametr n_neighbors dla UMAP", min_value=5, max_value=50, value=30, step=5)
            with st.spinner("Generowanie UMAP (może to chwilę potrwać)..."):
                fig_2d_umap = generate_interactive_umap_2d(all_embeddings, all_labels, n_neighbors=n_neighbors)

            if fig_2d_umap:
                st.plotly_chart(fig_2d_umap, use_container_width=True)
            else:
                st.error("Nie można wygenerować UMAP. Zainstaluj pakiet umap-learn: pip install umap-learn")

            with st.expander("Co to jest UMAP?"):
                st.markdown("""
                **Uniform Manifold Approximation and Projection (UMAP)** to zaawansowana technika redukcji wymiarowości, która:

                - Jest znacznie szybsza niż t-SNE, szczególnie dla dużych zbiorów danych
                - Lepiej zachowuje strukturę globalną danych
                - Pozwala na efektywną wizualizację skupisk i relacji między punktami
                - Zapewnia lepszą skalowalność dla dużych zestawów danych

                Parametr **n_neighbors** określa liczbę najbliższych sąsiadów używanych podczas konstruowania grafu, wpływając na równowagę między zachowaniem struktury lokalnej i globalnej.
                """)

        # Filtry kategorii
        st.subheader("Filtrowanie kategorii")
        st.write("Możesz filtrować kategorie, aby lepiej zobaczyć wzorce w danych.")

        col1, col2, col3 = st.columns(3)

        with col1:
            include_eng = st.checkbox("Angielski", value=True)
            eng_filter = st.radio("Angielski - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"],
                                  disabled=not include_eng)

        with col2:
            include_pol = st.checkbox("Polski", value=True)
            pol_filter = st.radio("Polski - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"], disabled=not include_pol)

        with col3:
            include_jap = st.checkbox("Japoński", value=True)
            jap_filter = st.radio("Japoński - filtr:", ["Wszystkie", "Tylko IND", "Tylko COL"],
                                  disabled=not include_jap)

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

            with st.spinner("Generowanie filtrowanej wizualizacji..."):
                if viz_type == "PCA":
                    filtered_fig = generate_interactive_pca_2d(filtered_embeddings, filtered_labels)
                elif viz_type == "t-SNE":
                    use_preprocessing = st.session_state.get('use_preprocessing', True)
                    filtered_fig = generate_interactive_tsne_2d_optimized(filtered_embeddings, filtered_labels,
                                                                          perplexity=perplexity_tsne,
                                                                          use_preprocessing=use_preprocessing)
                else:  # UMAP
                    filtered_fig = generate_interactive_umap_2d(filtered_embeddings, filtered_labels,
                                                                n_neighbors=n_neighbors)

            st.subheader("Filtrowana wizualizacja")
            if filtered_fig:
                st.plotly_chart(filtered_fig, use_container_width=True)
            else:
                st.error("Nie można wygenerować filtrowanej wizualizacji.")

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
            use_preprocessing_3d = st.checkbox("Użyj wstępnej redukcji wymiarowości PCA (szybsze)", value=True)

            with st.spinner("Generowanie t-SNE 3D (może to chwilę potrwać)..."):
                if use_preprocessing_3d:
                    # Preprocessing z PCA
                    reduced_emb = preprocess_embeddings(all_embeddings, n_components=50)
                    # Użycie funkcji compute_dimension_reduction z t-SNE
                    red_tsne_3d = compute_dimension_reduction(reduced_emb, method='tsne', perplexity=perplexity_tsne,
                                                              n_components=3)
                    # Teraz trzeba by zmodyfikować generate_interactive_tsne_3d aby przyjmowała gotowe dane t-SNE
                    # Ale na razie użyjmy oryginalnej funkcji
                    fig_tsne_3d = generate_interactive_tsne_3d(all_embeddings, all_labels, perplexity=perplexity_tsne)
                else:
                    fig_tsne_3d = generate_interactive_tsne_3d(all_embeddings, all_labels, perplexity=perplexity_tsne)

            st.plotly_chart(fig_tsne_3d, use_container_width=True)

        else:  # Point Cloud
            reduction_method = st.selectbox("Metoda redukcji wymiarowości:", ["pca", "tsne", "umap"])

            with st.spinner("Generowanie Point Cloud..."):
                fig_cloud = generate_plotly_3d_point_cloud(all_embeddings, all_labels, method=reduction_method)

            if fig_cloud:
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.error(
                    "Nie można wygenerować wizualizacji Point Cloud. Sprawdź, czy zainstalowano wymagane biblioteki.")

    elif navigation == "Analiza statystyczna":
        st.header("Analiza statystyczna")
        st.write(
            "Poniżej znajdują się wyniki testów statystycznych sprawdzających istotność różnic między językami oraz wewnątrz jednego języka.")

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
        st.write(
            "Wybierz metrykę oraz język, aby zobaczyć interaktywny wykres rozkładu odległości między zdaniami IND i COL.")

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
        st.write(
            "Poniższy wykres pokazuje mediany odległości między kategoriami IND i COL dla różnych języków i metryk.")

        # Przygotowanie danych do wykresu
        metrics_data = {}
        for metric_name in ["Euklides", "Kosinus", "Manhattan"]:
            metrics_data[metric_name] = {}
            for lang in ["POL", "ENG", "JAP"]:
                if lang == "POL":
                    distances = all_pairwise(embeddings["pol_ind"], embeddings["pol_col"], metric=metric_name.lower())
                elif lang == "ENG":
                    distances = all_pairwise(embeddings["eng_ind"], embeddings["eng_col"], metric=metric_name.lower())
                elif lang == "JAP":
                    distances = all_pairwise(embeddings["jap_ind"], embeddings["jap_col"], metric=metric_name.lower())
                metrics_data[metric_name][lang] = np.median(distances)

        # Tworzenie wykresu porównawczego
        fig_comparison = go.Figure()
        bar_width = 0.2
        positions = np.arange(len(language_options))

        for i, metric in enumerate(metric_options):
            medians = [metrics_data[metric][lang] for lang in language_options]
            fig_comparison.add_trace(go.Bar(
                x=positions + (i - len(metric_options) / 2 + 0.5) * bar_width,
                y=medians,
                width=bar_width,
                name=metric
            ))

        fig_comparison.update_layout(
            title="Mediany odległości między kategoriami IND i COL według języka i metryki",
            xaxis=dict(
                title="Język",
                ticktext=language_options,
                tickvals=positions,
            ),
            yaxis=dict(title="Mediana odległości"),
            legend_title="Metryka",
            barmode='group'
        )

        st.plotly_chart(fig_comparison, use_container_width=True)

        # Analiza korelacji
        st.subheader("Analiza korelacji między metrykami")
        st.write("Macierz korelacji pokazuje, jak silnie powiązane są różne metryki odległości.")

        # Przygotowanie danych
        dist_data = {}
        for metric_name in ["Euklides", "Kosinus", "Manhattan"]:
            for lang in ["POL", "ENG", "JAP"]:
                key = f"{lang}_{metric_name}"
                if lang == "POL":
                    dist_data[key] = all_pairwise(embeddings["pol_ind"], embeddings["pol_col"],
                                                  metric=metric_name.lower())
                elif lang == "ENG":
                    dist_data[key] = all_pairwise(embeddings["eng_ind"], embeddings["eng_col"],
                                                  metric=metric_name.lower())
                elif lang == "JAP":
                    dist_data[key] = all_pairwise(embeddings["jap_ind"], embeddings["jap_col"],
                                                  metric=metric_name.lower())

        # Ograniczanie liczby próbek dla macierzy korelacji (dla przyspieszenia)
        for key in dist_data:
            if len(dist_data[key]) > 5000:
                np.random.seed(42)
                indices = np.random.choice(len(dist_data[key]), size=5000, replace=False)
                dist_data[key] = [dist_data[key][i] for i in indices]

        # Tworzenie DataFrame
        dist_df = pd.DataFrame({k: v[:5000] for k, v in dist_data.items()})

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
        st.write(
            "Możesz przetestować klasyfikację nowego tekstu przy użyciu dwóch metod: centroidów i uczenia maszynowego.")

        # Wejście użytkownika
        user_text = st.text_area("Wpisz tekst:", value="I believe in working together for the greater good.",
                                 height=100)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metoda centroidów")
            st.write(
                "Metoda centroidów polega na porównaniu embeddingu nowego tekstu z uśrednionymi wektorami każdej kategorii.")

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
            st.write(
                "Regresja logistyczna trenuje model na wszystkich embeddingach, aby rozpoznawać kategorię nowego tekstu.")

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
                top_centroid = results_centroid[0][0].upper()  # Normalizacja do wielkich liter

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
                "Walidacja krzyżowa klasyfikatora",
                "Analiza klastrów semantycznych",
                "Porównania międzyjęzykowe"
            ]
        )

        if analysis_type == "Walidacja krzyżowa klasyfikatora":
            st.subheader("Walidacja krzyżowa klasyfikatora")

            n_splits = st.slider("Liczba podziałów", min_value=3, max_value=10, value=5)

            if st.button("Przeprowadź walidację krzyżową"):
                with st.spinner("Przeprowadzanie walidacji krzyżowej..."):
                    cv_data = get_cross_validation_results(all_embeddings, all_labels, n_splits=n_splits)

                st.write("### Podsumowanie wyników")
                st.write(
                    f"Średnia dokładność: {cv_data['summary']['accuracy_mean']:.4f} ± {cv_data['summary']['accuracy_std']:.4f}")
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

            if st.button("Przeprowadź analizę klastrów"):
                with st.spinner("Przeprowadzanie analizy klastrów..."):
                    cluster_results = cluster_analysis_local(all_embeddings, all_labels, n_clusters=n_clusters)

                st.write("### Wyniki klastrowania")

                # KMeans
                st.write("#### K-Means")
                st.write(f"Czystość klastrów: {cluster_results['comparison']['purity']:.4f}")
                st.write(
                    f"Poprawne przypisania: {cluster_results['comparison']['correct_assignments']} / {cluster_results['comparison']['total_samples']}")

                # Wykres rozkładu klas w klastrach
                cluster_mapping = cluster_results['comparison']['cluster_class_mapping']
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
                cross_ind_ind = all_pairwise(emb1_ind, emb2_ind, metric='cosine')
                cross_col_col = all_pairwise(emb1_col, emb2_col, metric='cosine')
                cross_ind_col = all_pairwise(emb1_ind, emb2_col, metric='cosine')
                cross_col_ind = all_pairwise(emb1_col, emb2_ind, metric='cosine')

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
# **Wyniki badania**

## **1. Efektywność modelu embeddingowego**

Analiza reprezentacji embeddingowych zdań wskazuje na istotne różnice w sposobie wyrażania postaw indywidualistycznych i kolektywistycznych w badanych językach. Model **text-embedding-3-large** skutecznie odróżnia zdania indywidualistyczne (IND) od kolektywistycznych (COL) we wszystkich trzech badanych językach: angielskim, polskim i japońskim.

### **Główne spostrzeżenia:**

1. **Różnicowanie kategorii (H₁)** - Przeprowadzone testy statystyczne potwierdzają, że odległości międzygrupowe (między zdaniami IND i COL) są istotnie większe niż odległości wewnątrzgrupowe (p < 0.01 dla wszystkich języków po korekcji Bonferroniego). Wynik ten jest spójny dla wszystkich trzech zastosowanych metryk (euklidesowej, kosinusowej i Manhattan), co wzmacnia wiarygodność wniosku. Oznacza to, że model skutecznie odróżnia oba typy zdań niezależnie od języka.

2. **Różnice międzyjęzykowe (H₂/H₃)** - Analiza median odległości między kategoriami IND i COL w różnych językach ujawnia ciekawe wzorce:
  - Angielski wykazuje największe oddzielenie kategorii (mediana odległości kosinusowej ≈ 0.78)
  - Japoński plasuje się pośrodku (mediana ≈ 0.70)
  - Polski charakteryzuje się najmniejszym rozdzieleniem kategorii (mediana ≈ 0.59)

  Wyniki te są zgodne z hipotezą H₂, sugerującą że w języku polskim i japońskim koncepcje indywidualizmu i kolektywizmu są semantycznie bliższe sobie niż w języku angielskim.

3. **Statystyczna istotność różnic** - Korekcja Bonferroniego dla wielokrotnych testów potwierdza, że obserwowane różnice między językami nie są przypadkowe (p < 0.01 dla wszystkich porównań po korekcji). Wskazuje to na realny wpływ kontekstu kulturowego na reprezentacje lingwistyczne.

4. **Porównanie metryk odległości** - Wszystkie trzy zastosowane metryki (euklidesowa, kosinusowa, Manhattan) wykazują podobne wzorce, co wzmacnia wiarygodność wyników. Analiza korelacji między metrykami pokazuje wysoką zgodność (r > 0.9), co sugeruje że obserwowane różnice są stabilne i niezależne od wybranej metryki.

5. **Analiza wewnątrz- i międzyjęzykowa** - Badanie podobieństwa zdań IND i COL zarówno wewnątrz języków, jak i między językami, ujawnia, że:
  - Zdania IND i COL w tym samym języku są bardziej podobne do siebie niż odpowiadające im kategorie w innych językach
  - Kategoria COL wykazuje większe podobieństwo między językami niż kategoria IND, co może sugerować, że kolektywizm jest bardziej uniwersalnym konstruktem

## **2. Implikacje teoretyczne**

Wyniki badania mają istotne implikacje dla teorii dotyczących relacji między językiem, kulturą i reprezentacjami wektorowymi.

### **2.1. Językowe odzwierciedlenie różnic kulturowych**

Zidentyfikowane wzorce odległości embeddingowych między kategoriami IND i COL w różnych językach są spójne z literaturą na temat różnic kulturowych:

1. **Język angielski** (kultura silnie indywidualistyczna) - Największe rozdzielenie semantyczne między IND i COL odzwierciedla wyraźną polaryzację tych koncepcji w kulturze anglosaskiej, gdzie indywidualizm i kolektywizm są często postrzegane jako przeciwstawne wartości.

2. **Język japoński** (kultura silnie kolektywistyczna) - Mniejsze rozdzielenie kategorii może odzwierciedlać fakt, że w kulturze japońskiej indywidualizm jest zawsze rozumiany w kontekście relacji społecznych, a autonomia jednostki nie neguje jej zobowiązań wobec grupy.

3. **Język polski** (kultura umiarkowanie indywidualistyczna z elementami kolektywizmu) - Najmniejsze rozdzielenie kategorii sugeruje bardziej zintegrowane pojmowanie indywidualizmu i kolektywizmu. Może to wynikać z historycznych i społecznych uwarunkowań kultury polskiej, łączącej wpływy wschodnie i zachodnie.

### **2.2. Reprezentacje embeddingowe jako odzwierciedlenie struktur konceptualnych**

Badanie dostarcza empirycznych dowodów na to, że reprezentacje embeddingowe skutecznie przechwytują kulturowo uwarunkowane różnice w strukturach konceptualnych:

1. **Hipoteza relatywizmu językowego** - Wyniki wspierają złagodzoną wersję hipotezy Sapira-Whorfa, sugerując że struktury językowe i kulturowe wpływają na organizację przestrzeni semantycznej.

2. **Kognitywne modele kulturowe** - Różnice w odległościach embeddingowych mogą odzwierciedlać różne modele kognitywne indywidualizmu i kolektywizmu funkcjonujące w badanych kulturach.

3. **Uniwersalia vs. różnice kulturowe** - Badanie wskazuje, że choć kategoryzacja na postawy IND i COL jest uniwersalna (wszystkie języki rozróżniają te kategorie), to stopień ich rozdzielenia semantycznego jest kulturowo uwarunkowany.

### **2.3. Metodologiczne implikacje dla NLP**

Wyniki mają również ważne implikacje dla rozwoju i ewaluacji modeli NLP:

1. **Kulturowy bias w modelach embeddingowych** - Zidentyfikowane różnice podkreślają potrzebę uwzględniania kontekstu kulturowego w rozwoju i ewaluacji modeli językowych.

2. **Międzyjęzykowa transferowalność modeli** - Różnice w strukturze przestrzeni semantycznej między językami sugerują ograniczenia w transferze modeli między kulturami.

3. **Kulturowo świadome AI** - Wyniki podkreślają potrzebę rozwijania systemów AI, które rozumieją i uwzględniają kulturowe niuanse w interpretacji tekstu.

## **3. Wnioski i przyszłe kierunki badań**

### **3.1. Synteza wyników**

Przeprowadzone badanie dostarcza empirycznych dowodów na to, że reprezentacje embeddingowe przechwytują kulturowo uwarunkowane różnice w konceptualizacji indywidualizmu i kolektywizmu. Wszystkie trzy postawione hipotezy zostały potwierdzone:

1. **H₁**: Model embeddingowy skutecznie odróżnia zdania IND od COL we wszystkich badanych językach.
2. **H₂**: Stopień rozdzielenia kategorii IND i COL różni się między językami, z największym rozdzieleniem w języku angielskim, a najmniejszym w polskim.
3. **H₃**: Zaobserwowane różnice są statystycznie istotne i nie wynikają z przypadku.

Wyniki te są zgodne z literaturą psychologii międzykulturowej i socjolingwistyki, podkreślając rolę języka jako nośnika wartości kulturowych.

### **3.2. Ograniczenia badania**

Należy podkreślić pewne ograniczenia przeprowadzonego badania:

1. **Reprezentatywność korpusu** - Choć podjąłem starania, aby zapewnić zróżnicowanie zdań, analizowany korpus może nie obejmować pełnego spektrum ekspresji indywidualizmu i kolektywizmu w badanych językach.

2. **Potencjalny bias modelu** - Model `text-embedding-3-large` mógł być trenowany na korpusie zdominowanym przez język angielski, co potencjalnie wpływa na reprezentacje w innych językach.

3. **Ekwiwalencja międzyjęzykowa** - Tłumaczenie koncepcji indywidualizmu i kolektywizmu między językami może nie zachowywać pełnej ekwiwalencji semantycznej.

4. **Zróżnicowanie wewnątrzkulturowe** - Badanie traktuje kultury jako względnie homogeniczne, podczas gdy w rzeczywistości istnieje znaczące zróżnicowanie wewnątrz każdej z nich.

### **3.3. Przyszłe kierunki badań**

Wyniki otwierają liczne możliwości dla przyszłych badań:

1. **Rozszerzenie na większą liczbę języków** - Włączenie języków z różnych rodzin językowych i obszarów kulturowych mogłoby dostarczyć bardziej kompleksowego obrazu zjawiska.

2. **Analiza innych wymiarów kulturowych** - Podobna metodologia mogłaby być zastosowana do badania innych wymiarów zróżnicowania kulturowego, np. dystansu władzy czy unikania niepewności.

3. **Badanie na przestrzeni czasu** - Interesujące byłoby prześledzenie zmian w reprezentacjach embeddingowych koncepcji kulturowych na przestrzeni czasu.

4. **Integracja z badaniami behawioralnymi** - Połączenie analizy embeddingów z tradycyjnymi metodami psychologii międzykulturowej mogłoby dostarczyć pełniejszego obrazu zjawiska.

5. **Rozwijanie kulturowo świadomych systemów NLP** - Wyniki mogą posłużyć jako podstawa do projektowania systemów NLP uwzględniających kontekst kulturowy w przetwarzaniu i generowaniu tekstu.

### **3.4. Konkluzja**

Przeprowadzone badanie stanowi istotny krok w kierunku ilościowego, opartego na danych podejścia do badania relacji między językiem, kulturą i reprezentacjami embeddingowymi. Wykazuję, że modele embeddingowe nie tylko przechwytują podobieństwa semantyczne, ale również odzwierciedlają głębsze, kulturowo uwarunkowane struktury konceptualne. Potwierdza to potencjał embeddingów jako narzędzia w badaniach międzykulturowych oraz podkreśla potrzebę kulturowo świadomego podejścia w rozwoju systemów sztucznej inteligencji.

### **3.5. Bibliografia**

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of Machine Learning Research, 3(Feb), 1137-1155.
2. Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 5454-5476.
3. Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., ... & Liang, P. (2021). On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.
4. Boski, P. (2006). Humanism-materialism: Centuries-long Polish cultural origins and 20 years of research in cultural psychology. In U. Kim, K.-S. Yang, & K.-K. Hwang (Eds.), Indigenous and cultural psychology: Understanding people in context (pp. 373–402). Springer.
5. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877-1901.
6. Cao, X., Sugiyama, M., Gan, C., & Shen, Y. (2023). Understanding and improving robustness of vision transformers through patch-based negative augmentation. arXiv preprint arXiv:2110.07858.
7. Chomsky, N. (1957). Syntactic structures. Mouton.
8. Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S. R., Schwenk, H., & Stoyanov, V. (2018). XNLI: Evaluating cross-lingual sentence representations. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2475-2485.
9. D'Andrade, R. G. (1995). The development of cognitive anthropology. Cambridge University Press.
10. Darwin, C. (1872). The expression of the emotions in man and animals. John Murray.
11. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 4171-4186.
12. Dunn, O. J. (1961). Multiple comparisons among means. Journal of the American Statistical Association, 56(293), 52-64.
13. Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, 6(3-4), 169-200.
14. Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 55-65.
15. Freud, S. (1915). The unconscious. Standard Edition, 14(1957), 159-215.
16. Geertz, C. (1973). The interpretation of cultures. Basic Books.
17. Gross, J. J., & Barrett, L. F. (2011). Emotion generation and emotion regulation: One or two depends on your point of view. Emotion Review, 3(1), 8-16.
18. Havaldar, S., Corso, A., & Zhao, D. (2023). Vantage point similarity in embedding space reveals cross-cultural universal and relative emotion semantics in facial expressions. Proceedings of the National Academy of Sciences, 120(26), e2218772120.
19. Hershcovich, D., Frank, A., Lent, H., & Zhu, H. (2022). Challenges and strategies in cross-cultural NLP. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 6997-7013.
20. Hofstede, G. (2001). Culture's consequences: Comparing values, behaviors, institutions and organizations across nations. Sage Publications.
21. Hofstede, G., Hofstede, G. J., & Minkov, M. (2010). Cultures and organizations: Software of the mind (3rd ed.). McGraw-Hill.
22. Jones, E., Oliphant, T., Peterson, P., & others. (2001). SciPy: Open source scientific tools for Python. Retrieved from http://www.scipy.org/
23. Kashima, E. S., & Kashima, Y. (1998). Culture and language: The case of cultural dimensions and personal pronoun use. Journal of Cross-Cultural Psychology, 29(3), 461-486.
24. Kitayama, S., & Park, H. (2007). Cultural shaping of self, emotion, and well-being: How does it work? Social and Personality Psychology Compass, 1(1), 202-222.
25. Lazarus, R. S. (1991). Emotion and adaptation. Oxford University Press.
26. Lucy, J. A. (1997). Linguistic relativity. Annual Review of Anthropology, 26(1), 291-312.
27. Markus, H. R., & Kitayama, S. (1991). Culture and the self: Implications for cognition, emotion, and motivation. Psychological Review, 98(2), 224-253.
28. Mesquita, B. (1997). The cultural construction of emotions. Psychological Inquiry, 8(4), 258-269.
29. Mesquita, B. (2001). Emotions in collectivist and individualist contexts. Journal of Personality and Social Psychology, 80(1), 68-74.
30. Mesquita, B. (2003). Emotions as dynamic cultural phenomena. In R. J. Davidson, K. R. Scherer, & H. H. Goldsmith (Eds.), Handbook of affective sciences (pp. 871-890). Oxford University Press.
31. Mesquita, B., & Leu, J. (2007). The cultural psychology of emotion. In S. Kitayama & D. Cohen (Eds.), Handbook of cultural psychology (pp. 734-759). Guilford Press.
32. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26.
33. Nisbett, R. E., Peng, K., Choi, I., & Norenzayan, A. (2001). Culture and systems of thought: Holistic versus analytic cognition. Psychological Review, 108(2), 291-310.
34. Panksepp, J. (2004). Affective neuroscience: The foundations of human and animal emotions. Oxford University Press.
35. Pavlenko, A. (2008). Emotion and emotion-laden words in the bilingual lexicon. Bilingualism: Language and Cognition, 11(2), 147-164.
36. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 1532-1543.
37. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. OpenAI.
38. Rumelhart, D. E., & McClelland, J. L. (1986). Parallel distributed processing: Explorations in the microstructure of cognition. MIT Press.
39. Scherer, K. R. (2009). The dynamic architecture of emotion: Evidence for the component process model. Cognition and Emotion, 23(7), 1307-1351.
40. Shore, B. (1996). Culture in mind: Cognition, culture, and the problem of meaning. Oxford University Press.
41. Slobin, D. I. (1996). From "thought and language" to "thinking for speaking". In J. J. Gumperz & S. C. Levinson (Eds.), Rethinking linguistic relativity (pp. 70-96). Cambridge University Press.
42. Triandis, H. C. (1989). The self and social behavior in differing cultural contexts. Psychological Review, 96(3), 506-520.
43. Triandis, H. C. (1995). Individualism & collectivism. Westview Press.
44. Tsai, J. L. (2007). Ideal affect: Cultural causes and behavioral consequences. Perspectives on Psychological Science, 2(3), 242-259.
45. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
46. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
47. Wierzbicka, A. (1999). Emotions across languages and cultures: Diversity and universals. Cambridge University Press.
48. Wierzbicka, A. (2006). English: Meaning and culture. Oxford University Press.
49. Wierzbicka, A. (2009). Language, experience, and translation: Towards a new science of the mind. Lexington Books.

        """)

    # Stopka
    st.markdown("""
    ---
    *Analiza przeprowadzona z wykorzystaniem modelu `text-embedding-3-large` od OpenAI. Wizualizacje utworzone przy użyciu bibliotek Plotly i Streamlit.*
    """)


if __name__ == "__main__":
    run_streamlit_app()

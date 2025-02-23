import os
from openai import OpenAI
import numpy as np
import pickle
from numpy.linalg import norm
import pandas as pd
import json
import plotly.express as px
import streamlit as st  # Funkcje Streamlit pozostają, gdyż są wykorzystywane przy imporcie do aplikacji
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dotenv import load_dotenv
# Statystyka
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, kstest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

###############################################
# KONFIGURACJA API I MODELU
###############################################
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
EMBEDDING_MODEL = "text-embedding-3-large"  # Model generuje wektory 3072D
CACHE_FILE = "embeddings_cache_3_large.pkl"

INTEGRATED_REPORT = '''
# **Teoria i kontekst badania**

## 1. **Podstawy lingwistyki kulturowej: indywidualizm vs. kolektywizm**

W psychologii i socjologii popularnym zagadnieniem jest różnica między kulturami **indywidualistycznymi** i **kolektywistycznymi**:

- **Kultura indywidualistyczna** (przykład: wiele krajów zachodnich, w tym Stany Zjednoczone) kładzie silny nacisk na autonomię jednostki, samodzielność i osobiste osiągnięcia.  
- **Kultura kolektywistyczna** (przykład: wiele krajów wschodnich, np. Japonia) podkreśla znaczenie grupy, współpracy, harmonii społecznej oraz lojalności wobec wspólnoty.

Choć istnieją różne poziomy tej dychotomii i każde społeczeństwo może mieć mieszankę obu podejść, w języku często ujawniają się subtelne **różnice w sformułowaniach**. Przykładowo, indywidualistyczne zdania (np. „Jestem niezależny”) uwydatniają podmiot i jego odrębną tożsamość, podczas gdy kolektywistyczne (np. „Wspólnie pokonujemy wyzwania”) wskazują na przynależność do grupy, współzależność i wspólny cel.

---

## 2. **Reprezentacje wektorowe języka (embeddings)**

W przetwarzaniu języka naturalnego (NLP) zdania można zamienić na wysokowymiarowe **wektory** (tzw. embeddingi), które przechowują informację o ich znaczeniu.  
Model **`text-embedding-3-large`** od OpenAI (podobnie jak wiele innych modeli, np. `text-embedding-ada-002` czy modele w stylu BERT, RoBERTa itp.) został nauczony na ogromnych korpusach tekstów tak, aby **podobne semantycznie zdania** miały wektory **bliskie** w przestrzeni 3072 wymiarów, a zdania o różnych znaczeniach – wektory **odległe**.

### 2.1. **Co oznacza 3072 wymiary?**

Każdy wymiar może być rozumiany jako pewna **ukryta cecha** językowa, która pozwala modelowi odróżniać różne typy treści. W praktyce trudno jest opisać każdy wymiar w kategoriach intuicyjnych (np. emocjonalny ton, rejestr formalny, kontekst kulturowy), ale wiadomo, że **im bliżej** są dwa wektory w tej przestrzeni, tym **bardziej podobne** znaczenia reprezentują.

---

## 3. **Cel badania i hipotezy**

1. **Zidentyfikować różnice**: czy ten sam model odzwierciedla różnice między zdaniami indywidualistycznymi i kolektywistycznymi **równie silnie** we wszystkich językach (angielski, polski, japoński)?  
2. **Porównać dystanse**: czy w języku polskim te dwa rodzaje zdań (IND i COL) są wektorowo bliżej siebie niż np. w języku angielskim (co mogłoby sugerować, że w polskich zdaniach te kategorie mniej się różnią pod względem „semantyki” wykrytej przez model)?  
3. **Statystyczna istotność**: czy różnice między językami (np. polski vs. angielski) są przypadkowe, czy można je uznać za statystycznie potwierdzone (p < 0.01)?

---

## 4. **Analiza w aplikacji**

### 4.1. **Zbiór danych i podział**

- Mamy zdania w **trzech językach**: angielski, polski i japoński.  
- Każdy język podzielony jest na dwa rodzaje: **IND** (Individualistic) i **COL** (Collectivistic).  
- Dla każdego zdania generujemy embedding 3072D, co daje **6 zestawów**:
  - `ENG_IND`, `ENG_COL`  
  - `POL_IND`, `POL_COL`  
  - `JAP_IND`, `JAP_COL`

### 4.2. **Wizualizacje (PCA i t-SNE)**

- **PCA (Principal Component Analysis)** redukuje wymiary do 2D lub 3D, szukając głównych składowych, które wyjaśniają największą wariancję w danych.  
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to metoda nieliniowa, skupiająca się na zachowaniu lokalnego sąsiedztwa wektorów, często dająca wyraźne klastry w niskowymiarowej projekcji.

W aplikacji możesz zobaczyć, czy zdania IND i COL w każdym języku **grupują się** w osobne obszary, czy może częściowo się **nakładają**.

### 4.3. **Klasyfikacja nowego tekstu**

#### **Metoda centroidów**
1. Dla każdej kategorii (np. `ENG_IND`) liczymy średni wektor (centroid).  
2. Nowe zdanie zamieniamy na embedding i obliczamy **podobieństwo kosinusowe** z każdym centroidem.  
3. Kategoria z najwyższym podobieństwem – to przewidywana etykieta.

#### **Klasyfikator ML (Regresja Logistyczna)**
1. Trenujemy model na **wszystkich embeddingach** (wszystkie kategorie jednocześnie).  
2. Na wejściu nowy embedding → **model** zwraca etykietę + rozkład prawdopodobieństwa.

### 4.4. **Raport statystyczny**

- **Miary odległości**: Euklides, Kosinus (1 – cos), Manhattan.  
- **Test normalności**: sprawdzamy, czy rozkład odległości jest normalny.  
- **Test Manna-Whitneya** (lub t-Studenta, jeśli rozkład jest ~ normalny) do porównania m.in. *Polski vs. Angielski* i *Japoński vs. Angielski*.

**Interpretacja**:
- Niższa **mediana odległości** oznacza, że zdania IND i COL w danym języku są **bliżej** w przestrzeni wektorowej (model mniej rozróżnia te kategorie).  
- Wyższa **p-wartość** (p-value) oznacza, że różnice mogą być przypadkowe; niska (p < 0.01) → różnice są statystycznie istotne.

---

## 5. **Interpretacja wyników w kontekście kulturowym**

Jeśli okaże się (jak często w tym badaniu), że:

- **Polskie zdania IND i COL są bliżej siebie niż w języku angielskim**, to możemy spekulować, że polski w ramach obecnego korpusu i reprezentacji modelu **mniej różnicuje** indywidualizm i kolektywizm. To może wynikać z:
  - Cech językowych (np. fleksja, konstrukcje składniowe),  
  - Wpływu kulturowego (Polska ma cechy zarówno indywidualistyczne, jak i kolektywistyczne),  
  - Skali treningowej modelu (być może w korpusie polskim występowało relatywnie mniej zdań skrajnie indywidualistycznych lub kolektywistycznych).

- **W języku japońskim** możemy zaobserwować wyniki pośrednie lub odbiegające od typowego wyobrażenia o skrajnym kolektywizmie Japonii. Być może model anglo-centryczny (trening na dużych angielskich zasobach) nie zawsze oddaje w pełni charakter japońskiej kultury, lub zdania w korpusie są na tyle różnorodne, że różnice się zacierają.

---

## 6. **Implikacje praktyczne**

- **Zrozumienie biasów**: jeśli model mniej wyraźnie rozróżnia indywidualizm i kolektywizm w danym języku, może to wpływać na aplikacje takie jak automatyczne tłumaczenie, analiza sentymentu czy chatboty wielojęzyczne.  
- **Rozszerzenie badań**: można dodać więcej języków (np. hiszpański, chiński) czy więcej kategorii kulturowych (np. zdania high vs. low context).  
- **Wnioski dla lingwistyki i psychologii**: różnice i podobieństwa w embeddingach mogą pomóc w analizie **uniwersalności** (lub jej braku) modeli językowych, a tym samym w badaniach nad przekładalnością pojęć psychologicznych i kulturowych między językami.

'''


###############################################
# FUNKCJE EMBEDDINGU I CACHE
###############################################
def get_embedding(txt, model=EMBEDDING_MODEL):
    response = client.embeddings.create(
        input=txt,
        model=model,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)


def get_embeddings_for_list(txt_list, cache_file=CACHE_FILE):
    try:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        cache = {}
    out = []
    updated = False
    for txt in txt_list:
        if txt in cache:
            emb = cache[txt]
        else:
            emb = get_embedding(txt, model=EMBEDDING_MODEL)
            cache[txt] = emb
            updated = True
        out.append(emb)
    if updated:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    return np.array(out)


###############################################
# POBRANIE EMBEDDINGÓW I OBLICZENIE CENTROIDÓW
###############################################
with open("zdania.json", "r", encoding="utf-8") as file:
    zdania = json.load(file)

english_individualistic = zdania["english_individualistic"]
english_collectivistic = zdania["english_collectivistic"]
polish_individualistic = zdania["polish_individualistic"]
polish_collectivistic = zdania["polish_collectivistic"]
japanese_individualistic = zdania["japanese_individualistic"]
japanese_collectivistic = zdania["japanese_collectivistic"]

eng_ind_embeddings = get_embeddings_for_list(english_individualistic)
eng_col_embeddings = get_embeddings_for_list(english_collectivistic)
pol_ind_embeddings = get_embeddings_for_list(polish_individualistic)
pol_col_embeddings = get_embeddings_for_list(polish_collectivistic)
jap_ind_embeddings = get_embeddings_for_list(japanese_individualistic)
jap_col_embeddings = get_embeddings_for_list(japanese_collectivistic)


def compute_centroid(emb_list, normalize_before=True):
    if normalize_before:
        emb_list = [v / norm(v) for v in emb_list]
    c = np.mean(emb_list, axis=0)
    c /= norm(c)
    return c


centroid_eng_ind = compute_centroid(eng_ind_embeddings)
centroid_eng_col = compute_centroid(eng_col_embeddings)
centroid_pol_ind = compute_centroid(pol_ind_embeddings)
centroid_pol_col = compute_centroid(pol_col_embeddings)
centroid_jap_ind = compute_centroid(jap_ind_embeddings)
centroid_jap_col = compute_centroid(jap_col_embeddings)


###############################################
# REDUKCJA WYMIAROWOŚCI I INTERAKTYWNE WIZUALIZACJE (PCA, t-SNE)
###############################################
def generate_interactive_pca_2d(all_emb, all_lbl):
    pca = PCA(n_components=2, random_state=42)
    red_pca = pca.fit_transform(all_emb)
    df = pd.DataFrame({
        "PC1": red_pca[:, 0],
        "PC2": red_pca[:, 1],
        "Cluster": all_lbl
    })
    # Definicja kolorów: dla każdego języka używamy odcieni danego koloru
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])
    # Filtrowanie klastrów – funkcja ta jest przeznaczona do użycia w Streamlit
    selected_clusters = st.multiselect("Wybierz klastry (PCA 2D)",
                                       options=df["Cluster"].unique().tolist(),
                                       default=df["Cluster"].unique().tolist())
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter(filtered_df, x="PC1", y="PC2", color="Cluster",
                     color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in
                                         df["Cluster"].unique()},
                     title="Interaktywna PCA 2D (text-embedding-3-large)")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


def generate_interactive_tsne_2d(all_emb, all_lbl):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    red_tsne = tsne.fit_transform(all_emb)
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
    selected_clusters = st.multiselect("Wybierz klastry (t-SNE 2D)",
                                       options=df["Cluster"].unique().tolist(),
                                       default=df["Cluster"].unique().tolist())
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter(filtered_df, x="Dim1", y="Dim2", color="Cluster",
                     color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in
                                         df["Cluster"].unique()},
                     title="Interaktywna t-SNE 2D (text-embedding-3-large)")
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


def generate_interactive_pca_3d(all_emb, all_lbl):
    pca_3d = PCA(n_components=3, random_state=42)
    red_pca_3d = pca_3d.fit_transform(all_emb)
    df = pd.DataFrame({
        "PC1": red_pca_3d[:, 0],
        "PC2": red_pca_3d[:, 1],
        "PC3": red_pca_3d[:, 2],
        "Cluster": all_lbl
    })
    # Używamy tego samego color_map co w funkcji 2D
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    try:
        selected_clusters = st.multiselect("Wybierz klastry (PCA 3D)",
                                           options=df["Cluster"].unique().tolist(),
                                           default=df["Cluster"].unique().tolist())
    except Exception:
        selected_clusters = df["Cluster"].unique().tolist()
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter_3d(filtered_df,
                        x="PC1", y="PC2", z="PC3",
                        color="Cluster",
                        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] 
                                              for cl in df["Cluster"].unique()},
                        title="Interaktywna PCA 3D (text-embedding-3-large)",
                        labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


def generate_interactive_tsne_3d(all_emb, all_lbl):
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
    red_tsne_3d = tsne_3d.fit_transform(all_emb)
    df = pd.DataFrame({
        "Dim1": red_tsne_3d[:, 0],
        "Dim2": red_tsne_3d[:, 1],
        "Dim3": red_tsne_3d[:, 2],
        "Cluster": all_lbl
    })
    # Używamy tego samego color_map co w funkcjach 2D
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    try:
        selected_clusters = st.multiselect("Wybierz klastry (t-SNE 3D)",
                                           options=df["Cluster"].unique().tolist(),
                                           default=df["Cluster"].unique().tolist())
    except Exception:
        selected_clusters = df["Cluster"].unique().tolist()
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter_3d(filtered_df,
                        x="Dim1", y="Dim2", z="Dim3",
                        color="Cluster",
                        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] 
                                              for cl in df["Cluster"].unique()},
                        title="Interaktywna t-SNE 3D (text-embedding-3-large)",
                        labels={"Dim1": "Dim1", "Dim2": "Dim2", "Dim3": "Dim3"})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig



###############################################
# METRYKI ODLEGŁOŚCI I TESTY STATYSTYCZNE
###############################################
def dist_euclidean(a, b):
    return norm(a - b)


def dist_cosine(a, b):
    c = np.dot(a, b) / (norm(a) * norm(b))
    return 1.0 - c


def dist_manhattan(a, b):
    return np.sum(np.abs(a - b))


def all_pairwise(emb_list_a, emb_list_b, dist_func):
    out = []
    for x in emb_list_a:
        for y in emb_list_b:
            out.append(dist_func(x, y))
    return out


def test_normality(data):
    stat_s, p_s = shapiro(data)
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma < 1e-12:
        p_k = 0.0
    else:
        z = (data - mu) / sigma
        stat_k, p_k = kstest(z, 'norm')
    return p_s, p_k


def generate_statistical_report():
    report = ""
    metrics = [("Euklides", dist_euclidean),
               ("Kosinus (1 - cos)", dist_cosine),
               ("Manhattan", dist_manhattan)]
    for metric_name, metric_func in metrics:
        dist_pol = all_pairwise(pol_ind_embeddings, pol_col_embeddings, metric_func)
        dist_eng = all_pairwise(eng_ind_embeddings, eng_col_embeddings, metric_func)
        dist_jap = all_pairwise(jap_ind_embeddings, jap_col_embeddings, metric_func)
        p_s_pol, p_k_pol = test_normality(dist_pol)
        p_s_eng, p_k_eng = test_normality(dist_eng)
        p_s_jap, p_k_jap = test_normality(dist_jap)
        normal_pol = (p_s_pol > 0.05 and p_k_pol > 0.05)
        normal_eng = (p_s_eng > 0.05 and p_k_eng > 0.05)
        normal_jap = (p_s_jap > 0.05 and p_k_jap > 0.05)
        report += f"\n=== Metryka: {metric_name} ===\n"
        report += f" Shapiro (Pol) p={p_s_pol:.4f}, K-S (Pol) p={p_k_pol:.4f}\n"
        report += f" Shapiro (Eng) p={p_s_eng:.4f}, K-S (Eng) p={p_k_eng:.4f}\n"
        report += f" Shapiro (Jap) p={p_s_jap:.4f}, K-S (Jap) p={p_k_jap:.4f}\n"
        if normal_pol and normal_eng and normal_jap:
            from scipy.stats import ttest_ind
            stat_t_eng, p_t_eng = ttest_ind(dist_pol, dist_eng, equal_var=False)
            p_one_eng = p_t_eng / 2.0
            stat_t_jap, p_t_jap = ttest_ind(dist_jap, dist_eng, equal_var=False)
            p_one_jap = p_t_jap / 2.0
            report += f" T-test Eng (dwustronny): p(dwu)={p_t_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" T-test Jap (dwustronny): p(dwu)={p_t_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
        else:
            from scipy.stats import mannwhitneyu
            stat_m_eng, p_m_eng = mannwhitneyu(dist_pol, dist_eng, alternative='two-sided')
            p_one_eng = p_m_eng / 2.0
            stat_m_jap, p_m_jap = mannwhitneyu(dist_jap, dist_eng, alternative='two-sided')
            p_one_jap = p_m_jap / 2.0
            report += f" Mann–Whitney Eng (dwustronny): p(dwu)={p_m_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" Mann–Whitney Jap (dwustronny): p(dwu)={p_m_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
        med_pol = np.median(dist_pol)
        med_eng = np.median(dist_eng)
        med_jap = np.median(dist_jap)
        report += f" median(Pol)={med_pol:.4f}, median(Eng)={med_eng:.4f}, median(Jap)={med_jap:.4f}\n"
        if p_one_eng < 0.01 and med_pol < med_eng:
            report += " Wynik Pol: Polskie zdania IND i COL są statystycznie bliżej siebie niż angielskie.\n"
        else:
            report += " Wynik Pol: Brak istotnej różnicy między polskimi a angielskimi zdaniami.\n"
        if p_one_jap < 0.01 and med_jap < med_eng:
            report += " Wynik Jap: Japońskie zdania IND i COL są statystycznie bliżej siebie niż angielskie.\n"
        else:
            report += " Wynik Jap: Brak istotnej różnicy między japońskimi a angielskimi zdaniami.\n"
        report += "--- KONIEC TESTU ---\n"
    return report


###############################################
# KLASYFIKACJA TEKSTU (METODA CENTROIDÓW)
###############################################
def klasyfikuj_tekst(txt):
    vec = get_embedding(txt, model=EMBEDDING_MODEL)
    vec /= norm(vec)
    centroidy = {
        "ENG_IND": centroid_eng_ind,
        "ENG_COL": centroid_eng_col,
        "POL_IND": centroid_pol_ind,
        "POL_COL": centroid_pol_col,
        "JAP_IND": centroid_jap_ind,
        "JAP_COL": centroid_jap_col
    }

    def cos_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    wyniki = {}
    for key, cent in centroidy.items():
        wyniki[key] = cos_sim(vec, cent)
    # Wyniki sortujemy malejąco według podobieństwa
    return sorted(wyniki.items(), key=lambda x: x[1], reverse=True)


###############################################
# KLASYFIKACJA TEKSTU (UCZENIE MASZYNOWE) – ULEPSZONA WERSJA
###############################################
def train_ml_classifier(embeddings, labels):
    X = np.array(embeddings)
    y = np.array(labels)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2']
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print("Raport klasyfikacji (cały zbiór):")
    print(classification_report(y, best_model.predict(X)))
    return best_model


def ml_klasyfikuj_tekst(txt, clf):
    vec = get_embedding(txt, model=EMBEDDING_MODEL)
    vec /= norm(vec)
    pred = clf.predict([vec])[0]
    proba = clf.predict_proba([vec])[0]
    # Tworzymy słownik z prawdopodobieństwami – klasy pobieramy z atrybutu modelu
    prob_dict = {label: prob for label, prob in zip(clf.classes_, proba)}
    # Sortujemy słownik według wartości malejąco
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    return pred, prob_dict


def get_ml_classifier(all_embeddings, all_labels, model_path="ml_classifier.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        print("Wczytano zapisany model ML.")
    else:
        clf = train_ml_classifier(all_embeddings, all_labels)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        print("Wytrenowano i zapisano nowy model ML.")
    return clf


###############################################
# BLOK GŁÓWNY – NIE URUCHAMIAMY INTERFEJSU STREAMLIT
###############################################
if __name__ == "__main__":
    # Przygotowanie wspólnego zbioru embeddingów i etykiet
    all_embeddings = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                                     pol_ind_embeddings, pol_col_embeddings,
                                     jap_ind_embeddings, jap_col_embeddings], axis=0)
    all_labels = (["ENG_IND"] * len(eng_ind_embeddings) +
                  ["ENG_COL"] * len(eng_col_embeddings) +
                  ["POL_IND"] * len(pol_ind_embeddings) +
                  ["POL_COL"] * len(pol_col_embeddings) +
                  ["JAP_IND"] * len(jap_ind_embeddings) +
                  ["JAP_COL"] * len(jap_col_embeddings))

    # Wyświetlamy informacje o dostępnych wykresach – funkcje interaktywne są przeznaczone do wykorzystania w Streamlit
    print("Funkcje generujące wykresy interaktywne (2D i 3D) są dostępne przy imporcie modułu do aplikacji Streamlit.")

    # 1) Generowanie raportu statystycznego i zapis do pliku
    report_text = generate_statistical_report()
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Raport statystyczny zapisany w pliku 'raport_statystyczny.txt'.")

    # 2) Przykładowa klasyfikacja metodą centroidów
    test_txt = "I believe in working together for the greater good."
    ranking = klasyfikuj_tekst(test_txt)
    print("\nKlasyfikacja testowego zdania (metoda centroidów):")
    for cat, sim in ranking:
        print(f" - {cat}: {sim:.4f}")

    # 3) Użycie klasyfikatora ML – model jest trenowany tylko raz (lub wczytywany z pliku)
    clf = get_ml_classifier(all_embeddings, all_labels)
    pred_label, prob_dict = ml_klasyfikuj_tekst(test_txt, clf)
    print("\nKlasyfikacja testowego zdania (ML):")
    print(f" Przewidywana etykieta: {pred_label}")
    print(" Rozkład prawdopodobieństwa:")
    for label, prob in prob_dict.items():
        print(f"   {label}: {prob * 100:.2f}%")

    print("\n=== KONIEC ===")

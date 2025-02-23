import os
from openai import OpenAI
import numpy as np
import pickle
from numpy.linalg import norm
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dotenv import load_dotenv
# Statystyka
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, kstest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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

W psychologii i socjologii często podkreśla się różnice między kulturami **indywidualistycznymi** i **kolektywistycznymi**:

- **Kultura indywidualistyczna** (przykład: wiele krajów zachodnich, w tym Stany Zjednoczone) kładzie nacisk na autonomię jednostki, niezależność i osobiste osiągnięcia.  
- **Kultura kolektywistyczna** (przykład: kraje wschodnie, np. Japonia) podkreśla znaczenie grupy, wspólnego działania i harmonii społecznej.

W tekstach pisanych można dostrzec delikatne różnice w sposobie wyrażania postaw. Przykładowo, zdania indywidualistyczne (np. „Jestem niezależny”) często koncentrują się na pierwszej osobie i jej autonomii, natomiast kolektywistyczne („Wspólnie pokonujemy wyzwania”) wskazują na współdziałanie i wzajemną zależność.

---

## **2. Reprezentacje wektorowe języka (embeddingi)**

W przetwarzaniu języka naturalnego (NLP) komputer musi "zrozumieć" tekst, który dla maszyny jest początkowo jedynie ciągiem znaków. Aby umożliwić komputerom analizę, interpretację i porównywanie tekstów, stosuje się metodę, która nazywana jest **embeddingiem**. Embedding to sposób reprezentacji zdania jako matematycznego opisu – wektora liczb. Każde zdanie, np. „Lubię pracować zespołowo”, zostaje przekształcone w ciąg liczb, które umieszczone są w **przestrzeni wysokowymiarowej**.

**Model `text-embedding-3-large` od OpenAI** (podobnie jak inne modele, np. `text-embedding-ada-002`, BERT czy RoBERTa) został wytrenowany na bardzo dużym zbiorze tekstów. Jego celem jest, aby zdania o podobnym znaczeniu miały wektory, które znajdują się blisko siebie, podczas gdy zdania znaczeniowo odmienne – były oddalone. W praktyce oznacza to, że przestrzeń wektorowa staje się odzwierciedleniem semantycznych relacji między zdaniami.

### **2.1. Co oznacza 3072 wymiary?**

W modelu `text-embedding-3-large` każde zdanie jest reprezentowane przez **3072 liczby**. Możemy traktować te liczby jako ukryte „cechy” języka czy informację zawarte w danym tekście, które model wyodrębnił podczas treningu. Każdy wymiar może odpowiadać (w dużym uproszczeniu):
 - Na przykład za ton emocjonalny zdania,
 - Za formalność lub rejestr językowy,
 - Za aspekty kulturowe czy specyficzne słownictwo,
 - Za kontekst semantyczny, umożliwiający rozróżnienie między wieloma znaczeniami słów (np. „zamek” jako budowla obronna, zamek od drzwi lub zamek błyskawiczny),
 - Za intencje wypowiedzi, pomagając określić, czy zdanie jest pytaniem, stwierdzeniem czy rozkazem,
 - Za specyficzne domeny tematyczne, takie jak język medyczny, prawniczy czy technologiczny,
 - Oraz za niuanse językowe, które pozwalają modelowi uchwycić gramatykę, składnię i idiomatyczność języka, co jest kluczowe dla rozumienia zarówno języka mówionego, jak i pisanego.

Z uwagi na bardzo wysoką liczbę wymiarów trudno jest bezpośrednio wizualizować takie dane. Dlatego stosuje się metody redukcji wymiarowości, takie jak **PCA** (Principal Component Analysis) czy **t-SNE** (t-Distributed Stochastic Neighbor Embedding). Pozwalają one „spłaszczyć” przestrzeń 3072-wymiarową do 2D lub 3D, co umożliwia nam wizualne porównanie położeń wektorów. W ten sposób, jeśli dwa wektory (czyli reprezentacje dwóch zdań) są blisko siebie w przestrzeni, uznajemy, że zdania te są semantycznie podobne.

---

## **3. Cel badania i hipotezy**

Celem tego projektu jest sprawdzenie, czy **istnieją rzeczywiste różnice kulturowe** w sposobie wyrażania postaw indywidualistycznych i kolektywistycznych, oraz czy te różnice są widoczne w reprezentacjach wektorowych generowanych przez model `text-embedding-3-large`. Badanie skupia się na trzech językach: angielskim, polskim i japońskim.

### **Hipotezy badawcze**

1. **Różnicowanie kategorii w różnych językach**  
   **Hipoteza H₁:** Model embeddingowy odróżnia zdania indywidualistyczne (IND) od kolektywistycznych (COL) w każdym języku.  
   Statystycznie, oznacza to, że rozkłady odległości (np. miara kosinusowa lub euklidesowa) pomiędzy embeddingami zdań IND i COL będą istotnie różne w ramach danego języka.

2. **Porównanie dystansów między kategoriami w zależności od języka**  
   **Hipoteza H₂:** W języku polskim (i ewentualnie japońskim) różnice między zdaniami IND i COL (mierzone odległościami wektorowymi) są mniejsze niż w języku angielskim.  
   Statystycznie, oznacza to, że mediana odległości między wektorami zdań IND a COL w polskim (lub japońskim) będzie niższa niż w angielskim, co można przetestować używając testu Manna–Whitneya (alternatywa dla t-testu, gdy rozkłady nie są normalne) przy poziomie istotności p < 0.01.

3. **Statystyczna istotność obserwowanych różnic**  
   **Hipoteza H₃:** Zaobserwowane różnice w dystansach między kategoriami nie są przypadkowe.  
   W tym celu stosujemy testy normalności (np. Shapiro-Wilka oraz test Kolmogorova-Smirnova), a następnie testy porównawcze (t-test lub test Manna–Whitneya) dla różnych metryk odległości (Euklides, Kosinus, Manhattan). Wynik p < 0.01 wskazuje, że różnice są statystycznie istotne.

### **Sposób testowania hipotez**

- **Obliczanie odległości:** Dla każdej pary zdań (między zdaniami IND i COL w danym języku, czyli każde zdanie IND jest połączone z każdym zdaniem COl. Daje to nam 100x100 kombinacji) obliczamy odległości przy użyciu wybranych metryk (np. kosinusowej – 1 − cosinus, euklidesowej czy Manhattan).
  
- **Analiza rozkładu:** Sprawdzamy, czy rozkłady obliczonych odległości są zgodne z rozkładem normalnym przy użyciu testów Shapiro-Wilka i Kolmogorova-Smirnova. Jeśli rozkład nie jest normalny, stosujemy test nieparametryczny, taki jak test Manna–Whitneya.

- **Porównanie median:** Porównujemy mediany odległości między embeddingami zdań IND i COL dla języka angielskiego, polskiego oraz japońskiego. Hipoteza H₂ przewiduje, że mediana dla języka polskiego oraz japońskiego będzie mniejsza niż dla języka angielskiego.

- **Test istotności:** Jeśli wynik testu (np. test Manna–Whitneya) dla porównania mediana_angielski vs. mediana_polski (lub japoński) daje p < 0.01, można odrzucić hipotezę zerową, że różnice są przypadkowe, i przyjąć, że różnice te są statystycznie istotne.

### **Wnioski teoretyczne**

W kontekście teoretycznym założenie, że model embeddingowy potrafi uchwycić subtelne różnice kulturowe, opiera się na następującej idei:  
- **Semantyczna reprezentacja:** Jeśli model został dobrze wytrenowany, to zdania o podobnym znaczeniu (np. wyrażające indywidualizm) powinny być reprezentowane przez wektory bliskie sobie w przestrzeni wektorowej.  
- **Różnice kulturowe:** W praktyce sposób, w jaki dana kultura wyraża indywidualizm lub kolektywizm, może się różnić – np. język angielski może silniej rozróżniać te kategorie, podczas gdy język polski lub japoński może wykazywać mniejsze różnice.  
- **Testy statystyczne:** Użycie testów statystycznych pozwala na ilościową weryfikację tej różnicy. Jeśli badania wykażą, że odległości między embeddingami zdań IND i COL w języku polskim są mniejsze i różnica ta jest istotna statystycznie, możemy wywnioskować, że model oddaje te subtelne różnice kulturowe.

Podsumowując, projekt opiera się na hipotezach mówiących o różnicach w sposobie reprezentacji semantycznej zdań w zależności od języka, a ich weryfikacja odbywa się przez porównanie rozkładów odległości w przestrzeni embeddingowej oraz zastosowanie odpowiednich testów statystycznych (przy założeniu poziomu istotności p < 0.01). Taki podejście pozwala nie tylko na wizualną eksplorację danych, ale również na ilościową analizę różnic, co stanowi solidne narzędzie do badań nad kulturą i językiem.

---

## **4. Analiza w aplikacji**

### **4.1. Zbiór danych i podział**

Zgromadziliśmy zdania w trzech językach: **angielskim**, **polskim** i **japońskim**.  
W każdym języku wyróżniono dwie grupy zdań, każda po 100 zdań:

- **IND (Individualistic)** – przykłady: „Działam samodzielnie”, „Jestem niezależny”.  
- **COL (Collectivistic)** – przykłady: „Zespół jest siłą”, „Wspólnie się wspieramy”.

Te grupy razem dają **6 kategorii**:
1. `ENG_IND`  
2. `ENG_COL`  
3. `POL_IND`  
4. `POL_COL`  
5. `JAP_IND`  
6. `JAP_COL`

Każde zdanie przekształciliśmy w wektor (embedding) **3072D** za pomocą modelu `text-embedding-3-large`. Następnie porównujemy i wizualizujemy te wektory, by sprawdzić, **jak daleko** (lub **jak blisko**) są zdania IND i COL w każdym języku.

---

### **4.2. Wizualizacje (PCA i t-SNE)**

#### **PCA (Principal Component Analysis)**
Jest to metoda liniowa, która szuka głównych kierunków maksymalnej różnorodności w danych i pozwala zobaczyć w 2D lub 3D, **gdzie** dane (wektory) układają się najdalej od siebie. Dzięki temu możemy dostrzec np. czy `POL_IND` i `POL_COL` tworzą zbliżone skupisko, czy bardziej rozchodzą się w przestrzeni.

#### **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
To metoda nieliniowa, której zadaniem jest **utrzymanie bliskości** punktów, które w oryginalnych 3072 wymiarach również były blisko. Jeśli w 3072D dwa zdania były podobne, to t-SNE stara się pokazać je blisko siebie również w niskim wymiarze (2D lub 3D). 

W aplikacji możemy sprawdzić, czy np. zdania indywidualistyczne i kolektywistyczne w języku japońskim **układają się** w osobne rejony, czy raczej się **mieszają**.

---

### **4.3. Klasyfikacja nowego tekstu (dodatkowa funkcja)**

#### **Metoda centroidów**
1. Każda kategoria (np. `ENG_IND`) ma swój **centroid** – czyli średnią wartość wszystkich zdań przypisanych do tej kategorii. Można to sobie wyobrazić jako "punkt środkowy" grupy zdań w przestrzeni liczb.  
2. Nowy tekst zamieniamy na wektor (ciąg **3072 liczb**) – jest to matematyczna reprezentacja znaczenia tego zdania.  
3. Sprawdzamy, do której kategorii tekst pasuje najlepiej, mierząc **podobieństwo** między jego wektorem a centroidami.  
   - Używamy do tego **miary kosinusowej** – sprawdza ona kąt między wektorami.  
   - Jeśli kąt między dwoma wektorami jest mały, oznacza to, że zdania są bardzo podobne.  
   - Jeśli kąt jest duży, teksty są różne.  
   - Można to porównać do porównywania kierunków dwóch strzałek – im bardziej są do siebie równoległe, tym bardziej pasują.  
4. Tekst przypisujemy do kategorii, której centroid jest **najbliżej** (czyli ma najmniejszy kąt względem wektora zdania).  
5. Dzięki tej metodzie możemy również analizować, jak dobrze nowy tekst pasuje do zbioru:  
   - Jeśli wektor jest blisko centroidu, to tekst dobrze wpisuje się w daną kategorię.  
   - Jeśli jest daleko, może być niejednoznaczny lub pasować do kilku kategorii jednocześnie.  

#### **Klasyfikator ML (Regresja Logistyczna)**
1. Zamiast liczyć średnią wartość kategorii (jak w metodzie centroidów), uczymy model analizować **cały zbiór** wektorów przypisanych do poszczególnych kategorii (`ENG_IND`, `POL_COL` itd.).  
2. Po przekształceniu nowego zdania na wektor (3072 liczby), model przewiduje, do której kategorii należy.  
3. Oprócz samej klasy zwraca także **prawdopodobieństwa**, czyli ocenę pewności swojej decyzji.  
   - Przykładowo:  
     - `ENG_IND`: 85%  
     - `POL_COL`: 10%  
     - `ENG_COL`: 5%  
   - To oznacza, że model jest w 85% pewien, że zdanie należy do kategorii `ENG_IND`.  
4. Model ML jest dokładniejszy od metody centroidów, ponieważ:  
   - Nie tylko sprawdza średnią wartość kategorii, ale analizuje **pełny rozkład wszystkich zdań**.  
   - Potrafi dostrzec **bardziej subtelne różnice**, np. wykryć, że dwa podobne zdania mogą jednak należeć do różnych kategorii ze względu na niuanse językowe.  
   - Może też wykrywać **mniej typowe przypadki**, które nie są blisko żadnego centroidu, ale mimo to pasują do konkretnej kategorii.  

---

### **4.4. Raport statystyczny**

- **Miary odległości**: Euklides, Kosinus (1 − cosinus), Manhattan.  
- **Test normalności**: sprawdzamy, czy rozkład jest zbliżony do normalnego (Shapiro-Wilka, K-S).  
- **Test Manna–Whitneya / t-Studenta**: jeśli p < 0.01, uznajemy, że zaobserwowana różnica jest znikomo prawdopodobna jako przypadek.

#### **Interpretacja wyników**  
- **Niższa mediana odległości** w danym języku (IND vs. COL) → zdania indywidualistyczne i kolektywistyczne są **bliżej** siebie w przestrzeni wektorowej (model słabiej je rozróżnia).  
- **Wartość p** < 0.01 → różnica między np. polskim a angielskim jest statystycznie **istotna**.
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
    # Tworzymy wykres z pełnym zbiorem danych.
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna PCA 2D (text-embedding-3-large)"
    )
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
    # Tworzymy wykres z pełnym zbiorem danych.
    fig = px.scatter(
        df, x="Dim1", y="Dim2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna t-SNE 2D (text-embedding-3-large)"
    )
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
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])
    # Tworzymy interaktywny wykres 3D z pełnym zbiorem danych.
    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna PCA 3D (text-embedding-3-large)",
        labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"}
    )
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
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])
    # Tworzymy interaktywny wykres 3D z pełnym zbiorem danych.
    fig = px.scatter_3d(
        df, x="Dim1", y="Dim2", z="Dim3", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna t-SNE 3D (text-embedding-3-large)",
        labels={"Dim1": "Dim1", "Dim2": "Dim2", "Dim3": "Dim3"}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


###############################################
# METRYKI ODLEGŁOŚCI I TESTY STATYSTYCZNE
###############################################
def plot_distribution(data1, data2, label1, label2, title, filename):
    plt.figure()
    plt.hist(data1, bins=30, alpha=0.5, label=label1, density=True)
    plt.hist(data2, bins=30, alpha=0.5, label=label2, density=True)
    plt.title(title)
    plt.xlabel("Dystans")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.savefig(filename)
    plt.close()

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
        report += f"\n=== Metryka: {metric_name} ===\n"
        
        def intra_category_dist(embeddings):
            dists = all_pairwise(embeddings, embeddings, metric_func)
            # odfiltrowujemy dystanse (0 lub bardzo małe), powstające przy porównaniu identycznych wektorów
            return [d for d in dists if d > 1e-12]

        # INTRA: dystanse w obrębie danej kategorii (IND lub COL), a następnie łączone
        intra_pol_ind = intra_category_dist(pol_ind_embeddings)
        intra_pol_col = intra_category_dist(pol_col_embeddings)
        intra_pol = intra_pol_ind + intra_pol_col

        intra_eng_ind = intra_category_dist(eng_ind_embeddings)
        intra_eng_col = intra_category_dist(eng_col_embeddings)
        intra_eng = intra_eng_ind + intra_eng_col

        intra_jap_ind = intra_category_dist(jap_ind_embeddings)
        intra_jap_col = intra_category_dist(jap_col_embeddings)
        intra_jap = intra_jap_ind + intra_jap_col

        # INTER: dystanse między różnymi podkategoriami w ramach tego samego języka (IND vs. COL)
        dist_pol = all_pairwise(pol_ind_embeddings, pol_col_embeddings, metric_func)
        dist_eng = all_pairwise(eng_ind_embeddings, eng_col_embeddings, metric_func)
        dist_jap = all_pairwise(jap_ind_embeddings, jap_col_embeddings, metric_func)

        # Rysujemy histogramy do wizualnej oceny
        plot_distribution(intra_pol, dist_pol,
                          "Intra POL", "Inter POL",
                          f"Dystrybucja (Polski) [{metric_name}]",
                          f"dist_pol_{metric_name.replace(' ','_')}.png")
        plot_distribution(intra_eng, dist_eng,
                          "Intra ENG", "Inter ENG",
                          f"Dystrybucja (Angielski) [{metric_name}]",
                          f"dist_eng_{metric_name.replace(' ','_')}.png")
        plot_distribution(intra_jap, dist_jap,
                          "Intra JAP", "Inter JAP",
                          f"Dystrybucja (Japoński) [{metric_name}]",
                          f"dist_jap_{metric_name.replace(' ','_')}.png")

        # =====================================================================
        # == BADANIE H1: Czy inter > intra? (różnicowanie kategorii w jednym języku)
        # =====================================================================
        report += "\n[H1] Porównanie INTRA vs. INTER w ramach jednego języka.\n"
        report += ("Hipoteza: odległości inter (IND vs. COL) są większe niż intra (w obrębie IND lub COL). "
                   "Inaczej mówiąc, model rozróżnia te podkategorie.\n\n")

        # Sprawdźmy normalność rozkładów inter i intra dla danego języka
        def compare_intra_inter(intra_distances, inter_distances, lang_code):
            """
            Zwraca (stat, pval, test_name) i buduje fragment raportu z testu.
            """

            # Najpierw test normalności obu rozkładów
            p_s_intra, p_k_intra = test_normality(intra_distances)
            p_s_inter, p_k_inter = test_normality(inter_distances)

            normal_intra = (p_s_intra > 0.05 and p_k_intra > 0.05)
            normal_inter = (p_s_inter > 0.05 and p_k_inter > 0.05)

            # Raport o normalności
            txt = (f"  -> Normalność {lang_code} (intra): Shapiro p={p_s_intra:.4f}, K-S p={p_k_intra:.4f}, "
                   f"czy normalny? {normal_intra}\n")
            txt += (f"  -> Normalność {lang_code} (inter): Shapiro p={p_s_inter:.4f}, K-S p={p_k_inter:.4f}, "
                    f"czy normalny? {normal_inter}\n")

            # Jeśli oba rozkłady normalne, używamy testu t-Studenta (jednostronnego).
            # Jeżeli przynajmniej jeden nie jest normalny -> Mann-Whitney (jednostronny).
            if normal_intra and normal_inter:
                # Test t-Studenta (jednostronny). Najpierw liczymy dwustronny, potem p/2
                stat_t, p_t = ttest_ind(inter_distances, intra_distances, equal_var=False)
                p_one_sided = p_t / 2.0  # jednostronny
                txt += (f"  -> Test t-Studenta (dwustronny) stat={stat_t:.4f}, p={p_t:.4f} => p(jednostronne)={p_one_sided:.4f}\n")
                stat_val = stat_t
                p_val = p_one_sided
                test_used = "t-test (jednostronny)"
            else:
                # Test Manna-Whitneya (jednostronny)
                stat_m, p_m = mannwhitneyu(inter_distances, intra_distances, alternative='greater')
                txt += (f"  -> Test Manna-Whitneya (jednostronny) stat={stat_m:.4f}, p={p_m:.4f}\n")
                stat_val = stat_m
                p_val = p_m
                test_used = "Mann-Whitney (jednostronny)"

            return stat_val, p_val, test_used, txt

        # Polski
        stat_h1_pol, p_h1_pol, test_name_pol, pol_txt = compare_intra_inter(intra_pol, dist_pol, "POL")
        report += pol_txt
        if p_h1_pol < 0.01:
            report += "  [H1] Polski: Model wyraźnie różnicuje zdania IND od COL.\n"
        else:
            report += "  [H1] Polski: Brak statystycznie istotnego rozróżnienia między IND a COL.\n"

        # Angielski
        stat_h1_eng, p_h1_eng, test_name_eng, eng_txt = compare_intra_inter(intra_eng, dist_eng, "ENG")
        report += eng_txt
        if p_h1_eng < 0.01:
            report += "  [H1] Angielski: Model wyraźnie różnicuje zdania IND od COL.\n"
        else:
            report += "  [H1] Angielski: Brak statystycznie istotnego rozróżnienia między IND a COL.\n"

        # Japoński
        stat_h1_jap, p_h1_jap, test_name_jap, jap_txt = compare_intra_inter(intra_jap, dist_jap, "JAP")
        report += jap_txt
        if p_h1_jap < 0.01:
            report += "  [H1] Japoński: Model wyraźnie różnicuje zdania IND od COL.\n"
        else:
            report += "  [H1] Japoński: Brak statystycznie istotnego rozróżnienia między IND a COL.\n"

        report += "--- KONIEC TESTU (H1: intra vs. inter) ---\n"
        
        p_s_pol, p_k_pol = test_normality(dist_pol)
        p_s_eng, p_k_eng = test_normality(dist_eng)
        p_s_jap, p_k_jap = test_normality(dist_jap)
        normal_pol = (p_s_pol > 0.05 and p_k_pol > 0.05)
        normal_eng = (p_s_eng > 0.05 and p_k_eng > 0.05)
        normal_jap = (p_s_jap > 0.05 and p_k_jap > 0.05)
        report += f"\n[H2/H3] Shapiro (Pol) p={p_s_pol:.4f}, K-S (Pol) p={p_k_pol:.4f}\n"
        report += f"[H2/H3] Shapiro (Eng) p={p_s_eng:.4f}, K-S (Eng) p={p_k_eng:.4f}\n"
        report += f"[H2/H3] Shapiro (Jap) p={p_s_jap:.4f}, K-S (Jap) p={p_k_jap:.4f}\n"
        
        plt.figure()
        plt.hist(dist_pol, bins=30, alpha=0.7, label="Polish")
        plt.hist(dist_eng, bins=30, alpha=0.7, label="English")
        plt.hist(dist_jap, bins=30, alpha=0.7, label="Japanese")
        plt.title(f"Dystrybucja dystansów między IND i COL ({metric_name} - H2/H3)")
        plt.xlabel("Dystans")
        plt.ylabel("Liczba przypadków")
        plt.legend()
        plt.savefig(f"dist_all_{metric_name.replace(' ','_')}.png")
        plt.close()
        
        if normal_pol and normal_eng and normal_jap:
            stat_t_eng, p_t_eng = ttest_ind(dist_pol, dist_eng, equal_var=False)
            p_one_eng = p_t_eng / 2.0
            stat_t_jap, p_t_jap = ttest_ind(dist_jap, dist_eng, equal_var=False)
            p_one_jap = p_t_jap / 2.0
            report += f" [H2/H3] T-test Eng (dwustronny): p(dwu)={p_t_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" [H2/H3] T-test Jap (dwustronny): p(dwu)={p_t_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
        else:
            stat_m_eng, p_m_eng = mannwhitneyu(dist_pol, dist_eng, alternative='two-sided')
            p_one_eng = p_m_eng / 2.0
            stat_m_jap, p_m_jap = mannwhitneyu(dist_jap, dist_eng, alternative='two-sided')
            p_one_jap = p_m_jap / 2.0
            report += f" [H2/H3] Mann–Whitney Eng (dwustronny): p(dwu)={p_m_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" [H2/H3] Mann–Whitney Jap (dwustronny): p(dwu)={p_m_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
        med_pol = np.median(dist_pol)
        med_eng = np.median(dist_eng)
        med_jap = np.median(dist_jap)
        report += f" [H2/H3] median(Pol)={med_pol:.4f}, median(Eng)={med_eng:.4f}, median(Jap)={med_jap:.4f}\n"
        if p_one_eng < 0.01 and med_pol < med_eng:
            report += " [H2/H3] Wynik Pol: Polskie zdania IND i COL są statystycznie bliżej siebie niż angielskie.\n"
        else:
            report += " [H2/H3] Wynik Pol: Brak istotnej różnicy między polskimi a angielskimi zdaniami.\n"
        if p_one_jap < 0.01 and med_jap < med_eng:
            report += " [H2/H3] Wynik Jap: Japońskie zdania IND i COL są statystycznie bliżej siebie niż angielskie.\n"
        else:
            report += " [H2/H3] Wynik Jap: Brak istotnej różnicy między japońskimi a angielskimi zdaniami.\n"
        report += "--- KONIEC TESTU (H2/H3: porównanie między językami) ---\n"
        
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
    prob_dict = {label: prob for label, prob in zip(clf.classes_, proba)}
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
# DODATKOWE FUNKCJE: INTERAKTYWNE WYKRESY ROZKŁADU
###############################################
import plotly.graph_objects as go

def generate_distribution_chart(data1, data2, label1, label2, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data1, histnorm='probability density', name=label1, opacity=0.5))
    fig.add_trace(go.Histogram(x=data2, histnorm='probability density', name=label2, opacity=0.5))
    fig.update_layout(title=title, xaxis_title="Dystans", yaxis_title="Gęstość", barmode="overlay")
    return fig

def generate_interactive_distribution_charts():
    metrics = [("Euklides", dist_euclidean),
               ("Kosinus", dist_cosine),
               ("Manhattan", dist_manhattan)]
    figures = {}
    for metric_name, metric_func in metrics:
        def intra_category_dist(embeddings):
            dists = all_pairwise(embeddings, embeddings, metric_func)
            return [d for d in dists if d > 1e-12]
        intra_pol = intra_category_dist(pol_ind_embeddings) + intra_category_dist(pol_col_embeddings)
        intra_eng = intra_category_dist(eng_ind_embeddings) + intra_category_dist(eng_col_embeddings)
        intra_jap = intra_category_dist(jap_ind_embeddings) + intra_category_dist(jap_col_embeddings)
        dist_pol = all_pairwise(pol_ind_embeddings, pol_col_embeddings, metric_func)
        dist_eng = all_pairwise(eng_ind_embeddings, eng_col_embeddings, metric_func)
        dist_jap = all_pairwise(jap_ind_embeddings, jap_col_embeddings, metric_func)

        fig_pol = generate_distribution_chart(intra_pol, dist_pol, "Intra POL", "Inter POL", f"Dystrybucja dystansów (Polski) - {metric_name}")
        fig_eng = generate_distribution_chart(intra_eng, dist_eng, "Intra ENG", "Inter ENG", f"Dystrybucja dystansów (Angielski) - {metric_name}")
        fig_jap = generate_distribution_chart(intra_jap, dist_jap, "Intra JAP", "Inter JAP", f"Dystrybucja dystansów (Japoński) - {metric_name}")
        figures[metric_name] = {"POL": fig_pol, "ENG": fig_eng, "JAP": fig_jap}
    return figures

###############################################
# BLOK GŁÓWNY – NIE URUCHAMIAMY INTERFEJSU STREAMLIT
###############################################
if __name__ == "__main__":
    all_embeddings = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                                     pol_ind_embeddings, pol_col_embeddings,
                                     jap_ind_embeddings, jap_col_embeddings], axis=0)
    all_labels = (["ENG_IND"] * len(eng_ind_embeddings) +
                  ["ENG_COL"] * len(eng_col_embeddings) +
                  ["POL_IND"] * len(pol_ind_embeddings) +
                  ["POL_COL"] * len(pol_col_embeddings) +
                  ["JAP_IND"] * len(jap_ind_embeddings) +
                  ["JAP_COL"] * len(jap_col_embeddings))

    print("Funkcje generujące wykresy interaktywne (PCA, t-SNE, UMAP 2D/3D) są dostępne przy imporcie modułu do aplikacji Streamlit.")
    report_text = generate_statistical_report()
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Raport statystyczny zapisany w pliku 'raport_statystyczny.txt'.")

    test_txt = "I believe in working together for the greater good."
    ranking = klasyfikuj_tekst(test_txt)
    print("\nKlasyfikacja testowego zdania (metoda centroidów):")
    for cat, sim in ranking:
        print(f" - {cat}: {sim:.4f}")

    clf = get_ml_classifier(all_embeddings, all_labels)
    pred_label, prob_dict = ml_klasyfikuj_tekst(test_txt, clf)
    print("\nKlasyfikacja testowego zdania (ML):")
    print(f" Przewidywana etykieta: {pred_label}")
    print(" Rozkład prawdopodobieństwa:")
    for label, prob in prob_dict.items():
        print(f"   {label}: {prob * 100:.2f}%")

    print("\n=== KONIEC ===")

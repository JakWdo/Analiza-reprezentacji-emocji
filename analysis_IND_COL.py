import os
import logging
import time
import numpy as np
import pickle
from numpy.linalg import norm
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, classification_report
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import wraps
# Statystyka
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, kstest
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from scipy.spatial.distance import cdist, pdist, squareform
from openai import OpenAI

###############################################
# KONFIGURACJA LOGOWANIA
###############################################
def setup_logging(log_level=logging.INFO, log_file="analysis.log"):
    """
    Konfiguruje system logowania z formatowaniem czasowym.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def timeit(func):
    """
    Dekorator mierzący czas wykonania funkcji.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Rozpoczęcie {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Zakończenie {func.__name__}, czas: {end_time - start_time:.2f}s")
        return result
    return wrapper

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

W tekstach pisanych można dostrzec delikatne różnice w sposobie wyrażania postaw. Przykładowo, zdania indywidualistyczne (np. „Jestem niezależny") często koncentrują się na pierwszej osobie i jej autonomii, natomiast kolektywistyczne („Wspólnie pokonujemy wyzwania") wskazują na współdziałanie i wzajemną zależność.

---

## **2. Reprezentacje wektorowe języka (embeddingi)**

W przetwarzaniu języka naturalnego (NLP) komputer musi "zrozumieć" tekst, który dla maszyny jest początkowo jedynie ciągiem znaków. Aby umożliwić komputerom analizę, interpretację i porównywanie tekstów, stosuje się metodę, która nazywana jest **embeddingiem**. Embedding to sposób reprezentacji zdania jako matematycznego opisu – wektora liczb. Każde zdanie, np. „Lubię pracować zespołowo", zostaje przekształcone w ciąg liczb, które umieszczone są w **przestrzeni wysokowymiarowej**.

**Model `text-embedding-3-large` od OpenAI** (podobnie jak inne modele, np. `text-embedding-ada-002`, BERT czy RoBERTa) został wytrenowany na bardzo dużym zbiorze tekstów. Jego celem jest, aby zdania o podobnym znaczeniu miały wektory, które znajdują się blisko siebie, podczas gdy zdania znaczeniowo odmienne – były oddalone. W praktyce oznacza to, że przestrzeń wektorowa staje się odzwierciedleniem semantycznych relacji między zdaniami.

### **2.1. Co oznacza 3072 wymiary?**

W modelu `text-embedding-3-large` każde zdanie jest reprezentowane przez **3072 liczby**. Możemy traktować te liczby jako ukryte „cechy" języka czy informację zawarte w danym tekście, które model wyodrębnił podczas treningu. Każdy wymiar może odpowiadać (w dużym uproszczeniu):
- Na przykład za ton emocjonalny zdania,
- Za formalność lub rejestr językowy,
- Za aspekty kulturowe czy specyficzne słownictwo,
- Za kontekst semantyczny, umożliwiający rozróżnienie między wieloma znaczeniami słów (np. „zamek" jako budowla obronna, zamek od drzwi lub zamek błyskawiczny),
- Za intencje wypowiedzi, pomagając określić, czy zdanie jest pytaniem, stwierdzeniem czy rozkazem,
- Za specyficzne domeny tematyczne, takie jak język medyczny, prawniczy czy technologiczny,
- Oraz za niuanse językowe, które pozwalają modelowi uchwycić gramatykę, składnię i idiomatyczność języka, co jest kluczowe dla rozumienia zarówno języka mówionego, jak i pisanego.

Z uwagi na bardzo wysoką liczbę wymiarów trudno jest bezpośrednio wizualizować takie dane. Dlatego stosuje się metody redukcji wymiarowości, takie jak **PCA** (Principal Component Analysis) czy **t-SNE** (t-Distributed Stochastic Neighbor Embedding). Pozwalają one „spłaszczyć" przestrzeń 3072-wymiarową do 2D lub 3D, co umożliwia nam wizualne porównanie położeń wektorów. W ten sposób, jeśli dwa wektory (czyli reprezentacje dwóch zdań) są blisko siebie w przestrzeni, uznajemy, że zdania te są semantycznie podobne.

---

## **3. Cel badania i hipotezy**

Celem tego projektu jest sprawdzenie, czy **istnieją rzeczywiste różnice kulturowe** w sposobie wyrażania postaw indywidualistycznych i kolektywistycznych, oraz czy te różnice są widoczne w reprezentacjach wektorowych generowanych przez model `text-embedding-3-large`. Badanie skupia się na trzech językach: angielskim, polskim i japońskim.

### **Hipotezy badawcze**

1. **Różnicowanie kategorii w różnych językach**  
  **Hipoteza H₁:** Model embeddingowy odróżnia zdania indywidualistyczne (IND) od kolektywistycznych (COL) w każdym języku.  
  Statystycznie, oznacza to, że rozkłady odległości (np. miara kosinusowa lub euklidesowa) pomiędzy embeddingami zdań IND i COL będą istotnie różne w ramach danego języka.

2. **Porównanie dystansów między kategoriami w zależności od języka**  
  **Hipoteza H₂:** W języku polskim i japońskim różnice między zdaniami IND i COL (mierzone odległościami wektorowymi) są mniejsze niż w języku angielskim.  
  Statystycznie, oznacza to, że mediana odległości między wektorami zdań IND a COL w polskim i japońskim będzie niższa niż w angielskim, co można przetestować używając testu Manna–Whitneya (alternatywa dla t-testu, gdy rozkłady nie są normalne) przy poziomie istotności p < 0.01.

3. **Statystyczna istotność obserwowanych różnic**  
  **Hipoteza H₃:** Zaobserwowane różnice w dystansach między kategoriami nie są przypadkowe.  
  W tym celu stosuję testy normalności (np. Shapiro-Wilka oraz test Kolmogorova-Smirnova), a następnie testy porównawcze (t-test lub test Manna–Whitneya) dla różnych metryk odległości (Euklides, Kosinus, Manhattan). Wynik p < 0.01 wskazuje, że różnice są statystycznie istotne.

### **Sposób testowania hipotez**

- **Obliczanie odległości:** Dla każdej pary zdań (między zdaniami IND i COL w danym języku, czyli każde zdanie IND jest połączone z każdym zdaniem COL. Daje to nam 100x100 kombinacji) obliczam odległości przy użyciu wybranych metryk (np. kosinusowej – 1 − cosinus, euklidesowej czy Manhattan).

- **Analiza rozkładu:** Sprawdzam, czy rozkłady obliczonych odległości są zgodne z rozkładem normalnym przy użyciu testów Shapiro-Wilka i Kolmogorova-Smirnova. Jeśli rozkład nie jest normalny, stosuję test nieparametryczny, taki jak test Manna–Whitneya.

- **Porównanie median:** Porównuję mediany odległości między embeddingami zdań IND i COL dla języka angielskiego, polskiego oraz japońskiego. Hipoteza H₂ przewiduje, że mediana dla języka polskiego oraz japońskiego będzie mniejsza niż dla języka angielskiego.

- **Test istotności:** Jeśli wynik testu (np. test Manna–Whitneya) dla porównania mediana_angielski vs. mediana_polski (lub japoński) daje p < 0.01, mogę odrzucić hipotezę zerową, że różnice są przypadkowe, i przyjąć, że różnice te są statystycznie istotne.

- **Korekcja dla wielokrotnych testów:** Stosuję poprawkę Bonferroniego dla wszystkich wykonywanych testów statystycznych, aby kontrolować łączny poziom błędu typu I (fałszywie pozytywnych wyników).

### **Wnioski teoretyczne**

W kontekście teoretycznym założenie, że model embeddingowy potrafi uchwycić subtelne różnice kulturowe, opiera się na następującej idei:  
- **Semantyczna reprezentacja:** Jeśli model został dobrze wytrenowany, to zdania o podobnym znaczeniu (np. wyrażające indywidualizm) powinny być reprezentowane przez wektory bliskie sobie w przestrzeni wektorowej.  
- **Różnice kulturowe:** W praktyce sposób, w jaki dana kultura wyraża indywidualizm lub kolektywizm, może się różnić – np. język angielski może silniej rozróżniać te kategorie, podczas gdy język polski lub japoński może wykazywać mniejsze różnice.  
- **Testy statystyczne:** Użycie testów statystycznych pozwala na ilościową weryfikację tej różnicy. Jeśli badania wykażą, że odległości między embeddingami zdań IND i COL w języku polskim są mniejsze i różnica ta jest istotna statystycznie, mogę wywnioskować, że model oddaje te subtelne różnice kulturowe.

Podsumowując, projekt opiera się na hipotezach mówiących o różnicach w sposobie reprezentacji semantycznej zdań w zależności od języka, a ich weryfikacja odbywa się przez porównanie rozkładów odległości w przestrzeni embeddingowej oraz zastosowanie odpowiednich testów statystycznych (przy założeniu poziomu istotności p < 0.01). Takie podejście pozwala nie tylko na wizualną eksplorację danych, ale również na ilościową analizę różnic, co stanowi solidne narzędzie do badań nad kulturą i językiem.

---

## **4. Analiza w aplikacji**

### **4.1. Zbiór danych i podział**

Zgromadziłem zdania w trzech językach: **angielskim**, **polskim** i **japońskim**.  
W każdym języku wyróżniłem dwie grupy zdań, każda po 100 zdań:

- **IND (Individualistic)** – przykłady: „Działam samodzielnie", „Jestem niezależny".  
- **COL (Collectivistic)** – przykłady: „Zespół jest siłą", „Wspólnie się wspieramy".

Te grupy razem dają **6 kategorii**:
1. `ENG_IND`  
2. `ENG_COL`  
3. `POL_IND`  
4. `POL_COL`  
5. `JAP_IND`  
6. `JAP_COL`

Dodatkowo, aby zwiększyć rozmiar zbioru danych i jego reprezentatywność, implementuję możliwość rozszerzenia istniejącego korpusu zdań poprzez generowanie semantycznych wariantów, które zachowują charakter indywidualistyczny lub kolektywistyczny oryginalnych zdań.

Każde zdanie przekształciłem w wektor (embedding) **3072D** za pomocą modelu `text-embedding-3-large`. Następnie porównuję i wizualizuję te wektory, by sprawdzić, **jak daleko** (lub **jak blisko**) są zdania IND i COL w każdym języku.

---

### **4.2. Wizualizacje (PCA i t-SNE)**

#### **PCA (Principal Component Analysis)**
Jest to metoda liniowa, która szuka głównych kierunków maksymalnej różnorodności w danych i pozwala zobaczyć w 2D lub 3D, **gdzie** dane (wektory) układają się najdalej od siebie. Dzięki temu mogę dostrzec np. czy `POL_IND` i `POL_COL` tworzą zbliżone skupisko, czy bardziej rozchodzą się w przestrzeni.

#### **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
To metoda nieliniowa, której zadaniem jest **utrzymanie bliskości** punktów, które w oryginalnych 3072 wymiarach również były blisko. Jeśli w 3072D dwa zdania były podobne, to t-SNE stara się pokazać je blisko siebie również w niskim wymiarze (2D lub 3D). 

W aplikacji mogę sprawdzić, czy np. zdania indywidualistyczne i kolektywistyczne w języku japońskim **układają się** w osobne rejony, czy raczej się **mieszają**.

---

### **4.3. Klasyfikacja nowego tekstu (dodatkowa funkcja)**

#### **Metoda centroidów**
1. Każda kategoria (np. `ENG_IND`) ma swój **centroid** – czyli średnią wartość wszystkich zdań przypisanych do tej kategorii. Można to sobie wyobrazić jako "punkt środkowy" grupy zdań w przestrzeni liczb.  
2. Nowy tekst zamieniam na wektor (ciąg **3072 liczb**) – jest to matematyczna reprezentacja znaczenia tego zdania.  
3. Sprawdzam, do której kategorii tekst pasuje najlepiej, mierząc **podobieństwo** między jego wektorem a centroidami.  
  - Używam do tego **miary kosinusowej** – sprawdza ona kąt między wektorami.  
  - Jeśli kąt między dwoma wektorami jest mały, oznacza to, że zdania są bardzo podobne.  
  - Jeśli kąt jest duży, teksty są różne.  
  - Można to porównać do porównywania kierunków dwóch strzałek – im bardziej są do siebie równoległe, tym bardziej pasują.  
4. Tekst przypisuję do kategorii, której centroid jest **najbliżej** (czyli ma najmniejszy kąt względem wektora zdania).  
5. Dzięki tej metodzie mogę również analizować, jak dobrze nowy tekst pasuje do zbioru:  
  - Jeśli wektor jest blisko centroidu, to tekst dobrze wpisuje się w daną kategorię.  
  - Jeśli jest daleko, może być niejednoznaczny lub pasować do kilku kategorii jednocześnie.  

#### **Klasyfikator ML (Regresja Logistyczna)**
1. Zamiast liczyć średnią wartość kategorii (jak w metodzie centroidów), uczę model analizować **cały zbiór** wektorów przypisanych do poszczególnych kategorii (`ENG_IND`, `POL_COL` itd.).  
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
- **Test normalności**: sprawdzam, czy rozkład jest zbliżony do normalnego (Shapiro-Wilka, K-S).  
- **Test Manna–Whitneya / t-Studenta**: jeśli p < 0.01, uznaję, że zaobserwowana różnica jest znikomo prawdopodobna jako przypadek.
- **Korekcja dla wielokrotnych testów**: stosuję poprawkę Bonferroniego, aby kontrolować łączny poziom błędu typu I.

#### **Interpretacja wyników**  
- **Niższa mediana odległości** w danym języku (IND vs. COL) → zdania indywidualistyczne i kolektywistyczne są **bliżej** siebie w przestrzeni wektorowej (model słabiej je rozróżnia).  
- **Wartość p** < 0.01 → różnica między np. polskim a angielskim jest statystycznie **istotna**.
- **Odrzucenie hipotezy zerowej** po korekcji Bonferroniego → obserwowane różnice mają małe prawdopodobieństwo wystąpienia przez przypadek.
'''

###############################################
# FUNKCJE EMBEDDINGU I CACHE
###############################################
@timeit
def get_embedding(txt, model=EMBEDDING_MODEL):
    """
    Generuje embedding dla pojedynczego tekstu.
    
    Parametry:
    ----------
    txt : str
        Tekst do przekształcenia na embedding.
    model : str, optional
        Nazwa modelu embeddingowego do użycia.
        
    Zwraca:
    -------
    numpy.ndarray
        Wektor embeddingu dla podanego tekstu.
    """
    response = client.embeddings.create(
        input=txt,
        model=model,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)


@timeit
def get_embeddings_for_list(txt_list, cache_file=CACHE_FILE, batch_size=50):
    """
    Generuje embeddingi dla listy tekstów z wykorzystaniem cache.
    
    Parametry:
    ----------
    txt_list : list
        Lista tekstów do przekształcenia na embeddingi.
    cache_file : str, optional
        Ścieżka do pliku cache.
    batch_size : int, optional
        Rozmiar partii do przetwarzania (dla optymalizacji).
        
    Zwraca:
    -------
    numpy.ndarray
        Macierz embeddingów dla podanej listy tekstów.
    """
    try:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        logger.info(f"Wczytano cache z {len(cache)} embeddingami")
    except FileNotFoundError:
        cache = {}
        logger.info("Nie znaleziono cache, tworzenie nowego")
    
    missing_texts = [txt for txt in txt_list if txt not in cache]
    logger.info(f"Liczba brakujących embeddingów: {len(missing_texts)}")
    
    if missing_texts:
        # Przetwarzanie w partiach dla większej wydajności
        batches = [missing_texts[i:i+batch_size] for i in range(0, len(missing_texts), batch_size)]
        
        for i, batch in enumerate(batches):
            logger.info(f"Przetwarzanie partii {i+1}/{len(batches)}")
            
            # Opcjonalnie: równoległe przetwarzanie dla większych partii
            if len(batch) > 10:
                num_cores = min(multiprocessing.cpu_count(), len(batch))
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(get_embedding, txt) for txt in batch]
                    for txt, future in zip(batch, futures):
                        cache[txt] = future.result()
            else:
                for txt in batch:
                    cache[txt] = get_embedding(txt, model=EMBEDDING_MODEL)
        
        # Zapisz zaktualizowany cache
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"Zapisano cache z {len(cache)} embeddingami")
    
    # Zwróć embeddingi w takiej kolejności, w jakiej były teksty wejściowe
    embeddings = np.array([cache[txt] for txt in txt_list])
    return embeddings


###############################################
# POBRANIE EMBEDDINGÓW I OBLICZENIE CENTROIDÓW
###############################################
@timeit
def load_sentences_and_embeddings(filepath="zdania.json"):
    """
    Wczytuje zdania z pliku JSON i generuje dla nich embeddingi.
    
    Parametry:
    ----------
    filepath : str, optional
        Ścieżka do pliku JSON z zdaniami.
        
    Zwraca:
    -------
    tuple
        (zdania, embeddingi) - słowniki zawierające zdania i ich embeddingi.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        zdania = json.load(file)
    
    english_individualistic = zdania["english_individualistic"]
    english_collectivistic = zdania["english_collectivistic"]
    polish_individualistic = zdania["polish_individualistic"]
    polish_collectivistic = zdania["polish_collectivistic"]
    japanese_individualistic = zdania["japanese_individualistic"]
    japanese_collectivistic = zdania["japanese_collectivistic"]
    
    embeddings = {}
    embeddings["eng_ind"] = get_embeddings_for_list(english_individualistic)
    embeddings["eng_col"] = get_embeddings_for_list(english_collectivistic)
    embeddings["pol_ind"] = get_embeddings_for_list(polish_individualistic)
    embeddings["pol_col"] = get_embeddings_for_list(polish_collectivistic)
    embeddings["jap_ind"] = get_embeddings_for_list(japanese_individualistic)
    embeddings["jap_col"] = get_embeddings_for_list(japanese_collectivistic)
    
    return zdania, embeddings


def compute_centroid(emb_list, normalize_before=True):
    """
    Oblicza centroid (średnią) dla listy embeddingów.
    
    Parametry:
    ----------
    emb_list : numpy.ndarray
        Lista embeddingów.
    normalize_before : bool, optional
        Czy normalizować embeddingi przed obliczeniem centroidu.
        
    Zwraca:
    -------
    numpy.ndarray
        Centroid (znormalizowany wektor średni).
    """
    if normalize_before:
        emb_list = [v / norm(v) for v in emb_list]
    c = np.mean(emb_list, axis=0)
    c /= norm(c)
    return c


def compute_all_centroids(embeddings):
    """
    Oblicza centroidy dla wszystkich kategorii embeddingów.
    
    Parametry:
    ----------
    embeddings : dict
        Słownik zawierający embeddingi dla różnych kategorii.
        
    Zwraca:
    -------
    dict
        Słownik zawierający centroidy dla każdej kategorii.
    """
    centroids = {}
    for key, emb in embeddings.items():
        centroids[key] = compute_centroid(emb)
    return centroids


###############################################
# FUNKCJE DO OBLICZANIA ODLEGŁOŚCI
###############################################
def dist_euclidean(a, b):
    """
    Oblicza odległość euklidesową między dwoma wektorami.
    
    Parametry:
    ----------
    a, b : numpy.ndarray
        Wektory do porównania.
        
    Zwraca:
    -------
    float
        Odległość euklidesowa.
    """
    return norm(a - b)


def dist_cosine(a, b):
    """
    Oblicza odległość kosinusową (1 - cosinus) między dwoma wektorami.
    
    Parametry:
    ----------
    a, b : numpy.ndarray
        Wektory do porównania.
        
    Zwraca:
    -------
    float
        Odległość kosinusowa.
    """
    c = np.dot(a, b) / (norm(a) * norm(b))
    return 1.0 - c


def dist_manhattan(a, b):
    """
    Oblicza odległość Manhattan między dwoma wektorami.
    
    Parametry:
    ----------
    a, b : numpy.ndarray
        Wektory do porównania.
        
    Zwraca:
    -------
    float
        Odległość Manhattan.
    """
    return np.sum(np.abs(a - b))


def optimized_pairwise_distances(embeddings_a, embeddings_b, metric='euclidean'):
    """
    Efektywnie oblicza macierz odległości między dwoma zbiorami embeddingów.
    
    Parametry:
    ----------
    embeddings_a, embeddings_b : numpy.ndarray
        Zbiory embeddingów do porównania.
    metric : str, optional
        Metryka odległości ('euclidean', 'cosine', 'cityblock').
        
    Zwraca:
    -------
    numpy.ndarray
        Spłaszczona macierz odległości między zbiorami.
    """
    # Mapowanie znanych funkcji dystansu na nazwy metryk używanych przez cdist
    metric_mapping = {
        'euclidean': 'euclidean',
        'cosine': 'cosine',
        'manhattan': 'cityblock'
    }
    
    scipy_metric = metric_mapping.get(metric, metric)
    
    # Obliczamy macierz odległości
    distance_matrix = cdist(embeddings_a, embeddings_b, metric=scipy_metric)
    
    return distance_matrix.flatten()


@timeit
def all_pairwise(emb_list_a, emb_list_b, dist_func):
    """
    Oblicza wszystkie parami odległości między dwoma zbiorami embeddingów.
    
    Parametry:
    ----------
    emb_list_a, emb_list_b : numpy.ndarray
        Zbiory embeddingów do porównania.
    dist_func : callable or str
        Funkcja odległości lub nazwa metryki.
        
    Zwraca:
    -------
    list
        Lista odległości między wszystkimi parami embeddingów.
    """
    # Mapowanie znanych funkcji dystansu na nazwy metryk używanych przez cdist
    metric = None
    if dist_func == dist_euclidean:
        metric = 'euclidean'
    elif dist_func == dist_cosine:
        metric = 'cosine'
    elif dist_func == dist_manhattan:
        metric = 'cityblock'
    
    if metric is not None:
        return cdist(np.array(emb_list_a), np.array(emb_list_b), metric=metric).flatten().tolist()
    elif isinstance(dist_func, str) and dist_func in ['euclidean', 'cosine', 'manhattan']:
        metric_map = {'euclidean': 'euclidean', 'cosine': 'cosine', 'manhattan': 'cityblock'}
        return cdist(np.array(emb_list_a), np.array(emb_list_b), metric=metric_map[dist_func]).flatten().tolist()
    else:
        # Fallback do oryginalnej implementacji dla niestandardowych funkcji
        out = []
        for x in emb_list_a:
            for y in emb_list_b:
                out.append(dist_func(x, y))
        return out


def calculate_intra_category_similarity(embeddings):
    """
    Oblicza podobieństwo wewnątrz kategorii.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Embeddingi należące do jednej kategorii.
        
    Zwraca:
    -------
    list
        Lista odległości między embeddingami w kategorii.
    """
    distances = all_pairwise(embeddings, embeddings, dist_func='cosine')
    # Usuwamy odległości od embeddingu do samego siebie (które wynoszą 0)
    return [d for d in distances if d > 1e-12]


def calculate_inter_category_similarity(embeddings_a, embeddings_b):
    """
    Oblicza podobieństwo między kategoriami.
    
    Parametry:
    ----------
    embeddings_a, embeddings_b : numpy.ndarray
        Embeddingi należące do dwóch różnych kategorii.
        
    Zwraca:
    -------
    list
        Lista odległości między embeddingami z różnych kategorii.
    """
    return all_pairwise(embeddings_a, embeddings_b, dist_func='cosine')


def calculate_intra_inter_distances(embeddings_ind, embeddings_col, metric_func):
    """
    Oblicza odległości wewnątrz- i międzygrupowe dla dwóch grup embeddingów.
    
    Parametry:
    -----------
    embeddings_ind : numpy.ndarray
        Macierz embeddingów zdań indywidualistycznych. 
        Każdy wiersz to jeden embedding (wektor w przestrzeni 3072D).
    embeddings_col : numpy.ndarray
        Macierz embeddingów zdań kolektywistycznych.
    metric_func : callable or str
        Funkcja odległości lub nazwa metryki.
        
    Zwraca:
    --------
    tuple
        (intra_ind, intra_col, inter)
        - intra_ind: lista odległości wewnątrz grupy indywidualistycznej
        - intra_col: lista odległości wewnątrz grupy kolektywistycznej
        - inter: lista odległości między grupami
    """
    intra_ind = all_pairwise(embeddings_ind, embeddings_ind, metric_func)
    intra_ind = [d for d in intra_ind if d > 1e-12]  # Usuwamy porównania embeddingu z samym sobą
    
    intra_col = all_pairwise(embeddings_col, embeddings_col, metric_func)
    intra_col = [d for d in intra_col if d > 1e-12]  # Usuwamy porównania embeddingu z samym sobą
    
    inter = all_pairwise(embeddings_ind, embeddings_col, metric_func)
    
    return intra_ind, intra_col, inter


###############################################
# REDUKCJA WYMIAROWOŚCI I INTERAKTYWNE WIZUALIZACJE
###############################################
@timeit
def generate_interactive_pca_2d(all_emb, all_lbl):
    """
    Generuje interaktywny wykres PCA 2D dla embeddingów.
    
    Parametry:
    ----------
    all_emb : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_lbl : list
        Lista etykiet dla embeddingów.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres PCA 2D.
    """
    pca = PCA(n_components=2, random_state=42)
    red_pca = pca.fit_transform(all_emb)
    logger.info(f"Wyjaśniona wariancja przez komponenty PCA: {pca.explained_variance_ratio_}")
    
    df = pd.DataFrame({
        "PC1": red_pca[:, 0],
        "PC2": red_pca[:, 1],
        "Cluster": all_lbl
    })
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    df["Color"] = df["Cluster"].apply(lambda x: color_map[x.split("_")[0]][x.split("_")[1]])
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna PCA 2D (text-embedding-3-large)"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


@timeit
def generate_interactive_tsne_2d(all_emb, all_lbl, perplexity=30):
    """
    Generuje interaktywny wykres t-SNE 2D dla embeddingów.
    
    Parametry:
    ----------
    all_emb : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_lbl : list
        Lista etykiet dla embeddingów.
    perplexity : int, optional
        Parametr perplexity dla t-SNE.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres t-SNE 2D.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
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
    fig = px.scatter(
        df, x="Dim1", y="Dim2", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title=f"Interaktywna t-SNE 2D (perplexity={perplexity})"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


@timeit
def generate_interactive_pca_3d(all_emb, all_lbl):
    """
    Generuje interaktywny wykres PCA 3D dla embeddingów.
    
    Parametry:
    ----------
    all_emb : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_lbl : list
        Lista etykiet dla embeddingów.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres PCA 3D.
    """
    pca_3d = PCA(n_components=3, random_state=42)
    red_pca_3d = pca_3d.fit_transform(all_emb)
    logger.info(f"Wyjaśniona wariancja przez komponenty PCA 3D: {pca_3d.explained_variance_ratio_}")
    
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
    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title="Interaktywna PCA 3D (text-embedding-3-large)",
        labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3"}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


@timeit
def generate_interactive_tsne_3d(all_emb, all_lbl, perplexity=30):
    """
    Generuje interaktywny wykres t-SNE 3D dla embeddingów.
    
    Parametry:
    ----------
    all_emb : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_lbl : list
        Lista etykiet dla embeddingów.
    perplexity : int, optional
        Parametr perplexity dla t-SNE.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres t-SNE 3D.
    """
    tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42)
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
    fig = px.scatter_3d(
        df, x="Dim1", y="Dim2", z="Dim3", color="Cluster",
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Cluster"].unique()},
        title=f"Interaktywna t-SNE 3D (perplexity={perplexity})",
        labels={"Dim1": "Dim1", "Dim2": "Dim2", "Dim3": "Dim3"}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig


def generate_plotly_3d_point_cloud(embeddings, labels, method='pca'):
    """
    Generuje interaktywną wizualizację 3D przestrzeni embeddingów.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Macierz embeddingów.
    labels : list
        Lista etykiet dla embeddingów.
    method : str, optional
        Metoda redukcji wymiarowości ('pca', 'tsne', 'umap').
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres 3D.
    """
    import plotly.express as px
    
    if method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=3, perplexity=30, random_state=42)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
        except ImportError:
            logger.warning("UMAP niedostępny. Zainstaluj pakiet: pip install umap-learn")
            return None
    
    reduced = reducer.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'z': reduced[:, 2],
        'Category': labels
    })
    
    color_map = {
        "ENG": {"IND": "#aec7e8", "COL": "#1f77b4"},
        "POL": {"IND": "#98df8a", "COL": "#2ca02c"},
        "JAP": {"IND": "#ff9896", "COL": "#d62728"}
    }
    
    fig = px.scatter_3d(
        df, x='x', y='y', z='z', 
        color='Category',
        color_discrete_map={cl: color_map[cl.split("_")[0]][cl.split("_")[1]] for cl in df["Category"].unique()},
        title=f"Interaktywna wizualizacja 3D ({method.upper()})",
        opacity=0.7, size_max=10
    )
    
    # Dodatkowe opcje dla lepszej interaktywności
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def interpret_pca_components(embeddings, labels, n_components=10):
    """
    Przeprowadza analizę i interpretację głównych komponentów PCA.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Macierz embeddingów.
    labels : list
        Lista etykiet dla embeddingów.
    n_components : int, optional
        Liczba komponentów PCA do analizy.
        
    Zwraca:
    -------
    dict
        Wyniki analizy PCA.
    """
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Analiza rozkładu wartości własnych
    eigenvalues = pca.explained_variance_
    
    # Analiza głównych komponentów dla każdej kategorii
    result = {
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'eigenvalues': eigenvalues,
        'category_analysis': {}
    }
    
    categories = np.unique(labels)
    for category in categories:
        cat_mask = np.array(labels) == category
        cat_embeddings = embeddings[cat_mask]
        cat_transformed = pca.transform(cat_embeddings)
        
        # Średnie wartości dla każdego komponentu
        component_means = cat_transformed.mean(axis=0)
        component_stds = cat_transformed.std(axis=0)
        
        result['category_analysis'][category] = {
            'component_means': component_means,
            'component_stds': component_stds
        }
    
    return result


###############################################
# METRYKI ODLEGŁOŚCI I TESTY STATYSTYCZNE
###############################################
def plot_distribution(data1, data2, label1, label2, title, filename):
    """
    Tworzy wykres porównujący rozkłady dwóch zbiorów danych.
    
    Parametry:
    ----------
    data1, data2 : list or numpy.ndarray
        Dane do porównania.
    label1, label2 : str
        Etykiety dla zbiorów danych.
    title : str
        Tytuł wykresu.
    filename : str
        Nazwa pliku do zapisania wykresu.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=30, alpha=0.5, label=label1, density=True)
    plt.hist(data2, bins=30, alpha=0.5, label=label2, density=True)
    plt.title(title)
    plt.xlabel("Dystans")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def generate_distribution_chart(data1, data2, label1, label2, title):
    """
    Generuje interaktywny wykres porównujący rozkłady dwóch zbiorów danych.
    
    Parametry:
    ----------
    data1, data2 : list or numpy.ndarray
        Dane do porównania.
    label1, label2 : str
        Etykiety dla zbiorów danych.
    title : str
        Tytuł wykresu.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres porównujący rozkłady.
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data1, histnorm='probability density', name=label1, opacity=0.5))
    fig.add_trace(go.Histogram(x=data2, histnorm='probability density', name=label2, opacity=0.5))
    fig.update_layout(
        title=title, 
        xaxis_title="Dystans", 
        yaxis_title="Gęstość", 
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def generate_metric_comparison_chart(distances_by_language, metrics, lang_pairs, title="Porównanie odległości między kategoriami"):
    """
    Generuje wykres porównujący wartości metryk dla różnych par języków.
    
    Parametry:
    ----------
    distances_by_language : dict
        Słownik zawierający odległości dla różnych par języków i metryk.
    metrics : list
        Lista nazw metryk.
    lang_pairs : list
        Lista par języków.
    title : str, optional
        Tytuł wykresu.
        
    Zwraca:
    -------
    plotly.graph_objects.Figure
        Interaktywny wykres porównujący metryki.
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    bar_width = 0.2
    positions = np.arange(len(lang_pairs))
    
    for i, metric in enumerate(metrics):
        medians = [np.median(distances_by_language[pair][metric]) for pair in lang_pairs]
        fig.add_trace(go.Bar(
            x=positions + (i - len(metrics)/2 + 0.5) * bar_width,
            y=medians,
            width=bar_width,
            name=metric
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Pary językowe",
            ticktext=lang_pairs,
            tickvals=positions,
        ),
        yaxis=dict(title="Mediana odległości"),
        legend_title="Metryka",
        barmode='group'
    )
    
    return fig


def test_normality(data):
    """
    Przeprowadza testy normalności dla danych.
    
    Parametry:
    ----------
    data : list or numpy.ndarray
        Dane do testowania.
        
    Zwraca:
    -------
    tuple
        (p_s, p_k) - p-wartości dla testów Shapiro-Wilka i Kolmogorova-Smirnova.
    """
    stat_s, p_s = shapiro(data)
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma < 1e-12:
        p_k = 0.0
    else:
        z = (data - mu) / sigma
        stat_k, p_k = kstest(z, 'norm')
    return p_s, p_k


def apply_multiple_testing_correction(p_values, method='bonferroni'):
    """
    Stosuje korekcję dla wielokrotnych testów statystycznych.
    
    Parametry:
    ----------
    p_values : list
        Lista wartości p z testów statystycznych.
    method : str, optional
        Metoda korekcji ('bonferroni', 'fdr_bh', 'holm').
        
    Zwraca:
    -------
    tuple
        (reject, pvals_corrected) - czy odrzucamy hipotezę zerową, skorygowane wartości p.
    """
    return multipletests(p_values, method=method)


@timeit
def generate_statistical_report(embeddings):
    """
    Generuje raport statystyczny na podstawie embeddingów.
    
    Parametry:
    ----------
    embeddings : dict
        Słownik zawierający embeddingi dla różnych kategorii.
        
    Zwraca:
    -------
    str
        Raport statystyczny w formie tekstu.
    """
    report = ""
    metrics = [("Euklides", dist_euclidean),
               ("Kosinus (1 - cos)", dist_cosine),
               ("Manhattan", dist_manhattan)]
    
    # Listy do przechowywania p-wartości dla korekcji wielokrotnych testów
    p_values_h1 = []
    p_values_h2h3 = []
    
    for metric_name, metric_func in metrics:
        report += f"\n=== Metryka: {metric_name} ===\n"

        def intra_category_dist(embeddings):
            dists = all_pairwise(embeddings, embeddings, metric_func)
            return [d for d in dists if d > 1e-12]

        intra_pol = intra_category_dist(embeddings["pol_ind"]) + intra_category_dist(embeddings["pol_col"])
        intra_eng = intra_category_dist(embeddings["eng_ind"]) + intra_category_dist(embeddings["eng_col"])
        intra_jap = intra_category_dist(embeddings["jap_ind"]) + intra_category_dist(embeddings["jap_col"])

        dist_pol = all_pairwise(embeddings["pol_ind"], embeddings["pol_col"], metric_func)
        dist_eng = all_pairwise(embeddings["eng_ind"], embeddings["eng_col"], metric_func)
        dist_jap = all_pairwise(embeddings["jap_ind"], embeddings["jap_col"], metric_func)

        plot_distribution(intra_pol, dist_pol,
                          "Intra POL", "Inter POL",
                          f"Dystrybucja (Polski) [{metric_name}]",
                          f"dist_pol_{metric_name.replace(' ', '_')}.png")
        plot_distribution(intra_eng, dist_eng,
                          "Intra ENG", "Inter ENG",
                          f"Dystrybucja (Angielski) [{metric_name}]",
                          f"dist_eng_{metric_name.replace(' ', '_')}.png")
        plot_distribution(intra_jap, dist_jap,
                          "Intra JAP", "Inter JAP",
                          f"Dystrybucja (Japoński) [{metric_name}]",
                          f"dist_jap_{metric_name.replace(' ', '_')}.png")

        report += "\n[H1] Porównanie INTRA vs. INTER w ramach jednego języka.\n"
        report += ("Hipoteza: odległości inter (IND vs. COL) są większe niż intra (w obrębie IND lub COL). "
                   "Inaczej mówiąc, model rozróżnia te podkategorie.\n\n")

        def compare_intra_inter(intra_distances, inter_distances, lang_code):
            p_s_intra, p_k_intra = test_normality(intra_distances)
            p_s_inter, p_k_inter = test_normality(inter_distances)
            normal_intra = (p_s_intra > 0.05 and p_k_intra > 0.05)
            normal_inter = (p_s_inter > 0.05 and p_k_inter > 0.05)
            txt = (f"  -> Normalność {lang_code} (intra): Shapiro p={p_s_intra:.4f}, K-S p={p_k_intra:.4f}, "
                   f"czy normalny? {normal_intra}\n")
            txt += (f"  -> Normalność {lang_code} (inter): Shapiro p={p_s_inter:.4f}, K-S p={p_k_inter:.4f}, "
                    f"czy normalny? {normal_inter}\n")
            if normal_intra and normal_inter:
                stat_t, p_t = ttest_ind(inter_distances, intra_distances, equal_var=False)
                p_one_sided = p_t / 2.0
                txt += (
                    f"  -> Test t-Studenta (dwustronny) stat={stat_t:.4f}, p={p_t:.4f} => p(jednostronne)={p_one_sided:.4f}\n")
                stat_val = stat_t
                p_val = p_one_sided
                test_used = "t-test (jednostronny)"
            else:
                stat_m, p_m = mannwhitneyu(inter_distances, intra_distances, alternative='greater')
                txt += (f"  -> Test Manna-Whitneya (jednostronny) stat={stat_m:.4f}, p={p_m:.4f}\n")
                stat_val = stat_m
                p_val = p_m
                test_used = "Mann-Whitney (jednostronny)"
            
            # Dodajemy p-wartość do listy do korekcji
            p_values_h1.append(p_val)
            
            return stat_val, p_val, test_used, txt

        stat_h1_pol, p_h1_pol, test_name_pol, pol_txt = compare_intra_inter(intra_pol, dist_pol, "POL")
        report += pol_txt
        if p_h1_pol < 0.01:
            report += "  [H1] Polski: Model wyraźnie różnicuje zdania IND od COL.\n"
        else:
            report += "  [H1] Polski: Brak statystycznie istotnego rozróżnienia między IND a COL.\n"

        stat_h1_eng, p_h1_eng, test_name_eng, eng_txt = compare_intra_inter(intra_eng, dist_eng, "ENG")
        report += eng_txt
        if p_h1_eng < 0.01:
            report += "  [H1] Angielski: Model wyraźnie różnicuje zdania IND od COL.\n"
        else:
            report += "  [H1] Angielski: Brak statystycznie istotnego rozróżnienia między IND a COL.\n"

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

        plt.figure(figsize=(10, 6))
        plt.hist(dist_pol, bins=30, alpha=0.7, label="Polish")
        plt.hist(dist_eng, bins=30, alpha=0.7, label="English")
        plt.hist(dist_jap, bins=30, alpha=0.7, label="Japanese")
        plt.title(f"Dystrybucja dystansów między IND i COL ({metric_name} - H2/H3)")
        plt.xlabel("Dystans")
        plt.ylabel("Liczba przypadków")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"dist_all_{metric_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()

        if normal_pol and normal_eng and normal_jap:
            stat_t_eng, p_t_eng = ttest_ind(dist_pol, dist_eng, equal_var=False)
            p_one_eng = p_t_eng / 2.0
            stat_t_jap, p_t_jap = ttest_ind(dist_jap, dist_eng, equal_var=False)
            p_one_jap = p_t_jap / 2.0
            report += f" [H2/H3] T-test Eng (dwustronny): p(dwu)={p_t_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" [H2/H3] T-test Jap (dwustronny): p(dwu)={p_t_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
            
            # Dodajemy p-wartości do listy do korekcji
            p_values_h2h3.extend([p_one_eng, p_one_jap])
        else:
            stat_m_eng, p_m_eng = mannwhitneyu(dist_pol, dist_eng, alternative='two-sided')
            p_one_eng = p_m_eng / 2.0
            stat_m_jap, p_m_jap = mannwhitneyu(dist_jap, dist_eng, alternative='two-sided')
            p_one_jap = p_m_jap / 2.0
            report += f" [H2/H3] Mann–Whitney Eng (dwustronny): p(dwu)={p_m_eng:.4f} => p(jednostronne)={p_one_eng:.4f}\n"
            report += f" [H2/H3] Mann–Whitney Jap (dwustronny): p(dwu)={p_m_jap:.4f} => p(jednostronne)={p_one_jap:.4f}\n"
            
            # Dodajemy p-wartości do listy do korekcji
            p_values_h2h3.extend([p_one_eng, p_one_jap])
            
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
    
    # Korekcja dla wielokrotnych testów
    reject_h1, p_corrected_h1 = apply_multiple_testing_correction(p_values_h1, method='bonferroni')
    reject_h2h3, p_corrected_h2h3 = apply_multiple_testing_correction(p_values_h2h3, method='bonferroni')
    
    report += "\n=== Korekcja Bonferroniego dla wielokrotnych testów ===\n"
    report += "[H1] P-wartości oryginalne: " + ", ".join([f"{p:.6f}" for p in p_values_h1]) + "\n"
    report += "[H1] P-wartości skorygowane: " + ", ".join([f"{p:.6f}" for p in p_corrected_h1]) + "\n"
    report += "[H1] Odrzucenie H0: " + ", ".join([str(r) for r in reject_h1]) + "\n\n"
    
    report += "[H2/H3] P-wartości oryginalne: " + ", ".join([f"{p:.6f}" for p in p_values_h2h3]) + "\n"
    report += "[H2/H3] P-wartości skorygowane: " + ", ".join([f"{p:.6f}" for p in p_corrected_h2h3]) + "\n"
    report += "[H2/H3] Odrzucenie H0: " + ", ".join([str(r) for r in reject_h2h3]) + "\n"

    return report


def generate_interactive_distribution_charts(embeddings):
    """
    Generuje interaktywne wykresy rozkładów odległości.
    
    Parametry:
    ----------
    embeddings : dict
        Słownik zawierający embeddingi dla różnych kategorii.
        
    Zwraca:
    -------
    dict
        Słownik zawierający interaktywne wykresy dla różnych metryk i języków.
    """
    metrics = [("Euklides", dist_euclidean),
               ("Kosinus", dist_cosine),
               ("Manhattan", dist_manhattan)]
    figures = {}
    
    for metric_name, metric_func in metrics:
        def intra_category_dist(embeddings):
            dists = all_pairwise(embeddings, embeddings, metric_func)
            return [d for d in dists if d > 1e-12]

        intra_pol = intra_category_dist(embeddings["pol_ind"]) + intra_category_dist(embeddings["pol_col"])
        intra_eng = intra_category_dist(embeddings["eng_ind"]) + intra_category_dist(embeddings["eng_col"])
        intra_jap = intra_category_dist(embeddings["jap_ind"]) + intra_category_dist(embeddings["jap_col"])
        dist_pol = all_pairwise(embeddings["pol_ind"], embeddings["pol_col"], metric_func)
        dist_eng = all_pairwise(embeddings["eng_ind"], embeddings["eng_col"], metric_func)
        dist_jap = all_pairwise(embeddings["jap_ind"], embeddings["jap_col"], metric_func)

        fig_pol = generate_distribution_chart(intra_pol, dist_pol, "Intra POL", "Inter POL",
                                              f"Dystrybucja dystansów (Polski) - {metric_name}")
        fig_eng = generate_distribution_chart(intra_eng, dist_eng, "Intra ENG", "Inter ENG",
                                              f"Dystrybucja dystansów (Angielski) - {metric_name}")
        fig_jap = generate_distribution_chart(intra_jap, dist_jap, "Intra JAP", "Inter JAP",
                                              f"Dystrybucja dystansów (Japoński) - {metric_name}")
        figures[metric_name] = {"POL": fig_pol, "ENG": fig_eng, "JAP": fig_jap}
    
    return figures


def perform_cross_linguistic_analysis(embeddings_by_language, labels_by_language, languages=['ENG', 'POL', 'JAP']):
    """
    Przeprowadza analizę wewnątrzjęzykową i międzyjęzykową.
    
    Parametry:
    ----------
    embeddings_by_language : list
        Lista macierzy embeddingów dla różnych języków.
    labels_by_language : list
        Lista etykiet dla embeddingów dla różnych języków.
    languages : list, optional
        Lista kodów języków.
        
    Zwraca:
    -------
    dict
        Wyniki analizy wewnątrzjęzykowej i międzyjęzykowej.
    """
    results = {}
    
    # Analiza wewnątrzjęzykowa
    for lang, (emb, lbl) in zip(languages, zip(embeddings_by_language, labels_by_language)):
        # Oblicz podobieństwa wewnątrz kategorii dla danego języka
        ind_mask = [l == f"{lang}_IND" for l in lbl]
        col_mask = [l == f"{lang}_COL" for l in lbl]
        
        ind_embeddings = emb[ind_mask]
        col_embeddings = emb[col_mask]
        
        intra_ind = calculate_intra_category_similarity(ind_embeddings)
        intra_col = calculate_intra_category_similarity(col_embeddings)
        inter = calculate_inter_category_similarity(ind_embeddings, col_embeddings)
        
        results[f"{lang}_intra_ind"] = intra_ind
        results[f"{lang}_intra_col"] = intra_col
        results[f"{lang}_inter"] = inter
    
    # Analiza międzyjęzykowa
    for lang1 in languages:
        for lang2 in languages:
            if lang1 != lang2:
                # Porównaj reprezentacje między językami
                emb1_ind = embeddings_by_language[languages.index(lang1)][
                    [l == f"{lang1}_IND" for l in labels_by_language[languages.index(lang1)]]
                ]
                emb2_ind = embeddings_by_language[languages.index(lang2)][
                    [l == f"{lang2}_IND" for l in labels_by_language[languages.index(lang2)]]
                ]
                cross_ling_ind = calculate_inter_category_similarity(emb1_ind, emb2_ind)
                results[f"{lang1}_{lang2}_ind"] = cross_ling_ind
                
                # To samo dla reprezentacji kolektywistycznych
                emb1_col = embeddings_by_language[languages.index(lang1)][
                    [l == f"{lang1}_COL" for l in labels_by_language[languages.index(lang1)]]
                ]
                emb2_col = embeddings_by_language[languages.index(lang2)][
                    [l == f"{lang2}_COL" for l in labels_by_language[languages.index(lang2)]]
                ]
                cross_ling_col = calculate_inter_category_similarity(emb1_col, emb2_col)
                results[f"{lang1}_{lang2}_col"] = cross_ling_col
    
    return results


def semantic_cluster_analysis(embeddings, labels, n_clusters=None):
    """
    Przeprowadza analizę klastrów semantycznych bez założeń o podziale na kategorie.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Macierz embeddingów.
    labels : list
        Lista etykiet dla embeddingów.
    n_clusters : int, optional
        Liczba klastrów. Jeśli None, zostanie określona automatycznie.
        
    Zwraca:
    -------
    dict
        Wyniki analizy klastrów.
    """
    from sklearn.metrics import silhouette_score
    
    if n_clusters is None:
        # Automatyczne określenie optymalnej liczby klastrów
        silhouette_scores = []
        k_range = range(2, 15)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, cluster_labels))
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        n_clusters = optimal_k
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    # DBSCAN clustering (alternatywne podejście)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(embeddings)
    
    # Analiza porównawcza z rzeczywistymi etykietami
    kmeans_comparison = compare_clusters_with_labels(kmeans_labels, labels)
    dbscan_comparison = compare_clusters_with_labels(dbscan_labels, labels)
    
    return {
        'kmeans': {
            'labels': kmeans_labels,
            'centroids': kmeans.cluster_centers_,
            'comparison': kmeans_comparison
        },
        'dbscan': {
            'labels': dbscan_labels,
            'comparison': dbscan_comparison
        }
    }


def compare_clusters_with_labels(cluster_labels, true_labels):
    """
    Porównuje etykiety klastrów z rzeczywistymi etykietami.
    
    Parametry:
    ----------
    cluster_labels : numpy.ndarray
        Etykiety klastrów z algorytmu klastrowania.
    true_labels : list
        Rzeczywiste etykiety.
        
    Zwraca:
    -------
    dict
        Wyniki porównania.
    """
    from collections import Counter
    
    # Mapowanie klastrów na klasy rzeczywiste
    clusters = np.unique(cluster_labels)
    cluster_class_mapping = {}
    
    for cluster in clusters:
        if cluster == -1:  # Obsługa szumu w DBSCAN
            continue
        mask = cluster_labels == cluster
        class_counts = Counter([true_labels[i] for i, is_in_cluster in enumerate(mask) if is_in_cluster])
        dominant_class = class_counts.most_common(1)[0][0]
        cluster_class_mapping[cluster] = {
            'dominant_class': dominant_class,
            'class_distribution': dict(class_counts)
        }
    
    # Obliczanie czystości klastrów
    correct = 0
    for i, (cluster, true_class) in enumerate(zip(cluster_labels, true_labels)):
        if cluster == -1:  # Obsługa szumu w DBSCAN
            continue
        if cluster_class_mapping[cluster]['dominant_class'] == true_class:
            correct += 1
    
    purity = correct / len(true_labels) if len(true_labels) > 0 else 0
    
    return {
        'cluster_class_mapping': cluster_class_mapping,
        'purity': purity,
        'correct_assignments': correct,
        'total_samples': len(true_labels)
    }


###############################################
# KLASYFIKACJA TEKSTU (METODA CENTROIDÓW)
###############################################
def klasyfikuj_tekst(txt, centroids):
    """
    Klasyfikuje tekst metodą centroidów.
    
    Parametry:
    ----------
    txt : str
        Tekst do klasyfikacji.
    centroids : dict
        Słownik zawierający centroidy dla różnych kategorii.
        
    Zwraca:
    -------
    list
        Lista par (kategoria, podobieństwo), posortowana malejąco według podobieństwa.
    """
    vec = get_embedding(txt, model=EMBEDDING_MODEL)
    vec /= norm(vec)

    def cos_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    wyniki = {}
    for key, cent in centroids.items():
        wyniki[key] = cos_sim(vec, cent)
    return sorted(wyniki.items(), key=lambda x: x[1], reverse=True)


###############################################
# KLASYFIKACJA TEKSTU (UCZENIE MASZYNOWE)
###############################################
@timeit
def train_ml_classifier(embeddings, labels):
    """
    Trenuje klasyfikator ML na embeddingach.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Macierz embeddingów.
    labels : list
        Lista etykiet dla embeddingów.
        
    Zwraca:
    -------
    sklearn.pipeline.Pipeline
        Wytrenowany klasyfikator ML.
    """
    X = np.array(embeddings)
    y = np.array(labels)
    
    # Podział danych na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2']
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    logger.info(f"Najlepsze parametry klasyfikatora: {grid.best_params_}")
    logger.info(f"Najlepszy wynik CV: {grid.best_score_:.4f}")
    
    # Ewaluacja na zbiorze testowym
    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"Raport klasyfikacji na zbiorze testowym:\n{report}")
    
    return grid.best_estimator_


def perform_cross_validation(embeddings, labels, n_splits=5):
    """
    Przeprowadza walidację krzyżową dla klasyfikatora ML.
    
    Parametry:
    ----------
    embeddings : numpy.ndarray
        Macierz embeddingów.
    labels : list
        Lista etykiet dla embeddingów.
    n_splits : int, optional
        Liczba podziałów w walidacji krzyżowej.
        
    Zwraca:
    -------
    list
        Lista wyników walidacji krzyżowej.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Konwersja do tablic NumPy, jeśli są potrzebne
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []
    
    for i, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        logger.info(f"Fold {i+1}/{n_splits}")
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Używamy prostszej konfiguracji niż pełne trenowanie
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
            'fold': i+1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        logger.info(f"Fold {i+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return cv_results


def ml_klasyfikuj_tekst(txt, clf):
    """
    Klasyfikuje tekst przy użyciu klasyfikatora ML.
    
    Parametry:
    ----------
    txt : str
        Tekst do klasyfikacji.
    clf : sklearn.pipeline.Pipeline
        Wytrenowany klasyfikator ML.
        
    Zwraca:
    -------
    tuple
        (przewidywana_etykieta, słownik_prawdopodobieństw)
    """
    vec = get_embedding(txt, model=EMBEDDING_MODEL)
    vec /= norm(vec)
    pred = clf.predict([vec])[0]
    proba = clf.predict_proba([vec])[0]
    prob_dict = {label: prob for label, prob in zip(clf.classes_, proba)}
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    return pred, prob_dict


@timeit
def get_ml_classifier(all_embeddings, all_labels, model_path="ml_classifier.pkl", force_retrain=False):
    """
    Pobiera lub trenuje klasyfikator ML.
    
    Parametry:
    ----------
    all_embeddings : numpy.ndarray
        Macierz wszystkich embeddingów.
    all_labels : list
        Lista etykiet dla wszystkich embeddingów.
    model_path : str, optional
        Ścieżka do pliku z modelem.
    force_retrain : bool, optional
        Czy wymusić ponowne trenowanie modelu.
        
    Zwraca:
    -------
    sklearn.pipeline.Pipeline
        Klasyfikator ML.
    """
    if os.path.exists(model_path) and not force_retrain:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        logger.info("Wczytano zapisany model ML.")
    else:
        logger.info("Trenowanie nowego modelu ML...")
        clf = train_ml_classifier(all_embeddings, all_labels)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        logger.info("Wytrenowano i zapisano nowy model ML.")
    return clf


###############################################
# GŁÓWNA FUNKCJA
###############################################
@timeit
def run_analysis(force_retrain_classifier=False):
    """
    Przeprowadza pełną analizę danych.
    
    Parametry:
    ----------
    force_retrain_classifier : bool, optional
        Czy wymusić ponowne trenowanie klasyfikatora ML.
        
    Zwraca:
    -------
    dict
        Wyniki analizy.
    """
    # Wczytaj zdania i embeddingi
    zdania, embeddings = load_sentences_and_embeddings()
    logger.info(f"Wczytano zdania i wygenerowano embeddingi dla {sum(len(e) for e in embeddings.values())} przykładów")
    
    # Oblicz centroidy
    centroids = compute_all_centroids(embeddings)
    logger.info("Obliczono centroidy dla wszystkich kategorii")
    
    # Przygotuj dane do wizualizacji i analizy
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
    
    # Generuj raport statystyczny
    report_text = generate_statistical_report(embeddings)
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Zapisano raport statystyczny w pliku 'raport_statystyczny.txt'")
    
    # Trenuj klasyfikator ML
    clf = get_ml_classifier(all_embeddings, all_labels, force_retrain=force_retrain_classifier)
    
    # Zwróć wyniki do dalszego użycia
    return {
        'zdania': zdania,
        'embeddings': embeddings,
        'centroids': centroids,
        'all_embeddings': all_embeddings,
        'all_labels': all_labels,
        'classifier': clf,
        'report': report_text
    }


###############################################
# BLOK GŁÓWNY
###############################################
if __name__ == "__main__":
    logger.info("Rozpoczęcie analizy...")
    
    # Uruchom analizę
    results = run_analysis(force_retrain_classifier=False)
    
    # Test klasyfikacji
    test_txt = "I believe in working together for the greater good."
    ranking = klasyfikuj_tekst(test_txt, results['centroids'])
    logger.info("\nKlasyfikacja testowego zdania (metoda centroidów):")
    for cat, sim in ranking:
        logger.info(f" - {cat}: {sim:.4f}")
    
    pred_label, prob_dict = ml_klasyfikuj_tekst(test_txt, results['classifier'])
    logger.info("\nKlasyfikacja testowego zdania (ML):")
    logger.info(f" Przewidywana etykieta: {pred_label}")
    logger.info(" Rozkład prawdopodobieństwa:")
    for label, prob in prob_dict.items():
        logger.info(f"   {label}: {prob * 100:.2f}%")
    
    logger.info("\n=== KONIEC ANALIZY ===")

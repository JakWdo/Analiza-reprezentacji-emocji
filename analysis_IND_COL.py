import os
from openai import OpenAI
import numpy as np
import pickle
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st
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
# Jak komputer rozumie tekst?

## 1.1. Jak komputer widzi zdania?
Kiedy my czytamy tekst, od razu rozumiemy jego sens dzięki naszym doświadczeniom i wiedzy. 
Komputer nie ma takich zdolności – dla niego tekst to tylko zbiór liter. 
Żeby umożliwić mu analizę, zamieniamy każde zdanie na zestaw liczb, czyli **wektor**.

W tym badaniu każde zdanie zamieniane jest na **wektor o 3072 liczbach**. 
To tak, jakby każde zdanie było punktem w ogromnej przestrzeni liczącej 3072 wymiary. 
Dzięki temu można porównywać zdania pod względem podobieństwa ich znaczenia.

### Dlaczego akurat 3072 liczby?
Ta liczba pochodzi z konfiguracji modelu, który generuje reprezentację zdań. 
Każda liczba w wektorze odpowiada pewnej cesze zdania – może wskazywać na jego emocjonalny ton, 
kontekst użycia albo zależności między słowami. 
To tak, jakby każde zdanie miało unikalny "odcisk palca" opisujący jego sens.

### Co daje taka reprezentacja?
Jeśli dwa zdania mają podobne znaczenie, ich wektory będą podobne. 
Możemy więc mierzyć, jak bardzo różnią się znaczeniowo, sprawdzając dystans między nimi.

# Model tworzący wektory: text-embedding-3-large

## 2.1. Jak działa model?
Model **text-embedding-3-large** od OpenAI to zaawansowany system sztucznej inteligencji, 
który uczył się na milionach zdań w wielu językach.
Jego zadaniem jest zamiana dowolnego tekstu na liczby w taki sposób, 
żeby odzwierciedlały jego znaczenie.

### Jak model się uczył?
Model analizował ogromne ilości tekstów, ucząc się zależności między słowami. 
Dzięki temu rozpoznaje podobieństwa i różnice między zdaniami.

### Czy język ma znaczenie?
Teoretycznie model powinien generować podobne liczby dla zdań o tym samym znaczeniu, 
nawet jeśli są napisane w różnych językach. 
W praktyce jednak może się okazać, że pewne języki niosą inne ukryte niuanse, 
które wpływają na sposób, w jaki model je reprezentuje.

# Cel badania

## 3.1. Co chcemy sprawdzić?
Chcemy zbadać, jak model przedstawia zdania wyrażające **indywidualizm** (np. „Jestem niezależny”) 
i **kolektywizm** (np. „Działamy razem”).

### Co porównujemy?
Porównujemy **dystanse** między wektorami zdań indywidualistycznych i kolektywistycznych w różnych językach.
Dzięki temu sprawdzamy, czy w każdym języku są one tak samo „oddalone” od siebie.

Jeśli tłumaczenia zdań są dobre, dystanse powinny być podobne we wszystkich językach. 
Jeśli jednak model nauczył się pewnych ukrytych różnic kulturowych, 
to dystanse mogą się różnić, nawet jeśli zdania znaczą to samo.

# Jak liczymy dystans między zdaniami?

## 4.1. Krok 1 – Zamiana tekstu na wektory

### Jak komputer przekształca tekst?
- **Tokenizacja** – zdanie jest dzielone na mniejsze części (np. słowa lub ich fragmenty).
- **Analiza kontekstu** – model bierze pod uwagę kolejność słów i ich znaczenie.
- **Zamiana na liczby** – każde słowo jest kodowane liczbami, a całość tworzy wektor.

## 4.2. Krok 2 – Obliczanie dystansu

### Jak mierzymy podobieństwo zdań?
Dla każdej pary zdań (jedno indywidualistyczne i jedno kolektywistyczne) liczymy dystans.
Jeśli mamy 100 zdań indywidualistycznych i 100 kolektywistycznych, to obliczamy **100 × 100 = 10 000 dystansów**.

Używamy trzech metod:
- **Dystans euklidesowy** – mierzy, jak daleko od siebie są dwa punkty w przestrzeni.
- **Dystans kosinusowy** – sprawdza, czy wektory są ustawione w podobnym kierunku.
- **Dystans Manhattan** – sumuje różnice między odpowiadającymi sobie liczbami wektorów.

## 4.3. Krok 3 – Porównanie dystansów w różnych językach
Sprawdzamy, czy dystanse między zdaniami indywidualistycznymi i kolektywistycznymi w językach 
polskim, angielskim i japońskim są podobne.

- Jeśli dystanse są podobne we wszystkich językach, oznacza to, że model dobrze odwzorowuje znaczenie.
- Jeśli dystanse się różnią, może to oznaczać, że model nauczył się pewnych ukrytych różnic między językami.

Przykładowo, jeśli w języku angielskim dystanse są większe niż w języku japońskim, 
może to sugerować, że model widzi większe różnice semantyczne między zdaniami w języku angielskim niż w japońskim.

# Wnioski z badania

Cały proces można podsumować w kilku krokach:
1. **Zamiana tekstu na liczby** – każde zdanie jest kodowane jako wektor 3072 liczb.
2. **Obliczenie odległości** – sprawdzamy, jak bardzo różnią się zdania indywidualistyczne od kolektywistycznych.
3. **Porównanie języków** – sprawdzamy, czy dystanse są podobne w angielskim, polskim i japońskim.
4. **Analiza wyników** – jeśli dystanse są różne, może to oznaczać, że model nauczył się specyficznych różnic kulturowych.

To badanie pozwala nam lepiej zrozumieć, jak komputer przetwarza znaczenie tekstu. 
Czy model faktycznie rozumie znaczenie zdań w różnych językach w taki sam sposób? 
Czy może jednak różnice kulturowe wpływają na sposób, w jaki widzi tekst? 
Odpowiedzi na te pytania mogą pomóc w dalszym ulepszaniu modeli językowych i ich zdolności do analizy semantycznej.
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
#LISTY ZDAŃ (ENG/PL/JP x INDIVIDUALISTIC/COLLECTIVISTIC)
###############################################
english_individualistic = [
    "I confidently choose my own path in life.",
    "I trust my instincts to guide my decisions.",
    "I rely on my own abilities to overcome challenges.",
    "I take pride in crafting my own destiny.",
    "I celebrate my unique strengths and individuality.",
    "I am determined to succeed through my own efforts.",
    "I value the freedom to pursue my personal goals.",
    "I embrace every opportunity to learn from my experiences.",
    "I stand by my decisions with unwavering confidence.",
    "I find fulfillment in creating my own success.",
    "I believe my personal drive is my greatest asset.",
    "I trust my inner wisdom to navigate through life.",
    "I enjoy setting and achieving my own milestones.",
    "I am responsible for shaping my future.",
    "I welcome challenges as chances to grow.",
    "I work hard to achieve my dreams on my own terms.",
    "I am proud of my independent spirit.",
    "I continuously strive to improve myself.",
    "I value my solitude as a time for self-reflection.",
    "I take ownership of my choices and their outcomes.",
    "I trust that my efforts will lead me to success.",
    "I enjoy the satisfaction of self-reliance.",
    "I confidently overcome obstacles through determination.",
    "I am committed to my personal growth and development.",
    "I trust my judgment when making important decisions.",
    "I celebrate every small victory along my journey.",
    "I am my own greatest supporter.",
    "I work independently to realize my ambitions.",
    "I believe in the power of my individual vision.",
    "I set clear goals and pursue them relentlessly.",
    "I take pride in my ability to adapt and learn.",
    "I am motivated by the challenge of self-improvement.",
    "I trust myself to make the best decisions for my life.",
    "I value the freedom to explore new ideas and directions.",
    "I am resilient in the face of adversity.",
    "I celebrate my achievements, knowing they are earned by me.",
    "I rely on my creativity to solve problems.",
    "I am driven by a desire to excel on my own.",
    "I confidently navigate life’s ups and downs.",
    "I embrace independence as a core value in my life.",
    "I am empowered by my ability to stand alone.",
    "I trust my experience to guide my future choices.",
    "I enjoy the journey of self-discovery.",
    "I value the lessons learned from my personal challenges.",
    "I take initiative to make my dreams a reality.",
    "I honor my individuality in every decision I make.",
    "I am committed to building a life that reflects my values.",
    "I trust my inner strength to carry me through tough times.",
    "I celebrate my ability to think independently.",
    "I am proud to follow a path uniquely my own.",
    "I find joy in the pursuit of personal excellence.",
    "I rely on my determination to push past obstacles.",
    "I embrace my personal journey with confidence and optimism.",
    "I am self-reliant and value my independence.",
    "I trust my inner voice when facing uncertainty.",
    "I celebrate my personal achievements with gratitude.",
    "I am proactive in seeking opportunities for growth.",
    "I believe in my ability to overcome any challenge.",
    "I value the freedom to make choices that suit my needs.",
    "I continuously invest in my personal development.",
    "I enjoy the process of setting and reaching my goals.",
    "I trust that my hard work will pave the way for success.",
    "I am proud of my self-discipline and focus.",
    "I embrace every opportunity to learn and evolve.",
    "I celebrate the power of my independent mind.",
    "I take pride in crafting my own life story.",
    "I trust my own insights to guide me forward.",
    "I value the journey as much as the destination.",
    "I am confident in my ability to make a difference.",
    "I honor my personal choices without seeking validation.",
    "I believe that every step I take is a step toward growth.",
    "I enjoy the challenge of navigating life on my own.",
    "I am committed to achieving my goals through perseverance.",
    "I trust my creative instincts to lead me to success.",
    "I celebrate my progress, no matter how small.",
    "I value the independence that allows me to be true to myself.",
    "I am constantly inspired by my own potential.",
    "I embrace self-reliance as a path to personal empowerment.",
    "I trust that my individual efforts will lead to meaningful achievements.",
    "I take pride in the unique perspective I bring to the world.",
    "I am driven by my personal values and aspirations.",
    "I value my ability to solve problems on my own.",
    "I confidently step forward, trusting my own capabilities.",
    "I celebrate my personal victories with humility and joy.",
    "I honor my journey of self-discovery and growth.",
    "I am determined to leave a mark defined by my individuality.",
    "I trust my intuition to steer me in the right direction.",
    "I value the freedom to live life on my own terms.",
    "I enjoy the satisfaction that comes from overcoming challenges alone.",
    "I am proud of my resilience and self-reliance.",
    "I embrace my individuality in every aspect of my life.",
    "I trust that my personal efforts will bring lasting rewards.",
    "I celebrate my unique journey with enthusiasm.",
    "I am committed to forging a path that is truly my own.",
    "I value every opportunity to express my authentic self.",
    "I trust my inner strength to help me achieve my dreams.",
    "I am inspired by my ability to shape my own destiny.",
    "I celebrate the freedom and empowerment that comes with self-reliance.",
    "I am dedicated to living a life defined by my own choices.",
    "I take pride in being the master of my own fate."
]

english_collectivistic = [
    "We work together to overcome challenges.",
    "Our strength lies in our unity.",
    "Together, we support each other through thick and thin.",
    "We celebrate every achievement as a team.",
    "Our collective efforts create a better future for all.",
    "We share our experiences to help each other grow.",
    "Together, we turn obstacles into opportunities.",
    "We trust each other and work hand in hand.",
    "Our success is built on mutual support and collaboration.",
    "We believe in the power of community.",
    "Together, we create solutions that benefit everyone.",
    "We value open communication and shared goals.",
    "We stand united in the face of challenges.",
    "Our teamwork brings out the best in all of us.",
    "We work side by side to achieve common goals.",
    "Together, we inspire each other to reach new heights.",
    "Our collective strength makes us unstoppable.",
    "We share responsibility and celebrate our successes together.",
    "Together, we create a supportive environment for all.",
    "We value each other's contributions and talents.",
    "Our community thrives on mutual trust and respect.",
    "Together, we build a future based on shared values.",
    "We support one another in every step we take.",
    "Our collective spirit drives us toward success.",
    "We unite to overcome any obstacle in our way.",
    "Together, we turn challenges into shared victories.",
    "We celebrate diversity as a strength of our team.",
    "Our shared efforts create lasting impact.",
    "We believe that together, we can achieve more.",
    "We work collaboratively to solve problems.",
    "Together, we value every voice in our community.",
    "Our unity is the foundation of our strength.",
    "We support each other to build a better tomorrow.",
    "Together, we foster a culture of cooperation.",
    "We trust in the power of collective wisdom.",
    "Our combined efforts lead to remarkable achievements.",
    "We stand together, united by common goals.",
    "Together, we create an environment of inclusion and care.",
    "We celebrate our shared successes with pride.",
    "Our community is built on teamwork and unity.",
    "Together, we strive for excellence in everything we do.",
    "We share our skills and knowledge for the benefit of all.",
    "Our strength is magnified when we work as one.",
    "We support each other in our individual and collective journeys.",
    "Together, we create a network of trust and collaboration.",
    "We believe in lifting each other up.",
    "Our shared vision guides us towards a brighter future.",
    "We work together to achieve common dreams.",
    "Together, we celebrate every milestone as a team.",
    "We value the synergy that comes from working together.",
    "Our community flourishes when we stand united.",
    "We share the burden and the rewards of success.",
    "Together, we create opportunities for everyone.",
    "We trust each other's abilities to make the right decisions.",
    "Our collective efforts pave the way for lasting change.",
    "We celebrate the unique contributions of every member.",
    "Together, we create a harmonious and supportive environment.",
    "We believe in the strength of our community spirit.",
    "Our teamwork transforms challenges into achievements.",
    "We stand together, committed to our shared values.",
    "Together, we foster an atmosphere of empathy and care.",
    "We work collectively to build a more inclusive future.",
    "Our unity is our greatest asset.",
    "We support each other in pursuing our common goals.",
    "Together, we create a legacy of mutual success.",
    "We share our experiences and learn from one another.",
    "Our collective voice is powerful and impactful.",
    "We believe that every contribution matters.",
    "Together, we overcome obstacles and celebrate victories.",
    "We create a community where everyone feels valued.",
    "Our shared commitment unites us in times of adversity.",
    "We work together to turn dreams into reality.",
    "Together, we build bridges that connect us all.",
    "We believe in a future shaped by our collective efforts.",
    "Our unity helps us to achieve common objectives.",
    "We trust each other to work towards the common good.",
    "Together, we create solutions that make a difference.",
    "We celebrate our shared journey with gratitude.",
    "Our teamwork leads to innovation and progress.",
    "We support one another in every endeavor.",
    "Together, we create an environment of shared success.",
    "We are committed to fostering a strong sense of community.",
    "Our collective efforts strengthen our bonds.",
    "We believe in working together for the greater good.",
    "Together, we celebrate the power of unity.",
    "We trust in our collective potential to drive change.",
    "Our shared vision unites us and inspires action.",
    "We work hand in hand to overcome any challenge.",
    "Together, we create a network of care and cooperation.",
    "We support each other in reaching our collective goals.",
    "Our unity paves the way for a promising future.",
    "We believe in the power of coming together as one.",
    "Together, we overcome challenges with resilience and grace.",
    "We celebrate every step of our collective journey.",
    "Our community thrives on shared responsibility and trust.",
    "We work together to create lasting positive change.",
    "Together, we harness the strength of our diverse talents.",
    "We believe in the power of collaboration and unity.",
    "Our collective spirit drives us forward every day.",
    "Together, we are building a future that benefits everyone."
]

polish_individualistic = [
    "Z pewnością wybieram własną ścieżkę w życiu.",
    "Ufam swoim instynktom przy podejmowaniu decyzji.",
    "Polegam na swoich umiejętnościach, aby pokonywać wyzwania.",
    "Jestem dumny, że sam kształtuję swój los.",
    "Świętuję swoje unikalne cechy i indywidualność.",
    "Jestem zdeterminowany, by osiągnąć sukces dzięki własnym wysiłkom.",
    "Cenię sobie wolność realizacji moich osobistych celów.",
    "Każdą okazję traktuję jako szansę na naukę z własnych doświadczeń.",
    "Stoję przy swoich decyzjach z pełnym przekonaniem.",
    "Odnajduję satysfakcję w tworzeniu własnego sukcesu.",
    "Wierzę, że moja determinacja jest moim największym atutem.",
    "Ufam swojej wewnętrznej mądrości, by prowadziła mnie przez życie.",
    "Lubię wyznaczać i osiągać własne kamienie milowe.",
    "Biorę odpowiedzialność za kształtowanie mojej przyszłości.",
    "Postrzegam wyzwania jako szansę na rozwój.",
    "Ciężko pracuję, aby spełniać marzenia na własnych zasadach.",
    "Jestem dumny z mojego niezależnego ducha.",
    "Nieustannie dążę do samodoskonalenia.",
    "Cenię czas spędzony w samotności, który pozwala mi na refleksję.",
    "Biorę odpowiedzialność za swoje wybory i ich konsekwencje.",
    "Ufam, że moje starania doprowadzą mnie do sukcesu.",
    "Czerpię radość z polegania na sobie.",
    "Pokonuję przeszkody z determinacją i pewnością siebie.",
    "Jestem zaangażowany w swój rozwój osobisty.",
    "Ufam swojemu osądowi przy podejmowaniu ważnych decyzji.",
    "Świętuję każdą małą wygraną na swojej drodze.",
    "Jestem swoim największym wsparciem.",
    "Pracuję samodzielnie, by zrealizować swoje ambicje.",
    "Wierzę w siłę mojej indywidualnej wizji.",
    "Stawiam sobie jasne cele i dążę do ich realizacji.",
    "Cenię swoją zdolność do adaptacji i nauki.",
    "Motywuje mnie wyzwanie związane z samodoskonaleniem.",
    "Ufam sobie, podejmując decyzje dotyczące mojego życia.",
    "Cenię wolność eksplorowania nowych pomysłów i dróg.",
    "Jestem odporny w obliczu trudności.",
    "Świętuję swoje osiągnięcia, wiedząc, że są efektem mojej pracy.",
    "Polegam na swojej kreatywności, aby rozwiązywać problemy.",
    "Motywuje mnie pragnienie osiągnięcia doskonałości na własną rękę.",
    "Z ufnością przechodzę przez wzloty i upadki życia.",
    "Niezależność to jedna z moich najważniejszych wartości.",
    "Czerpię siłę z umiejętności stania na własnych nogach.",
    "Ufam swojemu doświadczeniu, które kieruje moimi wyborami.",
    "Cieszę się z podróży w kierunku samopoznania.",
    "Cenię lekcje płynące z osobistych wyzwań.",
    "Podejmuję inicjatywę, aby spełniać swoje marzenia.",
    "Szanuję swoją indywidualność przy każdej decyzji.",
    "Jestem zaangażowany w budowanie życia zgodnego z moimi wartościami.",
    "Ufam swojej wewnętrznej sile, która pomaga mi przetrwać trudne chwile.",
    "Świętuję umiejętność samodzielnego myślenia.",
    "Jestem dumny, że podążam ścieżką, która jest wyłącznie moja.",
    "Odnajduję radość w dążeniu do osobistej doskonałości.",
    "Polegam na swojej determinacji, aby pokonywać przeszkody.",
    "Przyjmuję swoją osobistą podróż z pewnością i optymizmem.",
    "Jestem samowystarczalny i cenię swoją niezależność.",
    "Ufam swojemu wewnętrznemu głosowi w chwilach niepewności.",
    "Świętuję swoje osiągnięcia z wdzięcznością.",
    "Jestem proaktywny w poszukiwaniu okazji do rozwoju.",
    "Wierzę, że jestem w stanie pokonać każde wyzwanie.",
    "Cenię wolność dokonywania wyborów zgodnych z moimi potrzebami.",
    "Nieustannie inwestuję w mój rozwój osobisty.",
    "Lubię proces wyznaczania celów i ich realizację.",
    "Ufam, że ciężka praca otworzy przede mną drogę do sukcesu.",
    "Jestem dumny ze swojej dyscypliny i skupienia.",
    "Przyjmuję każdą okazję do nauki i rozwoju.",
    "Świętuję moc mojej niezależnej myśli.",
    "Jestem dumny, że sam tworzę swoją historię życia.",
    "Ufam swoim przemyśleniom, które kierują moimi działaniami.",
    "Cenię drogę, którą przebywam, tak samo jak cel.",
    "Jestem pewien, że potrafię wnieść pozytywne zmiany.",
    "Szanuję swoje wybory, nie szukając potwierdzenia u innych.",
    "Wierzę, że każdy krok przybliża mnie do rozwoju.",
    "Czerpię satysfakcję z wyzwań, które pokonuję samodzielnie.",
    "Jestem zaangażowany w osiąganie celów dzięki wytrwałości.",
    "Ufam swojej kreatywności, która prowadzi mnie do sukcesu.",
    "Świętuję każdy postęp, niezależnie od jego wielkości.",
    "Cenię niezależność, która pozwala mi być sobą.",
    "Nieustannie inspiruje mnie mój własny potencjał.",
    "Postrzegam samodzielność jako drogę do osobistej mocy.",
    "Ufam, że moje wysiłki przyniosą znaczące rezultaty.",
    "Jestem dumny z unikalnej perspektywy, którą wnoszę do świata.",
    "Kieruję się własnymi wartościami i aspiracjami.",
    "Cenię umiejętność rozwiązywania problemów na własną rękę.",
    "Pewnie stawiam czoła wyzwaniom, ufając swoim możliwościom.",
    "Świętuję swoje osobiste zwycięstwa z pokorą i radością.",
    "Szanuję swoją drogę do samopoznania i rozwoju.",
    "Jestem zdeterminowany, by pozostawić ślad mojej indywidualności.",
    "Ufam swojemu instynktowi, który kieruje mnie właściwym torem.",
    "Cenię wolność życia według własnych zasad.",
    "Odnajduję satysfakcję w pokonywaniu trudności samodzielnie.",
    "Jestem dumny ze swojej odporności i samowystarczalności.",
    "Przyjmuję swoją indywidualność w każdej dziedzinie życia.",
    "Ufam, że moje wysiłki przyniosą trwałe efekty.",
    "Świętuję moją unikalną podróż z entuzjazmem.",
    "Jestem oddany wykuwaniu ścieżki, która jest tylko moja.",
    "Cenię każdą okazję do wyrażenia swojego autentycznego ja.",
    "Ufam mojej wewnętrznej sile, która pomaga mi osiągać marzenia.",
    "Czerpię inspirację z możliwości kształtowania własnego losu.",
    "Świętuję wolność i siłę, jakie daje mi samodzielność.",
    "Jestem zaangażowany w życie, które definiuję własnymi wyborami.",
    "Jestem dumny z bycia panem własnego losu."
]

polish_collectivistic = [
    "Razem pokonujemy wyzwania, wspierając się nawzajem.",
    "Nasza siła tkwi w jedności i wzajemnym zaufaniu.",
    "Wspólnie przechodzimy przez trudne chwile, zawsze się wspierając.",
    "Świętujemy każde osiągnięcie jako zespół.",
    "Nasze wspólne wysiłki budują lepszą przyszłość dla wszystkich.",
    "Dzielimy się doświadczeniami, by razem się rozwijać.",
    "Razem przemieniamy przeszkody w szanse.",
    "Ufamy sobie nawzajem i działamy ramię w ramię.",
    "Nasz sukces opiera się na wzajemnym wsparciu i współpracy.",
    "Wierzymy w siłę naszej społeczności.",
    "Wspólnie tworzymy rozwiązania, które przynoszą korzyści każdemu.",
    "Cenimy otwartą komunikację i wspólne cele.",
    "Stajemy razem, niezłomni wobec wszelkich trudności.",
    "Nasza praca zespołowa wydobywa z nas to, co najlepsze.",
    "Działamy ramię w ramię, by osiągnąć wspólne cele.",
    "Wzajemnie inspirujemy się, osiągając nowe wyżyny.",
    "Nasza zbiorowa siła czyni nas niepowstrzymanymi.",
    "Dzielimy się odpowiedzialnością i świętujemy wspólne sukcesy.",
    "Razem tworzymy środowisko pełne wsparcia.",
    "Cenimy wkład każdego z nas i nasze talenty.",
    "Nasza społeczność rozwija się dzięki wzajemnemu szacunkowi.",
    "Wspólnie budujemy przyszłość opartą na wspólnych wartościach.",
    "Wspieramy się na każdym kroku naszej drogi.",
    "Nasza zbiorowa energia napędza nas do sukcesu.",
    "Razem pokonujemy każdą przeszkodę.",
    "Przemieniamy wyzwania w wspólne zwycięstwa.",
    "Cenimy różnorodność, która wzbogaca nasz zespół.",
    "Nasze wspólne wysiłki mają trwały wpływ.",
    "Wierzymy, że razem możemy osiągnąć więcej.",
    "Działamy razem, by rozwiązywać problemy.",
    "Razem cenimy każdą opinię w naszej społeczności.",
    "Jedność jest fundamentem naszej siły.",
    "Wspólnie budujemy lepsze jutro.",
    "Tworzymy kulturę współpracy i wzajemnego wsparcia.",
    "Ufamy mądrości wynikającej z naszej jedności.",
    "Nasze wspólne działania prowadzą do niezwykłych osiągnięć.",
    "Stajemy razem, kierowani wspólnymi celami.",
    "Razem tworzymy środowisko pełne włączenia i troski.",
    "Świętujemy nasze wspólne sukcesy z dumą.",
    "Nasza społeczność opiera się na współpracy i jedności.",
    "Razem dążymy do doskonałości we wszystkim, co robimy.",
    "Dzielimy się wiedzą i umiejętnościami dla dobra wszystkich.",
    "Nasza siła rośnie, gdy działamy razem.",
    "Wspieramy się w indywidualnych i zbiorowych dążeniach.",
    "Razem tworzymy sieć wzajemnego zaufania i współpracy.",
    "Wierzymy, że pomaganie sobie nawzajem umacnia nas wszystkich.",
    "Nasza wspólna wizja prowadzi nas ku lepszej przyszłości.",
    "Działamy razem, by spełniać wspólne marzenia.",
    "Razem świętujemy każdy wspólny kamień milowy.",
    "Cenimy synergię płynącą z pracy zespołowej.",
    "Nasza społeczność rozkwita, gdy jesteśmy zjednoczeni.",
    "Dzielimy się zarówno trudami, jak i radościami sukcesu.",
    "Razem tworzymy szanse dla każdego.",
    "Ufamy zdolnościom każdego członka naszej grupy.",
    "Nasze wspólne wysiłki otwierają drzwi do zmian.",
    "Świętujemy unikalny wkład każdego z nas.",
    "Razem budujemy przyjazne i harmonijne środowisko.",
    "Wierzymy w siłę ducha naszej społeczności.",
    "Nasza praca zespołowa przekształca wyzwania w osiągnięcia.",
    "Stajemy razem, wierząc w nasze wspólne wartości.",
    "Wspólnie tworzymy atmosferę empatii i zrozumienia.",
    "Działamy razem, by budować bardziej otwartą przyszłość.",
    "Nasza jedność to nasz największy atut.",
    "Wspieramy się, realizując wspólne cele.",
    "Tworzymy dziedzictwo wzajemnego sukcesu.",
    "Dzielimy się doświadczeniami i uczymy się od siebie nawzajem.",
    "Nasz zbiorowy głos ma siłę zmieniania świata.",
    "Wierzymy, że każdy wkład jest ważny.",
    "Razem pokonujemy przeszkody i świętujemy zwycięstwa.",
    "Budujemy społeczność, w której każdy czuje się doceniony.",
    "Nasze wspólne zobowiązanie jednoczy nas w trudnych chwilach.",
    "Działamy razem, by przemienić marzenia w rzeczywistość.",
    "Budujemy mosty, które łączą nasze serca i umysły.",
    "Wierzymy, że nasza wspólna praca kształtuje lepszą przyszłość.",
    "Jedność pozwala nam osiągać wspólne cele.",
    "Ufamy sobie nawzajem w dążeniu do dobra wspólnego.",
    "Razem tworzymy rozwiązania, które naprawdę zmieniają świat.",
    "Świętujemy naszą wspólną drogę z wdzięcznością.",
    "Nasza praca zespołowa rodzi innowacje i postęp.",
    "Wspieramy się w każdej inicjatywie, by osiągnąć sukces.",
    "Razem tworzymy przestrzeń, w której każdy może rozkwitać.",
    "Jesteśmy oddani budowaniu silnej i zjednoczonej społeczności.",
    "Nasze wspólne działania umacniają nasze więzi.",
    "Wierzymy, że współpraca jest kluczem do sukcesu.",
    "Razem świętujemy moc jedności.",
    "Ufamy wspólnemu potencjałowi, który popycha nas do przodu.",
    "Nasza wspólna wizja inspiruje nas do działania.",
    "Działamy razem, by pokonywać każdą przeszkodę.",
    "Razem tworzymy sieć wzajemnej troski i współpracy.",
    "Wspieramy się w dążeniu do wspólnych celów.",
    "Nasza jedność otwiera drzwi do obiecującej przyszłości.",
    "Wierzymy w siłę bycia razem jako jedna rodzina.",
    "Razem pokonujemy wyzwania z determinacją i wdziękiem.",
    "Świętujemy każdy krok naszej wspólnej podróży.",
    "Nasza społeczność opiera się na wzajemnej odpowiedzialności i zaufaniu.",
    "Działamy razem, aby wprowadzać pozytywne zmiany.",
    "Razem wykorzystujemy naszą różnorodność, by osiągać wspólne cele.",
    "Ufamy mocy współpracy i jedności.",
    "Nasz zbiorowy duch popycha nas do codziennych zwycięstw.",
    "Razem budujemy przyszłość, która przynosi korzyści wszystkim."
]

japanese_individualistic = [
    "私は自信を持って自分の人生の道を選びます。",
    "私は自分の本能を信じ、決断の指針とします。",
    "困難を乗り越えるために、自分の能力に頼ります。",
    "自分の運命を自ら切り拓くことに誇りを感じます。",
    "私の独自の強みと個性を称えます。",
    "自分の努力で成功することを固く決意しています。",
    "自分の目標を追求する自由を大切にしています。",
    "経験から学ぶあらゆる機会を受け入れます。",
    "揺るぎない自信を持って自分の決断を貫きます。",
    "自分自身の成功を築くことで充実感を得ています。",
    "私の原動力こそが最大の資産だと信じています。",
    "内なる知恵を信じ、人生を切り拓いています。",
    "自分の目標を設定し達成することに喜びを感じます。",
    "未来を形作る責任は自分にあります。",
    "挑戦を成長のチャンスとして歓迎します。",
    "自分の条件で夢を実現するために一生懸命努力します。",
    "自分の独立した精神を誇りに思います。",
    "常に自己改善に努めています。",
    "一人の時間を自己反省の貴重な時間と大切にしています。",
    "自分の選択とその結果に責任を持ちます。",
    "自分の努力が成功へと導くと信じています。",
    "自立の満足感を心から味わっています。",
    "決意で自信を持って障害を克服します。",
    "自己成長と発展に努めています。",
    "重要な決断を下す際、自分の判断を信頼します。",
    "旅路の小さな勝利を一つ一つ祝い上げます。",
    "私は自分自身の最大の支援者です。",
    "自分の野望を実現するために自立して努力します。",
    "自分だけのビジョンの力を信じています。",
    "明確な目標を設定し、妥協せず追求します。",
    "適応し学ぶ能力を誇りに思います。",
    "自己改善という挑戦に駆り立てられています。",
    "自分が人生に最適な決断を下すと信じています。",
    "新たなアイデアや方向性を探求する自由を大切にしています。",
    "逆境に立ち向かう強さを持っています。",
    "自分の努力で得た成果を認め、祝福します。",
    "問題解決のために創造力に頼ります。",
    "自ら卓越したいという情熱に突き動かされています。",
    "人生の起伏を自信を持って歩んでいます。",
    "独立性を人生の根幹的な価値として大切にしています。",
    "自分一人で立ち向かえる力に自信と誇りを感じます。",
    "自分の経験が未来の選択を導くと信じています。",
    "自己発見の旅路を楽しんでいます。",
    "個人的な困難から学んだ教訓を大切にしています。",
    "夢を現実にするために率先して行動します。",
    "あらゆる決断において、自分の個性を大切にします。",
    "自分の価値観を反映した人生を築くことに専念しています。",
    "困難な時を乗り越えるため、内なる強さを信じています。",
    "独立して考える力を称賛します。",
    "自分だけの道を歩むことを誇りに思います。",
    "自己の卓越性を追求することに喜びを感じます。",
    "決意で障害を乗り越える力に頼ります。",
    "自信と楽観をもって自分の旅路を受け入れています。",
    "自立しており、その独立性を大切にしています。",
    "不確かな時、内なる声を信じています。",
    "個人の成果を感謝の気持ちで祝います。",
    "成長の機会を積極的に探しています。",
    "どんな困難も克服できる自分の力を信じています。",
    "自分のニーズに合った選択をする自由を大切にしています。",
    "自分の成長に継続的に取り組んでいます。",
    "目標を設定し、その達成過程を楽しんでいます。",
    "努力が成功への道を切り開くと信じています。",
    "自己規律と集中力を誇りに思います。",
    "学び進化するあらゆる機会を受け入れます。",
    "独立した思考の力を讃えます。",
    "自分自身の人生物語を創り上げることに誇りを感じます。",
    "自分の洞察力が未来への指針になると信じています。",
    "目的地と同じくらい、その旅路も大切にします。",
    "変化をもたらす自分の力に自信があります。",
    "他人の承認を求めず、自分の選択を尊重します。",
    "一歩一歩が成長への大切な一歩だと信じています。",
    "自分一人で人生を歩む挑戦を楽しんでいます。",
    "忍耐強く目標達成に励んでいます。",
    "創造的な直感が成功へ導くと信じています。",
    "どんなに小さくても進歩を心から祝います。",
    "自分らしさを保てる独立性を大切にしています。",
    "自分自身の可能性に常に刺激を受けています。",
    "自立こそが自己実現への道だと受け入れています。",
    "自分の個人の努力が意味ある成果に繋がると信じています。",
    "世界に提供する自分ならではの視点を誇りに思います。",
    "自分の価値観と志に突き動かされています。",
    "自分で問題を解決する能力を大切にしています。",
    "自分の能力を信じ、自信を持って前進します。",
    "謙虚さと喜びをもって個人の勝利を祝います。",
    "自己発見と成長の旅路を尊重します。",
    "自分の独自性を示す足跡を残す決意があります。",
    "正しい方向へ導くため、直感を信じています。",
    "自分らしい生き方をする自由を大切にしています。",
    "一人で挑戦を乗り越えることで得る充実感を楽しんでいます。",
    "自分の回復力と自立を誇りに思います。",
    "人生のあらゆる面で自分の個性を受け入れています。",
    "自分の努力が永続的な成果をもたらすと信じています。",
    "自分だけのユニークな旅路を熱意をもって祝います。",
    "真に自分だけの道を切り拓くことに専念しています。",
    "自分らしさを表現するあらゆる機会を大切にしています。",
    "夢を叶えるために内なる強さを信じています。",
    "自分の運命を切り拓く力に刺激を受けています。",
    "自立から得られる自由と力強さを称賛します。",
    "自分の選択で定義される人生を生きることに専念しています。",
    "自分の運命を自分で切り拓くことに誇りを持っています。"
]

japanese_collectivistic = [
    "私たちは協力して困難を乗り越えます。",
    "私たちの強さは団結にあります。",
    "共に、良い時も悪い時もお互いを支えます。",
    "チームとしてすべての成果を祝います。",
    "私たちの共同の努力が、すべての人により良い未来をもたらします。",
    "お互いの成長を助けるために経験を共有します。",
    "共に、障害を機会に変えます。",
    "私たちは互いを信頼し、手を取り合って働きます。",
    "成功は相互支援と協力の上に築かれています。",
    "私たちはコミュニティの力を信じています。",
    "共に、すべての人に利益をもたらす解決策を生み出します。",
    "率直なコミュニケーションと共通の目標を大切にします。",
    "困難に直面しても私たちは団結しています。",
    "チームワークが、皆の最善を引き出します。",
    "並んで共通の目標を達成します。",
    "共に、互いに新たな高みを目指す刺激を与えます。",
    "団結した力が、私たちを止められないものにします。",
    "責任を分かち合い、成功を共に祝います。",
    "共に、すべての人に支え合える環境を作ります。",
    "互いの貢献と才能を尊重します。",
    "コミュニティは相互の信頼と尊敬の中で繁栄しています。",
    "共に、共有する価値観に基づいた未来を築きます。",
    "一歩一歩、お互いを支え合います。",
    "共同の精神が成功へと導きます。",
    "どんな障害も乗り越えるために団結します。",
    "共に、挑戦を共通の勝利に変えます。",
    "多様性をチームの強みとして祝福します。",
    "共通の努力が持続的な影響を生み出します。",
    "共にすれば、より多くのことが達成できると信じています。",
    "協力して問題を解決します。",
    "共に、コミュニティ内のすべての声を大切にします。",
    "団結が私たちの力の基盤です。",
    "より良い明日を築くため、互いを支え合います。",
    "共に、協力の文化を育んでいます。",
    "集合的な知恵の力を信じています。",
    "結集した努力が素晴らしい成果を生み出します。",
    "共通の目標で団結し共に立ちます。",
    "共に、包摂と配慮のある環境を作ります。",
    "共に達成した成功を誇りをもって祝います。",
    "コミュニティはチームワークと団結で成り立っています。",
    "共に、あらゆることにおいて卓越性を目指します。",
    "すべての人の利益のために技能と知識を共有します。",
    "一つになって働くと私たちの力は倍増します。",
    "個々と共同の旅路の中で互いを支え合います。",
    "共に、信頼と協力のネットワークを築きます。",
    "互いを支え上げることを信じています。",
    "共通のビジョンが明るい未来へと導きます。",
    "共通の夢を実現するために共に働きます。",
    "チームとしてすべての節目を共に祝います。",
    "共に働くことで生まれるシナジーを大切にします。",
    "団結するとコミュニティは繁栄します。",
    "成功の苦労も喜びも共に分かち合います。",
    "共に、すべての人に機会を創出します。",
    "正しい決断のために互いの能力を信頼します。",
    "共同の努力が持続可能な変革への道を開きます。",
    "各メンバーのユニークな貢献を称賛します。",
    "共に、調和と支援のある環境を作ります。",
    "コミュニティ精神の強さを信じています。",
    "チームワークが挑戦を成果に変えます。",
    "共通の価値観に基づき共に立っています。",
    "共に、思いやりと配慮のある雰囲気を育みます。",
    "より包括的な未来を築くため、共同で働きます。",
    "団結こそが私たちの最大の資産です。",
    "共通の目標追求の中で互いを支え合います。",
    "共に、相互の成功という遺産を築きます。",
    "経験を共有し互いに学び合います。",
    "共同の声は強力で影響力があります。",
    "すべての貢献が重要だと信じています。",
    "共に、障害を乗り越え勝利を祝います。",
    "すべての人が大切にされるコミュニティを創ります。",
    "困難な時こそ、共通の決意が私たちを団結させます。",
    "共に、夢を現実に変えるために働きます。",
    "共に、私たちをつなぐ架け橋を築きます。",
    "共同の努力で形作られる未来を信じています。",
    "団結が共通の目標達成を助けます。",
    "共通の利益のために互いに働くことを信頼します。",
    "共に、違いを生み出す解決策を創出します。",
    "共に歩む旅路を感謝の気持ちで祝います。",
    "チームワークが革新と進歩をもたらします。",
    "あらゆる努力の中で互いを支え合います。",
    "共に、成功を共有する環境を作ります。",
    "強いコミュニティ意識を育むことに専念しています。",
    "共同の努力が私たちの絆を強めます。",
    "共通の善のために共に働くことを信じています。",
    "共に、団結の力を祝います。",
    "共同の可能性が変革を促すと信じています。",
    "共通のビジョンが団結と行動を促します。",
    "手を取り合い、どんな挑戦も乗り越えます。",
    "共に、思いやりと協力のネットワークを築きます。",
    "共通の目標達成のために互いを支えます。",
    "団結が希望に満ちた未来への道を開きます。",
    "一つになる力を信じています。",
    "共に、しなやかさと優雅さで挑戦を乗り越えます。",
    "共同の旅路の一歩一歩を祝います。",
    "共有された責任感と信頼でコミュニティは繁栄します。",
    "共に、持続可能な前向きな変化を創出します。",
    "共に、私たちの多様な才能の力を活かします。",
    "協力と団結の力を信じています。",
    "共同の精神が日々私たちを前進させます。",
    "共に、すべての人に利益をもたらす未来を築いています。"
]

###############################################
# POBRANIE EMBEDDINGÓW I OBLICZENIE CENTROIDÓW
###############################################
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
# REDUKCJA WYMIAROWOŚCI (PCA, t-SNE)
###############################################
def min_max_normalize_2d(points):
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    dx = (x_max - x_min) if x_max != x_min else 1e-6
    dy = (y_max - y_min) if y_max != y_min else 1e-6
    points[:, 0] = (points[:, 0] - x_min) / dx
    points[:, 1] = (points[:, 1] - y_min) / dy
    return points


def generate_plots():
    all_emb = np.concatenate([eng_ind_embeddings, eng_col_embeddings,
                              pol_ind_embeddings, pol_col_embeddings,
                              jap_ind_embeddings, jap_col_embeddings], axis=0)
    all_lbl = (["ENG_IND"] * len(eng_ind_embeddings) +
               ["ENG_COL"] * len(eng_col_embeddings) +
               ["POL_IND"] * len(pol_ind_embeddings) +
               ["POL_COL"] * len(pol_col_embeddings) +
               ["JAP_IND"] * len(jap_ind_embeddings) +
               ["JAP_COL"] * len(jap_col_embeddings))
    colors = {"ENG_IND": "blue", "ENG_COL": "red",
              "POL_IND": "green", "POL_COL": "orange",
              "JAP_IND": "purple", "JAP_COL": "brown"}
    labs = list(set(all_lbl))

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    red_pca = pca.fit_transform(all_emb)
    red_pca = min_max_normalize_2d(red_pca)
    plt.figure(figsize=(6, 6))
    for lab in labs:
        idxs = [i for i, l in enumerate(all_lbl) if l == lab]
        plt.scatter(red_pca[idxs, 0], red_pca[idxs, 1],
                    color=colors.get(lab, "gray"), label=lab, alpha=0.7)
    plt.title("PCA 2D (text-embedding-3-large)")
    plt.legend()
    plt.savefig("all_pca_centroids.png", dpi=300)
    plt.close()

    # t-SNE 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    red_tsne = tsne.fit_transform(all_emb)
    red_tsne = min_max_normalize_2d(red_tsne)
    plt.figure(figsize=(6, 6))
    for lab in labs:
        idxs = [i for i, l in enumerate(all_lbl) if l == lab]
        plt.scatter(red_tsne[idxs, 0], red_tsne[idxs, 1],
                    color=colors.get(lab, "gray"), label=lab, alpha=0.7)
    plt.title("t-SNE 2D (text-embedding-3-large)")
    plt.legend()
    plt.savefig("all_tsne_centroids.png", dpi=300)
    plt.close()

    # PCA 3D
    pca_3d = PCA(n_components=3, random_state=42)
    red_pca_3d = pca_3d.fit_transform(all_emb)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for lab in labs:
        idxs = [i for i, l in enumerate(all_lbl) if l == lab]
        ax.scatter(red_pca_3d[idxs, 0], red_pca_3d[idxs, 1], red_pca_3d[idxs, 2],
                   color=colors.get(lab, "gray"), label=lab, alpha=0.7)
    ax.set_title("PCA 3D (text-embedding-3-large)")
    ax.legend()
    plt.savefig("all_pca_3d.png", dpi=300)
    plt.close()

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
    red_tsne_3d = tsne_3d.fit_transform(all_emb)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for lab in labs:
        idxs = [i for i, l in enumerate(all_lbl) if l == lab]
        ax.scatter(red_tsne_3d[idxs, 0], red_tsne_3d[idxs, 1], red_tsne_3d[idxs, 2],
                   color=colors.get(lab, "gray"), label=lab, alpha=0.7)
    ax.set_title("t-SNE 3D (text-embedding-3-large)")
    ax.legend()
    plt.savefig("all_tsne_3d.png", dpi=300)
    plt.close()


def generate_interactive_pca_3d(all_emb, all_lbl):
    pca_3d = PCA(n_components=3, random_state=42)
    red_pca_3d = pca_3d.fit_transform(all_emb)
    df = pd.DataFrame({
        "PC1": red_pca_3d[:, 0],
        "PC2": red_pca_3d[:, 1],
        "PC3": red_pca_3d[:, 2],
        "Cluster": all_lbl
    })
    available_clusters = df["Cluster"].unique().tolist()
    selected_clusters = st.multiselect("Wybierz klastry (PCA 3D)",
                                       options=available_clusters, default=available_clusters)
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter_3d(filtered_df,
                        x="PC1", y="PC2", z="PC3",
                        color="Cluster",
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
    available_clusters = df["Cluster"].unique().tolist()
    selected_clusters = st.multiselect("Wybierz klastry (t-SNE 3D)",
                                       options=available_clusters, default=available_clusters)
    filtered_df = df[df["Cluster"].isin(selected_clusters)]
    fig = px.scatter_3d(filtered_df,
                        x="Dim1", y="Dim2", z="Dim3",
                        color="Cluster",
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
    # Tworzę raport tylko dla części statystycznej
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
    """
    Przekształcam tekst na wektor przy użyciu modelu text-embedding-3-large (3072D),
    normalizuję go, a następnie obliczam kosinusowe podobieństwo do centroidów ustalonych grup.
    Zwracam ranking kategorii według podobieństwa.
    """
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
# KLASYFIKACJA TEKSTU (UCZENIE MASZYNOWE)
###############################################
def train_ml_classifier(embeddings, labels):
    """
    Trenuję klasyfikator (regresję logistyczną) przy użyciu Pipeline z optymalizacją hiperparametrów.
    Dane dzielę na zbiór treningowy i testowy, a następnie dostrajam model za pomocą GridSearchCV.
    Zwracam najlepszy wytrenowany model.
    """
    X = np.array(embeddings)
    y = np.array(labels)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l2']
    }

    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print("Raport klasyfikacji (cały zbiór):")
    print(classification_report(y, best_model.predict(X)))
    return best_model


def ml_klasyfikuj_tekst(txt, clf):
    """
    Przekształcam tekst na wektor, normalizuję go i używam wytrenowanego klasyfikatora do przypisania etykiety.
    Zwracam przewidywaną kategorię oraz rozkład prawdopodobieństwa.
    """
    vec = get_embedding(txt, model=EMBEDDING_MODEL)
    vec /= norm(vec)
    pred = clf.predict([vec])[0]
    proba = clf.predict_proba([vec])[0]
    return pred, proba


###############################################
# GŁÓWNA FUNKCJA URUCHOMIENIA
###############################################
if __name__ == "__main__":
    # 1) Generuję wykresy PCA i t-SNE
    generate_plots()
    print("✅ Wygenerowano all_pca_centroids.png oraz all_tsne_centroids.png")

    # 2) Generuję raport statystyczny i zapisuję go do pliku
    report_text = generate_statistical_report()
    with open("raport_statystyczny.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Raport statystyczny zapisany w 'raport_statystyczny.txt'")

    # 3) Przykładowa klasyfikacja tekstu (metoda centroidów)
    test_txt = "I believe in working together for the greater good."
    ranking = klasyfikuj_tekst(test_txt)
    print("Klasyfikacja testowego zdania (centroidy):")
    for item in ranking:
        print(" ", item)

    # 4) Trenowanie klasyfikatora ML (na przykładowym zbiorze)
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

    # 5) Klasyfikacja tekstu przy użyciu ML
    pred_label, proba = ml_klasyfikuj_tekst(test_txt, clf)
    print("Klasyfikacja testowego zdania (ML):")
    print(" Przewidywana etykieta:", pred_label)
    print(" Rozkład prawdopodobieństwa:", proba)

    print("=== KONIEC ===")

# Analiza reprezentacji wektorowych emocji

Repozytorium **Analiza reprezentacji emocji** to aplikacja napisana w Pythonie, wykorzystująca bibliotekę Streamlit, która umożliwia interaktywną analizę reprezentacji wektorowych emocji wyrażanych w trzech językach: angielskim, polskim i japońskim.

## Opis projektu

Aplikacja korzysta z modelu OpenAI `text-embedding-3-large`, który przekształca tekst na wektory o wymiarze 3072. Na podstawie tych reprezentacji aplikacja:
- Generuje interaktywne wizualizacje 2D i 3D (PCA i t-SNE), pozwalające na analizę relacji między zdaniami.
- Umożliwia klasyfikację nowego tekstu przy użyciu dwóch metod:
  - **Metoda centroidów:** Nowy tekst jest zamieniany na wektor, a następnie porównywany z centroidami wcześniej obliczonych grup (np. indywidualistyczne vs. kolektywistyczne w każdym języku). Ranking kosinusowego podobieństwa pokazuje, do której grupy tekst jest najbliższy.
  - **Klasyfikacja ML:** Wytrenowany model (regresja logistyczna) przypisuje nowy tekst do jednej z kategorii, generując jednocześnie rozkład prawdopodobieństwa. Model jest cache’owany (zapisywany do pliku), aby nie trenować go przy każdym uruchomieniu.

Dodatkowo, aplikacja generuje raport statystyczny, w którym obliczane są dystanse między embeddingami za pomocą trzech metryk: euklidesowej, kosinusowej (1 - cos) i Manhattan. Wyniki testów statystycznych (np. test Shapiro-Wilka, test Kolmogorova-Smirnova, testy t-test/Mann–Whitney) pozwalają ocenić, czy różnice między grupami (np. między językiem polskim a angielskim) są istotne.

## Jak to działa?

1. **Przetwarzanie tekstu:**  
   Każde zdanie jest zamieniane na wektor przez model `text-embedding-3-large`. Wektory są następnie normalizowane.

2. **Wyznaczanie centroidów:**  
   Dla każdej kategorii (np. `ENG_IND`, `POL_COL` itd.) obliczany jest centroid – średnia arytmetyczna wektorów z danej grupy.

3. **Klasyfikacja metodą centroidów:**  
   Nowy tekst jest przetwarzany na wektor, a następnie obliczane jest kosinusowe podobieństwo między tym wektorem a centroidami. Wyniki sortowane malejąco wskazują, która grupa jest najbliższa semantycznie.

4. **Klasyfikacja ML:**  
   Cały zbiór embeddingów wraz z etykietami służy do trenowania klasyfikatora (regresja logistyczna). Model jest optymalizowany za pomocą GridSearchCV i zapisywany do pliku, aby przy kolejnych uruchomieniach był wczytywany zamiast trenowany od nowa. Dla nowego tekstu model przypisuje etykietę oraz generuje rozkład prawdopodobieństwa, prezentowany w czytelnej formie tabelarycznej.

5. **Wizualizacje:**  
   Aplikacja generuje interaktywne wykresy:
   - **2D:** PCA i t-SNE z opcją filtrowania klastrów.
   - **3D:** PCA i t-SNE, gdzie użytkownik może obracać wykres i filtrować dane według klastrów.

6. **Analiza statystyczna:**  
   Dla każdej z metryk (Euklides, Kosinus, Manhattan) aplikacja:
   - Oblicza dystanse między embeddingami zdań indywidualistycznych i kolektywistycznych.
   - Przeprowadza testy normalności (Shapiro-Wilka, Kolmogorova-Smirnova) oraz testy istotności (t-test lub nieparametryczny Mann–Whitney).
   - Raport podsumowuje mediany dystansów oraz wnioski o istotności różnic między językami.

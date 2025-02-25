# Analiza reprezentacji wektorowych emocji

Repozytorium **Analiza reprezentacji emocji** to aplikacja napisana w Pythonie, wykorzystująca bibliotekę Streamlit, która umożliwia interaktywną analizę reprezentacji wektorowych emocji wyrażanych w trzech językach: angielskim, polskim i japońskim.

## Opis projektu

Ten projekt badawczy analizuje, w jaki sposób różnice kulturowe pomiędzy społeczeństwami indywidualistycznymi i kolektywistycznymi są odzwierciedlone w reprezentacjach wektorowych (embeddingach) zdań w trzech różnych językach: angielskim, polskim i japońskim.
Wykorzystując model `text-embedding-3-large` od **OpenAI**, badanie to dostarcza ilościowych dowodów na to, że kulturowo uwarunkowane koncepcje indywidualizmu i kolektywizmu są odmiennie reprezentowane w przestrzeni semantycznej różnych języków.

**Aplikacja**: https://jakwdo-col-vs-ind.streamlit.app/

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
   - Korekcję Bonferroniego dla wielokrotnych testów
   - Raport podsumowuje mediany dystansów oraz wnioski o istotności różnic między językami.

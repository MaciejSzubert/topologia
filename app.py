import streamlit as st
import numpy as np

def parse_points(text: str, n: int) -> np.ndarray:
    text = text.replace('\n', ';')
    segments = [seg.strip() for seg in text.split(';') if seg.strip()]

    if not segments:
        raise ValueError("Zbiór punktów nie może być pusty.")

    points = []
    for i, seg in enumerate(segments, start=1):
        seg = seg.replace(",", " ")
        parts = seg.split()

        if len(parts) != n:
            raise ValueError(
                f"Punkt w fragmencie {i} ('{seg}') ma {len(parts)} współrzędnych, a oczekiwano {n}."
            )

        try:
            point = [float(x) for x in parts]
        except ValueError:
            raise ValueError(f"Niepoprawne dane liczbowe we fragmencie {i} ('{seg}').")

        points.append(point)

    return np.array(points, dtype=float)

def are_equal(p: np.ndarray, q: np.ndarray) -> bool:
    return bool(np.allclose(p, q))

def distance(p: np.ndarray, q: np.ndarray, metric: str, minkowski_p: float = 2) -> float:
    if metric == "Minkowskiego":
        if minkowski_p < 1:
            raise ValueError("Parametr p w metryce Minkowskiego musi być >= 1.")
        diff = np.abs(p - q)
        return float(np.sum(diff ** minkowski_p) ** (1 / minkowski_p))

    if metric == "Hamminga":
        return float(np.sum(~np.isclose(p, q)))

    if metric == "Dyskretna":
        return 0.0 if are_equal(p, q) else 1.0

    raise ValueError("Nieznana metryka.")

def distance_matrix(points: np.ndarray, metric: str, minkowski_p: float = 2) -> np.ndarray:
    s = len(points)
    matrix = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            matrix[i, j] = distance(points[i], points[j], metric, minkowski_p)
    return matrix

def diameter(points: np.ndarray, metric: str, minkowski_p: float = 2) -> float:
    matrix = distance_matrix(points, metric, minkowski_p)
    return float(np.max(matrix))

def set_distance(E: np.ndarray, F: np.ndarray, metric: str, minkowski_p: float = 2) -> float:
    min_dist = float("inf")
    for x in E:
        for y in F:
            d = distance(x, y, metric, minkowski_p)
            if d < min_dist:
                min_dist = d
    return float(min_dist)

def format_matrix_latex(matrix: np.ndarray) -> str:
    lines = []
    for row in matrix:
        lines.append(" & ".join([f"{val:g}" for val in row]))
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

st.set_page_config(page_title="Metryki w R^n", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
    }

    [data-testid="stSidebar"] {
        min-width: 450px !important;
        width: 450px !important;
    }

    .katex-display, .katex-display > .katex {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    .katex-display::-webkit-scrollbar, .katex-display > .katex::-webkit-scrollbar {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

if "task1_clicked" not in st.session_state:
    st.session_state.task1_clicked = False
if "task2_clicked" not in st.session_state:
    st.session_state.task2_clicked = False

# --- PANEL BOCZNY ---
with st.sidebar:
    st.title("Metryki w przestrzeni $\mathbb{R}^n$")
    
    st.header("Parametry ogólne")
    n = st.number_input("Wymiar przestrzeni $n$", min_value=1, max_value=100, value=2, step=1)
    metric = st.selectbox("Wybierz metrykę $d$", ["Minkowskiego", "Hamminga", "Dyskretna"])
    
    st.divider()

    st.markdown(f"**Aktualna definicja metryki dla przestrzeni $\mathbb{{R}}^{{{n}}}$:**")
    
    math_container = st.container()
    
    minkowski_p = 2.0
    if metric == "Minkowskiego":
        st.write("")
        minkowski_p = st.number_input("Parametr $p$ (Minkowskiego)", min_value=1.0, value=2.0, step=0.5)

    with math_container:
        if metric == "Minkowskiego":
            if minkowski_p == 1.0:
                st.latex(rf"d(x,y) = \sum_{{i=1}}^{{{n}}} |x_i - y_i|")
            else:
                p_str = f"{minkowski_p:g}"
                st.latex(rf"d(x,y) = \left( \sum_{{i=1}}^{{{n}}} |x_i - y_i|^{{{p_str}}} \right)^{{1/{p_str}}}")
        elif metric == "Hamminga":
            st.latex(rf"d(x,y) = \sum_{{i=1}}^{{{n}}} \mathbf{{1}}_{{\\{{x_i \neq y_i\\}}}}")
        elif metric == "Dyskretna":
            st.latex(r"d(x,y) = \begin{cases} 0 & \text{dla } x = y \\ 1 & \text{dla } x \neq y \end{cases}")


# --- OKNO GŁÓWNE ---

# --- ZADANIE 1 ---
st.subheader("1. Macierz odległości i średnica zbioru $E$")

text_E = st.text_input(
    "Podaj punkty zbioru E (współrzędne oddzielaj spacją, a kolejne punkty średnikiem ;)", 
    value="0 0; 1 2; 3 4"
)


if st.button("Oblicz macierz odległości i diam(E)", type="primary"):
    st.session_state.task1_clicked = True

if st.session_state.task1_clicked:
    try:
        E = parse_points(text_E, int(n))
        D = distance_matrix(E, metric, minkowski_p)
        diam_E = diameter(E, metric, minkowski_p)

        st.success("Obliczenia zakończone poprawnie.")
        
        st.markdown("**Macierz odległości $\mathcal{D}(E)$:**")
        st.latex(format_matrix_latex(D))
        
        st.markdown("**Średnica zbioru:**")
        st.latex(rf"\operatorname{{diam}}(E) = {diam_E:g}")

    except Exception as e:
        st.error(f"Błąd w zbiorze E: {e}")

st.divider()

# --- ZADANIE 2 ---
st.subheader("2. Odległość między dwoma zbiorami $E$ i $F$")

col_e, col_f = st.columns(2)

with col_e:
    text_E2 = st.text_input("Podaj punkty zbioru E", value="0 0; 1 1", key="E2")
with col_f:
    text_F = st.text_input("Podaj punkty zbioru F", value="3 3; 5 5", key="F")

if st.button("Oblicz dist(E,F)", type="primary"):
    st.session_state.task2_clicked = True

if st.session_state.task2_clicked:
    try:
        E2 = parse_points(text_E2, int(n))
        F = parse_points(text_F, int(n))
        dist_EF = set_distance(E2, F, metric, minkowski_p)

        st.success("Obliczenia zakończone poprawnie.")
        
        st.markdown("**Odległość między zbiorami:**")
        st.latex(rf"\operatorname{{dist}}(E,F) = {dist_EF:g}")

    except Exception as e:
        st.error(f"Błąd w zbiorach E lub F: {e}")
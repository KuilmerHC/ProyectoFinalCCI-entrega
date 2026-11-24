# %% [markdown]
# # Proyecto Final — Confiabilidad y planeación de capacidad de un servicio de inferencia de IA
# **Ciencia Computacional Intermedia — Tercer corte**
#
# Integrantes: Kuilmer Santiago Hernández Cruz · Valerie Gutiérrez Zambrano · Juan Tinoco Penagos · Juan José Berrío Español

# %%
# ===== Identificación del grupo (requisito: primera celda) =====
C1 = 2060755245
C2 = 2034625406
S = C1 + C2
print(f"C1 = {C1}\nC2 = {C2}\nS  = {S}")

# "sinteticos" para probar el código; "reales" para los datos del docente
MODO_DATOS = "reales"

# %% [markdown]
# ## 0. Configuración, parámetros del grupo y carga de datos

# %%
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import gammaln

rng = np.random.default_rng(S)          # única fuente de aleatoriedad del proyecto

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.family": "DejaVu Sans",
                     "axes.grid": True, "grid.alpha": 0.3})        # cuadrícula en todas las figuras
pd.set_option("display.precision", 4)

RAIZ = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
DIR_DATOS = RAIZ / "data" / ("sinteticos" if MODO_DATOS == "sinteticos" else "")
DIR_FIGURAS = RAIZ / "figuras"
DIR_FIGURAS.mkdir(exist_ok=True)

# %%
# ===== Parámetros del grupo (Sección 3 del enunciado) =====
TAU_MS   = 180 + 10 * (S % 9)                  # umbral de latencia del SLO (ms)
EPSILON  = 0.01 * (1 + S % 3)                  # máximo tolerado de P(L > tau)
NIVEL    = {0: 0.90, 1: 0.95, 2: 0.99}[S % 3]  # 1 - alpha de todos los intervalos
ALPHA    = 1 - NIVEL
G_CREC   = 1.4 + 0.1 * (S % 7)                 # crecimiento de la tasa de llegadas
T1_KB    = 80 + 10 * (S % 5)                   # frontera pequeña / mediana
T2_KB    = 400 + 25 * (S % 5)                  # frontera mediana / grande
PI_PROD  = (4 + S % 6) / 1000                  # prevalencia de incidentes en producción
K_MAX    = 4 + (S % 5)                         # falsas alarmas/día toleradas
R_DESPL  = S % 6                               # desplazamiento del ítem B4
N_CONSEC = 10 + (S % 11)                       # solicitudes del ítem C4
C0       = 2                                   # GPUs del sistema real
Z = stats.norm.ppf(1 - ALPHA / 2)              # cuantil normal del nivel del grupo

pd.Series({"S": S, "tau": TAU_MS, "epsilon": EPSILON, "1-alpha": NIVEL, "g": G_CREC, "t1": T1_KB,
           "t2": T2_KB, "pi": PI_PROD, "k_max": K_MAX, "r": R_DESPL, "N": N_CONSEC}, dtype=object)

# %%
# ===== Funciones comunes =====
def ic_wilson(x, n):
    """IC de Wilson para una proporción x/n (funciona bien aunque la proporción sea pequeña)."""
    p = x / n
    centro = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    semi = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return p, centro - semi, centro + semi


def ic_media(x):
    """IC t clásico para la media (supone observaciones independientes)."""
    x = np.asarray(x, float)
    semi = stats.t.ppf(1 - ALPHA / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - semi, x.mean() + semi


def ic_dif_prop(x1, n1, x2, n2, z=Z):
    """IC para p1 - p2 (aproximación normal; válida con n grandes)."""
    p1, p2 = x1 / n1, x2 / n2
    semi = z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, p1 - p2 - semi, p1 - p2 + semi


def mostrar_ic(nombre, est, inf, sup, metodo):
    """Formato exigido: estimación, extremos, nivel y método."""
    print(f"{nombre}: {est:.4f}  IC {NIVEL:.0%} [{inf:.4f}, {sup:.4f}]  ({metodo})")


def guardar_figura(fig, numero, titulo, pie):
    """Título, pie con la conclusión y archivo PNG (normas de la Sección 6.1)."""
    fig.suptitle(f"Figura {numero}. {titulo}", fontweight="bold")
    fig.tight_layout()
    fig.text(0.01, -0.02, pie, fontsize=9, style="italic", va="top", wrap=True)
    fig.savefig(DIR_FIGURAS / f"fig_{numero}.png", bbox_inches="tight")
    plt.show()

# %%
# ===== Carga de datos =====
solicitudes_crudo = pd.read_csv(DIR_DATOS / "solicitudes.csv")
monitoreo = pd.read_csv(DIR_DATOS / "monitoreo.csv")
print("solicitudes:", solicitudes_crudo.shape, "| monitoreo:", monitoreo.shape)
solicitudes_crudo.head()

# %% [markdown]
# # Parte A — Auditoría y preparación de los datos
#
# ## A1 — Auditoría
# Una fila es inconsistente si viola la física del sistema (Sección 4.1). Se **corrige** solo si el valor real es
# deducible (versión "b" → "B"; duplicado exacto → una copia; tamaño registrado en bytes → dividir por 1024); si no,
# se **excluye**. No se eliminan latencias altas pero posibles: son la cola que mide el SLO.
#
# Dos detalles de los datos reales:
# - Los tiempos vienen redondeados a 0,01 ms: red + gpu puede superar a la latencia en hasta 0,015 ms solo por
#   redondeo (tres valores con error ≤ 0,005 cada uno). Por eso R9 usa una tolerancia de 0,015 ms.
# - Algunos tamaños son miles de veces mayores que el resto (hay un salto enorme entre el mayor tamaño normal y el
#   menor anómalo) y su tiempo de GPU corresponde al de tamaño/1024: fueron registrados en **bytes**. Se corrigen.
#
# Las solicitudes con mediciones inválidas igual **llegaron**: se conservan en `llegadas` (para B1 y A2) y se
# excluyen solo de `limpio` (para todo lo que usa tiempos o intentos).

# %%
LIMITE_BYTES = 20000      # KB: ningún tamaño normal llega a esto (ver la evidencia debajo de la tabla)

def auditar(crudo):
    df = crudo.drop_duplicates().copy()                              # R1: duplicados exactos
    r2 = df["id_solicitud"].duplicated(keep=False)
    df = df[~r2]                                                     # R2: mismo id, distinto contenido
    r3 = ~df["t_llegada_s"].between(0, 1200)
    df = df[~r3]                                                     # R3: fuera de la ventana
    df["version"] = df["version"].astype(str).str.strip().str.upper()  # R4: corrección de etiqueta
    en_bytes = df["tam_kb"] > LIMITE_BYTES
    df.loc[en_bytes, "tam_kb"] = df.loc[en_bytes, "tam_kb"] / 1024   # R4b: tamaño en bytes -> KB
    df = df.sort_values("t_llegada_s").reset_index(drop=True)

    repetidos = df["latencia_ms"].value_counts()
    centinelas = repetidos[(repetidos >= 3) & (repetidos.index > df["latencia_ms"].quantile(0.99))].index
    reglas = {
        "R5 versión ∉ {A, B}": ~df["version"].isin(["A", "B"]),
        "R6 valor faltante": df.isna().any(axis=1),
        "R7 valor negativo (incluye latencia = −1, código de dato faltante)": (df[["red_ms", "gpu_ms", "latencia_ms", "tam_kb"]] < 0).any(axis=1),
        "R8 caché=1 con gpu>0 o caché=0 con gpu=0": ((df["cache_hit"] == 1) & (df["gpu_ms"] > 0))
                                                   | ((df["cache_hit"] == 0) & (df["gpu_ms"] <= 0)),
        "R9 latencia < red + gpu (espera negativa)": df["latencia_ms"] < df["red_ms"] + df["gpu_ms"] - 0.015,
        "R10 intentos ∉ {1, 2, 3}": ~df["intentos"].isin([1, 2, 3]),
        "R11 exito=0 con menos de 3 intentos": (df["exito"] == 0) & (df["intentos"] < 3),
        "R12 latencia centinela (valor fijo repetido)": df["latencia_ms"].isin(centinelas),
    }
    tabla = [
        ["R1 duplicado exacto", len(crudo) - len(crudo.drop_duplicates()), "fila idéntica a otra", "conservar una copia"],
        ["R2 mismo id con distinto contenido", int(r2.sum()), "no se sabe cuál registro es el real", "excluir"],
        ["R3 llegada fuera de [0, 1200] s", int(r3.sum()), "fuera de la ventana observada", "excluir"],
        ["R4 versión mal escrita", (crudo["version"].astype(str) != crudo["version"].astype(str).str.strip().str.upper()).sum(),
         "'b' o espacios", "corregir"],
        ["R4b tamaño registrado en bytes", int(en_bytes.sum()), f"tam_kb > {LIMITE_BYTES}", "corregir: dividir por 1024"],
    ]
    excluir = pd.Series(False, index=df.index)
    for nombre, mascara in reglas.items():
        tabla.append([nombre, int(mascara.sum()), nombre.split(" ", 1)[1], "excluir de las mediciones"])
        excluir |= mascara
    tabla.append(["TOTAL excluidas de las mediciones", int(excluir.sum()), "unión de R5–R12", ""])
    tabla = pd.DataFrame(tabla, columns=["regla", "filas afectadas", "criterio", "tratamiento"])
    return df, df[~excluir].copy(), tabla


llegadas, limpio, tabla_a1 = auditar(solicitudes_crudo)
print(f"Filas crudas: {len(solicitudes_crudo)} | llegadas válidas: {len(llegadas)} | mediciones válidas: {len(limpio)}")
display(tabla_a1)

# Evidencia para R4b: salto en los tamaños y tiempo de GPU de las filas anómalas
tam = solicitudes_crudo["tam_kb"]
nc_ok = limpio[(limpio["cache_hit"] == 0) & (limpio["tam_kb"] <= LIMITE_BYTES)]
recta = np.polyfit(nc_ok["tam_kb"], nc_ok["gpu_ms"], 1)
raros = solicitudes_crudo[(tam > LIMITE_BYTES) & (solicitudes_crudo["cache_hit"] == 0)].dropna()
print(f"Mayor tamaño normal: {tam[tam <= LIMITE_BYTES].max():.0f} KB | menor tamaño anómalo: {tam[tam > LIMITE_BYTES].min():.0f}")
print(f"GPU media de las anómalas: {raros['gpu_ms'].mean():.1f} ms | esperada con tam: {np.polyval(recta, raros['tam_kb']).mean():.0f} ms"
      f" | esperada con tam/1024: {np.polyval(recta, raros['tam_kb'] / 1024).mean():.1f} ms")

# %% [markdown]
# ## A2 — ¿Un único régimen?
# **Criterio cuantitativo:** se divide la ventana en intervalos de 30 s y se prueba si tres variables son homogéneas:
# - **conteo de llegadas**: si la tasa es constante, los conteos son Poisson con igual media y
#   $\chi^2=\sum(n_i-\bar n)^2/\bar n\sim\chi^2_{k-1}$ (prueba de dispersión);
# - **tiempo de GPU** (no caché): prueba de Kruskal–Wallis entre intervalos;
# - **proporción con reintentos**: prueba χ² de homogeneidad de proporciones entre intervalos.
#
# Si alguna rechaza, el punto de cambio se ubica partiendo la serie en dos tramos con mínima suma de cuadrados dentro
# de los tramos. **Se repite dentro de cada tramo** hasta que todos sean homogéneos: así se detecta también un episodio
# transitorio (el sistema cambia y luego vuelve a la normalidad). **Sensibilidad:** se repite con intervalos de 60 y 120 s
# (con intervalos más anchos el corte se ubica peor, porque un intervalo puede quedar partido por el cambio).

# %%
VARIABLES = ["llegadas", "gpu", "reintentos"]

def mejor_corte(serie, ancho, inicio):
    """Instante que divide la serie en dos tramos con mínima suma de cuadrados dentro de los tramos."""
    sce = [np.sum((serie[:k] - serie[:k].mean())**2) + np.sum((serie[k:] - serie[k:].mean())**2)
           for k in range(2, len(serie) - 1)]
    return inicio + (np.argmin(sce) + 2) * ancho


def pruebas_tramo(lleg, lim, ini, fin, ancho):
    """p-valores de homogeneidad y punto de cambio de cada variable dentro de [ini, fin)."""
    bordes = np.arange(ini, fin + 1e-9, ancho)
    tramo = pd.cut(lim["t_llegada_s"], bordes, right=False)
    lim, tramo = lim[tramo.notna()], tramo[tramo.notna()]
    nc = lim["cache_hit"] == 0
    conteos = np.histogram(lleg["t_llegada_s"], bordes)[0].astype(float)
    gpu_tramo = lim[nc].groupby(tramo[nc], observed=True)["gpu_ms"]
    reint_tramo = lim.groupby(tramo, observed=True)["intentos"].apply(lambda v: (v > 1).mean())
    fila = {"tramo": f"[{ini:.0f}, {fin:.0f})", "ancho (s)": ancho,
            "p llegadas": stats.chi2.sf(np.sum((conteos - conteos.mean())**2) / conteos.mean(), len(conteos) - 1),
            "p gpu": stats.kruskal(*[g for _, g in gpu_tramo]).pvalue,
            "p reintentos": stats.chi2_contingency(pd.crosstab(tramo, lim["intentos"] > 1))[1]}
    for nombre, serie in [("llegadas", conteos), ("gpu", gpu_tramo.mean().to_numpy()), ("reintentos", reint_tramo.to_numpy())]:
        fila[f"corte {nombre} (s)"] = mejor_corte(serie, ancho, ini)
    return fila


def segmentar(lleg, lim, ancho=30):
    """Corta en dos cualquier tramo no homogéneo, hasta que todos lo sean. Devuelve los instantes de cambio."""
    cortes, pendientes = [], [(0.0, 1200.0)]
    while pendientes:
        ini, fin = pendientes.pop()
        if fin - ini < 4 * ancho:                      # tramo demasiado corto para probar
            continue
        fila = pruebas_tramo(lleg, lim, ini, fin, ancho)
        rechazan = [v for v in VARIABLES if fila[f"p {v}"] < ALPHA]
        if rechazan:
            corte = float(np.median([fila[f"corte {v} (s)"] for v in rechazan]))
            cortes.append(corte)
            pendientes += [(ini, corte), (corte, fin)]
    return sorted(cortes)


def tramo_estable(cortes, margen=30):
    """El tramo más largo entre cambios, dejando fuera `margen` segundos junto a cada cambio."""
    bordes = [0.0] + cortes + [1200.0]
    i = int(np.argmax(np.diff(bordes)))
    return bordes[i] + (margen if i > 0 else 0), bordes[i + 1] - (margen if i + 1 < len(bordes) - 1 else 0)


print("Ventana completa:")
display(pd.DataFrame([pruebas_tramo(llegadas, limpio, 0, 1200, a) for a in [30, 60, 120]]))
CAMBIOS = segmentar(llegadas, limpio)
print("Cambios de régimen (s), intervalos de 30 s:", CAMBIOS or "ninguno")
for ancho in [60, 120]:
    cortes = segmentar(llegadas, limpio, ancho)
    print(f"Sensibilidad — intervalos de {ancho} s: cambios {cortes}, régimen estable {tramo_estable(cortes)}")

# %% [markdown]
# **Régimen estable:** el tramo homogéneo más largo. Se dejan fuera 30 s junto a cada cambio (la cola tarda en
# estabilizarse). Se verifica que dentro del tramo elegido ninguna de las tres pruebas rechace.

# %%
INI_EST, FIN_EST = tramo_estable(CAMBIOS)
T_EST = FIN_EST - INI_EST
for d in (llegadas, limpio):
    d["regimen"] = [f"R{k + 1}" for k in np.searchsorted(CAMBIOS, d["t_llegada_s"], side="right")]
estable = limpio[limpio["t_llegada_s"].between(INI_EST, FIN_EST, inclusive="left")].copy()
t_est = llegadas.loc[llegadas["t_llegada_s"].between(INI_EST, FIN_EST, inclusive="left"), "t_llegada_s"].to_numpy()
nc_est = estable[estable["cache_hit"] == 0]                 # no cacheadas del régimen estable

print(f"Régimen estable: [{INI_EST:.0f}, {FIN_EST:.0f}) s | n = {len(estable)}")
print("Homogeneidad dentro del tramo estable (intervalos de 30 s):")
display(pd.DataFrame([pruebas_tramo(llegadas, limpio, INI_EST, FIN_EST, 30)])[["tramo", "p llegadas", "p gpu", "p reintentos"]])

# %% [markdown]
# ## A3 — Descriptivos por versión y régimen, y Figura 1

# %%
def resumen(x):
    return pd.Series({"media": x.mean(), "mediana": x.median(), "desv": x.std(),
                      "asimetría": stats.skew(x), "curtosis": stats.kurtosis(x)})

tabla_a3 = pd.concat({var: limpio.groupby(["regimen", "version"])[var].apply(resumen).unstack()
                      for var in ["latencia_ms", "gpu_ms", "red_ms", "tam_kb"]})
display(tabla_a3.round(2))

# %%
bordes_30 = np.arange(0, 1230, 30)
tramo_30 = pd.cut(limpio["t_llegada_s"], bordes_30, right=False)
serie = limpio.groupby(tramo_30, observed=False).agg(latencia=("latencia_ms", "mean"),
                                                    reintento=("intentos", lambda v: (v > 1).mean()))
serie["gpu"] = limpio[limpio["cache_hit"] == 0].groupby(tramo_30, observed=False)["gpu_ms"].mean()
serie["llegadas"] = np.histogram(llegadas["t_llegada_s"], bordes_30)[0] / 30
t_centro = bordes_30[:-1] + 15

fig, ejes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
columnas = [("llegadas", "Llegadas (sol/s)"), ("latencia", "Latencia media (ms)"),
            ("gpu", "G media no caché (ms)"), ("reintento", "P(intentos > 1)")]
for ax, (col, etiqueta) in zip(ejes, columnas):
    ax.plot(t_centro, serie[col], "o-", ms=3, label=etiqueta)
    for i, t_c in enumerate(CAMBIOS):
        ax.axvline(t_c, color="red", ls="--", label="cambio de régimen" if i == 0 else None)
    ax.set_ylabel(etiqueta)
    ax.legend(fontsize=8, loc="upper left")
ejes[-1].set_xlabel("Tiempo desde el inicio de la ventana (s)")
guardar_figura(fig, 1, "Evolución temporal del sistema (intervalos de 30 s)",
               "Las líneas rojas marcan los cambios de régimen detectados en A2: el lector debe ver qué variables cambian y cuándo.")

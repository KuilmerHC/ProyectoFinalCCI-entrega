# %% [markdown]
# # Parte G — Simulación y decisión de capacidad
#
# ## G1 — Simulador propio
# Modelos usados (todos ajustados antes):
# - llegadas: proceso de Poisson de tasa λ (B1);
# - tamaño de la imagen: tomado al azar de los tamaños observados;
# - caché: probabilidad según la clase de tamaño (D1);
# - G: Gamma con media $\beta_0+\beta_1\,\text{tam}$ y forma k (D3);
# - red: tomada al azar de los tiempos de red observados (remuestreo: no impone ninguna forma; ver C3).
#
# **Cola FIFO con c GPUs:** se guarda en una lista a qué hora queda libre cada GPU. Cada solicitud (en orden de
# llegada) empieza cuando llega o cuando se libera la primera GPU, lo que ocurra después.
# $L=\text{red}+\text{espera}+G$. Se descartan los primeros 200 s (el sistema arranca vacío).

# %%
def armar_modelo(datos, tasa):
    clase = np.searchsorted([T1_KB, T2_KB], datos["tam_kb"], side="right")      # 0, 1, 2
    p_cache = [datos["cache_hit"][clase == i].mean() for i in range(3)]
    nc = datos[datos["cache_hit"] == 0]
    beta = minimos_cuadrados(nc["tam_kb"].to_numpy(), nc["gpu_ms"].to_numpy())[0]
    for _ in range(20):                                                       # MCP con pesos 1/μ² (D3)
        beta, _, phi = minimos_cuadrados(nc["tam_kb"].to_numpy(), nc["gpu_ms"].to_numpy(),
                                         1 / (beta[0] + beta[1] * nc["tam_kb"].to_numpy())**2)
    return {"tasa": tasa, "tam": datos["tam_kb"].to_numpy(), "red": datos["red_ms"].to_numpy(),
            "p_cache": np.array(p_cache), "beta": beta, "k": 1 / phi,
            "carga_por_seg": (1 - datos["cache_hit"].mean()) * nc["gpu_ms"].mean() / 1000}


def simular(modelo, c, duracion=1200, calentamiento=200):
    """Una corrida del sistema con c GPUs. Devuelve las latencias (ms) después del calentamiento."""
    total = duracion + calentamiento
    n = rng.poisson(modelo["tasa"] * total)
    llegada = np.sort(rng.uniform(0, total, n))              # Poisson: dado n, instantes uniformes
    tam = rng.choice(modelo["tam"], n)
    clase = np.searchsorted([T1_KB, T2_KB], tam, side="right")
    cache = rng.random(n) < modelo["p_cache"][clase]
    media_g = np.maximum(modelo["beta"][0] + modelo["beta"][1] * tam, 1.0)
    g = np.where(cache, 0.0, rng.gamma(modelo["k"], media_g / modelo["k"]))
    red = rng.choice(modelo["red"], n)

    libre = [0.0] * c                                       # hora (s) en que queda libre cada GPU
    espera = np.zeros(n)
    for i in np.flatnonzero(~cache):                        # solo las no cacheadas usan GPU, en orden de llegada
        j = libre.index(min(libre))                         # la GPU que se libera primero
        inicio = max(llegada[i], libre[j])
        espera[i] = (inicio - llegada[i]) * 1000
        libre[j] = inicio + g[i] / 1000
    latencia = red + espera + g
    return latencia[llegada >= calentamiento]


modelo = armar_modelo(estable, lam)
print(f"Modelo: λ = {lam:.2f}/s, P(caché|clase) = {np.round(modelo['p_cache'], 3)}, "
      f"β = {np.round(modelo['beta'], 4)}, k = {modelo['k']:.2f}")

# %% [markdown]
# **Validación con c = 2 y la tasa actual:** cuantiles y P(L > τ) de 20 corridas vs datos reales.

# %%
cuantiles = [0.5, 0.9, 0.95, 0.99]
corridas = [simular(modelo, C0) for _ in range(20)]
sim_q = np.array([np.quantile(L, cuantiles) for L in corridas])
sim_p = np.array([np.mean(L > TAU_MS) for L in corridas])
validacion = pd.DataFrame({
    "datos": list(np.quantile(lat, cuantiles)) + [np.mean(lat > TAU_MS)],
    "simulación": list(sim_q.mean(axis=0)) + [sim_p.mean()],
}, index=["p50 (ms)", "p90 (ms)", "p95 (ms)", "p99 (ms)", "P(L > τ)"])
validacion["diferencia %"] = 100 * (validacion["simulación"] - validacion["datos"]) / validacion["datos"]
display(validacion)
mostrar_ic("P(L > τ) datos", *ic_wilson(np.sum(lat > TAU_MS), len(lat)), "Wilson")
mostrar_ic("P(L > τ) simulación", *ic_media(sim_p), "t sobre 20 corridas")

# %% [markdown]
# **Lectura:** si las diferencias son pequeñas y el IC de los datos contiene el valor simulado, el simulador
# reproduce el sistema. La mayor discrepancia suele estar en la cola (p99, P(L > τ)): hay pocas violaciones en los
# datos (IC ancho) y el modelo ignora efectos como ráfagas o reintentos que vuelven a la cola.
#
# ## G2 — Tasa × g: ¿cuántas GPUs?
# **Carga ofrecida** $a=\lambda g(1-p_c)E[G\mid\text{no caché}]$ = GPUs ocupadas en promedio. Con c GPUs la
# utilización es $\rho=a/c$; si $\rho\ge1$ la cola crece sin límite (inestable).
#
# **Réplicas:** dentro de una corrida las latencias están correlacionadas (comparten la cola), así que cada corrida
# aporta un solo valor $\hat P_r$. Las 20 corridas son independientes (números aleatorios distintos, cola vacía al
# inicio), así que el IC t sobre ellas es válido. 20 corridas dan un IC mucho más angosto que ε.
#
# Se elige el menor c cuyo IC completo queda por debajo de ε.

# %%
modelo_g = dict(modelo, tasa=lam * G_CREC)
carga = modelo_g["tasa"] * modelo["carga_por_seg"]
print(f"Tasa proyectada = {modelo_g['tasa']:.2f}/s, carga ofrecida = {carga:.2f} GPUs ocupadas en promedio")

filas_g2, c_estrella = [], None
for c in range(1, 8):
    p_r = np.array([np.mean(simular(modelo_g, c) > TAU_MS) for _ in range(20)])
    est, inf, sup = ic_media(p_r)
    filas_g2.append([c, carga / c, est, inf, sup])
    if sup <= EPSILON:
        c_estrella = c
        break
tabla_g2 = pd.DataFrame(filas_g2, columns=["c", "ρ", "P(L>τ)", "inf", "sup"]).set_index("c")
print(f"IC al {NIVEL:.0%}, método t sobre 20 réplicas independientes:")
display(tabla_g2)
print(f"c* = {c_estrella} GPUs")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
ejes[0].errorbar(tabla_g2.index, tabla_g2["P(L>τ)"], yerr=[tabla_g2["P(L>τ)"] - tabla_g2["inf"],
                 tabla_g2["sup"] - tabla_g2["P(L>τ)"]], fmt="o-", capsize=5, label=f"P(L > τ), IC {NIVEL:.0%}")
ejes[0].axhline(EPSILON, color="red", ls="--", label=f"ε = {EPSILON}")
ejes[0].set(xlabel="número de GPUs c", ylabel="P(L > τ)", title=f"Tasa × {G_CREC}")
ejes[0].legend()
ejes[1].bar(tabla_g2.index, tabla_g2["ρ"], color=["red" if r >= 1 else "C0" for r in tabla_g2["ρ"]])
ejes[1].axhline(1, color="k", ls="--", label="ρ = 1 (límite de estabilidad)")
ejes[1].set(xlabel="número de GPUs c", ylabel="utilización por GPU ρ", title="Carga por GPU")
ejes[1].legend()
guardar_figura(fig, 9, "Número de GPUs en el escenario de crecimiento",
               "Con ρ ≥ 1 (rojo) el sistema es inestable; el menor c con el IC bajo la línea ε es el recomendado.")

# %% [markdown]
# ## G3 — Robustez de c* frente a la incertidumbre de los parámetros
# Se repite 30 veces: (1) se remuestrean con reemplazo las solicitudes del régimen estable (bootstrap) y se toma
# $\lambda_b=N_b/T$ con $N_b\sim\text{Poisson}(n)$; (2) se reajusta todo el modelo; (3) se busca el menor c con
# $\hat P(L>\tau)\le\varepsilon$. La distribución de esos c* mide la robustez.
#
# **Modelo analítico de contraste: M/M/c (Erlang C).** Supone servicio exponencial con media $E[G]$. La probabilidad
# de esperar es $C(c,a)$ y la espera, dado que se espera, es Exp($c\mu-\lambda$). Se aproxima
# $P(L>\tau)\approx p_c\,P(\text{red}>\tau)+(1-p_c)\,P(\text{red}+W+S>\tau)$ por Montecarlo sobre red, W y S.

# %%
c_boot = []
for _ in range(30):
    remuestra = estable.iloc[rng.integers(0, len(estable), len(estable))]
    modelo_b = armar_modelo(remuestra, rng.poisson(n_lleg) / T_EST * G_CREC)
    c = int(modelo_b["tasa"] * modelo_b["carga_por_seg"]) + 1               # menor c estable
    while np.mean([np.mean(simular(modelo_b, c) > TAU_MS) for _ in range(5)]) > EPSILON:
        c += 1
    c_boot.append(c)
print("Distribución de c* bajo incertidumbre:", pd.Series(c_boot).value_counts(normalize=True).sort_index().to_dict())


def p_violacion_mmc(c, m=50000):
    lam_gpu = lam * G_CREC * (1 - p_c)                     # solicitudes por segundo que usan GPU
    mu = 1000 / mu_c                                        # capacidad de una GPU (sol/s)
    a = lam_gpu / mu
    if a >= c:
        return 1.0
    suma = sum(a**k / math.factorial(k) for k in range(c))
    ultimo = a**c / math.factorial(c) / (1 - a / c)
    prob_espera = ultimo / (suma + ultimo)                  # fórmula de Erlang C
    espera = np.where(rng.random(m) < prob_espera, rng.exponential(1 / (c * mu - lam_gpu), m), 0) * 1000
    servicio = rng.exponential(mu_c, m)
    red_m = rng.choice(red, m)
    return p_c * np.mean(red_m > TAU_MS) + (1 - p_c) * np.mean(red_m + espera + servicio > TAU_MS)

c_mmc = next(c for c in range(1, 20) if p_violacion_mmc(c) <= EPSILON)
print(f"c* simulación = {c_estrella} | c* M/M/c = {c_mmc} "
      f"(CV² del servicio real = {g_nc.var() / g_nc.mean()**2:.2f}; en M/M/c es 1)")

# %% [markdown]
# **Lectura:** si el CV² real es menor que 1, el servicio es menos variable que en M/M/c, así que M/M/c sobrestima las
# esperas y puede pedir más GPUs. Además M/M/c ignora que G y la caché dependen del tamaño.
#
# ## F6 — Validación con `solicitudes_validacion.csv`
# (Se presenta aquí porque usa el simulador.) Se aplican la misma auditoría (A1) y el mismo criterio de régimen
# estable (A2), y se comparan los parámetros
# principales con los originales; luego se recalcula c* con el modelo de la nueva ventana.

# %%
ruta_val = DIR_DATOS / "solicitudes_validacion.csv"
def p_dos_prop(a, b):
    """p-valor χ² para comparar dos proporciones (a y b son vectores de 0/1)."""
    return stats.chi2_contingency([[np.sum(a), len(a) - np.sum(a)], [np.sum(b), len(b) - np.sum(b)]])[1]


if ruta_val.exists():
    llegadas_v, limpio_v, _ = auditar(pd.read_csv(ruta_val))
    ini_v, fin_v = tramo_estable(segmentar(llegadas_v, limpio_v))     # mismo criterio de régimen que A2
    T_v = fin_v - ini_v
    print(f"Validación: régimen estable [{ini_v:.0f}, {fin_v:.0f}) s")
    val = limpio_v[limpio_v["t_llegada_s"].between(ini_v, fin_v, inclusive="left")]
    nc_v = val[val["cache_hit"] == 0]
    lam_v = llegadas_v["t_llegada_s"].between(ini_v, fin_v, inclusive="left").sum() / T_v
    filas_f6 = [
        ["λ (sol/s)", lam, lam_v, 2 * stats.norm.sf(abs(lam_v - lam) / np.sqrt(lam / T_EST + lam_v / T_v))],
        ["P(caché)", p_c, val["cache_hit"].mean(), p_dos_prop(estable["cache_hit"] == 1, val["cache_hit"] == 1)],
        ["q (EMV)", q_hat, (val["intentos"] - val["exito"]).sum() / val["intentos"].sum(),
         stats.chi2_contingency([[F, I_total - F], [(val["intentos"] - val["exito"]).sum(), val["exito"].sum()]])[1]],
        ["E[G no caché] (ms)", g_nc.mean(), nc_v["gpu_ms"].mean(), stats.ttest_ind(g_nc, nc_v["gpu_ms"], equal_var=False).pvalue],
        ["E[red] (ms)", red.mean(), val["red_ms"].mean(), stats.ttest_ind(red, val["red_ms"], equal_var=False).pvalue],
        ["P(L > τ)", np.mean(lat > TAU_MS), np.mean(val["latencia_ms"] > TAU_MS),
         p_dos_prop(lat > TAU_MS, val["latencia_ms"] > TAU_MS)],
    ]
    tabla_f6 = pd.DataFrame(filas_f6, columns=["parámetro", "original", "validación", "p-valor"]).set_index("parámetro")
    tabla_f6["cambio %"] = 100 * (tabla_f6["validación"] - tabla_f6["original"]) / tabla_f6["original"]
    tabla_f6["¿cambió?"] = tabla_f6["p-valor"] < ALPHA
    display(tabla_f6)

    modelo_v = armar_modelo(val, lam_v * G_CREC)
    c_v = int(modelo_v["tasa"] * modelo_v["carga_por_seg"]) + 1
    while ic_media([np.mean(simular(modelo_v, c_v) > TAU_MS) for _ in range(20)])[2] > EPSILON:
        c_v += 1
    print(f"c* con los datos de validación = {c_v} (original: {c_estrella})")
else:
    print("Aún no está solicitudes_validacion.csv: F6 se ejecutará cuando se copie a", DIR_DATOS)

# %% [markdown]
# ## G4 — Memorando ejecutivo
# Se entrega aparte (`memo/`, una página en PDF) con las cifras de este notebook.

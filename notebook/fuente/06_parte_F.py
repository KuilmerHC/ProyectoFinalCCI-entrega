# %% [markdown]
# # Parte F — Inferencia y comparación de versiones
#
# ## F1 — Distribución muestral de la media de la latencia
# **Población:** las latencias válidas del régimen estable (la mejor representación del servicio en operación
# normal). Se toman muestras con reemplazo, así que la media verdadera μ es conocida.
# **Cobertura:** proporción de intervalos $\bar x\pm z\,s/\sqrt n$ que contienen μ.

# %%
poblacion = estable["latencia_ms"].to_numpy()
mu_pob, sigma_pob = poblacion.mean(), poblacion.std()
R_F1 = 5000

def medias_y_cobertura(n):
    muestras = rng.choice(poblacion, size=(R_F1, n))
    medias, desv = muestras.mean(axis=1), muestras.std(axis=1, ddof=1)
    cobertura = np.mean(np.abs(medias - mu_pob) <= Z * desv / np.sqrt(n))
    return medias, cobertura

medias_n, filas_f1 = {}, []
for n in [5, 30, 200, 2000]:
    medias_n[n], cob = medias_y_cobertura(n)
    filas_f1.append([n, medias_n[n].std(), sigma_pob / np.sqrt(n), stats.skew(medias_n[n]),
                     stats.kurtosis(medias_n[n]), cob])
display(pd.DataFrame(filas_f1, columns=["n", "desv. de x̄", "σ/√n", "asimetría", "curtosis", "cobertura"]))

rejilla_n = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]
coberturas = [medias_y_cobertura(n)[1] for n in rejilla_n]
n_min = next((n for n, c in zip(rejilla_n, coberturas) if c >= NIVEL - 0.01), None)
print(f"Menor n con cobertura ≥ {NIVEL - 0.01:.2f}: {n_min}")

# %% [markdown]
# **Evolución:** al crecer n la desviación de $\bar x$ sigue a $\sigma/\sqrt n$ (dispersión), la asimetría y la
# curtosis bajan hacia 0 (la forma se vuelve normal, TLC) y la cobertura del IC normal sube hacia el nominal.

# %%
fig, ejes = plt.subplots(1, 5, figsize=(20, 4))
for ax, n in zip(ejes, [5, 30, 200, 2000]):
    ax.hist(medias_n[n], bins=50, density=True, alpha=0.6, label="x̄ simulada")
    x = np.linspace(medias_n[n].min(), medias_n[n].max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu_pob, sigma_pob / np.sqrt(n)), "r", label="normal (TLC)")
    ax.set(title=f"n = {n}", xlabel="media de latencia (ms)", ylabel="densidad (1/ms)")
    ax.legend(fontsize=7)
ejes[4].semilogx(rejilla_n, coberturas, "o-", label="cobertura")
ejes[4].axhline(NIVEL - 0.01, color="red", ls="--", label="objetivo")
ejes[4].set(title="Cobertura del IC normal", xlabel="n", ylabel="cobertura")
ejes[4].legend(fontsize=7)
guardar_figura(fig, 7, "Distribución muestral de la media de la latencia",
               "Con n pequeño la media es asimétrica y el IC normal cubre menos de lo nominal; al crecer n se vuelve normal.")

# %% [markdown]
# ## F2 — IC para la latencia media
# El IC t clásico supone **observaciones independientes**. Pero solicitudes consecutivas comparten la cola: si una
# espera mucho, la siguiente también (autocorrelación positiva) y el IC clásico queda demasiado estrecho.
# **Corrección: medias por lotes.** Se divide la serie (en orden de llegada) en 30 lotes contiguos; las medias de
# lotes largos son casi independientes, y se aplica el IC t a esas 30 medias.

# %%
lat = estable.sort_values("t_llegada_s")["latencia_ms"].to_numpy()
print(f"Autocorrelación de rezago 1: {np.corrcoef(lat[:-1], lat[1:])[0, 1]:.3f}")
m, inf_cl, sup_cl = ic_media(lat)
mostrar_ic("Latencia media (ms)", m, inf_cl, sup_cl, "t clásico")
medias_lotes = np.array([lote.mean() for lote in np.array_split(lat, 30)])
_, inf_l, sup_l = ic_media(medias_lotes)
mostrar_ic("Latencia media (ms)", m, inf_l, sup_l, "medias por lotes, 30 lotes")
print(f"El IC corregido es {(sup_l - inf_l) / (sup_cl - inf_cl):.2f} veces más ancho")

# %% [markdown]
# ## F3 — Versión B vs versión A
# Diferencias B − A en (i) media de G no caché, (ii) P(intentos > 1), (iii) P(L > τ). Como se reportan las tres a la
# vez, se usa **Bonferroni**: cada IC al nivel $1-\alpha/3$, para que las tres sean correctas a la vez con
# probabilidad ≥ 1 − α. Primero se verifica que la asignación aleatoria dejó grupos comparables.

# %%
A, B = estable[estable["version"] == "A"], estable[estable["version"] == "B"]
print("Comparabilidad (medias A vs B):")
display(pd.DataFrame({"A": [A["tam_kb"].mean(), A["cache_hit"].mean(), A["red_ms"].mean(), len(A)],
                      "B": [B["tam_kb"].mean(), B["cache_hit"].mean(), B["red_ms"].mean(), len(B)]},
                     index=["tamaño medio (KB)", "P(caché)", "red media (ms)", "n"]))
print(f"Prueba KS de tamaños A vs B: p = {stats.ks_2samp(A['tam_kb'], B['tam_kb']).pvalue:.3f}")

z_bonf = stats.norm.ppf(1 - ALPHA / 6)
gA, gB = A.loc[A["cache_hit"] == 0, "gpu_ms"], B.loc[B["cache_hit"] == 0, "gpu_ms"]
dif_g = gB.mean() - gA.mean()
semi_g = z_bonf * np.sqrt(gA.var() / len(gA) + gB.var() / len(gB))
resultados_f3 = {
    "(i) E[G no caché] B − A (ms)": (dif_g, dif_g - semi_g, dif_g + semi_g),
    "(ii) P(intentos > 1) B − A": ic_dif_prop((B["intentos"] > 1).sum(), len(B), (A["intentos"] > 1).sum(), len(A), z_bonf),
    "(iii) P(L > τ) B − A": ic_dif_prop((B["latencia_ms"] > TAU_MS).sum(), len(B), (A["latencia_ms"] > TAU_MS).sum(), len(A), z_bonf),
}
tabla_f3 = pd.DataFrame(resultados_f3, index=["diferencia", "inf", "sup"]).T
tabla_f3["veredicto"] = np.where(tabla_f3["sup"] < 0, "B mejor", np.where(tabla_f3["inf"] > 0, "B peor", "sin diferencia"))
print(f"IC conjuntos al {NIVEL:.0%} (Bonferroni, cada uno al {1 - ALPHA/3:.4f}):")
display(tabla_f3)

mejora, empeora = (tabla_f3["veredicto"] == "B mejor").sum(), (tabla_f3["veredicto"] == "B peor").sum()
print("Decisión:", "adoptar B" if mejora and not empeora else "no adoptar B" if empeora and not mejora
      else "compromiso: B mejora en unas métricas y empeora en otras" if mejora else "no hay evidencia para preferir B")

# %% [markdown]
# **Regla de decisión:** se adopta B si mejora en al menos una métrica y no empeora en ninguna.
#
# ## F4 — Wald vs Wilson para una proporción
# Wald: $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$. Wilson: ver `ic_wilson`. Para cada (n, p) se simulan $10^4$ muestras
# y se cuenta cuántos IC contienen el p verdadero.

# %%
filas_f4 = []
for n in [20, 50, 100, 500]:
    for p in [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]:
        x = rng.binomial(n, p, 10000)
        p_hat = x / n
        semi_wald = Z * np.sqrt(p_hat * (1 - p_hat) / n)
        _, w_inf, w_sup = ic_wilson(x, n)
        filas_f4.append([n, p, np.mean(np.abs(p_hat - p) <= semi_wald), np.mean((w_inf <= p) & (p <= w_sup))])
tabla_f4 = pd.DataFrame(filas_f4, columns=["n", "p", "Wald", "Wilson"])
display(tabla_f4.pivot(index="p", columns="n", values=["Wald", "Wilson"]).round(3))

fig, ejes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, (n, g) in zip(ejes, tabla_f4.groupby("n")):
    ax.plot(g["p"], g["Wald"], "o-", label="Wald")
    ax.plot(g["p"], g["Wilson"], "s-", label="Wilson")
    ax.axhline(NIVEL, color="k", ls="--", label="nominal")
    ax.set(title=f"n = {n}", xlabel="p verdadera", xscale="log")
ejes[0].set_ylabel("cobertura efectiva")
ejes[0].legend()
guardar_figura(fig, 8, "Cobertura de los IC para una proporción (10⁴ réplicas)",
               "Wald pierde cobertura cuando n·p es pequeño; Wilson se mantiene cerca del nivel nominal.")

# %% [markdown]
# ## F5 — Tamaño de muestra para semiancho ≤ 1 ms en F3(i)
# $z\sqrt{(\sigma_A^2+\sigma_B^2)/n}\le1\ \Rightarrow\ n\ge z^2(\sigma_A^2+\sigma_B^2)/1^2$ (por versión, no cacheadas),
# con el z de Bonferroni usado en F3.

# %%
n_f5 = int(np.ceil(z_bonf**2 * (gA.var() + gB.var()) / 1**2))
print(f"n necesario por versión (no cacheadas): {n_f5}")
semis = [z_bonf * np.sqrt(rng.choice(gA, n_f5).var() / n_f5 + rng.choice(gB, n_f5).var() / n_f5) for _ in range(500)]
print(f"Simulación (500 réplicas): semiancho medio = {np.mean(semis):.3f} ms")

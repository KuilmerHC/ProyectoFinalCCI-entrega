# %% [markdown]
# # Parte E — Monitoreo y decisión de alertas
#
# ## E1 — Sensibilidad y tasa de falsos positivos
# Sensibilidad $s=P(\text{alerta}\mid\text{incidente})$; tasa de falsos positivos $f=P(\text{alerta}\mid\text{normal})$.
# No dependen de la prevalencia, por eso se pueden estimar en el entorno de pruebas.

# %%
inc = monitoreo["incidente"] == 1
d1, d2 = monitoreo["D1"] == 1, monitoreo["D2"] == 1
reglas = {"D1": d1, "D2": d2, "D1 y D2": d1 & d2, "D1 o D2": d1 | d2}

filas_e1 = []
for nombre, alerta in reglas.items():
    s, s_inf, s_sup = ic_wilson((alerta & inc).sum(), inc.sum())
    f, f_inf, f_sup = ic_wilson((alerta & ~inc).sum(), (~inc).sum())
    filas_e1.append([nombre, s, s_inf, s_sup, f, f_inf, f_sup])
tabla_e1 = pd.DataFrame(filas_e1, columns=["regla", "s", "s inf", "s sup", "f", "f inf", "f sup"]).set_index("regla")
print(f"Intervalos al {NIVEL:.0%}, método de Wilson:")
display(tabla_e1)

# %% [markdown]
# ## E2 — P(incidente | alerta) en producción
# Teorema de Bayes con la prevalencia de producción π: $\text{VPP}=\dfrac{s\pi}{s\pi+f(1-\pi)}$.
# Intervalo: se evalúa el VPP en los extremos de los IC de s y f (peor y mejor caso).

# %%
def vpp(s, f, prev=PI_PROD):
    return s * prev / (s * prev + f * (1 - prev))

tabla_e2 = pd.DataFrame({
    "VPP producción": vpp(tabla_e1["s"], tabla_e1["f"]),
    "inf": vpp(tabla_e1["s inf"], tabla_e1["f sup"]),
    "sup": vpp(tabla_e1["s sup"], tabla_e1["f inf"]),
    "VPP en monitoreo.csv": [(a & inc).sum() / a.sum() for a in reglas.values()],
})
print(f"Intervalo al {NIVEL:.0%}: VPP evaluado en los extremos de los IC de Wilson de s y f")
display(tabla_e2)
# Verificación por simulación (regla D1): un millón de minutos de producción
hay_inc = rng.random(1_000_000) < PI_PROD
alerta_sim = rng.random(1_000_000) < np.where(hay_inc, tabla_e1.loc["D1", "s"], tabla_e1.loc["D1", "f"])
print(f"Verificación D1: simulación {hay_inc[alerta_sim].mean():.4f} vs Bayes {tabla_e2.loc['D1', 'VPP producción']:.4f}")
print(f"Prevalencia en pruebas = {inc.mean():.3f} vs producción π = {PI_PROD}")

# %% [markdown]
# **¿Por qué difieren?** En pruebas los incidentes se inyectaron con una prevalencia mucho mayor. Con π pequeño hay
# muchísimos más minutos normales, así que incluso una f pequeña produce más falsas alarmas que alarmas verdaderas.
#
# ## E3 — ¿D1 y D2 son condicionalmente independientes?
# Dentro de cada estado se compara $P(D_1\wedge D_2)$ con $P(D_1)P(D_2)$ y se aplica una prueba χ² a la tabla 2×2.

# %%
filas_e3 = []
for estado, m in [("con incidente", inc), ("sin incidente", ~inc)]:
    filas_e3.append([estado, (d1[m] & d2[m]).mean(), d1[m].mean() * d2[m].mean(),
                     stats.chi2_contingency(pd.crosstab(d1[m], d2[m]))[1]])
display(pd.DataFrame(filas_e3, columns=["estado", "P(D1 y D2)", "P(D1)·P(D2)", "p χ²"]))

# Efecto sobre E2: VPP de las reglas combinadas si se SUPONE independencia
s1, s2, f1, f2 = tabla_e1.loc["D1", "s"], tabla_e1.loc["D2", "s"], tabla_e1.loc["D1", "f"], tabla_e1.loc["D2", "f"]
display(pd.DataFrame({
    "VPP con datos (E2)": [tabla_e2.loc["D1 y D2", "VPP producción"], tabla_e2.loc["D1 o D2", "VPP producción"]],
    "VPP suponiendo independencia": [vpp(s1 * s2, f1 * f2), vpp(s1 + s2 - s1 * s2, f1 + f2 - f1 * f2)],
}, index=["D1 y D2", "D1 o D2"]))

# %% [markdown]
# ## E4 — Falsas alarmas por día
# Cada minuto produce una falsa alarma con probabilidad $f(1-\pi)$; en 1440 minutos independientes
# $FA\sim\text{Bin}(1440,\,f(1-\pi))$. Se exige $P(FA\le k_{máx})\ge0{,}90$.
# **Incertidumbre:** P(FA ≤ k_máx) disminuye con f, así que su IC sale de evaluar en los extremos del IC de f.
# Una regla cumple con seguridad si incluso con f en su extremo superior se llega a 0,90.

# %%
def p_cumple(f):
    return stats.binom.cdf(K_MAX, 1440, f * (1 - PI_PROD))

tabla_e4 = pd.DataFrame({
    "FA esperadas/día": 1440 * tabla_e1["f"] * (1 - PI_PROD),
    "P(FA ≤ k_máx)": p_cumple(tabla_e1["f"]),
    "inf": p_cumple(tabla_e1["f sup"]), "sup": p_cumple(tabla_e1["f inf"]),
    "sensibilidad": tabla_e1["s"],
})
tabla_e4["cumple"] = tabla_e4["inf"] >= 0.90
print(f"Intervalo al {NIVEL:.0%}: P(FA ≤ k_máx) evaluada en los extremos del IC de Wilson de f")
display(tabla_e4)
cumplen = tabla_e4[tabla_e4["cumple"]]
REGLA = cumplen["sensibilidad"].idxmax() if len(cumplen) else tabla_e4["P(FA ≤ k_máx)"].idxmax()
print("Regla recomendada:", REGLA, "(la más sensible entre las que cumplen)" if len(cumplen)
      else "(ninguna cumple; es la más cercana)")
# Verificación por simulación: 20 000 días de 1440 minutos con la regla recomendada
falsas = (rng.random((20000, 1440)) < tabla_e1.loc[REGLA, "f"] * (1 - PI_PROD)).sum(axis=1)
print(f"Verificación {REGLA}: simulación {np.mean(falsas <= K_MAX):.4f} vs binomial {tabla_e4.loc[REGLA, 'P(FA ≤ k_máx)']:.4f}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
f_eje = np.linspace(0, tabla_e1["f sup"].max() * 1.1, 300)
ax.plot(f_eje, p_cumple(f_eje), "k-", label="P(FA ≤ k_máx) según f")
ax.axhline(0.90, color="red", ls="--", label="exigencia 0,90")
for nombre, fila in tabla_e1.iterrows():
    ax.errorbar(fila["f"], p_cumple(fila["f"]), xerr=[[fila["f"] - fila["f inf"]], [fila["f sup"] - fila["f"]]],
                fmt="o", capsize=5, label=f"{nombre} (IC de f)")
ax.set(xlabel="tasa de falsos positivos por minuto f", ylabel="P(falsas alarmas en un día ≤ k_máx)")
ax.legend(fontsize=8)
guardar_figura(fig, 6, "Cumplimiento del límite de falsas alarmas",
               "Solo cumplen las reglas cuyo intervalo completo queda por encima de la línea roja.")

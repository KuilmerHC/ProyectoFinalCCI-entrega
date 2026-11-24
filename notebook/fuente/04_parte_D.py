# %% [markdown]
# # Parte D — Dependencias y esperanza condicional
#
# ## D1 — Conjunta de (clase de tamaño, `cache_hit`)
# Conjunta = proporción de cada combinación; marginales = totales; condicional $P(\text{caché}\mid\text{clase})$.
# Si son independientes, la condicional es igual en todas las clases. Evidencia: prueba χ² de independencia e IC
# de Wilson de cada condicional.

# %%
estable["clase"] = pd.cut(estable["tam_kb"], [0, T1_KB, T2_KB, np.inf], right=False,
                          labels=["pequeña", "mediana", "grande"])
conjunta = pd.crosstab(estable["clase"], estable["cache_hit"], normalize=True, margins=True)
print("Conjunta y marginales:"); display(conjunta.round(4))

conteos_d1 = pd.crosstab(estable["clase"], estable["cache_hit"])
cond = pd.DataFrame([ic_wilson(fila[1], fila.sum()) for _, fila in conteos_d1.iterrows()],
                    index=conteos_d1.index, columns=["P(caché | clase)", "IC inf", "IC sup"])
print(f"Condicionales P(caché | clase), IC {NIVEL:.0%} de Wilson:"); display(cond)
chi2_d1, p_d1, _, _ = stats.chi2_contingency(conteos_d1)
print(f"χ² de independencia = {chi2_d1:.1f}, p = {p_d1:.2e} ->",
      "NO son independientes" if p_d1 < ALPHA else "no hay evidencia contra la independencia")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(conjunta.iloc[:3, :2].to_numpy(), cmap="Blues")
for i in range(3):
    for j in range(2):
        ax.text(j, i, f"{conjunta.iloc[i, j]:.3f}", ha="center", va="center",
                color="white" if conjunta.iloc[i, j] > 0.3 else "black")
ax.set_xticks([0, 1], ["sin caché (0)", "caché (1)"])
ax.set_yticks([0, 1, 2], conjunta.index[:3])
ax.set(xlabel="cache_hit", ylabel="clase de tamaño")
fig.colorbar(im, label="probabilidad conjunta")
guardar_figura(fig, 4, "Conjunta de clase de tamaño y caché",
               "Las imágenes pequeñas se cachean más que las grandes: tamaño y caché no son independientes.")

# %% [markdown]
# ## D2 — Esperanza condicional, esperanza total y varianza total
# - $E[G\mid\text{clase},\text{no caché}]$: media por clase.
# - $E[G\mid\text{tam\_kb}]$ no paramétrica: media de G en cada decil de tamaño.
# - **Esperanza total:** $E[G]=\sum P(\text{clase})\,E[G\mid\text{clase}]$.
# - **Varianza total:** $\text{Var}(G)=E[\text{Var}(G\mid\text{clase})]+\text{Var}(E[G\mid\text{clase}])$; la fracción
#   explicada por la clase es $\text{Var}(E[G\mid\text{clase}])/\text{Var}(G)$.

# %%
nc_est = estable[estable["cache_hit"] == 0]
print("E[G | clase, no caché]:", nc_est.groupby("clase", observed=True)["gpu_ms"].mean().round(2).to_dict())
deciles = estable.groupby(pd.qcut(estable["tam_kb"], 10), observed=True)["gpu_ms"].mean()
print("E[G | tam_kb] por decil de tamaño:"); display(deciles.round(2))

grupo = estable.groupby("clase", observed=True)["gpu_ms"]
P_cl, E_cl, V_cl = grupo.size() / len(estable), grupo.mean(), grupo.var(ddof=0)
E_G, var_G = estable["gpu_ms"].mean(), estable["gpu_ms"].var(ddof=0)
var_dentro, var_entre = np.sum(P_cl * V_cl), np.sum(P_cl * (E_cl - E_G)**2)
print(f"Esperanza total: Σ P·E[G|clase] = {np.sum(P_cl * E_cl):.4f} | E[G] = {E_G:.4f}")
print(f"Varianza total: {var_dentro:.2f} + {var_entre:.2f} = {var_dentro + var_entre:.2f} | Var(G) = {var_G:.2f}")
print(f"Fracción de Var(G) explicada por la clase: {var_entre / var_G:.3f}")

# %% [markdown]
# ## D3 — Modelo lineal $E[G\mid\text{tam},\text{no caché}]=\beta_0+\beta_1\,\text{tam}$
# Ecuaciones normales: $X^\top X\hat\beta=X^\top y$. $\text{Cov}(\hat\beta)=\hat\sigma^2(X^\top X)^{-1}$,
# $\hat\sigma^2=\text{SCE}/(n-2)$.
#
# **Supuesto a revisar:** varianza constante de los residuos. Si G | tam es Gamma (C2), su varianza es
# $\mu^2/k$: crece con la media. **Corrección:** mínimos cuadrados ponderados con pesos $w=1/\hat\mu^2$:
# $\hat\beta_W=(X^\top WX)^{-1}X^\top Wy$, iterando porque $\hat\mu$ depende de $\hat\beta$.

# %%
def minimos_cuadrados(tam, g, pesos=None):
    X = np.column_stack([np.ones(len(tam)), tam])
    w = np.ones(len(g)) if pesos is None else pesos
    XtWX = X.T @ (X * w[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (w * g))                    # resuelve el sistema matricial
    resid = g - X @ beta
    sigma2 = np.sum(w * resid**2) / (len(g) - 2)
    ee = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtWX)))
    return beta, ee, sigma2


tam_nc, g_nc = nc_est["tam_kb"].to_numpy(), nc_est["gpu_ms"].to_numpy()
t_crit = stats.t.ppf(1 - ALPHA / 2, len(g_nc) - 2)
beta_mco, ee_mco, _ = minimos_cuadrados(tam_nc, g_nc)
for j, nombre in enumerate(["β0 (ms)", "β1 (ms/KB)"]):
    mostrar_ic(f"MCO {nombre}", beta_mco[j], beta_mco[j] - t_crit * ee_mco[j], beta_mco[j] + t_crit * ee_mco[j], "t")
print("Verificación numérica con np.polyfit (β0, β1):", np.round(np.polyfit(tam_nc, g_nc, 1)[::-1], 5))

# Residuos: ¿la varianza crece con el valor ajustado?
ajustado = beta_mco[0] + beta_mco[1] * tam_nc
resid = g_nc - ajustado
quintil = pd.qcut(ajustado, 5)
print("Desviación de los residuos por quintil del valor ajustado:",
      pd.Series(resid).groupby(quintil, observed=True).std().round(1).tolist())

# Corrección: pesos 1/μ², iterando hasta que β no cambie
beta_w = beta_mco
for _ in range(20):
    beta_w, ee_w, phi = minimos_cuadrados(tam_nc, g_nc, 1 / (beta_w[0] + beta_w[1] * tam_nc)**2)
for j, nombre in enumerate(["β0 (ms)", "β1 (ms/KB)"]):
    mostrar_ic(f"MCP {nombre}", beta_w[j], beta_w[j] - t_crit * ee_w[j], beta_w[j] + t_crit * ee_w[j], "t, pesos 1/μ²")
k_gamma = 1 / phi                                                  # forma Gamma implícita (se usa en G)
print(f"Forma Gamma implícita k = 1/φ = {k_gamma:.3f}")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
xs = np.linspace(0, np.quantile(tam_nc, 0.99), 100)
ejes[0].scatter(tam_nc, g_nc, s=2, alpha=0.2, color="gray", label="datos")
ejes[0].plot(xs, beta_mco[0] + beta_mco[1] * xs, label="MCO")
ejes[0].plot(xs, beta_w[0] + beta_w[1] * xs, "--", label="MCP (pesos 1/μ²)")
ejes[0].set(xlabel="tam_kb (KB)", ylabel="G no caché (ms)", title="Ajuste lineal",
            xlim=(0, xs.max()), ylim=(0, np.quantile(g_nc, 0.99)))
ejes[0].legend()
ejes[1].scatter(ajustado, resid, s=2, alpha=0.2)
ejes[1].axhline(0, color="k")
ejes[1].set(xlabel="valor ajustado (ms)", ylabel="residuo (ms)", title="Residuos MCO")
guardar_figura(fig, 5, "Regresión del tiempo de GPU sobre el tamaño",
               "Los residuos se abren en forma de embudo: la varianza crece con la media, por eso se corrige con pesos 1/μ².")

# %% [markdown]
# ## D4 — Covarianza y correlación

# %%
cov_d4 = np.cov(tam_nc, g_nc)[0, 1]
r_d4 = np.corrcoef(tam_nc, g_nc)[0, 1]
print(f"Cov = {cov_d4:.2f} KB·ms | r = {r_d4:.4f} | r² = {r_d4**2:.4f}")
print(f"β1 de MCO = Cov/Var(tam) = {cov_d4 / np.var(tam_nc, ddof=1):.5f} (MCO: {beta_mco[1]:.5f})")

# %% [markdown]
# **Lectura:** la pendiente de MCO es Cov/Var(tam) y su R² es r². La correlación mide solo la parte lineal de la
# relación; la fracción explicada por la clase (D2) usa solo 3 grupos, por eso suele ser menor que r².

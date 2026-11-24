# %% [markdown]
# # Parte C — Tiempo de cómputo y tiempo de red
#
# ## C1 — Distribución de G (todas las solicitudes)
# G es **mixta**: vale exactamente 0 si hubo caché (probabilidad $p_c>0$) y es continua si no. Con $F_c$ la FDA de
# G | no caché:
# $$F_G(x)=0\ (x<0),\qquad F_G(x)=p_c+(1-p_c)F_c(x)\ (x\ge0).$$
# Con $\mu_c,\sigma_c^2$ la media y varianza de G | no caché:
# $E[G]=(1-p_c)\mu_c$ y $\text{Var}(G)=(1-p_c)\sigma_c^2+p_c(1-p_c)\mu_c^2$.

# %%
g_todas = estable["gpu_ms"].to_numpy()
g_nc = nc_est["gpu_ms"].to_numpy()
p_c = estable["cache_hit"].mean()
mu_c, var_c = g_nc.mean(), g_nc.var()
print(f"p_c = {p_c:.4f}, μ_c = {mu_c:.3f} ms, σ²_c = {var_c:.3f} ms²")
print(f"E[G]:   fórmula = {(1 - p_c) * mu_c:.3f} | muestral = {g_todas.mean():.3f}")
print(f"Var(G): fórmula = {(1 - p_c) * var_c + p_c * (1 - p_c) * mu_c**2:.3f} | muestral = {g_todas.var():.3f}")

# %% [markdown]
# ## C2 — Máxima verosimilitud para la parte continua (modelo Gamma)
# Log-verosimilitud de Gamma(k, θ):
# $\ell(k,\theta)=-n\log\Gamma(k)-nk\log\theta+(k-1)\sum\log x_i-\sum x_i/\theta$.
# Se minimiza $-\ell$ con `optimize.minimize` sobre $(\log k,\log\theta)$, para que ambos sean positivos.
# **Otro método:** momentos, $k=\bar x^2/s^2$, $\theta=s^2/\bar x$.

# %%
def menos_logver(param_log, x):
    k, theta = np.exp(param_log)
    n = len(x)
    logver = -n * gammaln(k) - n * k * np.log(theta) + (k - 1) * np.sum(np.log(x)) - np.sum(x) / theta
    return -logver


k_mom, theta_mom = mu_c**2 / var_c, var_c / mu_c                  # método de momentos
res = optimize.minimize(menos_logver, x0=np.log([k_mom, theta_mom]), args=(g_nc,), method="Nelder-Mead")
k_emv, theta_emv = np.exp(res.x)
print(f"EMV:      k = {k_emv:.3f}, θ = {theta_emv:.3f}")
print(f"Momentos: k = {k_mom:.3f}, θ = {theta_mom:.3f}")
print(f"KS contra Gamma EMV: D = {stats.kstest(g_nc, 'gamma', args=(k_emv, 0, theta_emv)).statistic:.4f}, "
      f"p = {stats.kstest(g_nc, 'gamma', args=(k_emv, 0, theta_emv)).pvalue:.3g}")
print("Media de G (no caché) por clase de tamaño:",
      nc_est.groupby(pd.cut(nc_est["tam_kb"], [0, T1_KB, T2_KB, np.inf], right=False,
                             labels=["pequeña", "mediana", "grande"]), observed=True)["gpu_ms"].mean().round(1).to_dict())

# %%
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.linspace(0, np.quantile(g_nc, 0.995), 300)
ejes[0].step(np.sort(g_todas), np.arange(1, len(g_todas) + 1) / len(g_todas), label="FDA empírica")
ejes[0].plot(x, p_c + (1 - p_c) * stats.gamma.cdf(x, k_emv, scale=theta_emv), "r--", label="modelo mixto")
ejes[0].set(title="C1: FDA de G (salto en 0 = caché)", xlabel="G (ms)", ylabel="F(x)", xlim=(-5, x.max()))
ejes[1].hist(g_nc, bins=80, density=True, alpha=0.5, label="datos (no caché)")
ejes[1].plot(x, stats.gamma.pdf(x, k_emv, scale=theta_emv), label="Gamma EMV")
ejes[1].plot(x, stats.gamma.pdf(x, k_mom, scale=theta_mom), "--", label="Gamma momentos")
ejes[1].set(title="C2: densidad", xlabel="G (ms)", ylabel="densidad (1/ms)")
cuantiles = stats.gamma.ppf((np.arange(len(g_nc)) + 0.5) / len(g_nc), k_emv, scale=theta_emv)
ejes[2].plot(cuantiles, np.sort(g_nc), ".", ms=2, label="datos")
ejes[2].plot([0, g_nc.max()], [0, g_nc.max()], "k--", label="y = x")
ejes[2].set(title="C2: QQ-plot Gamma EMV", xlabel="cuantil teórico (ms)", ylabel="cuantil empírico (ms)")
for ax in ejes:
    ax.legend(fontsize=8)
guardar_figura(fig, 2, "Distribución del tiempo de GPU",
               "G tiene un salto en 0 (caché) y una parte continua asimétrica; el QQ-plot muestra si la Gamma ajusta, "
               "sobre todo en la cola.")

# %% [markdown]
# **Discusión:** el tiempo de GPU depende del tamaño de la imagen (medias muy distintas por clase). El modelo
# marginal mezcla tamaños distintos, por eso puede fallar en la cola. En el simulador (Parte G) se usa G
# condicionado al tamaño (Parte D3).
#
# ## C3 — ¿`red_ms` es normal?
# Criterios: (1) asimetría y curtosis (la normal tiene 0 y 0); (2) prueba de D'Agostino; (3) QQ-plot y
# comparación de probabilidades de cola.

# %%
red = estable["red_ms"].to_numpy()
media_red, sd_red = red.mean(), red.std()
p_normal = stats.normaltest(red).pvalue
print(f"asimetría = {stats.skew(red):.3f}, curtosis = {stats.kurtosis(red):.3f}, D'Agostino p = {p_normal:.3g}")
print("Conclusión:", "no se rechaza la normalidad" if p_normal > ALPHA else "se rechaza la normalidad",
      "(comparar además las probabilidades de cola de la tabla siguiente)")

xs = [round(media_red + c * sd_red) for c in (2, 3, 3.5)]
display(pd.DataFrame({"x (ms)": xs,
                      "P(red > x) empírica": [np.mean(red > x) for x in xs],
                      "P(red > x) normal": [stats.norm.sf(x, media_red, sd_red) for x in xs]}))

boot = [np.quantile(rng.choice(red, len(red)), 0.999) for _ in range(1000)]
mostrar_ic("Percentil 99,9 % de red (ms)", np.quantile(red, 0.999), *np.quantile(boot, [ALPHA / 2, 1 - ALPHA / 2]),
           "bootstrap, 1000 réplicas")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
xr = np.linspace(red.min(), red.max(), 300)
ejes[0].hist(red, bins=80, density=True, alpha=0.5, label="datos")
ejes[0].plot(xr, stats.norm.pdf(xr, media_red, sd_red), label="normal")
ejes[0].set(xlabel="red_ms (ms)", ylabel="densidad (1/ms)", title="Histograma")
ejes[0].legend()
stats.probplot(red, dist="norm", plot=ejes[1])
ejes[1].get_lines()[0].set_label("datos")
ejes[1].get_lines()[1].set_label("recta si fuera normal")
ejes[1].legend()
ejes[1].set(title="QQ-plot normal", xlabel="cuantil normal estándar", ylabel="red_ms (ms)")
guardar_figura(fig, 3, "¿Es normal el tiempo de red?",
               "Los puntos siguen la recta, también en los extremos: el tiempo de red es aproximadamente normal."
               if p_normal > ALPHA else
               "El QQ-plot se curva en los extremos: la normal no describe la cola del tiempo de red.")

# %% [markdown]
# ## C4 — Tiempo total de N solicitudes no cacheadas
# $T_N=\sum_{i=1}^N G_i$. Dos aproximaciones:
# 1. **TLC:** $T_N\approx\mathcal N(N\mu,N\sigma^2)$ ⇒ $P(T_N>N(\hat m+15))\approx1-\Phi(15\sqrt N/\sigma)$.
# 2. **Gamma exacta:** si $G_i\sim\text{Gamma}(k,\theta)$, la suma es $\text{Gamma}(Nk,\theta)$.
#
# Se comparan con una simulación (sumas de N valores tomados al azar de los datos).

# %%
m_hat, s_hat = g_nc.mean(), g_nc.std()
umbral = N_CONSEC * (m_hat + 15)
p_tlc = stats.norm.sf(15 * np.sqrt(N_CONSEC) / s_hat)
p_gamma = stats.gamma.sf(umbral, N_CONSEC * k_emv, scale=theta_emv)
sumas = rng.choice(g_nc, size=(100000, N_CONSEC)).sum(axis=1)
p_sim = np.mean(sumas > umbral)
print(f"N = {N_CONSEC}, umbral = {umbral:.1f} ms")
print(f"TLC = {p_tlc:.4f} | Gamma exacta = {p_gamma:.4f} | simulación = {p_sim:.4f}")
print("Más cercana a la simulación:", "TLC" if abs(p_tlc - p_sim) < abs(p_gamma - p_sim) else "Gamma exacta")

# %% [markdown]
# **¿Cuál es preferible?** La que más se acerque a la simulación. El TLC ignora la asimetría de G (con N pequeño la
# suma sigue siendo asimétrica) y la Gamma exacta solo es correcta si G realmente es Gamma (ver el QQ-plot de C2).

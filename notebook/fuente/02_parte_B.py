# %% [markdown]
# # Parte B — Llegadas, fallos y reintentos (régimen estable)
#
# ## B1 — Proceso de llegadas
# **Modelo:** proceso de Poisson de tasa λ (muchos clientes independientes). Implica: tiempos entre llegadas
# Exp(λ) y conteos por segundo Poisson(λ).
#
# **Estimación:** $\hat\lambda=n/T$. **IC exacto** (relación Poisson–χ²):
# $\left[\chi^2_{\alpha/2}(2n)/2T,\ \chi^2_{1-\alpha/2}(2n+2)/2T\right]$.
#
# **Validación (dos vías):** (1) KS de los tiempos entre llegadas contra Exp(λ̂); (2) χ² de bondad de ajuste de los
# conteos por segundo contra Poisson(λ̂).

# %%
n_lleg = len(t_est)
lam = n_lleg / T_EST
mostrar_ic("λ (sol/s)", lam, stats.chi2.ppf(ALPHA / 2, 2 * n_lleg) / (2 * T_EST),
           stats.chi2.ppf(1 - ALPHA / 2, 2 * n_lleg + 2) / (2 * T_EST), "exacto, χ²")

# Vía 1: tiempos entre llegadas vs exponencial
entre = np.diff(np.sort(t_est))
print(f"Vía 1 — KS vs Exp(λ̂): p = {stats.kstest(entre, 'expon', args=(0, 1 / lam)).pvalue:.3f} "
      f"(media {entre.mean()*1000:.1f} ms, CV = {entre.std() / entre.mean():.3f}; exponencial: CV = 1)")

# Vía 2: conteos por segundo vs Poisson (se agrupan las colas para que cada esperado sea >= 5)
conteos_1s = np.histogram(t_est, np.arange(INI_EST, FIN_EST - 1 + 1e-9, 1.0))[0]
k_lo, k_hi = int(stats.poisson.ppf(0.01, lam)), int(stats.poisson.ppf(0.99, lam))
obs = [np.sum(conteos_1s <= k_lo)] + [np.sum(conteos_1s == k) for k in range(k_lo + 1, k_hi)] + [np.sum(conteos_1s >= k_hi)]
prob = [stats.poisson.cdf(k_lo, lam)] + [stats.poisson.pmf(k, lam) for k in range(k_lo + 1, k_hi)] + [stats.poisson.sf(k_hi - 1, lam)]
esp = np.array(prob) * len(conteos_1s)
chi2_b1 = np.sum((np.array(obs) - esp)**2 / esp)
print(f"Vía 2 — χ² conteos por segundo vs Poisson: χ² = {chi2_b1:.1f}, gl = {len(obs) - 2}, "
      f"p = {stats.chi2.sf(chi2_b1, len(obs) - 2):.3f}; var/media = {conteos_1s.var() / conteos_1s.mean():.3f}")

# %% [markdown]
# ## B2 — Bloques de 100 solicitudes
# Si cada solicitud requiere más de un intento con probabilidad p, el número por bloque es $X\sim\text{Bin}(100,p)$:
# media $100p$ y varianza $100p(1-p)$. Se estima p en el régimen estable y se contrasta con los bloques de cada
# régimen. Índice de dispersión $D=s^2/[100\hat p(1-\hat p)]$: bajo el modelo, $(m-1)D\sim\chi^2_{m-1}$.

# %%
p_reint = (estable["intentos"] > 1).mean()
bloques = limpio.reset_index(drop=True)
bloques["bloque"] = bloques.index // 100
por_bloque = bloques.groupby("bloque").agg(x=("intentos", lambda v: (v > 1).sum()), n=("intentos", "size"),
                                           regimen=("regimen", lambda v: v.iloc[0] if v.nunique() == 1 else "mixto"))
por_bloque = por_bloque[(por_bloque["n"] == 100) & (por_bloque["regimen"] != "mixto")]

filas_b2 = []
for reg, g in por_bloque.groupby("regimen"):
    m, D = len(g), g["x"].var() / (100 * p_reint * (1 - p_reint))
    filas_b2.append({"régimen": reg, "bloques": m, "media obs": g["x"].mean(), "media modelo": 100 * p_reint,
                     "var obs": g["x"].var(), "var modelo": 100 * p_reint * (1 - p_reint), "D": D,
                     "p (dispersión)": 2 * min(stats.chi2.cdf((m - 1) * D, m - 1), stats.chi2.sf((m - 1) * D, m - 1))})
display(pd.DataFrame(filas_b2))
mostrar_ic("p (P(intentos > 1), estable)", *ic_wilson((estable["intentos"] > 1).sum(), len(estable)), "Wilson")
# Evidencia de la discrepancia: p estimada por separado en cada régimen
por_reg = limpio.groupby("regimen")["intentos"].agg(x=lambda v: (v > 1).sum(), n="size")
print(f"P(intentos > 1) por régimen, IC {NIVEL:.0%} de Wilson:")
display(pd.DataFrame([ic_wilson(f["x"], f["n"]) for _, f in por_reg.iterrows()],
                     index=por_reg.index, columns=["p", "inf", "sup"]))

# %% [markdown]
# **Lectura:** en los regímenes normales la media y la varianza coinciden con la binomial. En el régimen anómalo la
# media observada se aleja porque **p cambió** (sus IC no se solapan con el del régimen estable, ver A2): el modelo con
# el p del régimen estable no le sirve.
#
# ## B3 — Modelo de `intentos` y su estimador de máxima verosimilitud
# Cada intento falla con probabilidad q, independientemente; el SDK abandona tras 3 fallos (geométrica **censurada
# en 3**). Cada solicitud aporta $q^{f}(1-q)^{e}$ con $f$ = intentos fallidos y $e$ = 1 si tuvo éxito:
#
# | caso | probabilidad |
# |---|---|
# | 1 intento | $1-q$ |
# | 2 intentos | $q(1-q)$ |
# | 3 intentos con éxito | $q^2(1-q)$ |
# | 3 intentos, abandono | $q^3$ |
#
# $\ell(q)=F\log q+E\log(1-q)$ con $F=\sum(\text{intentos}-\text{exito})$, $E=\sum\text{exito}$.
# $\ell'(q)=0\Rightarrow\boxed{\hat q=F/(F+E)}$ = fallos totales / intentos totales.
# $\text{EE}(\hat q)=\sqrt{\hat q(1-\hat q)/\sum\text{intentos}}$.
#
# **Estimador inmediato:** si se ignora la censura, $E[I]=1/(1-q)$ ⇒ $\tilde q=1-1/\bar I$. Pero con censura
# $E[I]=1+q+q^2$, así que $\tilde q$ está sesgado. Se comprueba por simulación con $q=\hat q$.

# %%
F = (estable["intentos"] - estable["exito"]).sum()
I_total = estable["intentos"].sum()
q_hat = F / I_total
ee_q = np.sqrt(q_hat * (1 - q_hat) / I_total)
mostrar_ic("q (EMV)", q_hat, q_hat - Z * ee_q, q_hat + Z * ee_q, "Wald")

# Simulación: 5000 muestras del mismo tamaño generadas con q = q_hat
n = len(estable)
probs = [1 - q_hat, q_hat * (1 - q_hat), q_hat**2 * (1 - q_hat), q_hat**3]
muestras = rng.multinomial(n, probs, size=5000)                   # conteos de los 4 casos
fallos = muestras[:, 1] + 2 * muestras[:, 2] + 3 * muestras[:, 3]
intentos = muestras[:, 0] + 2 * muestras[:, 1] + 3 * (muestras[:, 2] + muestras[:, 3])
q_emv, q_ingenuo = fallos / intentos, 1 - n / intentos             # 1 - 1/promedio = 1 - n/total
display(pd.DataFrame({"sesgo": [q_emv.mean() - q_hat, q_ingenuo.mean() - q_hat],
                      "desv. est.": [q_emv.std(), q_ingenuo.std()],
                      "RECM": [np.sqrt(np.mean((q_emv - q_hat)**2)), np.sqrt(np.mean((q_ingenuo - q_hat)**2))]},
                     index=["EMV", "ingenuo 1 − 1/Ī"]))

# %% [markdown]
# **Lectura:** el ingenuo tiene un sesgo que no desaparece al crecer n (su límite es $1-1/(1+q+q^2)\neq q$): no es
# adecuado. El EMV es prácticamente insesgado.
#
# ## B4 — Al menos k solicitudes con reintento en 10 s
# Las llegadas son Poisson(λ) y cada una, de forma independiente, requiere más de un intento con probabilidad p.
# Por la **propiedad de adelgazamiento** del proceso de Poisson, las que reintentan forman un Poisson(λp); por
# **estacionariedad**, en 10 s su número es $X\sim\text{Poisson}(\mu)$, $\mu=10\lambda p$.

# %%
mu_hat = 10 * lam * p_reint
k = int(np.ceil(mu_hat)) + R_DESPL
p_analitica = stats.poisson.sf(k - 1, mu_hat)                     # P(X >= k) = 1 - P(X <= k-1)
llegan = rng.poisson(10 * lam, 100000)                            # llegadas en 100 000 ventanas de 10 s
x_sim = rng.binomial(llegan, p_reint)                             # cuántas de ellas reintentan
print(f"μ̂ = {mu_hat:.3f}, k = ⌈μ̂⌉ + r = {k}")
print(f"P(X ≥ {k}): analítica = {p_analitica:.4f}")
mostrar_ic(f"P(X ≥ {k}) Montecarlo", *ic_wilson(np.sum(x_sim >= k), len(x_sim)), "Wilson, 100 000 réplicas")

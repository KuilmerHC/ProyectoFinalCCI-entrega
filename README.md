# Proyecto Final — Ciencia Computacional Intermedia (Tercer corte)

**Confiabilidad y planeación de capacidad de un servicio de inferencia de IA**

Integrantes: Kuilmer Santiago Hernández Cruz · Valerie Gutiérrez Zambrano · Juan Tinoco Penagos · Juan José Berrío Español

## Entregables

| Entregable | Archivo |
|---|---|
| E1. Notebook | `notebook/proyecto_final.ipynb` |
| E2. Informe | `informe/` |
| E3. Deducciones manuscritas (B3, C1, D3, F5) | Anexo D del informe |
| E4. Memorando ejecutivo | `memo/` |
| E5. Bitácora de decisiones | Anexo B del informe |
| E6. Historial de versiones | historial de este repositorio |
| E7. Declaración de uso de IA | Anexo C del informe |

## Cómo ejecutar el notebook

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name proyecto-cci --display-name "Python (ProyectoFinalCCI)"
jupyter notebook notebook/proyecto_final.ipynb   # luego: Kernel → Restart & Run All
```

- Los datos del grupo están en `data/`. La única fuente de aleatoriedad es `np.random.default_rng(S)`.
- El notebook se arma a partir de `notebook/fuente/` (un archivo por parte del proyecto) con
  `python scripts/construir_notebook.py --ejecutar`, que además lo ejecuta completo.
- Las figuras generadas se guardan en `figuras/`.

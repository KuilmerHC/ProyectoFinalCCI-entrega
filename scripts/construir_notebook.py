"""Une notebook/fuente/*.py (formato percent de jupytext) en notebook/proyecto_final.ipynb.

Uso:
    .venv/bin/python scripts/construir_notebook.py            # solo construir
    .venv/bin/python scripts/construir_notebook.py --ejecutar # construir y ejecutar (Run All)
"""
import subprocess
import sys
from pathlib import Path

import jupytext

RAIZ = Path(__file__).resolve().parent.parent
FUENTE = RAIZ / "notebook" / "fuente"
SALIDA = RAIZ / "notebook" / "proyecto_final.ipynb"

partes = sorted(FUENTE.glob("*.py"))
texto = "\n\n".join(p.read_text(encoding="utf-8") for p in partes)
nb = jupytext.reads(texto, fmt="py:percent")
nb.metadata["kernelspec"] = {"name": "proyecto-cci", "display_name": "Python (ProyectoFinalCCI)",
                             "language": "python"}
nb.metadata.pop("jupytext", None)
jupytext.write(nb, SALIDA, fmt="ipynb")
print(f"Construido {SALIDA.name} con {len(nb.cells)} celdas a partir de {len(partes)} archivos")

if "--ejecutar" in sys.argv:
    cmd = [str(RAIZ / ".venv" / "bin" / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
           "--inplace", "--ExecutePreprocessor.timeout=-1", str(SALIDA)]
    subprocess.run(cmd, check=True, cwd=SALIDA.parent)
    print("Ejecutado sin errores")

"""El candidato del barrido en `modelos/reconocedor/` (TT-22; contracts/candidato.md, versión 2).

`modelo.pt` (contracts/modelo.md M-2 de TT-14) es el **modelo final**, entrenado con los dos pliegues
y las épocas elegidas, y `candidato.json` su ficha, con las cifras de la selección por rotación que lo
eligieron. Listos para versionarlos con DVC. No es registrar el modelo como aceptado (constitución IV):
eso lo hace la aceptación si alcanza ≥85 % Top-1 sobre la prueba por grabación.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from .errores import ErrorReconocedor
from .registro import CorridaRegistrada
from .seleccion import Agregado

DESTINO = Path("modelos") / "reconocedor"
PUNTERO = Path("modelos") / "reconocedor.dvc"
HUELLAS = ("corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella")
PROTOCOLO = "rotacion-por-grabacion"
VERSIONAR = (
    "Versiónalo ahora:\n"
    "  uv run dvc add modelos/reconocedor && uv run dvc push\n"
    "  git add modelos/reconocedor.dvc modelos/.gitignore"
)


def escribir_candidato(
    raiz: Path, barrido: str, agregado: Agregado, final: CorridaRegistrada, modelo_origen: Path, clips: int,
    duracion_s: float,
) -> Path:
    destino = Path(raiz) / DESTINO
    ficha_previa = destino / "candidato.json"
    if ficha_previa.is_file():
        anterior = json.loads(ficha_previa.read_text(encoding="utf-8")).get("barrido")
        if anterior != barrido and not (Path(raiz) / PUNTERO).is_file():
            raise ErrorReconocedor(
                f"Ya hay un candidato del barrido {anterior} sin versionar en {DESTINO}. Versiónalo o descártalo "
                "antes de escribir otro."
            )
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(modelo_origen, destino / "modelo.pt")
    ficha = {
        "version_candidato": 2,
        "protocolo": PROTOCOLO,
        "barrido": barrido,
        "configuracion": agregado.configuracion,
        "ronda": agregado.ronda,
        "semilla": agregado.semilla_final,
        "epocas_elegidas": agregado.epocas_elegidas,
        "seleccion": {
            **agregado.medias,
            "diferencia_rotaciones": agregado.diferencia_rotaciones,
            "por_rotacion": agregado.por_rotacion,
            "corridas": [c.corrida_id for c in agregado.corridas],
        },
        "final": {"run_id": final.id, "clips": clips, "duracion_s": duracion_s},
        "parametros": agregado.parametros,
        "huellas": {clave: final.etiquetas[clave] for clave in HUELLAS},
        "commit": final.etiquetas.get("codigo.commit"),
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "estado": "pendiente",
    }
    (destino / "candidato.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino

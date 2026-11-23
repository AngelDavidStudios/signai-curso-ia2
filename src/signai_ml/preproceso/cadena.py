"""El orquestador: clip → detección → keypoints → ventana (FR-002, FR-015, FR-022, FR-023).

Recibe el extractor por inyección (el binario real o un doble en las pruebas). Por clip:

    comprobar md5 → extractor (detección en caché) → leer y validar → seleccionar → guardar keypoints

Salta los clips con resultado vigente, así que reanudar es volver a lanzar lo mismo. Un clip que
falla se señala y no detiene a los demás. Las ventanas se calculan siempre desde los keypoints
guardados, de modo que renormalizar no exige volver a extraer.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .almacen import Almacen, Versiones
from .clips import ClipEtiquetado, comprobar_md5
from .decimacion import Decimacion
from .deteccion import leer_deteccion
from .errores import ErrorExtraccion, ErrorPreproceso
from .normalizacion import Normalizador, VentanaNormalizada
from .seleccion import seleccionar_clip


class Extractor(Protocol):
    def extraer(self, video: Path, destino: Path, cada: int, volcar: Path | None = None) -> None: ...


@dataclass(frozen=True)
class ResultadoClip:
    clip: ClipEtiquetado
    keypoints: np.ndarray
    ventana: VentanaNormalizada
    fallidos: int


@dataclass
class Resumen:
    procesados: list[str] = field(default_factory=list)
    saltados: list[str] = field(default_factory=list)
    errores: dict[str, str] = field(default_factory=dict)
    resultados: dict[str, ResultadoClip] = field(default_factory=dict)
    fotogramas_extraidos: int = 0


class Cadena:
    def __init__(
        self, extractor: Extractor, almacen: Almacen, cache: Path, versiones: Versiones,
        normalizador: Normalizador | None = None, decimacion: Decimacion | None = None,
    ) -> None:
        self.extractor = extractor
        self.almacen = almacen
        self.cache = Path(cache)
        self.versiones = versiones
        self.normalizador = normalizador or Normalizador()
        self.decimacion = decimacion or Decimacion()

    def _extraer(self, clip: ClipEtiquetado) -> tuple[np.ndarray, int, bool]:
        """Keypoints, fotogramas fallidos y si se llamó a Vision (False si salió de la caché)."""
        comprobar_md5(clip)
        destino = self.cache / clip.clase / f"{clip.clip_id}.json.gz"
        try:
            deteccion = leer_deteccion(destino, clip, self.decimacion) if destino.is_file() else None
        except (ErrorPreproceso, OSError, ValueError, KeyError):
            deteccion = None
        con_vision = deteccion is None or deteccion.macos != self.versiones.macos
        if con_vision:
            self.extractor.extraer(clip.ruta, destino, cada=self.decimacion.cada)
            deteccion = leer_deteccion(destino, clip, self.decimacion)
        if deteccion.macos != self.versiones.macos:
            raise ErrorExtraccion(
                clip.clip_id, f"extraído con macOS {deteccion.macos} y se esperaba {self.versiones.macos}"
            )
        keypoints = seleccionar_clip(deteccion)
        return keypoints, sum(f.fallido for f in deteccion.fotogramas), con_vision

    def _uno(self, clip: ClipEtiquetado) -> tuple[str, ResultadoClip | None, str | None, bool, bool]:
        try:
            con_vision = False
            if self.almacen.vigente(clip, self.versiones):
                keypoints = self.almacen.leer_keypoints(clip)
                fallidos = int(self.almacen.leer_meta(clip).get("fallidos", 0))
                saltado = True
            else:
                keypoints, fallidos, con_vision = self._extraer(clip)
                self.almacen.guardar_keypoints(clip, keypoints, self.versiones, fallidos=fallidos)
                saltado = False
            ventana = self.normalizador.normalizar(keypoints)
            return clip.clip_id, ResultadoClip(clip, keypoints, ventana, fallidos), None, saltado, con_vision
        except ErrorPreproceso as fallo:
            return clip.clip_id, None, str(fallo), False, False

    def procesar(self, clips: Sequence[ClipEtiquetado], procesos: int = 1) -> Resumen:
        resumen = Resumen()
        with ThreadPoolExecutor(max_workers=max(1, procesos)) as pool:
            for clip_id, resultado, error, saltado, con_vision in pool.map(self._uno, clips):
                if error is not None:
                    resumen.errores[clip_id] = error
                    continue
                assert resultado is not None
                resumen.resultados[clip_id] = resultado
                if saltado:
                    resumen.saltados.append(clip_id)
                else:
                    resumen.procesados.append(clip_id)
                if con_vision:
                    resumen.fotogramas_extraidos += resultado.keypoints.shape[0]
        return resumen

    def ensamblar(self, clips: Sequence[ClipEtiquetado], extra: Mapping) -> Path:
        """`ventanas.npy` en el orden de `clips`; solo si todos tienen resultado vigente."""
        pendientes = [c.clip_id for c in clips if not self.almacen.vigente(c, self.versiones)]
        if pendientes:
            muestra = ", ".join(pendientes[:5]) + (" …" if len(pendientes) > 5 else "")
            raise ErrorPreproceso(
                f"{len(pendientes)} clip(s) sin resultado vigente ({muestra}); vuelve a lanzar `procesar`."
            )
        ventanas = np.stack([self.normalizador.normalizar(self.almacen.leer_keypoints(c)).valores for c in clips])
        return self.almacen.escribir_ventanas(clips, ventanas, self.versiones, extra)

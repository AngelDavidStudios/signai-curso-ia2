"""Casos de uso de la copia de seguridad (contracts/cli.md).

La CLI solo traduce argumentos y salida; las reglas viven aquí y dependen de puertos (principio II).
Toda operación que copia, envía o comprueba deja su resultado en el registro, también si falla
(FR-008, FR-011).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import manifiesto, seguimiento
from .conjuntos import conjuntos, exclusiones, rutas_de_copia
from .dvc import EstadoDvc
from .envio import Envio, Envios
from .errores import ErrorCopia, ErrorUso, RechazoDominio
from .registro import Entrada, Registro, sanear
from .repositorio import Repositorio

VERIFICACION = "5%"
RETENCION = "30d"


class _VerificacionFallida(Exception):
    """Lleva el error de la verificación ya anotado, para no anotarlo dos veces."""

    def __init__(self, error: ErrorCopia) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True)
class EstadoCopia:
    """Lo que muestra `signai-copia estado` (data-model, «Estado de la copia»)."""

    ultima_actualizacion: str | None
    instantanea: str | None
    carpeta: str | None
    en_ipad: bool
    conjuntos: dict[str, str]
    corridas_en_curso: int
    dvc_sin_subir: bool

    @property
    def al_dia(self) -> bool:
        return (
            self.en_ipad
            and all(estado == "al-dia" for estado in self.conjuntos.values())
            and self.corridas_en_curso == 0
            and not self.dvc_sin_subir
        )


@dataclass(frozen=True)
class ResultadoActualizacion:
    instantanea: str
    conjuntos: tuple[str, ...]
    bytes_anadidos: int
    duracion_s: int
    envio: Envio


class ServicioCopia:
    def __init__(
        self,
        raiz: Path,
        *,
        registro: Registro,
        ahora: Callable[[], datetime],
        repositorio: Repositorio | None = None,
        estado_dvc: EstadoDvc | None = None,
        envios: Envios | None = None,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self._raiz = Path(raiz)
        self._registro = registro
        self._ahora = ahora
        self._repositorio = repositorio
        self._estado_dvc = estado_dvc
        self._envios = envios
        self._reloj = reloj

    # ── Actualizar y enviar ──────────────────────────────────────────────────────────────────

    def actualizar(self, ronda: str | None = None) -> ResultadoActualizacion:
        repositorio, estado_dvc, envios = self._dependencias()
        inicio = self._reloj()
        try:
            if seguimiento.corridas_en_curso(self._raiz / "mlflow" / "mlflow.db"):
                raise RechazoDominio("hay corridas en curso: actualiza la copia cuando terminen")
            if estado_dvc.sin_subir():
                raise RechazoDominio("hay objetos de DVC sin subir: uv run dvc push")
            etiquetas = self._etiquetas(repositorio, ronda)
            actual = manifiesto.inventario_ligero(self._raiz, [c.ruta for c in conjuntos()])
            anterior = manifiesto.leer(self._raiz)
            cambiados = (
                tuple(manifiesto.cambiados(anterior[1], actual))
                if anterior
                else tuple(c.nombre for c in conjuntos())
            )
            copia = repositorio.copiar(
                rutas_de_copia(self._raiz), base=self._raiz, etiquetas=etiquetas, excluir=exclusiones(self._raiz)
            )
            repositorio.olvidar(dentro_de=RETENCION, ultimas=1)
            self._verificar(repositorio, inicio)
            momento = self._ahora()
            manifiesto.guardar(self._raiz, momento.isoformat(), actual)
            envio = envios.preparar(copia.instantanea, momento)
        except _VerificacionFallida as fallo:
            raise fallo.error from None
        except ErrorCopia as fallo:
            self._anotar("actualizacion", inicio, ronda=ronda, motivo=str(fallo))
            raise
        duracion = self._segundos(inicio)
        self._anotar(
            "actualizacion",
            inicio,
            instantanea=copia.instantanea,
            ronda=ronda,
            conjuntos=cambiados,
            bytes_anadidos=copia.bytes_anadidos,
            detalle={"carpeta": envio.carpeta, "archivos": envio.archivos, "elementos": envio.elementos, "bytes": envio.bytes},
        )
        return ResultadoActualizacion(copia.instantanea, cambiados, copia.bytes_anadidos, duracion, envio)

    def confirmar(self, carpeta: str, recuento: int) -> str | None:
        """Registra que la carpeta llegó entera al iPad. Devuelve la carpeta confirmada antes, que
        ya se puede borrar del iPad."""
        envios = self._envios_requeridos()
        inicio = self._reloj()
        preparada = self._ultima("actualizacion", carpeta=carpeta)
        try:
            archivos, total = envios.confirmar(carpeta, recuento)
        except ErrorCopia as fallo:
            self._anotar("envio", inicio, detalle={"carpeta": sanear(carpeta), "recuento": recuento}, motivo=str(fallo))
            raise
        anterior = self._ultima("envio")
        self._anotar(
            "envio",
            inicio,
            instantanea=preparada.instantanea if preparada else None,
            detalle={"carpeta": carpeta, "archivos": archivos, "bytes": total},
        )
        return anterior.detalle.get("carpeta") if anterior else None

    # ── Estado, sin clave (research R-10) ────────────────────────────────────────────────────

    def estado(self) -> EstadoCopia:
        """No abre el repositorio: compara el origen con el inventario de la última actualización."""
        estado_dvc = self._estado_dvc_requerido()
        entradas = self._registro.leer()
        actualizaciones = [
            (posicion, entrada)
            for posicion, entrada in enumerate(entradas)
            if entrada.tipo == "actualizacion" and entrada.resultado == "correcto"
        ]
        posicion, ultima = actualizaciones[-1] if actualizaciones else (-1, None)
        carpeta = ultima.detalle.get("carpeta") if ultima else None
        en_ipad = carpeta is not None and any(
            entrada.tipo == "envio" and entrada.resultado == "correcto" and entrada.detalle.get("carpeta") == carpeta
            for entrada in entradas[posicion + 1 :]
        )
        anterior = manifiesto.leer(self._raiz)
        nombres = [conjunto.nombre for conjunto in conjuntos()]
        if anterior is None:
            pendientes = set(nombres)
        else:
            actual = manifiesto.inventario_ligero(self._raiz, [conjunto.ruta for conjunto in conjuntos()])
            pendientes = set(manifiesto.cambiados(anterior[1], actual))
        return EstadoCopia(
            ultima_actualizacion=ultima.fecha if ultima else None,
            instantanea=ultima.instantanea if ultima else None,
            carpeta=carpeta,
            en_ipad=en_ipad,
            conjuntos={nombre: "pendiente" if nombre in pendientes else "al-dia" for nombre in nombres},
            corridas_en_curso=seguimiento.corridas_en_curso(self._raiz / "mlflow" / "mlflow.db"),
            dvc_sin_subir=estado_dvc.sin_subir(),
        )

    # ── Restaurar y comprobar (research R-09) ────────────────────────────────────────────────

    def restaurar(
        self, repositorio: Repositorio, destino: Path, *, instantanea: str = "latest", ruta: str | None = None
    ) -> None:
        """Lee entera la copia (devuelta desde el iPad) y la restaura en `destino`."""
        destino = self._destino_permitido(destino)
        if destino.exists() and any(destino.iterdir()):
            raise RechazoDominio(f"el destino no está vacío: {destino}")
        inicio = self._reloj()
        try:
            repositorio.verificar(completo=True)
        except ErrorCopia as fallo:
            self._anotar("verificacion", inicio, detalle={"completa": True}, motivo=str(fallo))
            raise
        self._anotar("verificacion", inicio, detalle={"completa": True})
        try:
            repositorio.restaurar(instantanea, destino, ruta=ruta)
        except ErrorCopia as fallo:
            self._anotar("restauracion", inicio, detalle={"fase": "restaurar"}, motivo=str(fallo))
            raise

    def comprobar(self, restaurado: Path) -> str:
        """Compara lo restaurado con el origen (SC-003) y devuelve un resumen sin datos de señantes."""
        estado_dvc = self._estado_dvc_requerido()
        inicio = self._reloj()
        restaurado = Path(restaurado)
        rutas = [conjunto.ruta for conjunto in conjuntos()]
        base_origen, base_restaurada = self._raiz / "mlflow" / "mlflow.db", restaurado / "mlflow" / "mlflow.db"
        try:
            comparacion = manifiesto.comparar(
                manifiesto.manifiesto(self._raiz, rutas), manifiesto.manifiesto(restaurado, rutas)
            )
            ausentes = estado_dvc.ausentes_en(restaurado / "dvc-store")
            experimentos_origen, corridas_origen = seguimiento.recuento(base_origen)
            experimentos_restaurados, corridas_restauradas = (
                seguimiento.recuento(base_restaurada) if base_restaurada.is_file() else (0, 0)
            )
            integridad = seguimiento.integridad(base_restaurada) if base_restaurada.is_file() else "sin base"
        except ErrorCopia as fallo:
            # Un error al medir (DVC, una base ilegible) también es un resultado de la comprobación.
            self._anotar("restauracion", inicio, detalle={"fase": "comprobar"}, motivo=str(fallo))
            raise
        detalle = {
            "archivos": len(manifiesto.manifiesto(restaurado, rutas)),
            "faltan": len(comparacion.faltan),
            "sobran": len(comparacion.sobran),
            "distintos": len(comparacion.distintos),
            "dvc_ausentes": ausentes,
            "experimentos_origen": experimentos_origen,
            "experimentos_restaurados": experimentos_restaurados,
            "corridas_origen": corridas_origen,
            "corridas_restauradas": corridas_restauradas,
            "integridad": integridad,
        }
        problemas = [
            texto
            for hay, texto in (
                (not comparacion.identico,
                 f"faltan {detalle['faltan']}, sobran {detalle['sobran']} y difieren {detalle['distintos']} archivos"),
                (ausentes > 0, f"DVC echa en falta {ausentes} datos de los punteros"),
                ((experimentos_origen, corridas_origen) != (experimentos_restaurados, corridas_restauradas),
                 f"corridas: {corridas_origen} en el origen y {corridas_restauradas} restauradas"),
                (integridad != "ok", f"integrity_check: {integridad}"),
            )
            if hay
        ]
        motivo = "; ".join(problemas) or None
        self._anotar("restauracion", inicio, detalle=detalle, motivo=motivo)
        if motivo:
            raise RechazoDominio(f"la restauración no es idéntica: {motivo}")
        return self._resumen(detalle)

    def _resumen(self, detalle: dict[str, Any]) -> str:
        filas = [
            ("Archivos comparados (SHA-256)", detalle["archivos"]),
            ("Faltan", detalle["faltan"]),
            ("Sobran", detalle["sobran"]),
            ("Distintos", detalle["distintos"]),
            ("Datos de DVC ausentes", detalle["dvc_ausentes"]),
            ("Experimentos (origen / restaurados)", f"{detalle['experimentos_origen']} / {detalle['experimentos_restaurados']}"),
            ("Corridas (origen / restauradas)", f"{detalle['corridas_origen']} / {detalle['corridas_restauradas']}"),
            ("`PRAGMA integrity_check`", detalle["integridad"]),
        ]
        cuerpo = "\n".join(f"| {nombre} | {valor} |" for nombre, valor in filas)
        return f"Comprobación del {self._ahora():%Y-%m-%d %H:%M}\n\n| Comprobación | Resultado |\n|---|---|\n{cuerpo}\n"

    def _destino_permitido(self, destino: Path) -> Path:
        """Fuera de la raíz de datos, o dentro solo en `copia/restauracion-*` (excluido de la copia)."""
        destino = Path(destino).expanduser().resolve()
        raiz = self._raiz.resolve()
        if destino != raiz and raiz not in destino.parents:
            return destino
        partes = destino.relative_to(raiz).parts
        if len(partes) >= 2 and partes[0] == "copia" and partes[1].startswith("restauracion-"):
            return destino
        raise ErrorUso("destino no permitido: dentro de la raíz de datos solo se restaura en copia/restauracion-*")

    # ── Grabaciones crudas ───────────────────────────────────────────────────────────────────

    def anotar_crudas(self, donde: str, *, fuera: bool, cifrada: bool, completa: bool) -> None:
        """Comprobación de la copia existente de las grabaciones crudas (FR-001a)."""
        if "/" in donde or "~" in donde:
            raise ErrorUso("--donde describe el destino sin rutas (por ejemplo «disco externo cifrado»)")
        faltas = [
            descripcion
            for cumple, descripcion in ((fuera, "fuera del equipo"), (cifrada, "cifrada"), (completa, "completa"))
            if not cumple
        ]
        motivo = (
            f"la copia de las grabaciones crudas no está {', '.join(faltas)}: pendiente de TT-21" if faltas else None
        )
        self._anotar(
            "crudas",
            self._reloj(),
            detalle={"donde": donde, "fuera_del_equipo": fuera, "cifrada": cifrada, "completa": completa},
            motivo=motivo,
        )
        if motivo:
            raise RechazoDominio(motivo)

    # ── Apoyo ────────────────────────────────────────────────────────────────────────────────

    def _verificar(self, repositorio: Repositorio, inicio: float) -> None:
        """Una verificación fallida se anota como tal, no como una actualización fallida."""
        try:
            repositorio.verificar(subconjunto=VERIFICACION)
        except ErrorCopia as error:
            self._anotar("verificacion", inicio, detalle={"subconjunto": VERIFICACION}, motivo=str(error))
            raise _VerificacionFallida(error) from error

    def _dependencias(self) -> tuple[Repositorio, EstadoDvc, Envios]:
        if self._repositorio is None or self._estado_dvc is None:
            raise ErrorUso("esta operación necesita el repositorio y el estado de DVC")
        return self._repositorio, self._estado_dvc, self._envios_requeridos()

    def _estado_dvc_requerido(self) -> EstadoDvc:
        if self._estado_dvc is None:
            raise ErrorUso("esta operación necesita el estado de DVC")
        return self._estado_dvc

    def _envios_requeridos(self) -> Envios:
        if self._envios is None:
            raise ErrorUso("esta operación necesita los envíos al iPad")
        return self._envios

    @staticmethod
    def _etiquetas(repositorio: Repositorio, ronda: str | None) -> tuple[str, ...]:
        # «--ronda siempre gana»; sin ronda, la primera instantánea se marca como inicial.
        if ronda:
            return ("signai", f"ronda={ronda}")
        return ("signai",) if repositorio.instantaneas() else ("signai", "inicial")

    def _segundos(self, inicio: float) -> int:
        return max(0, round(self._reloj() - inicio))

    def _anotar(self, tipo: str, inicio: float, *, motivo: str | None = None, **campos: Any) -> None:
        self._registro.anadir(
            Entrada(
                fecha=self._ahora().isoformat(),
                tipo=tipo,
                resultado="fallo" if motivo else "correcto",
                duracion_s=self._segundos(inicio),
                motivo=sanear(motivo) if motivo else None,
                **campos,
            )
        )

    def _ultima(self, tipo: str, **detalle: Any) -> Entrada | None:
        for entrada in reversed(self._registro.leer()):
            if (
                entrada.tipo == tipo
                and entrada.resultado == "correcto"
                and all(entrada.detalle.get(k) == v for k, v in detalle.items())
            ):
                return entrada
        return None

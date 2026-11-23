"""Registro de la aceptación (contracts/registro.md; research R-05, R-10, R-11).

Un puerto (`RegistroAceptacion`) y dos adaptadores:

- `RegistroAceptacionMlflow`, sobre el almacén SQLite local de TT-03 y TT-14;
- `RegistroAceptacionEnMemoria`, el de las pruebas de la CLI.

- G-1: una corrida `consulta-NN` del experimento `aceptacion` por cada consulta a la prueba.
- G-2: el modelo registrado `reconocedor-kposeformer`, con una versión por candidato aceptado y el
  alias `aceptado` en la última. La versión apunta a `runs:/<consulta>/modelo`, que es el mismo
  `modelo.pt` evaluado. Solo se registra una consulta terminada con `decision=aceptado`.
- G-3: una corrida `exportacion-NN` del experimento `exportacion` por cada intento de exportar, con
  su informe de paridad (`superada` o `fallida`). No es una consulta a la prueba.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .. import seguimiento
from . import ALIAS, EXPERIMENTO_ACEPTACION, EXPERIMENTO_EXPORTACION, MODELO_REGISTRADO
from .consultas import EstadoConsulta, EvaluacionRegistrada

TIPO = "evaluacion-aceptacion"
TIPO_EXPORTACION = "exportacion"
PARIDADES = ("superada", "fallida")
ESTADO_DE_MLFLOW = {
    "FINISHED": EstadoConsulta.TERMINADA,
    "FAILED": EstadoConsulta.FALLIDA,
    "KILLED": EstadoConsulta.INTERRUMPIDA,
}


def nombre_consulta(numero: int) -> str:
    return f"consulta-{numero:02d}"


def _etiquetas_consulta(numero: int, etiquetas: Mapping[str, str]) -> dict[str, str]:
    if not etiquetas.get("modelo.sha256"):
        raise ValueError("Una consulta necesita la etiqueta modelo.sha256: es lo que identifica al candidato.")
    return {**{k: str(v) for k, v in etiquetas.items()}, "tipo": TIPO, "consulta": str(numero)}


def nombre_exportacion(numero: int) -> str:
    return f"exportacion-{numero:02d}"


def _comprobar_paridad(paridad: str) -> None:
    if paridad not in PARIDADES:
        raise ValueError(f"Paridad desconocida: {paridad!r}; es {' o '.join(PARIDADES)}.")


def _ficheros(artefactos: Sequence[Path]) -> list[str]:
    """Las rutas que quedan en el registro: un directorio (el paquete) aporta todos sus ficheros."""
    rutas = []
    for artefacto in map(Path, artefactos):
        if artefacto.is_dir():
            rutas += sorted(f"{artefacto.name}/{f.relative_to(artefacto).as_posix()}" for f in artefacto.rglob("*") if f.is_file())
        else:
            rutas.append(artefacto.name)
    return rutas


@dataclass(frozen=True)
class ExportacionRegistrada:
    run_id: str
    nombre: str
    estado: EstadoConsulta
    paridad: str | None
    metricas: dict[str, float]
    etiquetas: dict[str, str]


@dataclass(frozen=True)
class VersionRegistrada:
    version: int
    run_id: str
    etiquetas: dict[str, str]


def _exigir_aceptada(evaluacion: EvaluacionRegistrada | None, id_consulta: str) -> EvaluacionRegistrada:
    if evaluacion is None or evaluacion.estado is not EstadoConsulta.TERMINADA or evaluacion.decision != "aceptado":
        raise ValueError(
            f"La consulta {id_consulta} no está terminada con decision=aceptado: no se registra (constitución IV)."
        )
    return evaluacion


class RegistroAceptacion(Protocol):
    def abrir_consulta(self, numero: int, etiquetas: Mapping[str, str]) -> str: ...
    def terminar_consulta(
        self, id_consulta: str, metricas: Mapping[str, float], decision: str, artefactos: Sequence[Path], modelo: Path
    ) -> None: ...
    def fallar_consulta(self, id_consulta: str, motivo: str) -> None: ...
    def interrumpir_consulta(self, id_consulta: str, motivo: str = "interrumpida") -> None: ...
    def evaluaciones(self) -> list[EvaluacionRegistrada]: ...
    def artefactos(self, id_consulta: str) -> list[str]: ...
    def registrar_version(self, id_consulta: str, etiquetas: Mapping[str, str]) -> VersionRegistrada: ...
    def version_con_alias(self, alias: str = ALIAS) -> VersionRegistrada | None: ...
    def version_de(self, modelo_sha256: str) -> VersionRegistrada | None: ...
    def descargar_modelo(self, version: int, destino: Path) -> Path: ...
    def etiquetar_version(self, version: int, etiquetas: Mapping[str, str]) -> None: ...
    def version(self, numero: int) -> VersionRegistrada | None: ...
    def abrir_exportacion(self, etiquetas: Mapping[str, str]) -> str: ...
    def terminar_exportacion(
        self, id_exportacion: str, metricas: Mapping[str, float], artefactos: Sequence[Path], paridad: str
    ) -> None: ...
    def fallar_exportacion(self, id_exportacion: str, motivo: str) -> None: ...
    def exportaciones(self) -> list[ExportacionRegistrada]: ...


@dataclass
class _ConsultaEnMemoria:
    etiquetas: dict[str, str]
    estado: EstadoConsulta = EstadoConsulta.EN_CURSO
    metricas: dict[str, float] = field(default_factory=dict)
    artefactos: list[str] = field(default_factory=list)
    modelo: bytes = b""


class RegistroAceptacionEnMemoria:
    def __init__(self) -> None:
        self._consultas: dict[str, _ConsultaEnMemoria] = {}
        self._ids = itertools.count(1)
        self._versiones: list[VersionRegistrada] = []
        self._alias: dict[str, int] = {}
        self._exportaciones: dict[str, ExportacionRegistrada] = {}
        self._artefactos_exportacion: dict[str, list[str]] = {}

    def abrir_consulta(self, numero, etiquetas) -> str:
        id_consulta = f"memoria-{next(self._ids)}"
        self._consultas[id_consulta] = _ConsultaEnMemoria(_etiquetas_consulta(numero, etiquetas))
        return id_consulta

    def terminar_consulta(self, id_consulta, metricas, decision, artefactos, modelo) -> None:
        consulta = self._consultas[id_consulta]
        consulta.estado = EstadoConsulta.TERMINADA
        consulta.metricas = {k: float(v) for k, v in metricas.items()}
        consulta.etiquetas["decision"] = decision
        consulta.artefactos = [Path(a).name for a in artefactos] + [f"modelo/{Path(modelo).name}"]
        consulta.modelo = Path(modelo).read_bytes()

    def _cerrar(self, id_consulta: str, estado: EstadoConsulta, motivo: str) -> None:
        self._consultas[id_consulta].estado = estado
        self._consultas[id_consulta].etiquetas["estado.motivo"] = motivo

    def fallar_consulta(self, id_consulta, motivo) -> None:
        self._cerrar(id_consulta, EstadoConsulta.FALLIDA, motivo)

    def interrumpir_consulta(self, id_consulta, motivo="interrumpida") -> None:
        self._cerrar(id_consulta, EstadoConsulta.INTERRUMPIDA, motivo)

    def evaluaciones(self) -> list[EvaluacionRegistrada]:
        return [
            EvaluacionRegistrada(
                run_id=id_consulta,
                consulta=int(c.etiquetas["consulta"]),
                modelo_sha256=c.etiquetas["modelo.sha256"],
                estado=c.estado,
                decision=c.etiquetas.get("decision"),
                metricas=dict(c.metricas),
                etiquetas=dict(c.etiquetas),
            )
            for id_consulta, c in self._consultas.items()
        ]

    def artefactos(self, id_consulta) -> list[str]:
        if id_consulta in self._artefactos_exportacion:
            return list(self._artefactos_exportacion[id_consulta])
        return list(self._consultas[id_consulta].artefactos)

    def version(self, numero) -> VersionRegistrada | None:
        return self._versiones[numero - 1] if 1 <= numero <= len(self._versiones) else None

    def abrir_exportacion(self, etiquetas) -> str:
        id_exportacion = f"memoria-exportacion-{next(self._ids)}"
        self._exportaciones[id_exportacion] = ExportacionRegistrada(
            id_exportacion, nombre_exportacion(len(self._exportaciones) + 1), EstadoConsulta.EN_CURSO, None, {},
            {**{k: str(v) for k, v in etiquetas.items()}, "tipo": TIPO_EXPORTACION},
        )
        return id_exportacion

    def terminar_exportacion(self, id_exportacion, metricas, artefactos, paridad) -> None:
        _comprobar_paridad(paridad)
        anterior = self._exportaciones[id_exportacion]
        self._exportaciones[id_exportacion] = ExportacionRegistrada(
            id_exportacion, anterior.nombre, EstadoConsulta.TERMINADA, paridad,
            {k: float(v) for k, v in metricas.items()}, anterior.etiquetas | {"paridad": paridad},
        )
        self._artefactos_exportacion[id_exportacion] = _ficheros(artefactos)

    def fallar_exportacion(self, id_exportacion, motivo) -> None:
        anterior = self._exportaciones[id_exportacion]
        self._exportaciones[id_exportacion] = ExportacionRegistrada(
            id_exportacion, anterior.nombre, EstadoConsulta.FALLIDA, None, {},
            anterior.etiquetas | {"estado.motivo": motivo},
        )

    def exportaciones(self) -> list[ExportacionRegistrada]:
        return list(self._exportaciones.values())

    def registrar_version(self, id_consulta, etiquetas) -> VersionRegistrada:
        _exigir_aceptada(next((e for e in self.evaluaciones() if e.run_id == id_consulta), None), id_consulta)
        version = VersionRegistrada(len(self._versiones) + 1, id_consulta, {k: str(v) for k, v in etiquetas.items()})
        self._versiones.append(version)
        self._alias[ALIAS] = version.version
        return version

    def version_con_alias(self, alias=ALIAS) -> VersionRegistrada | None:
        return self._versiones[self._alias[alias] - 1] if alias in self._alias else None

    def version_de(self, modelo_sha256) -> VersionRegistrada | None:
        return next((v for v in self._versiones if v.etiquetas.get("modelo.sha256") == modelo_sha256), None)

    def descargar_modelo(self, version, destino) -> Path:
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        ruta = destino / "modelo.pt"
        ruta.write_bytes(self._consultas[self._versiones[version - 1].run_id].modelo)
        return ruta

    def etiquetar_version(self, version, etiquetas) -> None:
        anterior = self._versiones[version - 1]
        self._versiones[version - 1] = VersionRegistrada(
            anterior.version, anterior.run_id, anterior.etiquetas | {k: str(v) for k, v in etiquetas.items()}
        )


class RegistroAceptacionMlflow:
    """El almacén local de MLflow (`$SIGNAI_DATA_ROOT/mlflow`)."""

    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    def _cliente(self) -> MlflowClient:
        return MlflowClient(tracking_uri=seguimiento.uri_seguimiento(self._raiz))

    def abrir_consulta(self, numero, etiquetas) -> str:
        return seguimiento.abrir_corrida(
            EXPERIMENTO_ACEPTACION, nombre_consulta(numero), {}, _etiquetas_consulta(numero, etiquetas), self._raiz
        )

    def terminar_consulta(self, id_consulta, metricas, decision, artefactos, modelo) -> None:
        self._cliente().log_artifact(id_consulta, str(modelo), artifact_path="modelo")
        seguimiento.cerrar_corrida(
            id_consulta, "FINISHED", metricas, artefactos, {"decision": decision}, self._raiz
        )

    def fallar_consulta(self, id_consulta, motivo) -> None:
        seguimiento.cerrar_corrida(id_consulta, "FAILED", etiquetas={"estado.motivo": motivo}, raiz=self._raiz)

    def interrumpir_consulta(self, id_consulta, motivo="interrumpida") -> None:
        seguimiento.cerrar_corrida(id_consulta, "KILLED", etiquetas={"estado.motivo": motivo}, raiz=self._raiz)

    def evaluaciones(self) -> list[EvaluacionRegistrada]:
        cliente = self._cliente()
        experimento = cliente.get_experiment_by_name(EXPERIMENTO_ACEPTACION)
        if experimento is None:
            return []
        corridas = cliente.search_runs(
            [experimento.experiment_id], filter_string=f"tags.tipo = '{TIPO}'", max_results=1000
        )
        return [
            EvaluacionRegistrada(
                run_id=run.info.run_id,
                consulta=int(run.data.tags["consulta"]),
                modelo_sha256=run.data.tags["modelo.sha256"],
                estado=ESTADO_DE_MLFLOW.get(run.info.status, EstadoConsulta.EN_CURSO),
                decision=run.data.tags.get("decision"),
                metricas=dict(run.data.metrics),
                etiquetas=dict(run.data.tags),
            )
            for run in corridas
        ]

    def artefactos(self, id_consulta) -> list[str]:
        cliente = self._cliente()
        rutas, pendientes = [], [None]
        while pendientes:
            for entrada in cliente.list_artifacts(id_consulta, pendientes.pop()):
                (pendientes if entrada.is_dir else rutas).append(entrada.path)
        return rutas

    # ── G-2 ────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _version(modelo) -> VersionRegistrada:
        return VersionRegistrada(int(modelo.version), modelo.run_id, dict(modelo.tags))

    def registrar_version(self, id_consulta, etiquetas) -> VersionRegistrada:
        _exigir_aceptada(next((e for e in self.evaluaciones() if e.run_id == id_consulta), None), id_consulta)
        cliente = self._cliente()
        try:
            cliente.get_registered_model(MODELO_REGISTRADO)
        except MlflowException:
            cliente.create_registered_model(
                MODELO_REGISTRADO, description="KPoseFormer aceptado por TT-15 (≥85 % Top-1 sobre la prueba, ADR-0011)."
            )
        modelo = cliente.create_model_version(
            MODELO_REGISTRADO, source=f"runs:/{id_consulta}/modelo", run_id=id_consulta,
            tags={k: str(v) for k, v in etiquetas.items()},
        )
        cliente.set_registered_model_alias(MODELO_REGISTRADO, ALIAS, modelo.version)
        return self._version(modelo)

    def version_con_alias(self, alias=ALIAS) -> VersionRegistrada | None:
        try:
            return self._version(self._cliente().get_model_version_by_alias(MODELO_REGISTRADO, alias))
        except MlflowException:
            return None

    def version_de(self, modelo_sha256) -> VersionRegistrada | None:
        try:
            versiones = self._cliente().search_model_versions(f"name='{MODELO_REGISTRADO}'")
        except MlflowException:
            return None
        propias = sorted(
            (self._version(v) for v in versiones if v.tags.get("modelo.sha256") == modelo_sha256),
            key=lambda v: v.version,
        )
        return propias[0] if propias else None

    def descargar_modelo(self, version, destino) -> Path:
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        cliente = self._cliente()
        run_id = cliente.get_model_version(MODELO_REGISTRADO, str(version)).run_id
        return Path(cliente.download_artifacts(run_id, "modelo/modelo.pt", str(destino)))

    def etiquetar_version(self, version, etiquetas) -> None:
        cliente = self._cliente()
        for clave, valor in etiquetas.items():
            cliente.set_model_version_tag(MODELO_REGISTRADO, str(version), clave, str(valor))

    def version(self, numero) -> VersionRegistrada | None:
        try:
            return self._version(self._cliente().get_model_version(MODELO_REGISTRADO, str(numero)))
        except MlflowException:
            return None

    # ── G-3 ────────────────────────────────────────────────────────────────────────────────────

    def _corridas_exportacion(self) -> list:
        cliente = self._cliente()
        experimento = cliente.get_experiment_by_name(EXPERIMENTO_EXPORTACION)
        if experimento is None:
            return []
        return cliente.search_runs(
            [experimento.experiment_id], filter_string=f"tags.tipo = '{TIPO_EXPORTACION}'", max_results=1000
        )

    def abrir_exportacion(self, etiquetas) -> str:
        numero = len(self._corridas_exportacion()) + 1
        return seguimiento.abrir_corrida(
            EXPERIMENTO_EXPORTACION, nombre_exportacion(numero), {},
            {**{k: str(v) for k, v in etiquetas.items()}, "tipo": TIPO_EXPORTACION}, self._raiz,
        )

    def terminar_exportacion(self, id_exportacion, metricas, artefactos, paridad) -> None:
        _comprobar_paridad(paridad)
        cliente = self._cliente()
        ficheros = []
        for artefacto in map(Path, artefactos):
            if artefacto.is_dir():
                cliente.log_artifacts(id_exportacion, str(artefacto), artifact_path=artefacto.name)
            else:
                ficheros.append(artefacto)
        seguimiento.cerrar_corrida(id_exportacion, "FINISHED", metricas, ficheros, {"paridad": paridad}, self._raiz)

    def fallar_exportacion(self, id_exportacion, motivo) -> None:
        seguimiento.cerrar_corrida(id_exportacion, "FAILED", etiquetas={"estado.motivo": motivo}, raiz=self._raiz)

    def exportaciones(self) -> list[ExportacionRegistrada]:
        return [
            ExportacionRegistrada(
                run_id=run.info.run_id,
                nombre=run.info.run_name,
                estado=ESTADO_DE_MLFLOW.get(run.info.status, EstadoConsulta.EN_CURSO),
                paridad=run.data.tags.get("paridad"),
                metricas=dict(run.data.metrics),
                etiquetas=dict(run.data.tags),
            )
            for run in self._corridas_exportacion()
        ]

"""Registro de corridas y barridos (contracts/registro.md; research R-10).

Un puerto, `Registro`, con dos implementaciones: `RegistroMlflow` sobre el almacén local de TT-03
(`signai_ml.seguimiento`) y `RegistroEnMemoria` para probar el bucle y la CLI sin disco. Un barrido es
un experimento `reconocedor/<barrido>`; una corrida, un run `<configuracion>-s<semilla>` con la
etiqueta `estado`.
"""

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from mlflow import MlflowClient
from mlflow.entities import ViewType

from .. import seguimiento


class EstadoCorrida(StrEnum):
    EN_CURSO = "en_curso"
    TERMINADA = "terminada"
    FALLIDA = "fallida"
    INTERRUMPIDA = "interrumpida"


@dataclass(frozen=True)
class CorridaRegistrada:
    id: str
    nombre: str
    estado: EstadoCorrida
    parametros: dict[str, str]
    etiquetas: dict[str, str]
    metricas: dict[str, float]
    inicio: int = 0

    @property
    def configuracion(self) -> str:
        return self.etiquetas["configuracion"]

    @property
    def semilla(self) -> int:
        return int(self.parametros["semilla"])

    @property
    def ronda(self) -> int:
        return int(self.etiquetas["ronda"])

    @property
    def fase(self) -> str:
        """`seleccion` (una rotación) o `final` (el modelo final, TT-22); las de TT-14 son de selección."""
        return self.etiquetas.get("fase", "seleccion")

    @property
    def rotacion(self) -> str | None:
        return self.etiquetas.get("rotacion")


class Registro(Protocol):
    def abrir(self, barrido: str, nombre: str, parametros: Mapping, etiquetas: Mapping[str, str]) -> str: ...
    def epoca(self, id_corrida: str, epoca: int, metricas: Mapping[str, float]) -> None: ...
    def terminar(self, id_corrida: str, metricas: Mapping[str, float], artefactos: Sequence[Path],
                 etiquetas: Mapping[str, str]) -> None: ...
    def fallar(self, id_corrida: str, motivo: str) -> None: ...
    def interrumpir(self, id_corrida: str, motivo: str = "interrumpida") -> None: ...
    def corridas(self, barrido: str) -> list[CorridaRegistrada]: ...
    def marcar_interrumpidas(self, barrido: str) -> list[str]: ...
    def estado_barrido(self, barrido: str) -> dict[str, str]: ...
    def cerrar_barrido(self, barrido: str, etiquetas: Mapping[str, str]) -> None: ...
    def guardar_resumen(self, barrido: str, texto: str) -> None: ...
    def resumen(self, barrido: str) -> str | None: ...
    def descargar(self, id_corrida: str, artefacto: str, destino: Path) -> Path: ...
    def historia(self, id_corrida: str, metrica: str) -> list[tuple[int, float]]: ...


def experimento(barrido: str) -> str:
    return f"reconocedor/{barrido}"


# --- En memoria --------------------------------------------------------------------------------


@dataclass
class _CorridaEnMemoria:
    barrido: str
    nombre: str
    parametros: dict[str, str]
    etiquetas: dict[str, str]
    metricas: dict[str, float] = field(default_factory=dict)
    historias: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    artefactos: dict[str, bytes] = field(default_factory=dict)


class RegistroEnMemoria:
    def __init__(self) -> None:
        self._corridas: dict[str, _CorridaEnMemoria] = {}
        self._barridos: dict[str, dict[str, str]] = {}
        self._resumenes: dict[str, str] = {}

    def abrir(self, barrido, nombre, parametros, etiquetas) -> str:
        id_corrida = f"mem-{len(self._corridas) + 1}"
        self._corridas[id_corrida] = _CorridaEnMemoria(
            barrido, nombre, {k: str(v) for k, v in parametros.items()},
            {**{k: str(v) for k, v in etiquetas.items()}, "estado": EstadoCorrida.EN_CURSO},
        )
        return id_corrida

    def epoca(self, id_corrida, epoca, metricas) -> None:
        for nombre, valor in metricas.items():
            self._corridas[id_corrida].historias.setdefault(nombre, []).append((epoca, float(valor)))

    def terminar(self, id_corrida, metricas, artefactos, etiquetas) -> None:
        corrida = self._corridas[id_corrida]
        corrida.metricas.update({k: float(v) for k, v in metricas.items()})
        corrida.etiquetas.update({**etiquetas, "estado": EstadoCorrida.TERMINADA})
        corrida.artefactos.update({Path(a).name: Path(a).read_bytes() for a in artefactos})

    def fallar(self, id_corrida, motivo) -> None:
        self._corridas[id_corrida].etiquetas.update({"estado": EstadoCorrida.FALLIDA, "motivo": motivo})

    def interrumpir(self, id_corrida, motivo="interrumpida") -> None:
        self._corridas[id_corrida].etiquetas.update({"estado": EstadoCorrida.INTERRUMPIDA, "motivo": motivo})

    def corridas(self, barrido) -> list[CorridaRegistrada]:
        return [
            CorridaRegistrada(i, c.nombre, EstadoCorrida(c.etiquetas["estado"]), dict(c.parametros),
                              dict(c.etiquetas), dict(c.metricas), orden)
            for orden, (i, c) in enumerate(self._corridas.items()) if c.barrido == barrido
        ]

    def marcar_interrumpidas(self, barrido) -> list[str]:
        ids = [c.id for c in self.corridas(barrido) if c.estado is EstadoCorrida.EN_CURSO]
        for id_corrida in ids:
            self.interrumpir(id_corrida)
        return ids

    def estado_barrido(self, barrido) -> dict[str, str]:
        return dict(self._barridos.get(barrido, {}))

    def cerrar_barrido(self, barrido, etiquetas) -> None:
        self._barridos[barrido] = {"barrido.estado": "cerrado", **etiquetas}

    def guardar_resumen(self, barrido, texto) -> None:
        self._resumenes[barrido] = texto

    def resumen(self, barrido) -> str | None:
        return self._resumenes.get(barrido)

    def descargar(self, id_corrida, artefacto, destino) -> Path:
        Path(destino).mkdir(parents=True, exist_ok=True)
        ruta = Path(destino) / artefacto
        ruta.write_bytes(self._corridas[id_corrida].artefactos[artefacto])
        return ruta

    def historia(self, id_corrida, metrica) -> list[tuple[int, float]]:
        return list(self._corridas[id_corrida].historias.get(metrica, []))


# --- MLflow -------------------------------------------------------------------------------------

_TIPO_RESUMEN = "resumen"
_ESTADO_MLFLOW = {
    EstadoCorrida.TERMINADA: "FINISHED",
    EstadoCorrida.FALLIDA: "FAILED",
    EstadoCorrida.INTERRUMPIDA: "KILLED",
}


class RegistroMlflow:
    """Sobre el almacén de `$SIGNAI_DATA_ROOT/mlflow` (o el de `raiz`, en pruebas)."""

    def __init__(self, raiz: Path | None = None) -> None:
        self._raiz = raiz

    @property
    def _cliente(self) -> MlflowClient:
        return MlflowClient(tracking_uri=seguimiento.uri_seguimiento(self._raiz))

    def _id_experimento(self, barrido: str) -> str | None:
        existente = self._cliente.get_experiment_by_name(experimento(barrido))
        return None if existente is None else existente.experiment_id

    def abrir(self, barrido, nombre, parametros, etiquetas) -> str:
        return seguimiento.abrir_corrida(
            experimento(barrido), nombre, parametros, {**etiquetas, "estado": EstadoCorrida.EN_CURSO}, raiz=self._raiz
        )

    def epoca(self, id_corrida, epoca, metricas) -> None:
        seguimiento.registrar_epoca(id_corrida, epoca, metricas, raiz=self._raiz)

    def _cerrar(self, id_corrida, estado, metricas=None, artefactos=(), etiquetas=None) -> None:
        seguimiento.cerrar_corrida(
            id_corrida, _ESTADO_MLFLOW[estado], metricas=metricas, artefactos=artefactos,
            etiquetas={**(etiquetas or {}), "estado": estado}, raiz=self._raiz,
        )

    def terminar(self, id_corrida, metricas, artefactos, etiquetas) -> None:
        self._cerrar(id_corrida, EstadoCorrida.TERMINADA, metricas, artefactos, etiquetas)

    def fallar(self, id_corrida, motivo) -> None:
        self._cerrar(id_corrida, EstadoCorrida.FALLIDA, etiquetas={"motivo": motivo})

    def interrumpir(self, id_corrida, motivo="interrumpida") -> None:
        self._cerrar(id_corrida, EstadoCorrida.INTERRUMPIDA, etiquetas={"motivo": motivo})

    def _runs(self, barrido: str) -> list:
        id_experimento = self._id_experimento(barrido)
        if id_experimento is None:
            return []
        return self._cliente.search_runs([id_experimento], run_view_type=ViewType.ACTIVE_ONLY, max_results=50_000)

    def corridas(self, barrido) -> list[CorridaRegistrada]:
        return [
            CorridaRegistrada(
                r.info.run_id, r.info.run_name, EstadoCorrida(r.data.tags["estado"]), dict(r.data.params),
                dict(r.data.tags), dict(r.data.metrics), r.info.start_time,
            )
            for r in self._runs(barrido) if r.data.tags.get("tipo") != _TIPO_RESUMEN
        ]

    def marcar_interrumpidas(self, barrido) -> list[str]:
        ids = [c.id for c in self.corridas(barrido) if c.estado is EstadoCorrida.EN_CURSO]
        for id_corrida in ids:
            self.interrumpir(id_corrida)
        return ids

    def estado_barrido(self, barrido) -> dict[str, str]:
        id_experimento = self._id_experimento(barrido)
        if id_experimento is None:
            return {}
        etiquetas = self._cliente.get_experiment(id_experimento).tags
        return {k: v for k, v in etiquetas.items() if k.startswith(("barrido.", "cierre."))}

    def cerrar_barrido(self, barrido, etiquetas) -> None:
        cliente, id_experimento = seguimiento.cliente_y_experimento(experimento(barrido), self._raiz)
        for clave, valor in {"barrido.estado": "cerrado", **etiquetas}.items():
            cliente.set_experiment_tag(id_experimento, clave, str(valor))

    def _run_resumen(self, barrido: str):
        return next((r for r in self._runs(barrido) if r.data.tags.get("tipo") == _TIPO_RESUMEN), None)

    def guardar_resumen(self, barrido, texto) -> None:
        cliente, id_experimento = seguimiento.cliente_y_experimento(experimento(barrido), self._raiz)
        existente = self._run_resumen(barrido)
        id_run = existente.info.run_id if existente else cliente.create_run(
            id_experimento, run_name=_TIPO_RESUMEN, tags={"tipo": _TIPO_RESUMEN}
        ).info.run_id
        cliente.log_text(id_run, texto, "resumen.md")
        if existente is None:
            cliente.set_terminated(id_run)

    def resumen(self, barrido) -> str | None:
        existente = self._run_resumen(barrido)
        if existente is None:
            return None
        ruta = Path(self._cliente.download_artifacts(existente.info.run_id, "resumen.md"))
        return ruta.read_text(encoding="utf-8")

    def descargar(self, id_corrida, artefacto, destino) -> Path:
        Path(destino).mkdir(parents=True, exist_ok=True)
        bajado = Path(self._cliente.download_artifacts(id_corrida, artefacto, str(destino)))
        return bajado if bajado.parent == Path(destino) else Path(shutil.copy(bajado, Path(destino) / artefacto))

    def historia(self, id_corrida, metrica) -> list[tuple[int, float]]:
        return [(m.step, m.value) for m in self._cliente.get_metric_history(id_corrida, metrica)]

"""Dobles de los puertos de la copia, compartidos por las pruebas del servicio y de la CLI."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from signai_ml.copia.repositorio import Instantanea, ResultadoCopia


class RepositorioFalso:
    def __init__(self, raiz: Path, instantaneas: int = 0) -> None:
        self.raiz = raiz
        self._instantaneas = [Instantanea(f"i{n}", f"2026-10-0{n + 1}", ("signai",)) for n in range(instantaneas)]
        self.copias: list[dict[str, object]] = []
        self.verificaciones: list[str | None] = []
        self.restauraciones: list[tuple[str, Path, str | None]] = []

    def existe(self) -> bool:
        return True

    def instantaneas(self) -> list[Instantanea]:
        return list(self._instantaneas)

    def copiar(
        self,
        rutas: Sequence[str],
        *,
        base: Path,
        etiquetas: Sequence[str],
        excluir: Sequence[str] = (),
        cuando: str | None = None,
    ) -> ResultadoCopia:
        self.copias.append({"rutas": tuple(rutas), "base": base, "etiquetas": tuple(etiquetas), "excluir": tuple(excluir)})
        # El repositorio cobra forma para que el envío tenga algo que clonar.
        (self.raiz / "copia" / "repositorio" / "data").mkdir(parents=True, exist_ok=True)
        (self.raiz / "copia" / "repositorio" / "data" / f"p{len(self.copias)}").write_bytes(b"cifrado")
        self._instantaneas.append(Instantanea(f"nueva{len(self.copias)}", "2026-10-18", tuple(etiquetas), 2048))
        return ResultadoCopia(f"nueva{len(self.copias)}", 1234)

    def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None:
        self.verificaciones.append("completa" if completo else subconjunto)

    def restaurar(self, instantanea: str, destino: Path, *, ruta: str | None = None) -> None:
        self.restauraciones.append((instantanea, destino, ruta))
        destino.mkdir(parents=True, exist_ok=True)

    def olvidar(self, *, dentro_de: str, ultimas: int) -> None:
        pass


class DvcFalso:
    def __init__(self, sin_subir: bool = False, ausentes: int = 0) -> None:
        self._sin_subir = sin_subir
        self._ausentes = ausentes
        self.almacenes: list[Path] = []

    def sin_subir(self) -> bool:
        return self._sin_subir

    def ausentes_en(self, almacen: Path) -> int:
        self.almacenes.append(almacen)
        return self._ausentes


class ClonadorCopia:
    def clonar(self, origen: Path, destino: Path) -> None:
        shutil.copytree(origen, destino)

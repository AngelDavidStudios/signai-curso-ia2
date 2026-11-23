"""Inventarios del origen: el ligero para el estado sin clave (R-10) y el de huellas para restaurar (R-09)."""

import hashlib
import os
import shutil
from pathlib import Path

from signai_ml.copia.manifiesto import (
    EntradaLigera,
    cambiados,
    comparar,
    guardar,
    inventario_ligero,
    leer,
    manifiesto,
)

RUTAS = ("dvc-store", "mlflow")


def test_inventario_relativo_ordenado_y_con_separador_barra(raiz: Path) -> None:
    entradas = inventario_ligero(raiz, RUTAS)

    rutas = [entrada.ruta for entrada in entradas]
    assert rutas == sorted(rutas)
    assert all(not ruta.startswith("/") and "\\" not in ruta for ruta in rutas)
    assert "mlflow/mlflow.db" in rutas
    assert len([r for r in rutas if r.startswith("dvc-store/")]) == 20
    assert not any(r.startswith("revision/") for r in rutas)


def test_incluye_archivos_sueltos_y_omite_lo_que_no_existe(raiz: Path) -> None:
    (raiz / "copia").mkdir()
    (raiz / "copia" / "registro.jsonl").write_text("{}\n", encoding="utf-8")

    rutas = [e.ruta for e in inventario_ligero(raiz, (*RUTAS, "copia/registro.jsonl", "no-existe"))]
    assert "copia/registro.jsonl" in rutas


def test_guarda_y_lee_el_ultimo_manifiesto(raiz: Path) -> None:
    entradas = inventario_ligero(raiz, RUTAS)
    guardar(raiz, "2026-10-18T21:00:00+00:00", entradas)

    assert (raiz / "copia" / "ultimo-manifiesto.json").is_file()
    assert leer(raiz) == ("2026-10-18T21:00:00+00:00", entradas)


def test_sin_manifiesto_no_hay_nada_que_comparar(raiz: Path) -> None:
    assert leer(raiz) is None


def test_sin_cambios_no_hay_conjuntos_pendientes(raiz: Path) -> None:
    entradas = inventario_ligero(raiz, RUTAS)
    assert cambiados(entradas, inventario_ligero(raiz, RUTAS)) == []


def test_detecta_el_conjunto_que_cambia(raiz: Path) -> None:
    anterior = inventario_ligero(raiz, RUTAS)
    base = raiz / "mlflow" / "mlflow.db"
    estado = base.stat()
    os.utime(base, ns=(estado.st_atime_ns, estado.st_mtime_ns + 1_000_000_000))

    assert cambiados(anterior, inventario_ligero(raiz, RUTAS)) == ["seguimiento"]


def test_detecta_altas_bajas_y_cambios_de_tamano(raiz: Path) -> None:
    anterior = inventario_ligero(raiz, RUTAS)
    nuevo = raiz / "dvc-store" / "files" / "md5" / "ff" / ("f" * 30)
    nuevo.parent.mkdir(parents=True, exist_ok=True)
    nuevo.write_bytes(b"nuevo")
    assert cambiados(anterior, inventario_ligero(raiz, RUTAS)) == ["datos-versionados"]

    nuevo.unlink()
    (raiz / "mlflow" / "mlflow.db").write_bytes(b"otro contenido")
    assert cambiados(anterior, inventario_ligero(raiz, RUTAS)) == ["seguimiento"]


def test_la_entrada_es_ruta_bytes_y_mtime() -> None:
    assert EntradaLigera("mlflow/mlflow.db", 10, 5) == EntradaLigera(ruta="mlflow/mlflow.db", bytes=10, mtime=5)


# ── Manifiesto con huellas para la prueba de restauración (R-09) ───────────────────────────────


def test_manifiesto_con_sha256_relativo_y_ordenado(raiz: Path) -> None:
    entradas = manifiesto(raiz, RUTAS)
    rutas = [e.ruta for e in entradas]
    assert rutas == sorted(rutas)
    base = next(e for e in entradas if e.ruta == "mlflow/mlflow.db")
    assert base.sha256 == hashlib.sha256((raiz / "mlflow" / "mlflow.db").read_bytes()).hexdigest()
    assert base.bytes == (raiz / "mlflow" / "mlflow.db").stat().st_size


def test_comparar_dos_arboles_identicos(raiz: Path, tmp_path: Path) -> None:
    copia = tmp_path / "copia-identica"
    for ruta in RUTAS:
        shutil.copytree(raiz / ruta, copia / ruta)

    resultado = comparar(manifiesto(raiz, RUTAS), manifiesto(copia, RUTAS))
    assert resultado.identico
    assert (resultado.faltan, resultado.sobran, resultado.distintos) == ((), (), ())


def test_comparar_detecta_faltan_sobran_y_distintos(raiz: Path, tmp_path: Path) -> None:
    copia = tmp_path / "copia-alterada"
    for ruta in RUTAS:
        shutil.copytree(raiz / ruta, copia / ruta)
    objetos = sorted((copia / "dvc-store").rglob("*/*/*/*"))
    objetos[0].unlink()
    contenido = objetos[1].read_bytes()
    objetos[1].write_bytes(contenido[:-1] + bytes([contenido[-1] ^ 0xFF]))
    (copia / "mlflow" / "extra.bin").write_bytes(b"sobra")

    resultado = comparar(manifiesto(raiz, RUTAS), manifiesto(copia, RUTAS))
    assert not resultado.identico
    assert resultado.faltan == (objetos[0].relative_to(copia).as_posix(),)
    assert resultado.distintos == (objetos[1].relative_to(copia).as_posix(),)
    assert resultado.sobran == ("mlflow/extra.bin",)

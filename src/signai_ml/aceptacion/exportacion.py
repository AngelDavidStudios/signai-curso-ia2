"""Orden `exportar` (contracts/cli.md; spec US3, FR-009…FR-014; research R-06…R-08, R-12).

1. Exige el árbol limpio y comprueba que los pesos de la versión registrada son los evaluados.
2. Convierte, escribe los metadatos y valida el artefacto como la app.
3. Mide la paridad frente al original sobre la prueba y los casos de referencia, en CPU y con todas
   las unidades. No abre una consulta nueva: compara dos versiones de un modelo ya aceptado.
4. Registra el intento. Solo si la paridad se supera, escribe `modelos/coreml/` y etiqueta la versión.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..reconocedor.modelo import cargar
from . import CASOS_REFERENCIA, COREML, MODELO_REGISTRADO, TOLERANCIA_PARIDAD
from .artefacto import cifra, escribir_metadatos, ficha_artefacto, huella_paquete, unir, validar
from .candidato import Candidato, leer, sha256_fichero
from .conversion import OBJETIVO
from .datos import DatosPrueba, cargar_prueba
from .ejecucion import UNIDADES, casos_referencia
from .errores import ErrorAceptacion, ErrorUso, NoAceptado, ParidadFallida
from .evaluacion import probabilidades
from .formato import decimal, miles
from .paridad import InformeParidad, comparar, informe_paridad
from .registro import VersionRegistrada

if TYPE_CHECKING:  # pragma: no cover
    from .cli import Contexto

ETIQUETA_UNIDAD = {"cpu": "CPU_ONLY", "todas": "ALL"}
VERSIONAR_COREML = "Versiónalo:  uv run dvc add modelos/coreml && uv run dvc push"


def _ruta_casos(ctx: "Contexto", casos: Path | None) -> Path:
    ruta = Path(casos) if casos else ctx.raiz / CASOS_REFERENCIA
    if not ruta.is_file():
        raise ErrorUso(f"No existen los casos de referencia {ruta}: indica otros con --casos.")
    return ruta


def _version(ctx: "Contexto", numero: int | None) -> VersionRegistrada:
    version = ctx.registro.version(numero) if numero else ctx.registro.version_con_alias()
    if version is None:
        detalle = f"No existe la versión {numero}" if numero else "No hay ninguna versión con alias aceptado"
        raise NoAceptado(f"{detalle} de {MODELO_REGISTRADO}: lanza antes `uv run signai-aceptacion registrar`.")
    if version.etiquetas.get("estado") != "aceptado":
        raise NoAceptado(f"La versión {version.version} no está aceptada: no se exporta (constitución IV).")
    return version


def _comprobar_pesos(pesos: Path, version: VersionRegistrada, candidato: Candidato) -> None:
    huella = sha256_fichero(pesos)
    if not huella == version.etiquetas.get("modelo.sha256") == candidato.modelo_sha256:
        raise ErrorAceptacion(
            f"Los pesos de la versión {version.version} ({huella[:12]}…) no son los del candidato evaluado "
            f"({candidato.modelo_sha256[:12]}…): FR-009. No se exporta."
        )


def _metadatos(version: VersionRegistrada, candidato: Candidato) -> dict[str, str]:
    return {
        "clases": unir(candidato.clases, "clases"),
        "glosas": unir(candidato.glosas, "glosas"),
        "cifra": cifra(float(version.etiquetas["top1_prueba"]), int(version.etiquetas["consulta"])),
        "modelo.version": str(version.version),
        "modelo.sha256": candidato.modelo_sha256,
        "aceptacion.run_id": version.run_id,
        "particion.huella": candidato.huellas["particion.huella"],
    }


def _paridad(ctx: "Contexto", modelo, paquete: Path, datos: DatosPrueba, referencia) -> InformeParidad:
    original = {"prueba": probabilidades(modelo, datos.x), "referencia": probabilidades(modelo, referencia)}
    resultados = {}
    for unidad in UNIDADES:
        ejecutor = ctx.ejecutor(unidad)
        resultados[unidad] = {
            "prueba": comparar(original["prueba"], ejecutor.probabilidades(paquete, datos.x), datos.y),
            "referencia": comparar(original["referencia"], ejecutor.probabilidades(paquete, referencia)),
        }
    return informe_paridad(resultados)


def _metricas(informe: InformeParidad, bytes_paquete: int) -> dict[str, float]:
    metricas = {"coreml.bytes": float(bytes_paquete)}
    for unidad, conjuntos in informe.unidades.items():
        for conjunto, i in conjuntos.items():
            base = f"paridad.{unidad}.{conjunto}"
            metricas[f"{base}.max_diferencia"] = i.max_diferencia
            metricas[f"{base}.clase_distinta"] = float(len(i.clase_distinta))
            metricas[f"{base}.cambios_95"] = float(sum(i.cambios_95.values()))
    return metricas


def tabla_paridad(informe: InformeParidad) -> list[str]:
    unidades = list(informe.unidades)
    lineas = [f"Paridad (tolerancia {decimal(TOLERANCIA_PARIDAD, 2)})".ljust(33)
              + "".join(ETIQUETA_UNIDAD.get(u, u).ljust(20) for u in unidades)]
    conjuntos = list(dict.fromkeys(c for u in unidades for c in informe.unidades[u]))
    for conjunto in conjuntos:
        filas = {
            "clase distinta": lambda i: f"{len(i.clase_distinta)} / {miles(i.casos)}",
            "Top-1": lambda i: (
                f"{miles(i.aciertos_original)} = {miles(i.aciertos_exportado)}"
                if i.aciertos_original is not None else "—"
            ),
            "máx. diferencia": lambda i: decimal(i.max_diferencia, 4),
            "cambios al 95 %": lambda i: str(sum(i.cambios_95.values())),
        }
        for numero, (nombre, valor) in enumerate(filas.items()):
            etiqueta = (f"  {conjunto}" if numero == 0 else "").ljust(13) + nombre.ljust(20)
            lineas.append((etiqueta + "".join(
                (valor(informe.unidades[u][conjunto]) if conjunto in informe.unidades[u] else "—").ljust(20)
                for u in unidades
            )).rstrip())
    return [linea.rstrip() for linea in lineas]


def cmd_exportar(ctx: "Contexto", casos: Path | None = None, numero: int | None = None) -> int:
    commit, sucio = ctx.comprobar_codigo(ctx.raiz)
    ruta_casos = _ruta_casos(ctx, casos)
    candidato = leer(ctx.raiz)
    version = _version(ctx, numero)
    with tempfile.TemporaryDirectory(prefix="signai-exportacion-") as temporal:
        trabajo = Path(temporal)
        pesos = ctx.registro.descargar_modelo(version.version, trabajo / "modelo")
        _comprobar_pesos(pesos, version, candidato)
        modelo, _ = cargar(pesos)
        datos = cargar_prueba(ctx.raiz, candidato, lector=ctx.lector, total=ctx.total, por_clase=ctx.por_clase)
        referencia = casos_referencia(ruta_casos)
        id_exportacion = ctx.registro.abrir_exportacion({
            "modelo.version": str(version.version), "modelo.sha256": candidato.modelo_sha256,
            "aceptacion.run_id": version.run_id, "coreml.formato": "mlprogram", "coreml.precision": "float16",
            "coreml.objetivo": OBJETIVO, "codigo.commit": commit, "codigo.sucio": str(sucio).lower(),
        })
        try:
            conversion = ctx.convertidor.convertir(modelo, candidato.clases, trabajo / "coreml")
            paquete = conversion.paquete
            escribir_metadatos(paquete, _metadatos(version, candidato))
            validar(paquete, candidato.clases, candidato.glosas)
            informe = _paridad(ctx, modelo, paquete, datos, referencia)
            paridad = "superada" if informe.superada else "fallida"
            ficha = ficha_artefacto(paquete, conversion.avisos) | {"paridad": paridad}
            (paquete.parent / "paridad.json").write_text(
                json.dumps(informe.a_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (paquete.parent / "artefacto.json").write_text(
                json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (paquete.parent / "avisos.txt").write_text("\n".join(conversion.avisos) + "\n", encoding="utf-8")
            ctx.registro.terminar_exportacion(
                id_exportacion, _metricas(informe, ficha["bytes"]),
                [paquete, *(paquete.parent / n for n in ("paridad.json", "artefacto.json", "avisos.txt"))], paridad,
            )
        except BaseException as error:
            ctx.registro.fallar_exportacion(id_exportacion, str(error) or type(error).__name__)
            raise
        print(
            f"Versión {version.version} de {MODELO_REGISTRADO} · modelo {candidato.modelo_sha256[:8]}… · "
            f"consulta {version.etiquetas.get('consulta')}"
        )
        print(f"Artefacto: {paquete.name} · mlprogram · FLOAT16 · {OBJETIVO} · {decimal(ficha['bytes'] / 1e6, 2)} MB")
        for linea in tabla_paridad(informe):
            print(linea)
        if not informe.superada:
            print("Paridad: FALLIDA → no se escribe nada en modelos/coreml/")
            raise ParidadFallida(
                f"La paridad no se supera (informe en exportacion · {id_exportacion}): el artefacto no da los mismos "
                "resultados que el original. Revisa la conversión."
            )
        destino = ctx.raiz / COREML
        if destino.exists():
            shutil.rmtree(destino)
        shutil.copytree(paquete.parent, destino)
    ctx.registro.etiquetar_version(version.version, {
        "coreml.sha256": huella_paquete(destino / paquete.name), "coreml.run_id": id_exportacion, "paridad": "superada",
    })
    print("Paridad: SUPERADA → modelos/coreml/")
    print(VERSIONAR_COREML)
    return 0

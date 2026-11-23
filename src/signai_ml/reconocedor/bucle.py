"""Una corrida de selección y el modelo final (TT-14 · spec FR-003, FR-007, FR-008, FR-014; TT-22 · R-06).

Entrena en el equipo local (MPS si está disponible, si no CPU), desde pesos iniciales que solo
dependen de la semilla. En cada época mide la validación sin aumento y avisa por `al_terminar_epoca`;
conserva el modelo de la época con mejor Top-1 (empate: mejor F1 macro). Si la pérdida deja de ser
finita, la corrida termina como fallida.
"""

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .aumento import aplicar
from .configuracion import Configuracion
from .datos import DatosEntrenamiento, DatosFinal
from .metricas import calcular, top1_por_clase
from .modelo import KPoseFormer
from .registro import EstadoCorrida

LOTE_EVALUACION = 512
DESFASE_AUMENTO = 1_000_000  # el generador del aumento no coincide con el de los lotes de otra semilla


@dataclass(frozen=True)
class ResultadoCorrida:
    estado: EstadoCorrida
    epoca_elegida: int | None = None
    metricas: dict[str, float] | None = None
    por_clase: list[float] | None = None
    modelo: KPoseFormer | None = None
    duracion_s: float = 0.0
    motivo: str | None = None


def dispositivo() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _perdida_lote(logits: torch.Tensor, etiquetas: torch.Tensor, criterio: nn.Module) -> torch.Tensor:
    return criterio(logits, etiquetas)


def _evaluar(modelo: KPoseFormer, x: torch.Tensor, y: torch.Tensor, dev: torch.device) -> tuple:
    modelo.eval()
    probabilidades, perdida = [], 0.0
    criterio = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for inicio in range(0, len(x), LOTE_EVALUACION):
            logits = modelo(x[inicio : inicio + LOTE_EVALUACION].to(dev))
            perdida += criterio(logits, y[inicio : inicio + LOTE_EVALUACION].to(dev)).item()
            probabilidades.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(probabilidades).numpy(), perdida / len(x)


def _preparar(cfg: Configuracion, clases: int, semilla: int, dev: torch.device):
    """Modelo, optimizador, programación, criterio y generadores, en el mismo orden para la selección y el
    final: con la misma semilla y los mismos datos, las dos dan los mismos pesos época a época."""
    e = cfg.entrenamiento
    torch.manual_seed(semilla)
    modelo = KPoseFormer(cfg.forma, clases=clases, dropout=e.dropout).to(dev)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=e.tasa, weight_decay=e.regularizacion)
    programador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=e.epocas) if e.programacion == "coseno" else None
    criterio = nn.CrossEntropyLoss(label_smoothing=e.suavizado_etiquetas)
    lotes = torch.Generator().manual_seed(semilla)
    azar_aumento = torch.Generator().manual_seed(semilla + DESFASE_AUMENTO)
    return modelo, optimizador, programador, criterio, lotes, azar_aumento


def _una_epoca(modelo, optimizador, criterio, lotes, azar_aumento, x, y, cfg: Configuracion,
               dev: torch.device) -> tuple[float, float] | None:
    """Una pasada de entrenamiento: `(pérdida media, Top-1)`, o `None` si la pérdida deja de ser finita."""
    e = cfg.entrenamiento
    modelo.train()
    suma_perdida, aciertos, n = 0.0, 0, len(x)
    for indices in torch.randperm(n, generator=lotes).split(e.lote):
        xb = aplicar(x[indices], cfg.aumento, azar_aumento).to(dev)
        yb = y[indices].to(dev)
        logits = modelo(xb)
        perdida = _perdida_lote(logits, yb, criterio)
        if not math.isfinite(perdida.item()):
            return None
        optimizador.zero_grad()
        perdida.backward()
        if e.recorte_gradiente > 0:
            nn.utils.clip_grad_norm_(modelo.parameters(), e.recorte_gradiente)
        optimizador.step()
        suma_perdida += perdida.item() * len(indices)
        aciertos += (logits.argmax(dim=1) == yb).sum().item()
    return suma_perdida / n, aciertos / n


def _copia(modelo: KPoseFormer) -> dict:
    return copy.deepcopy({k: v.detach().cpu() for k, v in modelo.state_dict().items()})


def entrenar(
    datos: DatosEntrenamiento,
    cfg: Configuracion,
    semilla: int,
    al_terminar_epoca: Callable[[int, dict[str, float]], None],
    dev: torch.device | None = None,
) -> ResultadoCorrida:
    """Una corrida de selección: valida cada época con el pliegue que no entrenó y conserva la mejor."""
    dev = dev or dispositivo()
    inicio_corrida = time.perf_counter()
    e = cfg.entrenamiento
    clases = len(datos.clases)
    modelo, optimizador, programador, criterio, lotes, azar_aumento = _preparar(cfg, clases, semilla, dev)
    y_validacion = datos.y_validacion.numpy()

    mejor_clave, mejor_epoca, mejor_estado, mejores_metricas, mejor_por_clase = None, None, None, None, None
    for epoca in range(1, e.epocas + 1):
        pasada = _una_epoca(modelo, optimizador, criterio, lotes, azar_aumento, datos.x_entrenamiento,
                            datos.y_entrenamiento, cfg, dev)
        if pasada is None:
            return ResultadoCorrida(
                EstadoCorrida.FALLIDA, duracion_s=time.perf_counter() - inicio_corrida,
                motivo=f"pérdida no finita en la época {epoca}",
            )
        if programador is not None:
            programador.step()

        probabilidades, perdida_validacion = _evaluar(modelo, datos.x_validacion, datos.y_validacion, dev)
        validacion = calcular(probabilidades, y_validacion, clases)
        al_terminar_epoca(epoca, {
            "entrenamiento.perdida": pasada[0],
            "entrenamiento.top1": pasada[1],
            "validacion.perdida": perdida_validacion,
            **{f"validacion.{k}": v for k, v in validacion.items()},
        })
        clave = (validacion["top1"], validacion["f1_macro"])
        if mejor_clave is None or clave > mejor_clave:
            mejor_clave, mejor_epoca = clave, epoca
            mejor_estado = _copia(modelo)
            mejores_metricas = {**validacion, "perdida": perdida_validacion}
            mejor_por_clase = top1_por_clase(probabilidades, y_validacion, clases)

    elegido = KPoseFormer(cfg.forma, clases=clases, dropout=e.dropout)
    elegido.load_state_dict(mejor_estado)
    return ResultadoCorrida(
        EstadoCorrida.TERMINADA, mejor_epoca, mejores_metricas, mejor_por_clase, elegido.eval(),
        time.perf_counter() - inicio_corrida,
    )


def entrenar_final(
    datos: DatosFinal,
    cfg: Configuracion,
    semilla: int,
    epocas: int,
    al_terminar_epoca: Callable[[int, dict[str, float]], None],
    dev: torch.device | None = None,
) -> ResultadoCorrida:
    """El modelo final (TT-22, research R-06): los dos pliegues, la programación de toda la configuración
    (coseno con `T_max = epocas` de la configuración) y se detiene al terminar la época `epocas`, sin
    validar. Esos pesos son el candidato."""
    e = cfg.entrenamiento
    if not 1 <= epocas <= e.epocas:
        raise ValueError(f"El modelo final se entrena entre 1 y {e.epocas} épocas (las de la configuración), no {epocas}.")
    dev = dev or dispositivo()
    inicio_corrida = time.perf_counter()
    clases = len(datos.clases)
    modelo, optimizador, programador, criterio, lotes, azar_aumento = _preparar(cfg, clases, semilla, dev)
    for epoca in range(1, epocas + 1):
        pasada = _una_epoca(modelo, optimizador, criterio, lotes, azar_aumento, datos.x_entrenamiento,
                            datos.y_entrenamiento, cfg, dev)
        if pasada is None:
            return ResultadoCorrida(
                EstadoCorrida.FALLIDA, duracion_s=time.perf_counter() - inicio_corrida,
                motivo=f"pérdida no finita en la época {epoca}",
            )
        if programador is not None:
            programador.step()
        al_terminar_epoca(epoca, {"entrenamiento.perdida": pasada[0], "entrenamiento.top1": pasada[1]})
    final = KPoseFormer(cfg.forma, clases=clases, dropout=e.dropout)
    final.load_state_dict(_copia(modelo))
    return ResultadoCorrida(
        EstadoCorrida.TERMINADA, epocas, None, None, final.eval(), time.perf_counter() - inicio_corrida,
    )

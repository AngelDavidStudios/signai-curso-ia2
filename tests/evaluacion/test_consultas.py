"""Número de consulta a la prueba (spec FR-005, SC-003; research R-05; casos límite de repetición)."""

from signai_ml.aceptacion.consultas import EstadoConsulta, EvaluacionRegistrada, Siguiente, siguiente, total

A, B = "a" * 64, "b" * 64


def _evaluacion(consulta: int, huella: str, estado: EstadoConsulta, run_id: str = "r") -> EvaluacionRegistrada:
    return EvaluacionRegistrada(run_id=run_id, consulta=consulta, modelo_sha256=huella, estado=estado)


def test_la_primera_consulta_es_la_1() -> None:
    assert siguiente([], A) == Siguiente(numero=1, terminada=None, reutiliza=False)
    assert total([]) == 0


def test_una_consulta_terminada_no_se_repite() -> None:
    terminada = _evaluacion(1, A, EstadoConsulta.TERMINADA, "r1")

    resultado = siguiente([terminada], A)

    assert resultado.terminada == terminada
    assert resultado.numero == 1


def test_una_interrumpida_reutiliza_su_numero() -> None:
    evaluaciones = [_evaluacion(1, A, EstadoConsulta.INTERRUMPIDA)]

    assert siguiente(evaluaciones, A) == Siguiente(numero=1, terminada=None, reutiliza=True)


def test_una_fallida_reutiliza_su_numero() -> None:
    evaluaciones = [_evaluacion(1, B, EstadoConsulta.TERMINADA), _evaluacion(2, A, EstadoConsulta.FALLIDA)]

    assert siguiente(evaluaciones, A) == Siguiente(numero=2, terminada=None, reutiliza=True)


def test_una_en_curso_abandonada_reutiliza_su_numero() -> None:
    assert siguiente([_evaluacion(1, A, EstadoConsulta.EN_CURSO)], A).reutiliza


def test_otro_candidato_es_la_consulta_siguiente() -> None:
    evaluaciones = [_evaluacion(1, A, EstadoConsulta.TERMINADA)]

    assert siguiente(evaluaciones, B) == Siguiente(numero=2, terminada=None, reutiliza=False)
    assert total(evaluaciones) == 1


def test_la_terminada_gana_a_los_intentos_fallidos_del_mismo_candidato() -> None:
    terminada = _evaluacion(1, A, EstadoConsulta.TERMINADA, "r2")
    evaluaciones = [_evaluacion(1, A, EstadoConsulta.INTERRUMPIDA, "r1"), terminada]

    assert siguiente(evaluaciones, A).terminada == terminada


def test_el_total_es_la_mayor_consulta() -> None:
    evaluaciones = [
        _evaluacion(1, A, EstadoConsulta.TERMINADA),
        _evaluacion(2, B, EstadoConsulta.INTERRUMPIDA),
        _evaluacion(2, B, EstadoConsulta.TERMINADA),
    ]

    assert total(evaluaciones) == 2

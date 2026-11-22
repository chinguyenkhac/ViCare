from vicare_safe.safety import (DOSE_POS, EMERG_NEG, EMERG_POS, is_dose_request, is_emergency,
                                is_emergency_baseline)
from vicare_safe.text import fold


def test_emergency_v2_recall_accented_and_unaccented():
    assert all(is_emergency(q) for q in EMERG_POS)
    assert all(is_emergency(fold(q)) for q in EMERG_POS)


def test_emergency_v2_no_false_alarm():
    assert not any(is_emergency(q) for q in EMERG_NEG)
    assert not any(is_emergency(fold(q)) for q in EMERG_NEG)   # 'ngạt mũi', 'từ từ' không bị nhầm


def test_baseline_misses_unaccented():
    assert not any(is_emergency_baseline(fold(q)) for q in EMERG_POS)


def test_dose():
    assert all(is_dose_request(q) for q in DOSE_POS)
    assert not is_dose_request("Liệu tôi có bị viêm gan không?")   # 'liệu' ≠ 'liều'

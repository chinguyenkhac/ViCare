from vicare_safe.text import fold, has_diacritics, normalize_teencode, split_sentences, to_teencode


def test_fold():
    assert fold("Khó Thở, Đau ngực") == "kho tho, dau nguc"


def test_has_diacritics():
    assert has_diacritics("bị sốt")
    assert not has_diacritics("bi sot cao")


def test_teencode_basic():
    out, changed = normalize_teencode("bs ơi con e bị sốt có sao ko, uống thuốc j dc z")
    assert changed
    assert out == "bác sĩ ơi con e bị sốt có sao không, uống thuốc gì được vậy"


def test_teencode_guards_medical_letters():
    for text in ["Thiếu vitamin K ở trẻ sơ sinh", "Nhóm máu R có sao không", "Viêm gan C có lây không"]:
        out, _ = normalize_teencode(text)
        assert out == text
    out, _ = normalize_teencode("AMH 8.59 ng/mL có cao không")      # 'ng' là đơn vị, không phải 'người'
    assert "ng/mL" in out


def test_teencode_roundtrip():
    s = "Bác sĩ cho tôi hỏi uống thuốc này có được không vậy"
    t = to_teencode(s)
    assert "bs" in t and "ko" in t and "dc" in t
    assert normalize_teencode(t)[0].lower() == s.lower()


def test_split_sentences_fixes_glued_period():
    sents = split_sentences("Đây là bệnh mạn tính.Em cần đưa bố đến gặp bác sĩ chuyên khoa Hô hấp để được hướng dẫn.")
    assert len(sents) == 2 and sents[1].startswith("Em cần")

from vicare_safe.diacritics import DiacriticRestorer, needs_restoration, syllable_accuracy

TRAIN = [
    "Bé bị sốt cao phải làm sao", "Con tôi bị sốt cao co giật", "Tôi bị đau bụng bên phải",
    "Bà bầu uống thuốc hạ sốt được không", "Viêm gan B có lây không", "Đau bụng có phải viêm ruột thừa không",
    "Bàn ghế trong phòng", "Bạn nên đi khám", "Tiêm vaccine Covid được không",
] * 3


def test_restore_simple():
    r = DiacriticRestorer().fit(TRAIN)
    assert r.restore("con toi bi sot cao co giat") == "con tôi bị sốt cao co giật"
    assert r.restore("ba bau uong thuoc ha sot duoc khong") == "bà bầu uống thuốc hạ sốt được không"


def test_keeps_user_diacritics_and_case():
    r = DiacriticRestorer().fit(TRAIN)
    assert r.restore("Bạn nen di kham") == "Bạn nên đi khám"
    assert r.restore("Viem gan B co lay khong?") == "Viêm gan B có lây không?"


def test_unknown_tokens_untouched():
    r = DiacriticRestorer().fit(TRAIN)
    assert r.restore("xyz 123 abc") == "xyz 123 abc"


def test_needs_restoration():
    assert needs_restoration("bi dau bung")
    assert not needs_restoration("bị đau bụng")
    assert not needs_restoration("123")


def test_accuracy_metric():
    r = DiacriticRestorer().fit(TRAIN)
    acc = syllable_accuracy(r, ["Con tôi bị sốt cao co giật"])
    assert acc["syllable_acc"] == 1.0 and acc["sentence_exact"] == 1.0


def test_bigram_and_trigram_agree_on_easy_sentence():
    for order in (2, 3):
        r = DiacriticRestorer(order=order).fit(TRAIN)
        assert r.restore("con toi bi sot cao co giat") == "con tôi bị sốt cao co giật"

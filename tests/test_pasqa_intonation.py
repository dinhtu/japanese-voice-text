from app.services.pasqa_intonation import _VENDOR_SRC, morae_to_katakana, score_pasqa


def test_morae_to_katakana():
    assert morae_to_katakana(["あ", "し", "た"]) == ["ア", "シ", "タ"]


def test_score_pasqa_none_without_checkpoint(tmp_path):
    assert score_pasqa(tmp_path / "x.wav", ["あ"], None) is None
    assert score_pasqa(tmp_path / "x.wav", ["あ"], tmp_path / "missing.pkl") is None


def test_vendor_package_is_present():
    assert (_VENDOR_SRC / "pasqa" / "predictor.py").is_file()
    assert (_VENDOR_SRC / "pasqa" / "vocab.txt").is_file()

from tinytransformer.data import CharVocab


def test_vocab_roundtrip(tmp_path):
    v = CharVocab("hello\nworld\n")
    p = tmp_path / "vocab.txt"
    v.save(p)
    v2 = CharVocab.load(p)
    assert v.stoi == v2.stoi
    assert v2.decode(v2.encode("hello")) == "hello"


def test_encode_unknown_char_raises_valueerror():
    v = CharVocab("abc\n")
    try:
        v.encode("z")
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-vocab char")

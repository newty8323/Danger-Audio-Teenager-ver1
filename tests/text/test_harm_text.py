from text.harm_text import load_lexicon, score_text, vocabulary_prompt


def test_vocabulary_prompt_contains_terms():
    """Intent: every category is represented, in BOTH scripts, within the token budget.

    (Previously pinned to "full house"; the lists are Korean-first/English-last, so growing
    the Korean side pushed specific English terms out of any fixed-size sample. The sampling
    guarantee is coverage, not a particular term.)
    """
    p = vocabulary_prompt()
    assert "제삿날" in p and "필로폰" in p                    # Korean, from two categories
    assert any(w in p for w in ("kill you", "casino", "slot machine", "cocaine"))  # English
    lex = load_lexicon()
    for cat, spec in lex["categories"].items():
        assert any(t["t"] in p for t in spec["terms"]), cat
    assert len(p) <= 1000


def test_gambling_korean():
    r = score_text("와 풀하우스 떴다")
    assert r.top_category == "gambling" and r.text_risk > 0.5


def test_threat_korean_explicit():
    r = score_text("너는 내가 기필코 죽인다")
    assert r.top_category == "threat" and r.text_risk >= 0.9


def test_threat_korean_idiom():
    r = score_text("너는 오늘이 제삿날이다")
    assert r.top_category == "threat" and "제삿날" in r.matched["threat"]


def test_threat_english():
    r = score_text("I swear I will kill you tonight")
    assert r.top_category == "threat" and r.text_risk >= 0.9


def test_gambling_english_case_insensitive():
    r = score_text("Dude I hit the JACKPOT with a Full House")
    assert r.top_category == "gambling"
    assert "jackpot" in r.matched["gambling"] and "full house" in r.matched["gambling"]


def test_sexual_category():
    assert score_text("send nudes right now").top_category == "sexual"
    assert score_text("우리 야동 보자").top_category == "sexual"


def test_drug_category():
    assert score_text("코카인이랑 필로폰 팔아").top_category == "drug"
    assert score_text("let's go buy drugs and get high").top_category == "drug"


def test_safe_text_is_zero():
    assert score_text("오늘 날씨가 좋네요 산책 갈까요").text_risk == 0.0
    assert score_text("Let's meet for coffee tomorrow").text_risk == 0.0


def test_empty_text():
    assert score_text("").text_risk == 0.0


def test_multiple_categories():
    r = score_text("카지노에서 올인했는데 너 죽여버릴거야")
    assert {"gambling", "threat"} <= set(r.categories)


def test_shipped_lexicon_has_expected_categories():
    lex = load_lexicon()
    assert set(lex["categories"]) == {"abuse", "threat", "gambling", "sexual", "drug"}


def test_explicit_abuse_without_ambiguous_bare_term():
    assert score_text("우리 형님이 거짓할게 이 새끼야!").text_risk > 0.9
    assert score_text("개새끼야 당장 나와").text_risk > 0.9
    assert score_text("강아지 새끼가 태어났어요").text_risk == 0.0

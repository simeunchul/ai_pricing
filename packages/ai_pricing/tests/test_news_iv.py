from ai_pricing.news_iv.classify import classify_rule
from ai_pricing.news_iv.iv_shift import adjust_iv, dominant_event, IV_SHIFT_RULES


def test_rule_classifier():
    assert classify_rule("삼성전자 어닝쇼크 실적 부진") == "earnings_miss"
    assert classify_rule("금리 인상 결정") == "macro_shock"
    assert classify_rule("normal quarterly filing") == "neutral"
    assert classify_rule("SK하이닉스 M&A 추진") == "mna"


def test_iv_shift_bounds():
    assert adjust_iv(0.20, "earnings_miss", 1.0) == 0.20 + IV_SHIFT_RULES["earnings_miss"]
    assert 0.01 <= adjust_iv(0.20, "macro_shock", 0.5) <= 2.0


def test_dominant_event_picks_largest():
    evs = [("neutral", 0.9), ("earnings_miss", 0.5), ("macro_shock", 0.3)]
    ev, _ = dominant_event(evs)
    # macro_shock (0.05*0.3=0.015) vs earnings_miss (0.03*0.5=0.015) tie — either ok,
    # but neutral (0.0) must NOT win.
    assert ev != "neutral"


def test_adjust_iv_clips_negative():
    assert adjust_iv(0.02, "earnings_beat", 10.0) >= 0.01

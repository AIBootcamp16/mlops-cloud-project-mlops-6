from src.reco.metrics import ndcg_at_k, region_match_at_k

def test_ndcg_simple():
    gains = [2,1,0,0,0]
    val = ndcg_at_k(gains, 5)
    assert 0.0 < val <= 1.0

def test_region_match():
    countries = ["France","Italy","USA","Spain","France"]
    preferred = {"France","Italy"}
    assert abs(region_match_at_k(countries, preferred, 5) - 0.6) < 1e-9

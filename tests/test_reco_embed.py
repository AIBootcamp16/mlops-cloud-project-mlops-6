from src.reco.embed import fit_tfidf, transform

def test_fit_and_transform():
    corpus = ["merlot napa valley", "cabernet france bordeaux", "prosecco italy veneto"]
    vec, X = fit_tfidf(corpus, ngram=(1,2), min_df=1)
    assert X.shape[0] == 3
    q = transform(vec, ["merlot napa"])
    assert q.shape[0] == 1
    assert q.nnz > 0

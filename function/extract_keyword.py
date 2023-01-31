import yake

def extract_keyword(captions, deduplication_threshold = 0.9, max_ngram_size=3, num_keywords=20):
    language = "en"
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=num_keywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(captions)
    keywords = [keyword[0] for keyword in keywords]
    return keywords
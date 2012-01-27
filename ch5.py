import nltk
from nltk.corpus import treebank, brown

def preceds_past_participle(corpus=brown):
	"""find all word-tag pairs that preceds a past participle (VN)"""
	words = corpus.tagged_words(simplify_tags=True)
	bigrams = nltk.bigrams(words)
	return nltk.ConditionalFreqDist((word2[0], word1) for (word1, word2) in bigrams if word2[1] == 'VN')
	
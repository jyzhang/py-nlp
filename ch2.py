import nltk
from nltk.corpus import brown, names, state_union, cmudict, stopwords, wordnet as wn

days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
def days_of_week_in_brown():
	cfd = nltk.ConditionalFreqDist((genre, day.lower()) 
		for genre in brown.categories()
		for day in brown.words(categories=genre) if day.lower() in days)
	return cfd 
	
def state_union_ts(word_list):
	cfd = nltk.ConditionalFreqDist((word.lower(), fileid[:4]) 
		for fileid in state_union.fileids()
		for word in state_union.words(fileid) if word.lower() in word_list)
	return cfd
	
def gender_initials_plot():
	"""shows a plot of the distribution of first name initials of males and females"""
	cfd = nltk.ConditionalFreqDist((gender, name[:1])
		for gender in names.fileids()
		for name in names.words(gender))
	return cfd.plot()
	
def cmu_multi_pronounciation():
	"""calculates the proportion of words in the cmu dictionary that has more than 1 pronounciation"""
	freq = nltk.FreqDist([entry[0] for entry in cmudict.entries()])
	multiples = freq.B() - freq.Nr(1) # total_types - 1_pronounciation
	return multiples * 1.0 / freq.B()
	
def noun_synsets_hyponyms():
	"""calculates the proportion of noun synsets that have no hyponyms"""
	freq = nltk.FreqDist([len(synset.hyponyms()) for synset in wn.all_synsets('n')])
	return freq[0] * 1.0 / freq.N()
	
def supergloss(synset):
	"""returns definition of a synset concated with all the definitions of its hyponyms and hypernyms"""
	definitions = [synset.definition]
	definitions += [hyponym.definition for hyponym in synset.hyponyms()]
	definitions += [hypernym.definition for hypernym in synset.hypernyms()]
	return '; '.join(definitions)
	
def brown_occurrences(n=3):
	"""returns a list of words that occurred n times in the brown corpus"""
	freq = nltk.FreqDist(brown.words())
	return [pair[0] for pair in freq.items() if pair[1] == n]
	
def brown_diversity():
	"""calculate and display lexical diversity score (token/token_type) for each brown corpus category"""
	cfd = nltk.ConditionalFreqDist((category, word)
		for category in brown.categories()
		for word in brown.words(categories=category))
	print "{0:15s} {1:10s}".format("CATEGORY", "DIVERSITY")
	for category in cfd.conditions():
		print "{0:15s} {1:10f}".format(category, (cfd[category].N() * 1.0 / cfd[category].B()))

def frequent_bigrams(text, num=50):
	"""returns the 50 most common bigrams in a text that don't contain stopwords"""
	sw = stopwords.words()
	sw += ['.', ',', '!', ';', '"', "'", '``', '?', "''", '--', '(', ')', ":"] # add punctuations to stop words
	bigrams = []
	for bigram in nltk.bigrams(text):
		use = True
		for word in bigram:
			if word in sw:
				use = False
				break
		if use:
			bigrams.append(bigram)
	freq = nltk.FreqDist(bigrams)
	for item in freq.items()[:num]:
		print "{0:25s} {1:4d}".format(" ".join(item[0]), item[1])
		
def polysemy_analysis():
	"""returns the average polysemy (number of senses) for nouns, verbs, adjectives, and adverbs in wordnet"""
	conditions = (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV)
	cfd = nltk.ConditionalFreqDist((pos, len(wn.synsets(lemma_name, pos)))
		for pos in conditions
		for lemma_name in wn.all_lemma_names(pos))
	for pos in cfd.conditions():
		print "{0:2s} {1:10f}".format(pos, sum([item[0] * item[1] for item in cfd[pos].items()]) * 1.0 / cfd[pos].N())

	
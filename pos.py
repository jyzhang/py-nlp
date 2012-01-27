# this is used to play and evaluate parts-of-speech taggers under various 
# training and testing conditions

import nltk
from random import randint
from math import log, exp
from numpy import zeros, empty, logaddexp
from time import time
from cPickle import dump, load
from progressbar import ProgressBar, Percentage, Bar, Timer
from nltk.corpus import brown
from nltk.tag import TaggerI, UnigramTagger, NgramTagger
from nltk.probability import ConditionalProbDist, MLEProbDist, ELEProbDist, DictionaryConditionalProbDist, DictionaryProbDist

TEST_PROPORTION = 0.10 # use 10% for testing, rest can be used for training
_NINF = float('-inf') # negative infinity

def training_sentences(use=1.0, categories=[]):
	"""returns a training sentence set: [[(word, tag), ..], [(word, tag), ..], ..]"""
	if len(categories) == 0:
		categories = brown.categories() # use all of the brown categories
	sents = []
	for category in categories:
		total = len(brown.tagged_sents(categories=category))
		max = int((1-TEST_PROPORTION) * use * total) - 1 # use the first n sentences for training
		sents += brown.tagged_sents(categories=category, simplify_tags=True)[0:max]
	return sents

def unlabel(sentences):
	return [[word for (word, tag) in sentence] for sentence in sentences]
	
def test_sentences(categories=[]):
	"""returns a test sentence set: [[(word, tag), ..], [(word, tag), ..], ..]"""
	if len(categories) == 0:
		categories = brown.categories() # use all of the brown categories
	sents = []
	for category in categories:
		total = len(brown.tagged_sents(categories=category))
		start = int(TEST_PROPORTION * total) # use the last k sentences for test
		sents += brown.tagged_sents(categories=category, simplify_tags=True)[-start:-1]
	return sents

def word_count(sentence_set):
	"""counts the total number of word tokens in the sentence set"""
	return sum(len(sentence) for sentence in sentence_set)

def log_sum(array):
	total = _NINF
	for num in array:
		total = logaddexp(num, total)
	return total

# Classes
	
class Trainer(object):
	"""Mixin used for training POS taggers"""
	
	def train(cls, examples, **kwargs):
		"""
		trains tagger using labelled examples (supervised training)
		:param examples: POS tagged sentence examples
		:type examples: list[list[tuple(word, tag)]]
		:return: a POS tagger trained on the examples
		:rtype: instance that subclasses nltk.tag.TaggerI
		"""
		raise NotImplementedError, "train(...) method must be implemented by subclass."
	
	def unsupervised_train(cls, examples, **kwargs):
		raise NotImplementedError, "unsupervised_train(...) method must be implemented by subclass."
	

class BaselineTrainer(Trainer):
	
	def train(self, examples, **kwargs):
		self.__tagger = UnigramTagger(examples)
		return self.__tagger
	
	def unsupervised_train(self, examples, **kwargs):
		return self.__tagger # we don't know how to train using unlabeled data
	

class SimpleNGramTrainer(Trainer):
		
	def __init__(self, **kwargs):
		self.__n = kwargs.get('n', 2)
		self.__cutoff = kwargs.get('cutoff', 0)
	
	def train(self, examples, **kwargs):
		prev = UnigramTagger(examples)
		for i in range(1, self.__n):
			self.__tagger = NgramTagger(i+1, train=examples, backoff=prev, cutoff=self.__cutoff)
			prev = self.__tagger
		return self.__tagger
	
	def unsupervised_train(self, examples, **kwargs):
		return self.__tagger # we don't know how to train using unlabeled data
	

class HmmModel(object):
	"""Default Hidden Markov Model for parts of speech tagging"""
	@classmethod
	def load(cls, filename):
		"""load core model data from pickled file"""
		pkl_file = open(filename, 'rb')
		data = load(pkl_file)
		return cls(data['tags'], data['vocab'], data['transitions'], data['emissions'])
	
	def __init__(self, tags, vocab, transitions, emissions):
		assert isinstance(tags, nltk.FreqDist), "tags must be an instance of nltk.FreqDist"
		self._tags = tags
		assert isinstance(vocab, nltk.FreqDist), "vocab must be an instance of nltk.FreqDist"
		self._vocab = vocab
		assert isinstance(transitions, nltk.ConditionalFreqDist), "transitions must be an instance of nltk.ConditionalFreqDist"
		self._transitions = transitions
		self.transitions = nltk.ConditionalProbDist(transitions, self._transition_smoothing)
		assert isinstance(emissions, nltk.ConditionalFreqDist), "emissions must be an instance of nltk.ConditionalFreqDist"
		self._emissions = emissions
		self.emissions = nltk.ConditionalProbDist(emissions, self._emission_smoothing)
		# initialize the log probability caches
		self.__t_cache = dict((tag, {}) for tag in self.transitions.conditions())
		self.__e_cache = dict((tag, {}) for tag in self.emissions.conditions())
		
	def __repr__(self):
		return "<{0:s}@{1:x} tags:{2:d} words:{3:d}>".format(self.__class__.__name__, id(self), self._tags.B(), self._vocab.N())
		
	def dump(self, filename):
		"""dumps core model information to a file"""
		data = {'tags': self._tags, 'vocab': self._vocab, 'transitions': self._transitions, 'emissions': self._emissions}
		pkl_file = open(filename, 'wb')
		dump(data, pkl_file)
	
	def tags(self):
		return self._tags.samples()
	
	def log_transition(self, prev, curr):
		if self.__t_cache[prev].has_key(curr):
			return self.__t_cache[prev][curr]
		else:
			p = self.transitions[prev].prob(curr)
			if p <= 0: return _NINF
			else:
				log_p = log(p)
				self.__t_cache[prev][curr] = log_p
				return log_p
	
	def log_emission(self, tag, word):
		if self.__e_cache[tag].has_key(word):
			return self.__e_cache[tag][word]
		else:
			p = self.emissions[tag].prob(word)
			if p <= 0: return _NINF
			else:
				log_p = log(p)
				self.__e_cache[tag][word] = log_p
				return log_p
	
	def _emission_smoothing(self, freqdist):
		return MLEProbDist(freqdist)
		
	def _transition_smoothing(self, freqdist):
		return MLEProbDist(freqdist)
	

class SmoothedHmmModel(HmmModel):
	"""Extension of the default HMM by adding smoothing some of the emission probabilities"""
	
	def _emission_smoothing(self, freqdist):
		ratio = freqdist.B() * 1.0 / self._vocab.B()
		if ratio > 0.01: 
			return ELEProbDist(freqdist)
		else:
			return MLEProbDist(freqdist)
	

class HmmTagger(TaggerI, Trainer):
	"""Hidden Markov Model POS Tagger"""
	
	def __init__(self, model_cls=SmoothedHmmModel, model=None):
		self.model = model # we start out with no model
		self.__model_cls = model_cls
		self.__runtime = 0
		self.__traintime = 0
		self.__tagged = 0
		self.__trained = 0
	
	def __repr__(self):
		return "<{0:s}@{1:x} model={2:s}>".format(self.__class__.__name__, id(self), str(self.model))
	
	def _alpha(self, sentence):
		"""calculates the forward probabilities"""
		assert isinstance(self.model, HmmModel), "invalid or untrained model"
		alpha = {0: {".": 0.0}} # we start at the beginning of a sentence
		for t in range(1, len(sentence)+1):
			word = sentence[t-1]
			alpha[t] = {}
			for curr in self.model.tags():
				total = _NINF
				for prev in self.model.tags():
					if alpha[t-1].has_key(prev):
						a = self.model.log_transition(prev, curr)
						if a == _NINF:
							continue
						b = self.model.log_emission(curr, word)
						if b == _NINF:
							continue
						prob = alpha[t-1][prev] + a + b
						total = logaddexp(prob, total) # calculate the log sum
				# print t, curr, total
				if total > _NINF:
					alpha[t][curr] = total
		return alpha
	
	def _beta(self, sentence):
		"""calculates the backward probabilities"""
		assert isinstance(self.model, HmmModel), "invalid or untrained model"
		beta = {len(sentence): dict((tag, 0.0) for tag in self.model.tags())}
		for t in range(len(sentence)-1, -1, -1):
			word = sentence[t]
			beta[t] = {}
			for prev in self.model.tags():
				total = _NINF
				for curr in self.model.tags():
					if beta[t+1].has_key(curr):
						a = self.model.log_transition(prev, curr)
						if a == _NINF:
							continue
						b = self.model.log_emission(curr, word)
						if b == _NINF:
							continue
						prob = beta[t+1][curr] + a + b
						total = logaddexp(prob, total) # calculate the log sum
				if total > _NINF:
					beta[t][prev] = total
		return beta
	
	def log_probability(self, sentence, use='alpha'):
		"""calculates the log probability of a sentence"""
		if use == 'alpha':
			return log_sum(self._alpha(sentence)[len(sentence)].values())
		elif use == 'beta':
			return self._beta(sentence)[0]['.']
		else:
			# this is very inefficient, we would never actually use this in practice
			t = randint(0, len(sentence)) # pick a random t
			alpha = self._alpha(sentence)[t]
			beta = self._beta(sentence)[t]
			values = [(at + bt) for (tag1, at) in alpha.items() for (tag2, bt) in beta.items() if tag1 == tag2]
			return log_sum(values)
	
	def train(self, examples, **kwargs):
		"""supervised training of this HMM, creates a maximum likelihood HMM model using the tagged examples"""
		start_t = time() # keep track of training time
		transition = nltk.ConditionalFreqDist()
		emission = nltk.ConditionalFreqDist()
		vocab = nltk.FreqDist()
		tags = nltk.FreqDist()
		self.__trained = 0
		for sentence in examples:
			prev = '.' # previous tag is always the end of the last sentence
			for (word, tag) in sentence:
				vocab.inc(word) # count this word in the vocab
				tags.inc(tag) # count this tag
				emission[tag].inc(word) # count the tag=>word emission
				transition[prev].inc(tag) # count the tag(i-1)=>tag(i) transition
				prev = tag # now current tag is prev for the next iteration
				self.__trained += 1
		self.model = self.__model_cls(tags, vocab, transition, emission) # create new model
		self.__traintime += (time() - start_t)
		return self 
	
	def unsupervised_train(self, examples, **kwargs):
		"""unsupervised training via the EM (forward-backward) algorithm using unlabeled examples"""
		assert isinstance(self.model, self.__model_cls), "a model must already exist, use train(..) to train a new model first."
		itr = 1
		prev_log_prob = _NINF
		max_iterations = kwargs.get('max', 10) # max number of iterations to run
		min_delta = kwargs.get('delta', len(examples) * 1) # minimum convergence
		dump = kwargs.get('dump', None) # dump each iteration into a model file
		stop = False
		while not stop:
			start_t = time() # keep track of training time
			log_prob = 0.0
			transition_counter = dict((tag, {}) for tag in self.model.tags()) # used to sum the log probabilities
			emission_counter = dict((tag, {}) for tag in self.model.tags()) 
			# show a progress bar for this long running task
			pbar = ProgressBar(widgets=["EM Itr " + str(itr) + ":", Percentage(), Bar(), Timer()], maxval=len(examples)).start()
			progress = 0
			for sentence in examples:
				progress += 1
				alpha = self._alpha(sentence)
				beta = self._beta(sentence)
				log_prob += beta[0]['.'] # calculate the total log probability
				# now we calculate the expected counts using the current model
				for t in range(0, len(sentence)):
					word = sentence[t]
					for prev in self.model.tags():
						if alpha[t].has_key(prev):
							for curr in self.model.tags():
								if beta[t+1].has_key(curr):
									a = self.model.log_transition(prev, curr)
									if a == _NINF:
										continue
									b = self.model.log_emission(curr, word)
									if b == _NINF:
										continue
									val = alpha[t][prev] + a + b + beta[t+1][curr] - beta[0]['.'] # prob of prev=>curr at time t
									# accumulate for transition: this is the expectation when summed of all t
									if transition_counter[prev].has_key(curr):
										transition_counter[prev][curr] = logaddexp(val, transition_counter[prev][curr])
									else:
										transition_counter[prev][curr] = val
									# accumulate for emission: this is the expectation when summed over all t
									if emission_counter[curr].has_key(word):
										emission_counter[curr][word] = logaddexp(val, emission_counter[curr][word])
									else:
										emission_counter[curr][word] = val
				pbar.update(progress) # update progress bar
			pbar.finish()
			# these are part of the new model that we create from the maximized likelihood (counts)
			tags = nltk.FreqDist()
			vocab = nltk.FreqDist()
			transition = nltk.ConditionalFreqDist()
			emission = nltk.ConditionalFreqDist()
			# we have all the counts, now we convert them to probability distributions
			for (prev, counter) in transition_counter.items():
				for (curr, val) in counter.items():
					count = exp(val) # take val out of log space
					transition[prev].inc(curr, count) 
					tags.inc(curr, count)
			for (curr, counter) in emission_counter.items():
				for (word, val) in counter.items():
					count = exp(val) # take val out of log space
					emission[curr].inc(word, count)
					vocab.inc(word, count)
			# maximum iteration reached, we stop
			if itr >= max_iterations: 
				stop = True 
			else:
				itr += 1
			# convergence criteria reached, we stop
			if (log_prob - prev_log_prob) < min_delta: 
				stop = True
			else: 
				prev_log_prob = log_prob
			if stop:
				# create new model based on these counts and use this new model
				self.model = self.__model_cls(tags, vocab, transition, emission)
			else:
				# we don't use the smoothed model, because there will be no unknown words in subsequent iterations
				# this will let us converge faster (I think...)
				self.model = HmmModel(tags, vocab, transition, emission)
			# dump the models out if requested
			if dump != None: self.model.dump('hmm_models/' + dump + "_" + str(itr-1) + ".pkl")
			print "log probability: ", log_prob
			self.__traintime += (time() - start_t)
		return self						
	
	def tag(self, sentence):
		"""use viterbi algorithm to determine the most likely tag sequence for a sentence"""
		assert isinstance(self.model, HmmModel), "invalid or untrained model"
		start_t = time() # used to measure tag performance
		sigma = {-1: {'.': 0.0}} # initial conditions
		backt = {} # backtrace
		for t in range(len(sentence)):
			word = sentence[t]
			sigma[t] = {}
			backt[t] = {}
			for curr in self.model.tags():
				curr_max = _NINF # keep a running max
				curr_argmax = -1 # keep a running argmax
				for prev in self.model.tags():
					if sigma[t-1].has_key(prev):
						a = self.model.log_transition(prev, curr) # aij
						if a == _NINF:
							continue # optimization - no need to calculate
						b = self.model.log_emission(curr, word) # bjo
						if b == _NINF:
							continue # optimization - no need to calculate
						val = a + b + sigma[t-1][prev]
						if val > curr_max:
							curr_max = val
							curr_argmax = prev
				if curr_max > _NINF:
					sigma[t][curr] = curr_max
					backt[t][curr] = curr_argmax
			# print t, sigma[t], backt[t]
		last = sigma[len(sentence)-1]
		log_prob = max(last) # the log probability of the most likely tag sequence
		prev = max(last, key=last.get) 
		decoded = [prev] # use backtrace to find the most likely tag sequence
		for j in range(len(sentence)-1, 0, -1):
			prev = backt[j][prev]
			decoded.insert(0, prev) # add to front
		self.__runtime += (time() - start_t)
		self.__tagged += len(decoded)
		return zip(sentence, decoded) # return [(word, tag), ...]
	
	def stats(self):
		"""prints performance statistics for this tagger"""
		print "-"*33
		if self.__trained > 0:
			ttime_per_tag = self.__traintime / self.__trained
		else:
			ttime_per_tag = 0.0
		print "|{0:20s}|{1:10f}|".format("training per tag", ttime_per_tag)
		if self.__tagged > 0:
			rtime_per_tag = self.__runtime / self.__tagged
		else:
			rtime_per_tag = 0.0
		print "|{0:20s}|{1:10f}|".format("runtime per tag", rtime_per_tag)
		print "-"*33
	

class TaggerEvaluator(object):
	
	def __init__(self):
		self.__taggers = [] # cache the taggers that gets created
		self.__results = [] # the evalutation results
		self.__supervised_regimen = [0.2, 0.4, 0.6, 0.8, 1.0]
		
	def run(self, trainer):
		tagger = trainer(training_sentences) 
		

class POSTester(object):
	
	def __init__(self, tagger_cls):
		self.__tagger_cls = tagger_cls
		self.__regimen = [0.2, 0.4, 0.6, 0.8, 1.0] # different training regimen sizes
		self.__categories = [] # use all categories
	
	def set_training_program(self, regimen, categories=[]):
		self.__regimen = regimen
		self.__categories = categories
	
	def set_test_sentences(self, test_sentences):
		self.__test = test_sentences
	
	def run(self):
		print "%8s | %10s" % ("words", "accuracy")
		print "%8s | %10s" % ('-' * 8, '-' * 10)
		for proportion in self._regimen:
			train = training_sentences(proportion, self._categories)
			tagger = self._tagger_cls(train)
			val = tagger.evaluate(self._test)
			words = set_word_count(train)
			print "%8i | %10f" % (words, val) 
	
	
		
	

		
		
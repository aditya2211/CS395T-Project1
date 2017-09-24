# models.py

from nerdata import *
from utils import *

import numpy as np
import pickle as pi


# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]
    



class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        scorematrix = np.ones((len(sentence),len(self.tag_indexer)),dtype=float)
        backtracking = np.ones((len(sentence),len(self.tag_indexer)),dtype=float)
        cnt=1
        curr = sentence.tokens[0].word
        if self.word_indexer.contains(curr)==0:
            curr = "UNK"
        for tidx1 in range(len(self.tag_indexer)):
            scorematrix[0][tidx1] = self.init_log_probs[tidx1] + self.emission_log_probs[tidx1][(self.word_indexer).index_of(curr)]
        for tok in sentence.tokens[1:]:
            curr = tok.word
            if self.word_indexer.contains(tok.word)==0:
                curr = "UNK"
            for tidx1 in range(0,len(self.tag_indexer)):
                maxval = scorematrix[cnt-1][0] + self.transition_log_probs[0][tidx1]
                maxidx = 0;
                for tidx2 in range(1,len(self.tag_indexer)):
                    if(scorematrix[cnt-1][tidx2] + self.transition_log_probs[tidx2][tidx1] > maxval):
                        maxidx = tidx2
                        maxval = scorematrix[cnt-1][tidx2] + self.transition_log_probs[tidx2][tidx1]
                scorematrix[cnt][tidx1] = maxval + self.emission_log_probs[tidx1][self.word_indexer.index_of(curr)]
                backtracking[cnt][tidx1] = maxidx
            cnt+=1



        maxval = scorematrix[cnt-1][0]
        maxidx = 0

        for tidx2 in range(1,len(self.tag_indexer)):
            if(scorematrix[cnt-1][tidx2] > maxval):
                maxidx = tidx2
                maxval = scorematrix[cnt-1][tidx2]

        pred_tags  =np.ones(len(sentence))

        pred_tags[len(sentence)-1] = maxidx    

        for n in range(len(sentence)-1,0,-1):
            pred_tags[n-1] = backtracking[n][int(pred_tags[n])]


        final_pred=[]

        for n in range(len(sentence)):
            final_pred.append(self.tag_indexer.get_object(pred_tags[n]))               


        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(final_pred))
        


        
      







        #raise Exception("IMPLEMENT ME")


# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print repr(init_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print "Tag indexer: " + repr(tag_indexer)
    print "Initial state log probabilities: " + repr(init_counts)
    print "Transition log probabilities: " + repr(transition_counts)
    print "Emission log probs too big to print..."
    print "Emission log probs for India: " + repr(emission_counts[:,word_indexer.get_index("India")])
    print "Emission log probs for Phil: " + repr(emission_counts[:,word_indexer.get_index("Phil")])
    print "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)"
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights,transitionvector):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transitionvector = transitionvector

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    
    
        
        
    
    def decode(self, sentence):
        te = self.transitionvector
        tlen = len(self.tag_indexer)
        scorematrix = np.ones((len(sentence),len(self.tag_indexer)),dtype=float)
        backtracking = np.ones((len(sentence),len(self.tag_indexer)),dtype=float)
        cnt=1
        feature_cache = [[[] for k in xrange(0, len(self.tag_indexer))] for j in xrange(0, len(sentence))]
        for word_idx in xrange(0, len(sentence)):
            for tag_idx in xrange(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)
                #print(feature_cache[word_idx][tag_idx])
        #curr = sentence.tokens[0].word
        maxval = -1*float("inf")
        for i in xrange(0,len(sentence)):
            for j in xrange(0,len(self.tag_indexer)):
                scorematrix[i][j] = maxval
        
        
        for tidx1 in range(len(self.tag_indexer)):
            if(isB(self.tag_indexer.get_object(tidx1)) or isO(self.tag_indexer.get_object(tidx1))):
                scorematrix[0][tidx1] = score_indexed_features(feature_cache[0][tidx1],self.feature_weights)
            
        for tok in xrange(1,len(sentence)):
            #curr = tok.word
            for tidx1 in range(0,len(self.tag_indexer)):
                maxval = -1*float("inf")
                maxidx = -1;
                for tidx2 in range(0,len(self.tag_indexer)):
                    
                    if(isI(self.tag_indexer.get_object(tidx1))):
                        if(isO(self.tag_indexer.get_object(tidx2))):
                            continue
                        if(isB(self.tag_indexer.get_object(tidx2))):
                            curr = get_tag_label(self.tag_indexer.get_object(tidx2))
                            prev = get_tag_label(self.tag_indexer.get_object(tidx1))
                            if prev!=curr:
                                continue
                        if(isI(self.tag_indexer.get_object(tidx2))):
                            curr = get_tag_label(self.tag_indexer.get_object(tidx2))
                            prev = get_tag_label(self.tag_indexer.get_object(tidx1))
                            if prev!=curr:
                                continue
                           
                           
                    if(scorematrix[cnt-1][tidx2]+score_indexed_features([tidx1*tlen+tidx2],te) > maxval):
                        maxidx = tidx2
                        maxval = scorematrix[cnt-1][tidx2]+score_indexed_features([tidx1*tlen+tidx2],te)
                scorematrix[cnt][tidx1] = maxval + score_indexed_features(feature_cache[tok][tidx1],self.feature_weights)
                backtracking[cnt][tidx1] = maxidx
            cnt+=1



        maxval = scorematrix[cnt-1][0]
        maxidx = 0

        for tidx2 in range(1,len(self.tag_indexer)):
            if(scorematrix[cnt-1][tidx2] > maxval):
                maxidx = tidx2
                maxval = scorematrix[cnt-1][tidx2]

        pred_tags  =np.ones(len(sentence))

        pred_tags[len(sentence)-1] = maxidx    

        for n in range(len(sentence)-1,0,-1):
            pred_tags[n-1] = backtracking[n][int(pred_tags[n])]


        final_pred=[]

        for n in range(len(sentence)):
            final_pred.append(self.tag_indexer.get_object(pred_tags[n]))               

        
        
        
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(final_pred))
        
        
        
        
        #raise Exception("IMPLEMENT ME")
        

def forwardbackwardZ(senlength,taglength,feature_cache,fe,te):
    alpha = np.zeros((senlength,taglength),dtype=float)
    
    
    
    
    
    
    
    
    for i in xrange(0,taglength):
        alpha[0][i] = score_indexed_features(feature_cache[0][i],fe)
        
    
    for i in xrange(1,senlength):
        for j in xrange(0,taglength):
            
            alpha[i][j] = alpha[i-1][0]+score_indexed_features([j*taglength],te)
            for k in xrange(1,taglength):
                alpha[i][j]=np.logaddexp(alpha[i-1][k]+score_indexed_features([j*taglength+k],te),alpha[i][j])
            
            alpha[i][j]+= score_indexed_features(feature_cache[i][j],fe)
    Z = 0  
    Z = alpha[senlength-1][0]
    
    for i in xrange(1,taglength):
        
        Z=np.logaddexp(Z,alpha[senlength-1][i])
        

    


    beta = np.zeros((senlength,taglength),dtype=float)
    
    for i in xrange(0,taglength):
        beta[senlength-1][i] = 0
    
    for i in xrange(senlength-2,-1,-1):
        for j in xrange(0,taglength):
            beta[i][j] = beta[i+1][0] + score_indexed_features(feature_cache[i+1][0],fe) + score_indexed_features([j],te)
            for k in xrange(1,taglength):
                beta[i][j]=np.logaddexp(beta[i+1][k] + score_indexed_features(feature_cache[i+1][k],fe) + score_indexed_features([k*taglength+j],te),beta[i][j])
            
  
    
    return alpha,beta,Z







        
        
 


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    
    
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print "Extracting features"
    feature_indexer = Indexer()
    tagtrans = np.zeros((len(tag_indexer),len(tag_indexer)),dtype=float)
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]

    for sentence_idx in xrange(0, len(sentences)):
        tags = sentences[sentence_idx].get_bio_tags()
        
        
        if sentence_idx % 100 == 0:
            print "Ex " + repr(sentence_idx) + "/" + repr(len(sentences))
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            
            if word_idx>0:
                tagtrans[tag_indexer.get_index(tags[word_idx-1])][tag_indexer.get_index(tags[word_idx])]+=1
            for tag_idx in xrange(0, len(tag_indexer)):
                
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
                
    
    feature_size = len(feature_cache[0][0][0])
    weightvector = np.ones((len(feature_indexer)),dtype = float)
    transtionvector = np.ones((len(tag_indexer)*len(tag_indexer)),dtype=float)
    
    
    tlen = len(tag_indexer)
    for tidx1 in xrange(0,tlen):
        for tidx2 in xrange(0,tlen):
            if(isI(tag_indexer.get_object(tidx1))):
                        if(isO(tag_indexer.get_object(tidx2))):
                            transtionvector[tlen*tidx1+tidx2] = -1*float("inf")
                        if(isB(tag_indexer.get_object(tidx2))):
                            curr = get_tag_label(tag_indexer.get_object(tidx2))
                            prev = get_tag_label(tag_indexer.get_object(tidx1))
                            if prev!=curr:
                                transtionvector[tlen*tidx1+tidx2] = -1*float("inf")
                        if(isI(tag_indexer.get_object(tidx2))):
                            curr = get_tag_label(tag_indexer.get_object(tidx2))
                            prev = get_tag_label(tag_indexer.get_object(tidx1))
                            if prev!=curr:
                                transtionvector[tlen*tidx1+tidx2] = -1*float("inf")
            
            if(tagtrans[tidx2][tidx1]==0):
                transtionvector[tlen*tidx1+tidx2] = -1*float("inf")
            else:
                transtionvector[tlen*tidx1+tidx2] = np.log(tagtrans[tidx2][tidx1])
    
    #transtionvector = np.ones((len(tag_indexer)*len(tag_indexer)),dtype=float)
    
    #forwardbackward(len(sentences[0]),len(tag_indexer),feature_cache[0],weightvector)

    epochs=10
    alpha2 =.1
    alpha=.1
    flag = 1
    for i  in xrange(0,epochs):

        for sentence_idx in xrange(0,len(sentences)):
           
            te = np.ones((len(tag_indexer)*len(tag_indexer)),dtype=float)
            we = np.ones((len(feature_indexer)),dtype = float)
            te = np.copy(transtionvector)
          
            we = np.copy(weightvector)
          
            
            a,b,Z = forwardbackwardZ(len(sentences[sentence_idx]),len(tag_indexer),feature_cache[sentence_idx],weightvector,transtionvector)
            #print(te)
            
            #print(a,b,Z)
            goldval = sentences[sentence_idx].get_bio_tags()
            for word_idx in xrange(0,len(sentences[sentence_idx])):
                #print("Here it goes")
                #print(we)
                
                for widx in feature_cache[sentence_idx][word_idx][tag_indexer.get_index(goldval[word_idx])]:
                    #goldtag = tag_indexer.get_index(goldval[word_idx])
                    weightvector[widx] = weightvector[widx] + alpha
                for possibletags in xrange(0,len(tag_indexer)):
                    for widx in feature_cache[sentence_idx][word_idx][possibletags]:
                        send = a[word_idx][possibletags]+b[word_idx][possibletags]-Z
                        weightvector[widx] = weightvector[widx] + alpha*(-1*np.exp(send))
         
                #print(te)  
                
                if flag:
                    if word_idx >0:
                        sum = 0
                        curr = tag_indexer.get_index(goldval[word_idx])
                        prev = tag_indexer.get_index(goldval[word_idx-1])
                        transtionvector[tlen*curr + prev ] = transtionvector[tlen*curr + prev] + alpha2
                        #print(te)
                        
                        for p1 in xrange(0,len(tag_indexer)):
                            for p2 in xrange(0,len(tag_indexer)):
                                '''
                                tidx1 = p2
                                tidx2 = p1
                                if(isI(tag_indexer.get_object(tidx1))):
                                    if(isO(tag_indexer.get_object(tidx2))):
                                        continue
                                    if(isB(tag_indexer.get_object(tidx2))):
                                        curr = get_tag_label(tag_indexer.get_object(tidx2))
                                        prev = get_tag_label(tag_indexer.get_object(tidx1))
                                        if prev!=curr:
                                            continue
                                    if(isI(tag_indexer.get_object(tidx2))):
                                        curr = get_tag_label(tag_indexer.get_object(tidx2))
                                        prev = get_tag_label(tag_indexer.get_object(tidx1))
                                        if prev!=curr:
                                            continue
                                
                                
                                
                                '''
                                #print(te)
                                send = a[word_idx-1][p1] + score_indexed_features([tlen*p2 + p1],te)+score_indexed_features(feature_cache[sentence_idx][word_idx][p2],we) + b[word_idx][p2] -Z
                                transtionvector[tlen*p2 + p1] = transtionvector[tlen*p2 + p1] + alpha2*(-1*np.exp(send))
                                
                                #if word_idx==2:
                                    #print(np.exp(send))
                                #sum+=np.exp(send)
                        #print(sum)
                
                   
                
        #print(transtionvector)
                
                
                
        print("Epoch# "+str(i+1)+" done")
        #print(i)
                
    #transtionvector = np.zeros((tlen*tlen),dtype = float)
    '''
    for p1 in xrange(0,tlen):
        for p2 in xrange(0,tlen):
            print(tag_indexer.get_object(p1)+" to "+tag_indexer.get_object(p2)+" is "+str(transtionvector[tlen*p2+p1]))
            
    #transtionvector = np.zeros((len(tag_indexer)*len(tag_indexer)),dtype=float)
    
    '''
    
    #pickling learnt vectors for future use
    f1 = open('emmisionweights.p','w')
    f2 = open('transitionweights.p','w')
    
    pi.dump(weightvector,f1)
    pi.dump(transtionvector,f2)
    
    f1.close()
    f2.close()
    
    return CrfNerModel(tag_indexer,feature_indexer, weightvector,transtionvector)

                
                
            
    
    
    
    
    
    
        
        
        
    #raise Exception("IMPLEMENT THE REST OF ME")


# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).

            
            

    
            


            
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    
    context = []
    for i in xrange(max(0,word_index-2),min(len(sentence),word_index+3)):
        context+=sentence.tokens[i].word
        
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":ContextWindow=" + repr(context))
    #maybe_add_feature(feats, feature_indexer, add_to_indexer, ":ContextWindow=" + repr(context))
    return np.asarray(feats, dtype=int)

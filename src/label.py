import csv 
import re 
import nltk 
import xgboost
import numpy as np  

def normalization(token):
	#remove all English punctuation 
	punctuation = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
	if re.match(punctuation,token) is not None:
		token = ""
	#convert to lower case 
	token = token.lower()
	return token 

def preprocess(sentence,word_dict):
	tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
	tokens = tokenizer.tokenize(sentence)
	temp = []
	for i in range(len(tokens)):
		tokens[i] = normalization(tokens[i])
		if tokens[i] !='':
			if tokens[i] not in word_dict:
				word_dict[tokens[i]]=[]
			temp.append(tokens[i])
	return np.array(temp), word_dict

def construct_glove(filename,word_dict):
	with open(filename) as vocab_txt:
		for line in vocab_txt:
			word = line.split(" ")[0]
			if word in word_dict:
				vector = np.array(line.strip('\n').split(" ")[1:])
				word_dict[word]=vector.astype(np.float)
	print "Finish Glove Construction"
	return word_dict

def construct_worddict(word_dict):
	undefined = []
	for word in word_dict:
		if word_dict[word] is None:
			undefined.append(word)
			word_dict[word] = np.zeroes(300)
	print "Find Undefined Word"
	return undefined

def window_join(sentence,word_dict,window_token):
	sentence_len = sentence.size
	temp=np.empty(sentence_len-2,dtype=object)
	for i in range(sentence_len-2):
		new_token = "-".join(sentence[i:i+3])
		if new_token not in window_token:
			word_1 = word_dict[sentence[i]]
			word_2 = word_dict[sentence[i+1]]
			word_3 = word_dict[sentence[i+2]]
			seq_word = np.concatenate((word_1,word_2,word_3),axis = 0)
			window_token[new_token] = seq_word
		temp[i]=new_token
	return window_token,temp 

def save_file(window_token):
	token_len = len(window_token)
	un_label_token = np.empty(token_len,dtype=object)
	data = np.empty([token_len,900])
	i = 0 
	for token in window_token:
		if window_token[token].size !=900:
			print token 
		else:
			un_label_token[i] = token
			data[i,]=window_token[token]
			i = i+1

	np.savetxt("un_label_token.csv",un_label_token,delimiter=",",fmt="%s")
	np.savetxt("data.csv",data,delimiter=",")
	print "Finish Save Two Files"

def main():
	# read text file
	sentences = []
	with open("label.txt") as file_txt:
		for line in file_txt:
			sentences.append(line)
	print "Finish Load File"

	# preprocess the sentences
	word_dict = {}
	sentence_token = np.empty(len(sentences),dtype=object)
	for j in range(len(sentences)):
		temp,word_dict = preprocess(sentences[j],word_dict)
		sentence_token[j] = temp
	print "Finish Preprocess"

	# read glove file 
	word_dict = construct_glove("vocab.txt",word_dict)
	print "Finish New Vocabulary"

	# construct word_dict 
	undefined = construct_worddict(word_dict)

	# new dataset
	join_token = np.empty(len(sentences),dtype=object)
	window_token = {}
	k = 0
	for sentence in sentence_token:
		if sentence.size>=3:
			window_token,temp = window_join(sentence,word_dict,window_token)
			join_token[k]=temp
			k=k+1
	np.savetxt("join_token.csv",join_token,delimiter=",",fmt="%s")
	print "Finish New Token Construction"

	# save new data file 
	save_file(window_token)

if __name__ == '__main__':
	main()
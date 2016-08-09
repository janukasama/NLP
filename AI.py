import threading
import nltk
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
from nltk.corpus import wordnet
import csv
from collections import Counter
from nltk.tag.perceptron import PerceptronTagger
import re
import collections
from flask import Flask
from flask import request
from flask.ext.cors import CORS
from os.path import exists
from pip._vendor import requests, os
from pymongo import MongoClient


#--------------------------------Answering part-------------------------------------------#
# _________________________________________________________________________________________ #
# ----------------------------- CloudentDB Configuration ------------------------------------- #
app = Flask(__name__)
CORS(app)

client = MongoClient()
db = client.AI

alphabet = 'abcdefghijklmnopqrstuvwxyz'
tagger = PerceptronTagger()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = [
    'about',
    'has',
    'a',
    'an',
    'are',
    'as',
    'at',
    'can',
    'be',
    'by',
    'from',
    'is',
    'it',
    'of',
    'in',
    'on',
    'or',
    'that',
    'the',
    'this',
    'to',
    'was',
    'then',
    'www']

keywords = ['timesheet', 'timesheets', 'route']
keywordsinquestion = []
questionwords = [ "what", "where", "why", "how", "who", "which", "when", "whom", "whose", "can" ]#, "is", "are", "am", "have", "did", "do", "does", "tell", "from", ]

class AI(object):

    def evaluate_Db(self):
        try:
            dictionary = {}
            cursor = db.questionAndAnswer.find()
            for i in cursor:
                for j in range(1):
                    keyindictionary = i['question']
                    valueindictionary = i['answer']
                    dictionary[keyindictionary] = valueindictionary
        except Exception as e:
            print(e)
            print("Error on evaluate_Db_Json()")
        return dictionary

    def evaluate_Db_keywords(self, sendword):
        try:
            dictionary = {}
            search = db.questionAndAnswer.find({"keyword": '' + sendword + ''})
            for i in search:
                for j in range(1):
                    keyindictionary = i['question']
                    valueindictionary = i['answer']
                    dictionary[keyindictionary] = valueindictionary
            del keywordsinquestion[:]
            if not dictionary:
                dictionary = self.evaluate_Db()
        except Exception as e:
            print(e)
            print("Error on evaluate_Db_keywords()")
        return dictionary
# ____________________________________________________________________________________________________________________ #
# ----------------------------------------- Spell Correction Component ----------------------------------------------- #

    def __init__(self):
        self.NWORDS = self.train(self.words(open('TechwordDict.txt').read()))

    def words(self, text):
        try:
            return re.findall('[a-z]+', text.lower())
        except Exception as e:
            print(e)

    def train(self, features):
        try:
            model = collections.defaultdict(lambda: 1)
            for f in features:
                model[f] += 1
        except Exception as e:
            print(e)
        return model

    def edits1(self, word):
        try:
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [a + b[1:] for a, b in splits if b]
            transposes = [a + b[1] + b[0] + b[2:]
                          for a, b in splits if len(b) > 1]
            replaces = [a + c + b[1:]
                        for a, b in splits for c in alphabet if b]
            inserts = [a + c + b for a, b in splits for c in alphabet]
        except Exception as e:
            print(e)
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        try:
            return set(e2 for e1 in self.edits1(word)
                       for e2 in self.edits1(e1) if e2 in self.NWORDS)
        except Exception as e:
            print(e)

    def known(self, words):
        try:
            return set(w for w in words if w in self.NWORDS)
        except Exception as e:
            print(e)

    def correct(self, word):
        try:
            candidates = self.known(
                [word]) or self.known(
                self.edits1(word)) or self.known_edits2(word) or [word]
            answer = max(candidates, key=self.NWORDS.get)
        except Exception as e:
            print(e)
        return answer

    def correctQuestion(self, text):
        try:
            final_answer = ""
            splited_words = text.split()
            english_vocab = set(w.lower() for w in nltk.corpus.words.words())
            for i in splited_words:
                if i in english_vocab:
                    final_answer = final_answer + i + " "
                else:
                    final_answer = final_answer + self.correct(i) + " "
        except Exception as e:
            print(e)
            print("Error on correctQuestion()")
        return final_answer

# --------------------------------- END - Spell Correction Component ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Natural Language -------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

    def getCosineDist(self, a, b):
        try:
            a_vals = Counter(a)
            b_vals = Counter(b)
            words = list(set(a_vals) | set(b_vals))
            a_vect = [a_vals.get(word, 0)
                      for word in words]
            b_vect = [b_vals.get(word, 0)
                      for word in words]
            len_a = sum(av * av for av in a_vect) ** 0.5
            len_b = sum(bv * bv for bv in b_vect) ** 0.5
            dot = sum(av * bv for av, bv in zip(a_vect, b_vect))
            cosine = dot / (len_a * len_b)
        except Exception as e:
            print(e)
            print("Error on getCosineDist()")
        return cosine

    def removePunctuations(self, Question):
        try:
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            no_punct = ""
            for char in Question:
                if char not in punctuations:
                    no_punct = no_punct + char
        except Exception as e:
            print(e)
            print("Error on removePunctuations()")
        return no_punct

    def pos_tagging(self, X):
        try:
            tagset = None
            tokens = nltk.word_tokenize(X)
            tags = nltk.tag._pos_tag(tokens, tagset, tagger)
            pos_X = map(self.get_wordnet_pos, tags)
        except Exception as e:
            print(e)
            print("Error on pos_tagging()")
        return pos_X

    def lemmatization_stopwordsRemoval(self, pos_X):
        try:
            lemmae_X = [
                lemmatizer.lemmatize(
                    token.lower().strip(
                        string.punctuation), pos) for token, pos in pos_X if (
                    pos == wordnet.NOUN or pos == wordnet.VERB or pos == wordnet.ADJ or pos == wordnet.ADV) and token.lower().strip(
                            string.punctuation) not in stopwords]
        except Exception as e:
            print(e)
            print("Error on lemmatization_stopwordsRemoval()")
        return lemmae_X

    def get_wordnet_pos(self, pos_tag):
        try:
            if pos_tag[1].startswith('J'):
                return (pos_tag[0], wordnet.ADJ)
            elif pos_tag[1].startswith('V'):
                return (pos_tag[0], wordnet.VERB)
            elif pos_tag[1].startswith('N'):
                return (pos_tag[0], wordnet.NOUN)
            elif pos_tag[1].startswith('R'):
                return (pos_tag[0], wordnet.ADV)
            else:
                return (pos_tag[0], wordnet.NOUN)
        except Exception as e:
            print(e)
            print("Error on get_wordnet_pos()")

    def arrangeText(self, text):
        try:
            pos_text = self.pos_tagging(text)
            lemmae_text = self.lemmatization_stopwordsRemoval(pos_text)
        except Exception as e:
            print(e)
            print("Error on arrangeText()")
        return lemmae_text

    def similarity_jaccrd(self, a, b):
        try:
            arrangedText_a = self.arrangeText(a)
            arrangedText_b = self.arrangeText(b)
            jaccrdRatio = len(set(arrangedText_a).intersection(
                arrangedText_b)) / float(len(set(arrangedText_a).union(arrangedText_b)))
        except Exception as e:
            print(e)
            print("Error on similarity_jaccrd()")
        return jaccrdRatio

    def similarity_cosine(self, a, b):
        try:
            arrangedText_a = self.arrangeText(a)
            arrangedText_b = self.arrangeText(b)
            cosineRatio = self.getCosineDist(arrangedText_a, arrangedText_b)
        except Exception as e:
            print(e)
            print("Error on similarity_cosine()")
        return cosineRatio
# ------------------------------------------- END - Natural Language------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

    def get_suggestions(self, question, best_match):
        splitedwords = question.split()
        isrelevant = False
        isquestion = False
        for word in splitedwords:
            if word in keywords:
                isrelevant = True
            if word in questionwords:
                isquestion = True

        if isrelevant and not isquestion:
            return 'Your question is ambiguous, you can ask specific questions like, "'+best_match+"'."

        return 'DIRROUTE'

    def get_Ai_Answer(self, question, user_question):
        try:
            max_coisne_db = 0
            answer = ""
            best_question = ""
            best_jac = 0 #only needed for making suggestions when jac < 0.5
            splitedwords = question.split()
            for i in keywords:
                for j in splitedwords:
                    if(i == j):
                        keywordsinquestion.append(i)

            if not keywordsinquestion:
                dictionary = self.evaluate_Db()
            else:
                if(len(keywordsinquestion) == 0):
                    word = ''.join(str(v) for v in keywordsinquestion)
                    sendword = '' + word + ''
                    dictionary = self.evaluate_Db_keywords(sendword)
                else:
                    word = ', '.join(keywordsinquestion)
                    sendword = '' + word + ''
                    dictionary = self.evaluate_Db_keywords(sendword)

            for item in enumerate(dictionary):
                print ("DBQUESTION >>>>> " + item[1])
                print("USER_QUESTION >>>>" + question)
                dbmessage = item[1]
                jac_value = self.similarity_jaccrd(question, dbmessage)
                print("JACCARD VALUE >>>>" + str(jac_value))

                if (jac_value >= 0.5):
                    print("IN JACCARD")
                    dist_db = self.similarity_cosine(question, dbmessage)
                    print("COSINE >>>>> " + str(dist_db))
                    if(max_coisne_db <= dist_db):
                        max_coisne_db = dist_db
                        print("IN COS")
                        answer = dictionary.get(dbmessage)
                        best_question = dbmessage
                elif (jac_value >= best_jac):
                    best_jac = jac_value
                    best_question = dbmessage

            if(max_coisne_db > 0.6):
                print("QUESTION MATCHED")
                answer = answer
                print("COSINE >>>>> " + str(max_coisne_db))
                print("ANSWER >>>>> " + answer)

            else:
                answer = self.get_suggestions(question, best_question)
        except Exception as e:
            print(e)
            print("Error on get_Ai_Answer()")
        return answer
#-------------------------------End of Answering part------------------------------------#

#------------------------------API CALLS---------------------------------------------------#


@app.route("/get_answer", methods=['POST'])
def getAnswer():
    try:
        a = AI()
        user_question = request.values['user_question']
        user_question = user_question.lower()
        question_remove_punc = a.removePunctuations(user_question)
        question = a.correctQuestion(question_remove_punc)
        Answer = a.get_Ai_Answer(question, user_question)
    except Exception as e:
        print(e)
        print("Error on getAnswer()")
    return Answer

#------------------------------End of API CALLS---------------------------------------------------#

def createDocumentDbWithKeywords(question, answer, keyword):
    try:
        data = {
            'question': '' + question + '',
            'answer': '' + answer + '', 'keyword': '' + keyword + ''}
        print(data)
        result = db.questionAndAnswer.insert_one(data)
        del keywordsinquestion[:]
        print("Object ID >>> " + str(result.inserted_id))

    except Exception as e:
        print(e)
        print("Error On createDocumentDbWithKeywords()")


def insertCsvToDb():
    b = AI()
    try:
        if exists('QnA.csv'):
            csv_file = open('QnA.csv')
            reader = csv.DictReader(csv_file)
            for i in reader:
                question = i['Question'].lower()
                question_remove_punc = b.removePunctuations(question)
                question_spell_correct = b.correctQuestion(
                    question_remove_punc)

                spilited_words = question_spell_correct.split()

                answer = i['Answer'].lower()
                answer_remove_punc = b.removePunctuations(answer)
                answer_spell_correct = b.correctQuestion(answer_remove_punc)

                for i in keywords:
                    for j in spilited_words:
                        if(i == j):
                            keywordsinquestion.append(i)
                if(len(keywordsinquestion) == 0):
                    word = ''.join(str(v) for v in keywordsinquestion)
                    keyword = '' + word + ''
                    createDocumentDbWithKeywords(
                        question_spell_correct, answer_spell_correct, keyword)
                else:
                    word = ', '.join(keywordsinquestion)
                    keyword = '' + word + ''
                    createDocumentDbWithKeywords(
                        question_spell_correct, answer_spell_correct, keyword)

            f = open('QnA.csv', "w+")
            f.close()
        else:
            pass

    except Exception as e:
        print(e)
        print("Error On insertCsvToDb()")

port = os.getenv('PORT', '5000')
if __name__ == '__main__':
    insertCsvToDb()
app.run(host='0.0.0.0', port=int(port))

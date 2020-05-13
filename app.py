"""
nlp_tf_idf_hadoop.py
NLP analysis of Term Frequency - Inverse Document Frequency using Hadoop
Handles the primary functions
"""

import sys
import re
import math
import pyspark

# Local[*] is used to run spark locally with as many worker threads as logical cores on your machine
# nlp_tf_idf is the name of the application
#Gateway to pyspark
sc = pyspark.SparkContext('local[*]', 'nlp_tf_idf')

#This module provides regular expression matching operations similar to those found in Perl.
DIS_REGEX = re.compile('^(dis)_[^ ]+_\\1$')
QUERY = ""

# Convert every line in the text to a document
def text_to_document(txt):
    #Python verb which splits text into words
    splitted = txt.split()
    # return (document id, words in doc)
    return splitted[0], [w for w in splitted[1:] if DIS_REGEX.match(w) or w == QUERY]

# Split a document into words 
def document_to_words(doc):
    words = doc[1]
    num_words = len(words)
    ret = []
    for word in words:
        ret.append(((doc[0], num_words, word), 1))
    return ret

#Reject # of arguments if not 3
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Incorrect number of arguments. Must have file name and query term.')
        exit(0)

    filename = sys.argv[1]

    QUERY = sys.argv[2]
    output = open('output', 'w')
    # Write F-String to file 
    output.write(f'Query: {QUERY}\n')

    #Get file as input
    text = sc.textFile(filename)
    documents = txt.map(text_to_document)

    document_count = documents.count()
    output.write(f'Number of documents: {doc_count}\n')
      #Add  a and b
      #The flatMap() is used to produce multiple output elements for 
      #each input element. When using map(), the function we provide 
      #to flatMap() is called individually for each element in our input RDD. 
      #Instead of returning a single element, an iterator with the return values is returned.
   
    words_by_document = documents.flatMap(document_to_words) \
            .reduceByKey(lambda a, b: a + b) # Number of terms per document

    # 1 Generate the term frequency - Split the document into words and divide by the total term count 
    tf = words_by_document.map(lambda word: (word[0][2], [(word[0][0], word[1]/word[0][1])])) \
            .reduceByKey(lambda a, b: a + b)

    #Operates on key/value pairs and merges the value for each key

    # 2 key: word, value: (idf, [(docid, tf)])
    idf = tf.map(lambda word: (word[0],(math.log(document_count / len(word[1]), 10), word[1])))
 
    # 3 key: word, value: [(docid, tf*idf)]
    # score = tf_idf_merged = tf * idf 
    # score = term frequency * (number of docs/number of docs appeared)
    tf_idf_merged = idf.map(lambda word: (word[0], {i[0]: word[1][0] * i[1] for i in word[1][1]}))


    sorted_tf_idf = tf_idf.sortByKey()
    q = sorted_tf_idf.lookup(QUERY)

    q = [i for i in q]
    q_norm = sum(map(lambda x: x ** 2, q[0].values())) ** (1/2)
    #Similarity
    similar = tf_idf.map(lambda w: (w[0], sum([q[0][element] * w[1][element] for element in q[0].keys() & w[1].keys()]) / (sum(map(lambda x: x ** 2, w[1].values())) ** (1/2) * q_norm)))
      
    # take top 5 from ordered the pair in descending order
    terms = similar.takeOrdered(6, key=lambda word: -word[1])

    output.write(f'\nTop 5 similar to {QUERY}:\n')
    # We skip the query term because that matches itself (~1.0 score)
    output.writelines([f'{word} {item}\n' for (word, item) in terms if word != QUERY])
    output.close()

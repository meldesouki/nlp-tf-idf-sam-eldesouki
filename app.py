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
def txt_to_doc(txt):
    #Python verb which splits text into words
    #If no arguments are given, it splits on repeated runs of whitespace.
    splitted = txt.split()
    # return (document id, words in doc)
    return splitted[0], [w for w in splitted[1:] if DIS_REGEX.match(w) or w == QUERY]

# Split a document into words 
def doc_to_words(doc):
    words = doc[1]

    #len returns the number of items in an object
    num_words = len(words)
    ret = []
    for word in words:
        ret.append(((doc[0], num_words, word), 1))
    return ret

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
    txt = sc.textFile(filename)
    #docs = a tuple of the doc_id and the rest of the string 
    # apply a function on all the elements of specified iterable 
    # and return map object. 
  
    # docs is a map object is an iterator, so 
    # we can iterate over its elements. We can also convert map object to sequence objects such as list, tuple
    docs = txt.map(txt_to_doc)

    #The count() method returns the number of occurrences of an element in a list - perform count on map-object
    doc_count = docs.count()
    output.write(f'Number of documents: {doc_count}\n')
      #Add  a and b
      #The flatMap() is used to produce multiple output elements for 
      #each input element. When using map(), the function we provide 
      #to flatMap() is called individually for each element in our input RDD. 
      #Instead of returning a single element, an iterator with the return values is returned.
      #Flattened into an array - key/value pair turned into an array
    words_by_doc = docs.flatMap(doc_to_words) \
            .reduceByKey(lambda a, b: a + b) # term count per doc
    tf = words_by_doc.map(lambda word: (word[0][2], [(word[0][0], word[1]/word[0][1])])) \
            .reduceByKey(lambda a, b: a + b) #tf formula from Lecture 9, slide 64

    #Spark RDD reduceByKey function merges the values for each key using an associative 
    #reduce function. Basically reduceByKey function works only for RDDs which contains key 
    #and value pairs kind of elements



    #Operates on key/value pairs and merges the value for each key

    # key: word, value: (idf, [(docid, tf)])
    tf_idf = tf.map(lambda word: (word[0],
        (math.log(doc_count / len(word[1]), 10), word[1]))) #idf formula from Lecture
 
    # key: word, value: [(docid, tf*idf)]
    # Given a key, we score each document using the tf*idf formula
    tf_idf_merged = tf_idf.map(lambda word: (word[0], {i[0]: word[1][0] * i[1] for i in word[1][1]}))

    #sortByKey function maintains the order of elements. 
    #It receives key-value pairs (K, V) as an input, sorts the elements in 
    #ascending or descending order and generates a dataset in an order.
    
    sorted_tf_idf = tf_idf_merged.sortByKey()
    q = sorted_tf_idf.lookup(QUERY)

    q = [i for i in q]
    q_norm = sum(map(lambda x: x ** 2, q[0].values())) ** (1/2)

    # Apply our formula to retrieve the term - term frequency
    similartities = tf_idf_merged.map(lambda w: (w[0], sum([q[0][elem] * w[1][elem] for elem in q[0].keys() & w[1].keys()]) / (sum(map(lambda x: x ** 2, w[1].values())) ** (1/2) * q_norm)))
      
    # take top 5 from ordered the pair in descending order
    terms = similartities.takeOrdered(6, key=lambda word: -word[1])

    output.write(f'\nTop 5 similar to {QUERY}:\n')
    # We skip the query term because that matches itself (~1.0 score)
    output.writelines([f'|    {word} -------- {item}   |\n' for (word, item) in terms if word != QUERY])
    output.close()

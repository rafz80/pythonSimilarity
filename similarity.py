import gensim 
import logging
import colorama
from colorama import Fore, Style, init

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

init(autoreset=True)


# --------------------
# Type the word to find
# --------------------
wordFind = input(Fore.YELLOW + Style.BRIGHT + "Please, type the word to find similarities in text: " + Fore.WHITE)


# --------------------
# file to be analyzed
# --------------------
data_file="bbc-text.csv"

with open (data_file, 'r') as f:
    for i,line in enumerate (f):
        print(line)
        break
        
def read_input(input_file):  
    logging.info("reading file {0}... this may take a while...".format(input_file))
    
    with open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            yield gensim.utils.simple_preprocess (line)

documents = list (read_input (data_file))
logging.info ("Read file completed!")



# --------------------
# train it...
# --------------------

model = gensim.models.Word2Vec (documents, size=200, window=3, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)



# --------------------
# looking for the similarity
# --------------------
w1 = wordFind
model.wv.most_similar (positive=w1)
print(Fore.MAGENTA + 'Results: ')
print(model.wv.most_similar (positive=w1))





import fasttext
import sys
import logging
import os

## https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz


logging.basicConfig(level=logging.INFO)

model_location= sys.argv[1]
model_output_file=f"{model_location}/cooking.model.bin"
training_file=f"{model_location}/cooking.train"
validation_file=f"{model_location}/cooking.valid"


def loadOrBuild(training_file:str,model_file:str) -> fasttext.FastText:
    if(os.path.exists(model_output_file)):
        logging.info(f"Model {model_output_file} exists")
        return fasttext.load_model(model_output_file)
    else:    
        #model = fasttext.train_supervised(input=training_file)
        #model = fasttext.train_supervised(input=training_file,lr=1.0, epoch=25, wordNgrams=2)
        model = fasttext.train_supervised(input=training_file,lr=1.0, epoch=25, wordNgrams=2,bucket=200000, dim=50, loss='hs')
        logging.info(f"Model {model}")
        logging.info(f"Model is saved @ {model_output_file}")
        logging.info(f"Model words: {len(model.get_words())} ")
        logging.info(f"Model Labels: {len(model.get_labels())} ")
        model.save_model(model_output_file)
        return model


model=loadOrBuild(training_file,model_location)

result = model.test(validation_file)

logging.info(f"Model accuracy: {result}")

while True:
    inp_question = input("Please enter a question: ")
    results = model.predict(inp_question, 5)
    logging.info(f"Results: \n {results}")
    print("\n")
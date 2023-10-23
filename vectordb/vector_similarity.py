from sentence_transformers import SentenceTransformer,util


import logging


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    model_name="all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logging.info(model)

    vec1 = model.encode("This is a red cat with a hat")
    vec2 = model.encode("have you seen my red cat")


    score = util.cos_sim(vec1, vec2)

    logging.info(f"Score is {score}")

    sentences = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.'
          ]
    
    embeddings = model.encode(sentences)
    co_sin_scores = util.cos_sim(embeddings, embeddings)

    logging.info(co_sin_scores)

    all_sentence_combination=[]
    for i in range(len(co_sin_scores)-1):
        for j in range(i+1, len(co_sin_scores)):
            all_sentence_combination.append([co_sin_scores[i,j],i,j])

    all_sentence_combinations = sorted(all_sentence_combination, key=lambda x: x[0], reverse=True)

    logging.info("Top 5 similar pairs")
    for score, i, j in all_sentence_combinations[0:5]:
        logging.info("{} \t {} \t Score: {:.4f}".format(sentences[i], sentences[j], score))        
    


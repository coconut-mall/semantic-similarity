# Import module and load language model
import spacy

nlp = spacy.load("en_core_web_sm")

# Extract 1
print("//Extract 1//")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Extract 2
print("//Extract 2//")
tokens = nlp("cat apple monkey banana ")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Extract 3
print("//Extract 3//")

sentence_to_compare = "Why is my cat on the car"

sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana",
]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

"""
I found it interesting that despite monkey and banana being different
types of things, they still had a strong similarity which shows the
relative sophistication of the language model to pick up more abstract
relationships between words.
"""

"""
With the simpler language model the relationship between monkey
and banana is weaker. Cat also now has a much higher similarity to
apple which doesn't make much sense. The similarity between the sentences
also gives completely different results. I think these aspects show that 
the model is much less sophisticated.
"""

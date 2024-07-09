# main.py
import os
from atp import ATP
from utils import load_german_CHILDES, load_pairs, load_word_to_ipa

# test the directory and if relative path is correct
print("Current directory:", os.getcwd())
file_path = os.path.join(os.getcwd(), '../data/german/CHILDES-DE.txt')
print("Does the file exist?", os.path.exists(file_path))

pairs, feature_space = load_german_CHILDES()
atp = ATP(feature_space=feature_space).train(pairs)
atp.plot_tree('../temp/german', open_pdf=True) # plot the tree and save it to a PDF file
atp.inflect('Sache', ('F'))
atp.inflect('Gleis', ('N'))

# test the inflection with features
print(atp.inflect('Sache', ('F')))
print(atp.inflect('Gleis', ('N')))
print()

# test the inflection without features
print(atp.inflect_no_feat('Sache', ())) # the result is still correct

print(atp.inflect_no_feat('Gleis', ()))

print(atp.inflect_no_feat('Kach', ())) # for a nonce word with unknown gender, ATP produces the -er suffix, as do a majority of humans


pairs = [('a', 'a-', ('Noun',)), 
            ('b', 'b-', ('Noun',)), 
            ('c', 'c-', ('Noun',)),
            ('d', 'd*', ('Noun',)),
            ('a', 'a+', ('Verb',)),
            ('b', 'b+', ('Verb',)),
            ('c', 'c+', ('Verb',)),
            ('d', 'd**', ('Verb',))]

feature_space = {'Noun', 'Verb'}


atp = ATP(feature_space)
atp.train(pairs)
print(atp.inflect('a', ('Noun',)))

print(atp.inflect('e', ('Noun',)))

print(atp.inflect('e', ('Verb',)))


print(atp.inflect('d', ('Noun',)))

print(atp.inflect('d', ('Verb',)))

word_to_ipa = load_word_to_ipa() # load a dictionary of english word-to-IPA mappings
pairs, features = load_pairs('../data/english/growth/child-0/100.txt', sep=' ')
print(pairs[0])

pairs, features = load_pairs('../data/english/growth/child-0/100.txt', sep=' ', preprocessing=lambda s: word_to_ipa[s]) # map every lemma/inflection to its IPA
print(pairs[0])



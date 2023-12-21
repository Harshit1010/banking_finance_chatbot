import numpy as np
import pandas as pd

if __name__  == "__main__":
    #load faq text file
    with open('faq.txt', 'r') as file:
        data = file.read()

    #store each newline in list
    sentences = [line.strip() for line in data.strip().split('\n')]
    sentences = [i for i in sentences if len(i)>0]
    
    index_list = []
    for i in range(1, len(sentences)):
        if sentences[i].startswith('Question: ') or sentences[i].startswith('Answer: '):
            continue
        else:
            sentences[i-1] = sentences[i-1] + ' ' + sentences[i]
            index_list.append(i)
    
    index_list.sort(reverse=True)
    for index in index_list:
        sentences.pop(index)

    ques = []
    ans = []
    for sentence in sentences:
        if sentence.startswith('Q'):
            ques.append(sentence[10:])
        else:
            ans.append(sentence[8:])

    df = pd.DataFrame({'Questions': ques, 'Answers': ans})
    df.to_csv('faq_new_1.csv', index = False)
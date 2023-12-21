import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from profanity_check import predict
import time

def check_abusive_words(input_question):
    return predict([input_question])

def print_dashed_line():
    print("\n--------------------------------------------------------------------------------------------- \n")

def get_user_query():
    return input("Please go ahead writing your query, I'll try my best to answer it: ")

def get_faq_database():
    df = pd.read_csv('faq_new.csv')
    questions = df['Questions'].tolist()
    answers = df['Answers'].tolist()
    return questions, answers

def chatbot():
        #Get user input question
        user_input = get_user_query()
        if len(user_input)==0:
            time.sleep(1)
            print("I got no response. Closing the chat for now. Have a good day!")
            print_dashed_line()
            return
        print_dashed_line()
        
        #Profanity check on question
        is_abusive = check_abusive_words(user_input)
        if is_abusive:
            time.sleep(1)
            flag = input("Sorry, I can't answer that. Do you have any other question? (Y/N): ")
            if flag.lower()=='y' or flag.lower()=='yes':
                chatbot()
            else:
                print('\nAs you do not wish to continue, I am closing the chat. Have a good day!')
                print_dashed_line()

        else:
            print("Surely, let me check if I know FAQs similar to what you asked. I'll be right back.")
            time.sleep(5)
            print_top_3_matching_faqs(user_input)

def print_top_3_matching_faqs(input_question):
    questions, answers = get_faq_database()

    nlp = spacy.load("en_core_web_sm")
    vectorizer = TfidfVectorizer()

    # process faq questions
    lower_questions = [ques.lower() for ques in questions]
    processed_questions = [nlp(q) for q in lower_questions]
    questions_lis = [str(doc) for doc in processed_questions]
    ques_vectors = vectorizer.fit_transform(questions_lis)

    # process input question
    lower_query = input_question.lower()
    processed_query = nlp(lower_query)
    query_lis = [str(processed_query)]
    query_vector = vectorizer.transform(query_lis)

    # Getting cosine similarity between the input question and FAQ questions
    similarities_list = cosine_similarity(query_vector, ques_vectors).flatten()

    # Get indices of top 3 matches
    top_indices = similarities_list.argsort()[-3:][::-1]

    # Get the top 3 matching FAQs and similarity score value
    top_similarity_scores = [similarities_list[i] for i in top_indices]
    top_questions = [questions[i] for i in top_indices]
    top_answers = [answers[i] for i in top_indices]

    # Check if similarity scores are above threshold
    similarity_threshold = 0.2
    if top_similarity_scores[2]<similarity_threshold:
        print("I don't think I have very similar FAQs in my knowledge. \n")
        flag = input('Do you have any other question? (Y/N): \n')
        if flag.lower()=='y' or flag.lower()=='yes':
            chatbot()
        else:
            print('\nAs you do not wish to continue, I am closing the chat. Have a good day!')
            print_dashed_line()

    else:
        print("Here are the most relevant FAQs which I feel might answer your question \n")
        #print the 3 top matching FAQs
        for counter in range(3):
            print(f"{counter+1}. Question: {top_questions[counter]} ")
            print(f"  Answer: {top_answers[counter]}")
        print('\n I hope I helped you with your query. Have a good day!')
        print_dashed_line()

if __name__  == "__main__":
    #Print starting message
    print("\n Hi, I'm your banking and finance assistant!")
    print_dashed_line()
    time.sleep(0.5)

    #Get input from user, check profanity, predict top 3 FAQs
    chatbot()
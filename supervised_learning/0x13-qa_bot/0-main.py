#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('/home/rodrigo/Documents/ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))

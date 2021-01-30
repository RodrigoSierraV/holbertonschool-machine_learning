#!/usr/bin/env python3
"""
Takes in input from the user with the prompt "Q:"
"""


while True:
    question = input("Q: ")

    if question.lower().strip() in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    print("A:")

import os
import string

UNCLEANED_DIR = "./uncleaned_texts/"
CLEANED_DIR = './cleaned_texts/'
TRANSLATED_DIR = './data/'

REPLACE = {
    '“':'"',
    '”': '"',
    '‘': "'",
    '’': "'",
    ' ': ' ',
}



BLACKLIST = '…_—﻿™•£·'

def main():
    illegal_chars = ''
    for file in os.listdir(UNCLEANED_DIR):
        with open(UNCLEANED_DIR + file, "r", encoding="utf-8") as in_file:
            contents = in_file.read()
            
            for char in REPLACE.keys():
                contents = contents.replace(char, REPLACE[char])

            for char in BLACKLIST:
                contents = contents.replace(char, '')
            
            for char in contents:
                if not char.isalnum() and not char in string.punctuation and not char == ' ' and not char == '\n' and not char == '\t':
                    illegal_chars += char + "\n"
            
            with open(CLEANED_DIR + file, "w", encoding='utf-8') as out_file:
                out_file.write(contents)
                
            # with open(TRANSLATED_DIR + file, 'w', encoding='utf-8') as t_file:
            #     t_file.write('')


    if (illegal_chars != ''):   
        print("Illegal characters found:", len(illegal_chars))            
        with open("./illegal_chars.txt", "w", encoding='utf-8') as i_file:
            i_file.write(illegal_chars)


if __name__ == "__main__":
    main()
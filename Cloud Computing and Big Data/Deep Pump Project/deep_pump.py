import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
import os
import pronouncing
import markovify
import re
import random


def find_syllables(line):
    count = 0
    for word in line.split(" "):
        vowels = 'aeiouy'
        word = word.lower().strip(".:;?!")
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
    return count / maxsyllables


def find_rhyme(line, rhyme_list):
    word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
    rhymes_list = pronouncing.rhymes(word)
    rhymes_list = [x.encode('UTF8') for x in rhymes_list]
    rhymes_list_ends = []
    for i in rhymes_list:
        rhymes_list_ends.append(i[-2:])
    try:
        rhyme_scheme = max(set(rhymes_list_ends), key=rhymes_list_ends.count)
    except Exception:
        rhyme_scheme = word[-2:]
    try:
        float_rhyme = rhyme_list.index(rhyme_scheme)
        float_rhyme = float_rhyme / float(len(rhyme_list))
        return float_rhyme
    except Exception:
        return None


def create_nn(depth):
    model = Sequential()
    model.add(layer=(LSTM(4,
                          input_shape=(2, 2),
                          return_sequences=True)))
    for item in range(depth):
        model.add(layer=LSTM(8,
                             return_sequences=True))
    model.add(layer=LSTM(2,
                         return_sequences=True))
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='mse')
    if artist + ".rap" in os.listdir(".") and train_mode is False:
        model.load_weights(str(artist + ".rap"))
    return model


def find_rhyme_list(lyrics):
    if str(artist) + ".rhymes" in os.listdir(".") and train_mode is False:
        return open(str(artist) + ".rhymes", "r").read().split("\n")
    else:
        rhyme_master_list = []
        for i in lyrics:
            word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()
            rhymes_list = pronouncing.rhymes(word)
            rhymes_list = [x.encode('UTF8') for x in rhymes_list]
            rhymes_list_ends = []
            for item in rhymes_list:
                rhymes_list_ends.append(item[-2:])
            try:
                rhyme_scheme = max(set(rhymes_list_ends),
                                   key=rhymes_list_ends.count)
            except Exception:
                rhyme_scheme = word[-2:]
            rhyme_master_list.append(rhyme_scheme)
        rhyme_master_list = list(set(rhyme_master_list))

        reverse_list = [x[::-1] for x in rhyme_master_list]
        reverse_list = sorted(str(reverse_list))
        rhyme_list = [x[::-1] for x in reverse_list]

        f = open(str(artist) + ".rhymes", "w")
        f.write("\n".join(rhyme_list))
        f.close()
        return rhyme_list


def make_dataset(lyrics, rhyme_list):
    dataset = []
    for line in lyrics:
        line_list = [line,
                     find_syllables(line),
                     find_rhyme(line,
                                rhyme_list)]
        dataset.append(line_list)

    x_data = []
    y_data = []

    for item in range(len(dataset) - 3):
        line1 = dataset[item][1:]
        line2 = dataset[item + 1][1:]
        line3 = dataset[item + 2][1:]
        line4 = dataset[item + 3][1:]

        x = [line1[0], line1[1],
             line2[0], line2[1]]
        x = np.array(x)
        x = x.reshape(2, 2)
        x_data.append(x)

        y = [line3[0], line3[1],
             line4[0], line4[1]]
        y = np.array(y)
        y = y.reshape(2, 2)
        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def train_model(x_data, y_data, model):
    model.fit(np.array(x_data),
              np.array(y_data),
              batch_size=4,
              epochs=20,
              verbose=1)
    model.save_weights(artist + ".rap")


def generate_lyrics(lyrics_file):
    bars = []
    last_words = []
    lyric_length = len(open(lyrics_file).read().split("\n"))
    count = 0
    read = open(lyrics_file, "r").read()
    markov_model = markovify.NewlineText(read)
    while len(bars) < lyric_length / 9 and count < lyric_length * 2:
        bar = markov_model.make_sentence()
        if type(bar) != type(None) and find_syllables(bar) < 1:
            def get_last_word(bar):
                last_word = bar.split(" ")[-1]
                if last_word[-1] in "!.?,":
                    last_word = last_word[:-1]
                return last_word

            last_word = get_last_word(bar)
            if bar not in bars and last_words.count(last_word) < 3:
                bars.append(bar)
                last_words.append(last_word)
                count += 1
    return bars


def write_rap(rhyme_list, lyrics_file, model):
    rap_vectors = []
    text = open(lyrics_file).read()
    text = text.split("\n")

    while "" in text:
        text.remove("")

    human_lyrics = text
    initial_index = random.choice(range(len(human_lyrics) - 1))
    initial_lines = human_lyrics[initial_index:initial_index + 8]
    starting_input = []

    for line in initial_lines:
        starting_input.append([find_syllables(line),
                               find_rhyme(line, rhyme_list)])

    starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(4, 2, 2))
    rap_vectors.append(starting_vectors)

    for i in range(49):
        rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).flatten().reshape(4, 2, 2)))

    return rap_vectors


def vectors_into_song(vectors, generated_lyrics, rhyme_list):
    def last_word_compare(rap, line2):
        penalty = 0
        for line1 in rap:
            word1 = line1.split(" ")[-1]
            word2 = line2.split(" ")[-1]
            while word1[-1] in "?!,. ":
                word1 = word1[:-1]

            while word2[-1] in "?!,. ":
                word2 = word2[:-1]

            if word1 == word2:
                penalty += 0.2

        return penalty

    def calculate_score(vector_half, syllables, rhyme, penalty):
        desired_syllables = vector_half[0]
        desired_rhyme = vector_half[1]
        desired_syllables = desired_syllables * maxsyllables
        desired_rhyme = desired_rhyme * len(rhyme_list)

        score = 1.0 - (abs((float(desired_syllables) - float(syllables))) + abs(
            (float(desired_rhyme) - float(rhyme)))) - penalty

        return score

    dataset = []

    for line in generated_lyrics:
        line_list = [line,
                     find_syllables(line),
                     find_rhyme(line,
                                rhyme_list)]
        dataset.append(line_list)

    rap = []

    vector_halves = []
    for vector in vectors:
        vector_halves.append(list(vector[0][0]))
        vector_halves.append(list(vector[0][1]))

    for vector in vector_halves:
        score_list = []

        for item in dataset:
            line = item[0]

            if len(rap) != 0:
                penalty = last_word_compare(rap, line)
            else:
                penalty = 0
            total_score = calculate_score(vector,
                                          item[1],
                                          item[2],
                                          penalty)
            score_entry = [line, total_score]
            score_list.append(score_entry)

        fixed_score_list = []

        for score in score_list:
            fixed_score_list.append(float(score[1]))

        max_score = max(fixed_score_list)

        for item in score_list:
            if item[1] == max_score:
                rap.append(item[0])
                for i in dataset:
                    if item[0] == i[0]:
                        dataset.remove(i)
                        break
                break

        return rap


if __name__ == "__main__":
    depth = 8
    maxsyllables = 20
    train_mode = int(input("Please, choose a program mode. Type 1 if you want to train your model first "
                           "(Do this if you dont have a trained model) or 2 if you want to create a rap lyrics: "))
    if train_mode == 1:
        train_mode = True
    elif train_mode == 2:
        train_mode = False

    input_file = "lyrics.txt"
    output_file = "neural_rap.txt"
    artist = "mix"

    model = create_nn(depth=depth)

    if train_mode is True:
        text = open(input_file).read()
        text = text.split("\n")
        while "" in text:
            text.remove("")
        bars = text
    elif train_mode is False:
        bars = generate_lyrics(lyrics_file=input_file)

    rhyme_list = find_rhyme_list(lyrics=bars)
    if train_mode is True:
        x_data, y_data = make_dataset(lyrics=bars,
                                      rhyme_list=rhyme_list)
        train_model(x_data=x_data,
                    y_data=y_data,
                    model=model)

    elif train_mode is False:
        vectors = write_rap(rhyme_list=rhyme_list,
                            lyrics_file=input_file,
                            model=model)
        rap = vectors_into_song(vectors=vectors,
                                generated_lyrics=bars,
                                rhyme_list=rhyme_list)
        f = open(output_file, "w")
        for bar in rap:
            f.write(bar)
            f.write("\n")

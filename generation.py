import pandas as pd
import openai
import random
import sys
import time
import os
import csv

# Directly setting the API key
# openai.api_key = 

# Constants
SHUFFLED_DATA_PATH = 'shuffled_data.csv'
FIXED_SAMPLES = 35

def load_translations(csv_filepath):
    """Load both Chinese and Nushu translations from the CSV."""
    try:
        # Load the CSV file
        data = pd.read_csv(csv_filepath)

        # Return the 'Perfect Chinese Sentences' and 'Perfect Nvshu Sentences' columns
        return data['Chinese'].tolist(), data['Nushu'].tolist()
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return [], []
    except KeyError:
        print("Error: CSV does not contain the required columns. Ensure that 'Chinese' and 'Nushu' columns are present.")
        return [], []
    

def shuffle_and_save_data(chinese_phrases, nushu_phrases):
    """Shuffle the data once and save it to a CSV file."""
    data = list(zip(chinese_phrases, nushu_phrases))
    random.shuffle(data)
    
    shuffled_df = pd.DataFrame(data, columns=["Chinese", "Nushu"])
    shuffled_df.to_csv(SHUFFLED_DATA_PATH, index=False)
    
    print(f"Shuffled data saved to {SHUFFLED_DATA_PATH}.")
    return data


def load_shuffled_data():
    """Load the shuffled data from the CSV file."""
    data = pd.read_csv(SHUFFLED_DATA_PATH)
    return data['Chinese'].tolist(), data['Nushu'].tolist()
    

def load_chinese_only(csv_filepath):
    """Load only Chinese phrases from the CSV."""
    try:
        data = pd.read_csv(csv_filepath)
        return data['Chinese'].tolist()
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return []
    except KeyError:
        print(f"Error: CSV does not contain the required 'Chinese' column.")
        return []


def load_dictionary(csv_filepath):
    """Load the dictionary used for translations."""
    dictionary = {}
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            csv_reader = pd.read_csv(file)
            for row in csv_reader.itertuples():
                # Split the string into individual characters
                chinese_list = [char for char in getattr(row, '对应汉字')]
                # Validate that each entry is a single Chinese character
                if all(len(char) == 1 for char in chinese_list):
                    dictionary[getattr(row, '女书字符')] = chinese_list
                else:
                    print(f"Validation error: Not all entries are single characters in the dictionary for {getattr(row, '女书字符')}")
        return dictionary
    except FileNotFoundError:
        print("Error: Dictionary CSV file not found.")
        return {}
    except KeyError:
        print("Error: Dictionary CSV does not contain the required columns.")
        return {}

    
def combined_prompt_translation(chinese_phrases, nushu_phrases, dictionary, sample_size=35):
    """Provide 35 perfect examples for the API to learn from before translating other sentences."""
    messages = [
        {"role": "system", "content": "You are a translation assistant for Chinese to Nushu. Use these examples and the provided dictionary as reference for future translations. "}
    ]
    messages.append({"role": "system", "content": f"The provided dictionary is: {dictionary}"})
    for i in range(sample_size):
        messages.append({"role": "user", "content": f"Example Chinese: {chinese_phrases[i]}"})
        messages.append({"role": "assistant", "content": f"Example Nushu: {nushu_phrases[i]}"})
    return messages


def translate_chinese_with_examples(chinese_phrases, dictionary, base_messages, retries=7, retry_delay=10):
    """Translate new Chinese phrases with explicit instructions on each translation to ensure task consistency."""
    output_translations = []
    needs_redo = []
    output_file = 'output_translations.csv'
    failed_file = 'failed_translations.csv'

    # Check if output files exist and create them with headers if they don't
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Chinese', 'Nushu'])

    if not os.path.exists(failed_file):
        with open(failed_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Failed Chinese Sentence'])

    for i, phrase in enumerate(chinese_phrases):
        print(f"Translating Chinese sentence {i+1}: {phrase}")

        # Build the current context with the base instructions and the current translation task
        messages = base_messages + [
            {"role": "user",
             "content": (
                 f"Translate '{phrase}' from Chinese to Nushu, one character at a time. "
                 f"One Chinese character can only map to one Nushu character. "
                 f"Use the provided dictionary for allowed mappings, and your best judgment from example learning, to choose the single best Nushu character for each Chinese character. "
                 f"Return only the Nushu translation and nothing else."
             )}
        ]

        translation_successful = False
        attempts = 0

        while not translation_successful and attempts < retries:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.5
                )

                nushu_translation = response.choices[0].message['content']
                print(f"Translated Nushu: {nushu_translation}")

                # Check if the character lengths of the input and output match
                if len(phrase) == len(nushu_translation):
                    output_translations.append((phrase, nushu_translation))
                    # Save the translation to CSV
                    with open(output_file, mode='a', encoding='utf-8', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([phrase, nushu_translation])
                    translation_successful = True
                else:
                    print("Length mismatch detected, redoing translation...")
                    attempts += 1
                    if attempts == retries:
                        print(f"Maximum retries reached for {phrase}, moving to next.")
                        with open(failed_file, mode='a', encoding='utf-8', newline='') as fail_file:
                            writer = csv.writer(fail_file)
                            writer.writerow([phrase])
                    continue

            except openai.error.RateLimitError as e:
                # Handle rate limit and retry
                attempts += 1
                print(f"Rate limit reached: {e}. Retrying in {retry_delay} seconds... (Attempt {attempts}/{retries})")
                time.sleep(retry_delay)

        if translation_successful:
            # After a successful translation, wait for 10 seconds before the next one
            print(f"Waiting 10 seconds before next translation...")
            time.sleep(10)


def main(learning_csv='85perfect.csv', translation_csv=None, sample_size=35, start_index=2, end_index=52):
    dictionary = load_dictionary('data.csv')
    
    # Load and prepare initial data for learning
    if os.path.exists(SHUFFLED_DATA_PATH):
        print(f"Loading shuffled data from {SHUFFLED_DATA_PATH}.")
        chinese_phrases, nushu_phrases = load_shuffled_data()
    else:
        chinese_phrases, nushu_phrases = load_translations(learning_csv)
        shuffle_and_save_data(chinese_phrases, nushu_phrases)

    # Use specified number of examples for the API to learn from
    example_messages = combined_prompt_translation(chinese_phrases, nushu_phrases, dictionary, sample_size=sample_size)
    
    # Adjust for zero-based indexing by subtracting 1 from the 1-based start and end indices
    if translation_csv:
        new_chinese_phrases = load_chinese_only(translation_csv)[start_index-2:end_index-1]
    else:
        new_chinese_phrases = chinese_phrases[start_index-2:end_index-1]

    print(f"Translating sentences {start_index} to {end_index-1} from {translation_csv if translation_csv else learning_csv}.")
    translate_chinese_with_examples(new_chinese_phrases, dictionary, example_messages)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]  # Ignore the script name
    # Default values
    learning_csv = '85perfect.csv'
    translation_csv = None
    sample_size = 35
    start_index = 2
    end_index = 51

    # Parsing command line arguments
    if args:
        learning_csv = args[0]
        if len(args) > 1:
            translation_csv = args[1]
            if len(args) > 2:
                sample_size = int(args[2])
                if len(args) > 3:
                    start_index = int(args[3])
                    if len(args) > 4:
                        end_index = int(args[4])

    main(learning_csv, translation_csv, sample_size, start_index, end_index)

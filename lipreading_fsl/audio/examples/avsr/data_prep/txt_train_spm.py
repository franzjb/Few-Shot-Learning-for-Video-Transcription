import os
import string


def create_txt_spm(input_folder, output_file):
    """
    Iterates through the input_folder (and its subfolders), reads every .txt file (ignoring the output file
    if it's located in the input folder), and writes an output file where:
      - The content of each .txt file is copied exactly,
      - Files that are empty (or contain only whitespace) are skipped,
      - All punctuation (e.g., . ! ? , ; etc.) is removed from the content.
    After each file's content, a newline is added so that the next file's content starts on a new line.

    Parameters:
      - input_folder: The root directory to search for .txt files.
      - output_file: The full path (directory + filename) for the output file.
    """
    # Get the absolute path of the output file for comparison.
    output_file_abs = os.path.abspath(output_file)

    # Create a translation table for removing all punctuation.
    punctuation_table = str.maketrans('', '', string.punctuation)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Walk through the input folder and its subdirectories.
        for dirpath, _, filenames in os.walk(input_folder):
            for filename in filenames:
                if filename.lower().endswith('.txt'):
                    file_path = os.path.join(dirpath, filename)
                    # Skip the output file if it's within the input folder.
                    if os.path.abspath(file_path) == output_file_abs:
                        continue
                    try:
                        with open(file_path, 'r', encoding='utf-8') as in_f:
                            content = in_f.read()
                        # Skip files that are empty or contain only whitespace.
                        if not content.strip():
                            print(f"Skipping empty file: {file_path}")
                            continue

                        # Remove all punctuation using the translation table.
                        filtered_content = content.translate(punctuation_table)

                        # Write the filtered content as is.
                        out_f.write(filtered_content)
                        # Ensure that we end with a newline so that the next file's content starts on a new line.
                        if not filtered_content.endswith("\n"):
                            out_f.write("\n")

                        print(f"Processed file: {file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    print(f"Combined file created at: {output_file}")


create_txt_spm('/home/frb6002/Documents/lipread_files_wav', '/home/frb6002/Documents/lipread_files_wav/txt_training_spm1.txt')

'''
spm_train 
--input=/home/frb6002/Documents/lipread_files_wav/txt_training_spm1.txt --model_prefix=spm_model1 --vocab_size=1023 --character_coverage=1.0 --model_type=unigram

spm_encode
--model=spm_model1.model --output_format=piece < test_corpus.txt | spm_decode --model=spm.model --generate_perplexity


'''
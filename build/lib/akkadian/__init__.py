import os

this_dir, this_filename = os.path.split(__file__)
output_path = os.path.join(this_dir, "output")

hmm_path = os.path.join(output_path, "hmm_model.pkl")
memm_path = os.path.join(output_path, "memm_model.pkl")
bilstm_path = os.path.join(output_path, "bilstm_model.pkl")

dictionary_path = os.path.join(output_path, "dictionary.txt")
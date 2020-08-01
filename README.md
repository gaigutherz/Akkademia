# Translating-Akkadian-using-NLP
Translating Akkadian signs to transliteration using NLP algorithms such as HMM, MEMM and BiLSTM neural networks.

## Getting Started
There are 3 main ways to deploy the project:

	- Website
	
	- Python package
	
	- Github clone

## Website
Use this link to access the website: https://babylonian.herokuapp.com/#/

This project is under "Translit" tab.

## Python Package
These instructions will get you a copy of the project up and running on your local machine testing purposes using akkadian python package.

### Prerequisites
Install Python 3.6 or 3.7 - Link for example (version 3.7.1): https://www.python.org/downloads/release/python-371/

### Installing
Install akkadian package:
```
pip install akkadian
```

### Running
Few examples for running sessions.

Tranliterating akkadian signs:
```
import akkadian.transliterate as akk
print(akk.transliterate("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
```

Tranliterating akkadian signs using BiLSTM:
```
import akkadian.transliterate as akk
print(akk.transliterate_bilstm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
```

Top three options of tranliterating akkadian signs using BiLSTM:
```
import akkadian.transliterate as akk
print(akk.transliterate_bilstm_top3("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
```

Tranliterating akkadian signs using MEMM:
```
import akkadian.transliterate as akk
print(akk.transliterate_memm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
```

Tranliterating akkadian signs using HMM:
```
import akkadian.transliterate as akk
print(akk.transliterate_hmm("𒁹𒀭𒌍𒋀𒈨𒌍𒌷𒁀"))
```

## Github
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
Install Python 3.6 or 3.7 - Link for example (version 3.7.1): https://www.python.org/downloads/release/python-371/

Install git - https://git-scm.com/downloads (Choose your Operating system)

Create a GitHub user - https://github.com/join?source=header-home

### Installing
A step by step series of examples that tell you how to get a development env running.

Install torch:
Windows - 
```
pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Linux and MAC - 
```
pip3 install torch torchvision
```

Install allennlp:
```
pip install allennlp
```

Open pycharm and choose "Check out from version control"

Insert the GitHub project path  - https://github.com/gaigutherz/Translating-Akkadian-using-NLP.git

Insert your GitHub credentials

Clone

Now you can run the project and insert the Akkadian signs!

### Project structure

**BiLSTM_input**: 

	Contains  dictionaries used for transliteration by BiLSTM.
	
**NMT_input**:

	Contains dictionaries used for natural machine translation.
	
**akkadian.egg-info**:

	Inforamtion  and settings for akkadian python package.
	
**akkadian**:

	Sources and train's output.
	
	output:	Train's output for HMM, MEMM and BiLSTM - mostly pickles.
		
	__init__.py: Init script for akkadian python package. Initializes global variables.
	
	bilstm.py:  Class for BiLSTM train and prediction using AllenNLP implementation.
	
	build_data.py: Code for organizing the data in dictionaries.
	
	check_translation.py: Code for translation accuracy checking.
	
	combine_algorithms.py: Code for prediction using both HMM, MEMM and BiLSTM.
	
	data.py: Utils for accuracy checks and dictionaries interpretations.
	
	full_translation_build_data.py: Code for organizing the data for full translation task.
	
	get_texts_details.py: Util for getting more information about the text.
	
	hmm.py: Implementation of HMM for train and prediction.
	
	memm.py: Implementation of MEMM for train and prediction.
	
	parse_json: Json parsing used for data organizing.
	
	parse_xml.py: XML parsing used for data organizing.
	
	train.py: API for training all 3 algorithms and store the output.
	
	translation_tokenize.py: Code for tokenization for translation task.
	
	transliterate.py: API for transliterating using all 3 algorithms.
	
**build/lib/akkadian**:

	Inforamtion  and settings for akkadian python package.
	
**dist**:

	Akkadian python package - wheel and tar.
	
**raw_data**:

	Databases used for  training the models.
	
	random: 4 Texts used for cross era testing.
		
	riao: This project intends to present annotated editions of the entire corpus of Assyrian royal inscriptions, texts that were published in RIMA 1-3.
		
	ribo: This project intends to present annotated editions of the entire corpus of Babylonian royal inscriptions from the Second Dynasty of Isin to the Neo-Babylonian Dynasty (1157-539 BC).
		
	rinap: Presents fully searchable, annotated editions of the royal inscriptions of Neo-Assyrian kings Tiglath-pileser III (744-727 BC), Shalmaneser V (726-722 BC), Sennacherib (704-681 BC), Esarhaddon (680-669 BC), Ashurbanipal (668-631 BC), Aššur-etel-ilāni (630-627 BC), and Sîn-šarra-iškun (626-612 BC).
		
	saao: The online counterpart to the State Archives of Assyria series.
		
	suhu: This project presents annotated editions of the officially commissioned texts of the extant, first-millennium-BC inscriptions of the rulers of Suhu, texts published in Frame, RIMB 2 pp. 275-331.
		
	tei: Databases used for full translation.
		

### Authors
Gai Gutherz

Ariel Elazary

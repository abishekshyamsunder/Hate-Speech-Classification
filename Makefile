run: 
	pip install -r requirements.txt
	python extract_and_clean.py
	python -W ignore baseline_model.py
	python -W ignore simple_model.py
	python -W ignore simple_rnn.py 
	python -W ignore lstm_model.py
	python -W ignore bi_lstm_model.py
	python -W ignore gru_model.py
	python -W ignore bi_gru_model.py
	python -W ignore conv_model.py
	python -W ignore glove_model.py
	python -W ignore TFBERT_model.py

################### PACKAGE ACTIONS #######################

update_requirements:
	@pip install -r requirements.txt

run_preprocess:
	python -c 'from depth_planes.main import preprocess; preprocess()'

run_load_processed_data:
	python -c 'from depth_planes.main import load_processed_data; load_processed_data()'

run_train:
	python -c 'from depth_planes.main import train; train()'

run_evaluate:
	python -c 'from depth_planes.main import evaluate; evaluate()'

run_predict:
	python -c 'from depth_planes.main import predict; predict()'

run_all:
	run_load_processed_data run_train run_evaluate

################### TESTS #################################



################### DATA SOURCES ACTIONS ##################
ML_DIR=~/.lewagon/project

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir ${ML_DIR}/training_outputs
	mkdir ${ML_DIR}/training_outputs/metrics
	mkdir ${ML_DIR}/training_outputs/models
	mkdir ${ML_DIR}/training_outputs/params

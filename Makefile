################### PACKAGE ACTIONS #######################

update_requirements:
	@pip install -r requirements.txt

run_preprocess:
	python -c 'from depth_planes.main import preprocess; preprocess()'

run_train:
	python -c 'from depth_planes.main import train; train()'

################### TESTS #################################



################### DATA SOURCES ACTIONS ##################
ML_DIR=~/.lewagon/project

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir ${ML_DIR}/training_outputs
	mkdir ${ML_DIR}/training_outputs/metrics
	mkdir ${ML_DIR}/training_outputs/models
	mkdir ${ML_DIR}/training_outputs/params

################### PACKAGE ACTIONS #######################

update_requirements:
	@pip install -r requirements.txt

################### TESTS #################################



################### DATA SOURCES ACTIONS ##################
ML_DIR=~/.lewagon/project

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir ${ML_DIR}
	mkdir ${ML_DIR}/training_outputs
	mkdir ${ML_DIR}/training_outputs/metrics
	mkdir ${ML_DIR}/training_outputs/models
	mkdir ${ML_DIR}/training_outputs/params

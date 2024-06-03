################### PACKAGE ACTIONS #######################
reinstall_package:
	@pip uninstall -y 2d-to-plane || :
	@pip install -e .

################### TESTS #################################



################### DATA SOURCES ACTIONS ##################

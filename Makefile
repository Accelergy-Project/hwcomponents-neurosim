build:
	cd hwcomponents_neurosim && rm -rf NeuroSim
	cd hwcomponents_neurosim && mkdir NeuroSim
	cp -r hwcomponents_neurosim/DNN_NeuroSim_V1.3/Inference_pytorch/NeuroSIM/* hwcomponents_neurosim/NeuroSim/
	cd hwcomponents_neurosim && cp -rf drop_in/* ./NeuroSim/
	cd hwcomponents_neurosim && cd NeuroSim ; make

install:
	make
	pip install .

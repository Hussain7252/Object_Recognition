Author:-
Hussain Kanchwala 

OS :
Hussain - Linux (Ubuntu) w/ VSCode and CMake

Instructions for running executables:
CMAKE COMMANDS:
add_executable(vidDisplay src/thresholding.cpp header_files/objfun.h src/function_implement.cpp header_files/csv_util.h src/csv_util.cpp)

-> Run the vidDisplay.exe executable by following the above CMake command, this allows real-time 2D object recognition.
-> The user can select either of the 2 options:
		1 for Nearest Neighbour based classification
		2 for DNN based classificiation
-> User then needs to enter the minimum area for segmented regions to be displayed
-> Then user needs to provide a CSV file of the format : LABEL, FEATURE_VECTOR
-> Display any of the mentioned objects on white background and our system will recognize it.
-> The user can press the 'N' key to add a new object to our system DB with appropriate label
-> The user can create confusion matrix by pressing 'C' and then providing true label and press 'S' for visualization of confusion matrix.
-> network path is the .onnx fie provided

NOTE : All header files should be in .\header_files folder and code files in .\src folder
The .onnx file for DNN should also be in the .\src folder







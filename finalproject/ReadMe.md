Dependents:
	ros-kinetic
	Gazebo7
	openai	: which I made some change and put it in the folder


Directory Structure:
	config: 
		turtlebot2_openai_qlearn_params.yaml : contain all configuration data
	launch:
		learning:
			python_train.launch : used for runing openai environment
		simulation:
			turtlebot: contains all robot launch file
			main.launch: used for runing Gazebo environment
	models: contain all gazebo robots configuration
	worlds: contain all words could be used in project
	Qvalue: store q-tables when doing training
	scripts: contain all python scripts needed in the projects
	report: contain all results of project

Check result of the project:
	in the folder: report
		record.mp4 :video shows the best testing result of training
		report
		sildes	: presentation


Runing Instructions:
	For training:
		roslaunch finalproject main.launch
		roslaunch finalproject python_train.launch
		in directory scripts:
			python my_start_qlearning_maze.py

	For showing training result:
		roslaunch finalproject main.launch
		roslaunch finalproject python_train.launch
		in directory scripts:
			python run_demos.py

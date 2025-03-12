

run:
	@echo "Running main.py"
	@${shell} clear
	@python main.py --config_dir "src/calibration" --videos_path "./videos"

activate:
	@echo "Activating virtual envoriment"
	@pyenv activate compvis

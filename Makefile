

run:
	@echo "Running main.py"
	@python main.py --config_dir "src/calibration" --videos_path "./videos"

activate:
	@echo "Activating virtual envoriment"
	@pyenv activate compvis

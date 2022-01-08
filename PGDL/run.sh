# Toy data
python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission

# Private leaderboard data for a single task
python ingestion_program/ingestion.py /mnt/ssd3/ronan/double_descent/phase_two_data/input_data/ private_complexities ingestion_program sample_code_submission task8

# compute score on private leaderboard task
python scoring_program/score.py /mnt/ssd3/ronan/double_descent/phase_two_data/reference_data/ private_complexities private_scores
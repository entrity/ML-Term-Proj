import os, torch, logging
from . import trainer as sae_trainer

if __name__ == '__main__':
	runner = sae_trainer.Runner(sae_trainer.Trainer)

	master_model = runner.model
	def trainer_factory(*args, **kwargs):
		trainer = sae_trainer.Trainer(*args, **kwargs)
		trainer.master_model = master_model
		return trainer
	runner.trainer_factory = trainer_factory

	runner.model = master_model.get_encoder()
	logging.info('Fine-tuning encoder...')
	runner.run()

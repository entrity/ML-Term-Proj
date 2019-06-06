import os, torch, logging
from . import trainer as sae_trainer

class PreTrainer(sae_trainer.Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.mse_loss = torch.nn.MSELoss()

	def _loss(self, batch, do_acc=False):
		X = batch['X'].cuda()
		out = self.net(X)
		loss = self.mse_loss(out, X)
		if do_acc: self.acc = -1
		return loss

	def test(self):
		loss = self._loss_for_dataloader(self.testloader, 'TEST')
		self.test_losses.append(loss)
		if loss.item() < self.best_test:
			self.best_test = loss.item()
			self.save_model()
		if self.scheduler is not None:
			self.scheduler.step(loss)


if __name__ == '__main__':
	runner = sae_trainer.Runner(PreTrainer)

	master_model = runner.model
	def trainer_factory(*args, **kwargs):
		trainer = PreTrainer(*args, **kwargs)
		trainer.master_model = master_model
		return trainer
	runner.trainer_factory = trainer_factory

	# Run Runner for each level of SAE
	for n in range(1, len(runner.args.arch)):
		runner.args.lr = 1e-1
		runner.model = master_model.subnet(n)
		logging.info(' Running with subnet %d' % (n))
		runner.run()

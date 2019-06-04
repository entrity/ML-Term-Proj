from shared import util
from shared import trainer
from . import net

def add_args(parser):
	parser.add_argument('--randinit', action='store_true', help='Don\'t use pretrained model (Resnet-18)')

if __name__ == '__main__':
	args = util.init(add_args)
	if args.load_path is not None:
		model = net.load(args.load_path)
	elif args.randinit:
		model = net.new(False)
	else:
		model = net.new(True)
	trainer.run(args, model)

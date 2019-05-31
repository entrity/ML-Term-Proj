from shared import util
from shared import trainer
import net

if __name__ == '__main__':
	args = util.init()
	model = net.load(args.load_path) if args.load_path is not None else net.new()
	trainer.run(args, model)

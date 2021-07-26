import argparse
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=0.07, help='learning rate')
    parser.add_argument('--epochs', type=int, default=101, help='number of epochs to train')
    parser.add_argument('--bn', action='store_true', help='is training on remote server')

    opt = parser.parse_args()
    print(opt)

    trainer = Trainer(learning_rate=opt.learning_rate, bn=opt.bn, workers=opt.workers)
    trainer.train(opt.epochs)
    trainer.final_print()

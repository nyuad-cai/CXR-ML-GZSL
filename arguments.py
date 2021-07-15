import argparse

def parse_args():
    argParser = argparse.ArgumentParser(description='arguments')

    argParser.add_argument('--data-root', default='data/nih_chest_xrays', type=str, help='the path to dataset')
    argParser.add_argument('--save-dir', default='checkpoints', type=str, help='the path to save the checkpoints')
    argParser.add_argument('--train-file', default='dataset_splits/train.txt', type=str, help='the path to train list ')
    argParser.add_argument('--val-file', default='dataset_splits/val.txt', type=str, help='the path to val list ')
    argParser.add_argument('--test-file', default='dataset_splits/test.txt', type=str, help='the path to test list')

    argParser.add_argument('--pretrained', dest='pretrained', action='store_true',  help='load imagenet pretrained model')
    argParser.add_argument('--bce-only', dest='bce_only', help='train with only binary cross entropy loss', action='store_true')

    argParser.add_argument('--num-classes', default=14, type=int, help='number of classes')
    argParser.add_argument('--batch-size', default=16, type=int, help='training batch size')
    argParser.add_argument('--epochs', default=40, type=int, help='number of epochs to train')
    argParser.add_argument('--vision-backbone', default='densenet121', type=str, help='[densenet121, densenet169, densenet201]')
    argParser.add_argument('--resume-from', default=None, type=str, help='path to checkpoint to resume the training from')
    argParser.add_argument('--load-from', default=None, type=str, help='path to checkpoint to load the weights from')

    argParser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    argParser.add_argument('--crop', default=224, type=int, help='number of epochs to train')
    argParser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    argParser.add_argument('--steps', default='20, 40, 60, 80', type=str, help='learning rate decay steps comma separated')

    argParser.add_argument('--beta-map', default=0.1, type=float, help='learning rate')
    argParser.add_argument('--beta-con', default=0.1, type=float, help='learning rate')
    argParser.add_argument('--beta-rank', default=1, type=float, help='learning rate')
    argParser.add_argument('--neg-penalty', default=0.03, type=float, help='learning rate')

    argParser.add_argument('--wo-con', dest='wo_con', help='train with out semantic consistency regularizer loss', action='store_true')
    argParser.add_argument('--wo-map', dest='wo_map', help='train with out alignement loss', action='store_true')
    

    argParser.add_argument('--textual-embeddings', default='../text_embeddings/embeddings/nih_chest_xray_biobert.npy', type=str, help='the path to labels embeddings')
    args = argParser.parse_args()
    return args

from .cnn4conv import CNN4Conv
from .mobilenet import MobileNetCifar
from .resnet import *
    
    
def get_model(args):
    if args.model == 'cnn4conv':
        net_glob = CNN4Conv(in_channels=args.in_channels, num_classes=args.num_classes, args=args).to(args.device)
    elif args.model == 'mobilenet':
        net_glob = MobileNetCifar(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet10':
        net_glob = resnet10(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet18':
        net_glob = resnet18(in_channels=args.in_channels, num_classes=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)

    return net_glob
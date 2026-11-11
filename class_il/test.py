import argparse
import json

from trainer import train


ARCHITECTURES = {
    'resac-a': [10, 144, [3, 7, 8, 10, 10], [3, 8, 10, 10, 10]],
    'resac-b': [9, 12, [4, 6, 8, 8, 9], [1, 3, 5, 8, 9]],
}


def run_test():
    parser = argparse.ArgumentParser(description='Evaluate a Class IL configuration.')
    parser.add_argument('--config', default='./exps/wa.json')
    parser.add_argument('--dataset', choices=['cifar100', 'imagenet100'])
    parser.add_argument('--data-root', help='ImageNet-100 folder containing train/ and val/')
    parser.add_argument('--increment', type=int, choices=[5, 10])
    parser.add_argument('--architecture', choices=[*ARCHITECTURES, 'resnet18', 'resnet32'])
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--gpu', help='Override the configured GPU with one GPU ID')
    cli = parser.parse_args()

    with open(cli.config, encoding='utf-8') as config:
        args = json.load(config)
    for key in ('dataset', 'data_root'):
        value = getattr(cli, key)
        if value is not None:
            args[key] = value
    if args['dataset'] == 'imagenet100' and not args.get('data_root'):
        parser.error('--data-root or data_root in the config is required for ImageNet-100')
    if cli.increment is not None:
        args.update(init_cls=cli.increment, increment=cli.increment)
    if cli.seeds is not None:
        args['seed'] = cli.seeds
    if cli.gpu is not None:
        args['device'] = [cli.gpu]

    architecture = cli.architecture
    if architecture in ('resnet18', 'resnet32'):
        args['convnet_type'] = f'memo_{architecture}' if args['model_name'] == 'memo' else architecture
    elif architecture in ARCHITECTURES:
        args['convnet_type'] = 'memo_arch_craft' if args['model_name'] == 'memo' else 'arch_craft'
    elif args['convnet_type'] in ('arch_craft', 'memo_arch_craft'):
        architecture = 'resac-a'
    if architecture in ARCHITECTURES:
        args.update(zip(('depth', 'width', 'pool', 'double'), ARCHITECTURES[architecture]))

    architecture = architecture or args['convnet_type']
    file_id = f"{architecture}-{args['model_name']}-{args['dataset']}-inc{args['increment']}"
    train(args, file_id)


if __name__ == '__main__':
    run_test()

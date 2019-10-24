from data import make_data_loader
from model import AAnet
import torch.optim as optim
import MinkowskiEngine as ME

def crit(pred, gt):
    return nn.CrossEntropyLoss()

def train(net=AAnet(), device, config):
    print(net)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    train_dataloader = make_data_loader(
        'train',
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config)

    val_dataloader = make_data_loader(
        'val',
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config)


    net.train()
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)

    def iter_init():
        optimizer.zero_grad()
        return train_iter.next()

    def iter_update(input):
        output = net(input)
        loss = crit(output.F,data_dict['labels'].to(device))
        loss.backward()
        optimizer.step()

    for i in range(curr_iter, config.max_iter):

        batch = iter_init()
        iter_update(batch)

        if i % config.val_freq == 0 and i > 0:
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'curr_iter': i,
            }, config.weights)
            # Validation
#            logging.info('Validation')
#            test(net, val_iter)

#           scheduler.step()
#           net.train()

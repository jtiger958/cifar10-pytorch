class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LambdaLR:
    def __init__(self, n_epoch, offset, decay_epoch):
        self.n_epoch = n_epoch
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        if int(self.offset + epoch / self.decay_epoch) > 4:
            factor = pow(0.5, 4)
        else:
            factor = pow(0.5, int(self.offset + epoch / self.decay_epoch))
        return factor

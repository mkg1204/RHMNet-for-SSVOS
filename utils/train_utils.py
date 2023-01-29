import time
import torch.distributed as dist


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


def reduce_mean(stats, nprocs):
    for name, value in stats.items():
        rt = value.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        stats[name] = rt / nprocs
    return stats


def update_stats(old_stats, new_stats, batch_size):
    for name, value in new_stats.items():
        if name not in old_stats.keys():
            old_stats[name] = AverageMeter()
        old_stats[name].update(value, n=batch_size)
    return old_stats


def print_stats(epoch, i, loader, times, batch_size, num_frames, stats, log_file=None):
    curr_time = time.time()
    batch_fps = batch_size / (curr_time - times[1])
    avg_fps = num_frames / (curr_time - times[0])
    print_str = '[epoch: %d, %d / %d] ' % (epoch, i, len(loader))
    print_str += 'FPS: %.1f (%.1f)  ,  ' % (avg_fps, batch_fps)
    for name, value in stats.items():
        if hasattr(value, 'avg'):
            print_str += '%s: %.5f  ,  ' % (name, value.avg)
    print(print_str[:-5])

    if log_file:
        log_str = print_str[:-5] + '\n'
        with open(log_file, 'a') as f:
            f.write(log_str)


def write_tb(writer, stats, iter_num):
    for name, value in stats.items():
        writer.add_scalars(name, {name: value.avg}, iter_num)
        stats[name].reset()
    return stats
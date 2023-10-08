from torch.optim.lr_scheduler import LambdaLR
import math

def create_scheduler(args, optimizer):
    print(f'### use {args.sched} scheduler')
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)
    print("### peak_learning_rate, ", args['lr'], flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif args.sched == 'cos':
        eta_min = args.get('eta_min', 0.0)
        eta_max = args.lr
        T_max = args.num_training_steps - args.num_warmup_steps
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return (eta_min + 0.5 * (eta_max - eta_min) * (
                    1.0 + math.cos((current_step - args.num_warmup_steps) / T_max * math.pi))) / eta_max

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler

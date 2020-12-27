import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib;
import time
import datetime

matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=6, num_workers=4, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Hack because of previous bug in code
# TODO: remove, because shouldn't be necessary
if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

# TODO: remove this switch
# metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)
for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

ori_it = it
while True:
    epoch_it += 1
    #     scheduler.step()

    for batch in train_loader:
        it += 1
        now = time.time()
        loss, loss_ori, loss_head1, loss_head2, loss_head3, loss_head4 = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/loss_ori', loss_ori, it)
        logger.add_scalar('train/loss_head1', loss_head1, it)
        logger.add_scalar('train/loss_head2', loss_head2, it)
        logger.add_scalar('train/loss_head3', loss_head3, it)
        logger.add_scalar('train/loss_head4', loss_head4, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print(
                '[Epoch %02d] it=%03d, loss=%.4f,loss_ori=%.4f,loss_head1=%.4f,loss_head2=%.4f,loss_dog3=%.4f,loss_attention=%.4f,per_iter_time=%.3f'
                % (epoch_it, it, loss, loss_ori, loss_head1, loss_head2, loss_head3, loss_head4, time.time() - now))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(data_vis)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            try:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('|WARING:run out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            # eval_dict = trainer.evaluate(val_loader, args, agent, batch, it, ori_it)
            # metric_val = eval_dict[model_selection_metric]
            try:
                fobj = open('/home/lab105/lilei/occupancy_networks-master_batchhead/vaild_value.txt', mode='a')
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
                fobj.write(str(eval_dict) + nowTime)
                fobj.write('\n')
                fobj.close()
            except Exception as e:
                print(e)
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)

'''
Author: Dongchao Yang. 
Modified based on UniAudio training framework: https://github.com/yangdongchao/UniAudio
Target: A Flow-matching based audio codec trainer that transfer the tokens into audio (mel or VAE latent)
Features: support multi-node multi-gpu training. Based on FSDP.
'''
import os
import time
import math
import pickle
import numpy as np
import torch
import argparse
import logging
import json
import functools
import inspect
import torch.distributed as dist
from utils.train_utils import ( seed_everything,  setup_logging, yaml_no_alias_safe_dump,
    save_checkpoint, maybe_resume_checkpoint, WarmupLR, str2bool, find_data_jsons, maybe_resume_checkpoint_fine)
from data.dataloader_cfm import get_data_iterator_tokenizer_vocabulary
from utils.reporter import Reporter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
import torch.nn as nn
from models.AudioDiffusion1D import AudioDiffusion1D
from utils import torch_tools
from models.scalar24k import ScalarAE
from torch.utils.tensorboard import SummaryWriter


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}

    # create optim groups.
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(
        f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(
        f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(
        f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def get_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        base_lr: float = 1e-4,
        end_lr: float = 0.0,
):
    num_warmup_steps = int(num_training_steps * 0.03) # we set the warmup steps to 3% of the total training steps
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_args():
    # TODO: move all argument parsing into utils/arguments.py and make them grouped
    parser = argparse.ArgumentParser()

    # args for randomness
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', default=False, action='store_true', help='set cudnn.deterministic True')

    # args for model
    parser.add_argument('--model', type=str, default='diffusion_transformer_1D', help='the model of name [diffusion_transformer_1D, diffusion_transformer_2D]')
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--num-output-layer", type=int, default=2, help="number of transformer layer for local transformer")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="drop path rate for stochastic depth")
    parser.add_argument("--statistical_prior_path", type=str, help="the statistical prior file path for normalization in LDM")

    # args for data
    parser.add_argument('--train_data_path', type=str, help="the scp path pf training data")
    parser.add_argument('--val_data_path', type=str, help="the scp path pf val data")
    parser.add_argument('--batch_size', type=int, default=2, help="summed sequence length of each batch")
    parser.add_argument('--max_length', type=int, default=8000, help="maximum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--min_length', type=int, default=100, help="minimum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--n_worker', type=int, default=4, help='number of loading workers for each GPU')
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    parser.add_argument('--segment_duration', type=float, default=20, help="the segment duration for training. 20s is the default value")
    
    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=500, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--fine_decoder', type=str2bool, default=False, help='whether to use fine_decoder')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=1000, help="step of warmup")
    parser.add_argument('--data-parallel', type=str, default='fsdp', help='data parallel strategy: fsdp, sdp, hsdp. ')
    parser.add_argument('--mixed-precision', type=str, default='bf16', help='mixed precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--grad-precision', type=str, default='bf16', help='gradient precision: fp32, tf32, bf16, fp16')
    parser.add_argument('--activation-checkpointing', type=bool, default=True, help='use activation checkpointing')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay for AdamW')

    # args for model setting
    parser.add_argument('--n_layer', type=int, default=16, help='the layer of transformer')
    parser.add_argument('--n_head', type=int, default=12, help='the number of multiple head in transformer')
    parser.add_argument('--n_embd', type=int, default=768, help='The embedding dim for transformer')
    parser.add_argument('--dropout', type=float, default=0.0, help='for pretraining 0 is good, for finetuning try 0.1+')
    parser.add_argument('--bias', type=bool, default=False, help='do we use bias inside LayerNorm and Linear layers?')
    parser.add_argument('--block_size', type=int, default=8192, help='exact allowed sequence length in the model')
    parser.add_argument('--prefix-lm', type=str2bool, default=False, help="If true, use prefix LM.")
    parser.add_argument('--num-codebooks', default=1, type=int, help="The number of codec number codebooks for MIMO prediction")
    parser.add_argument('--num-channels', default=32, type=int, help="The channel number of flow-matching target")
    parser.add_argument('--unet-model-name', default='transformer-2d', type=str, help="The name of unet-model")
    parser.add_argument('--transformer_diffusion_config', default='/data4/ydc/music_tokenizer/ckpts/model_config.json', help='the config path for unet model')
    parser.add_argument('--sq-config', type=str, default='config.yaml', help="the path of sqcodec path")
    parser.add_argument('--sq-resume', type=str, default='ckpt', help='the path for SQCodec')
    parser.add_argument('--whisper_path', type=str, default='/data4/ydc/music_tokenizer/ckpts/whisper-large-v2', help='the path for whisper model')


    # args for save model and log: 
    parser.add_argument('--exp_dir', type=str, help='directory of this experiment')
    parser.add_argument('--print_freq', type=int, default=500, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=5000, help='save a checkpoint within an epoch')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')

    args = parser.parse_args()
    return args

def main():
    # (1) use DDP anyway (even for 1 GPU)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(local_rank)

    # (2) arg parsing and logging
    args = get_args()
    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/logs', exist_ok=True)
    else:
        time.sleep(3)
    
    # writer
    if rank==0:
        writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'tensorboard_logs'))
    args.rank = rank
    log_file = args.exp_dir + '/logs/RANK.log'
    setup_logging(rank, world_size, log_file)
    reporter = Reporter()

    # (3) randomness & cudnn settings 
    torch.manual_seed(1337 + args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # (4) data related objects: data iterator, tokenizers, vocabulary
    train_iter, valid_iter = \
        get_data_iterator_tokenizer_vocabulary(
            args=args,
            rank=rank,
            world_size=world_size,
        )
    # (5) save config
    if rank == 0:
        with open(args.exp_dir + "/config.yaml", "w", encoding="utf-8") as f:
            logging.warning(f'Saving the configuration in {args.exp_dir}/config.yaml')
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

    # (6) model, wrapped in FSDP
    
    model = AudioDiffusion1D(
        num_channels= args.num_channels,
        statistical_prior = args.statistical_prior_path,
        pre_trained_model_name = 'whisper&bestrq',
        features_type = 'continuous',
        vq_training = True,
        unet_model_name = args.unet_model_name,
        unet_model_config_path = args.transformer_diffusion_config,
        whisper_path = args.whisper_path,
        snr_gamma = None,
        uncondition = True,
        out_paint = False,
        fine_decoder = args.fine_decoder,
    )
    #vae, stft = build_pretrained_models(ckpt='/data6/ydc/music_tokenizer/ckpts/audioldm_48k.pth')
    SQCodec = ScalarAE(scalar_config=args.sq_config, resume_path=args.sq_resume)
    # define the vae compression model

    print(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    logging.info(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    
    model = setup_fsdp_sync(model, args, torch.cuda.current_device()) # set FSDP
    
    SQCodec = SQCodec.to(torch.cuda.current_device()) # for the vae model, we only use it to exteact features
    for v in SQCodec.parameters(): # fix vae model
        v.requires_grad = False
    print(
        "num. SQCodec params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in SQCodec.parameters()),
            sum(p.numel() for p in SQCodec.parameters() if p.requires_grad),
            sum(p.numel() for p in SQCodec.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in SQCodec.parameters()),
        )
    )
    # (7) objects related to optimization: optimizer and scheduler
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.learning_rate, (0.9, 0.95), rank, logging)
    scheduler = get_cosine_scheduler(
        optimizer,
        num_training_steps=args.n_epoch * len(train_iter) // args.grad_accum,
        num_cycles=0.5,
        last_epoch=-1,
        base_lr=args.learning_rate,
    )
    # (8) Resume model, optimizer, scaler, etc, if needed. 
    if args.fine_decoder:
        maybe_resume_checkpoint_fine(args, model, optimizer, scheduler, reporter, train_iter)
    else:
        maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_iter)
    # statistics
    logging.info(f'model arch: {model}')
    # TODO: more model statistics, like param budget? 

    # (9) training and evaluation
    start_epoch = reporter.get_epoch() + 1
    if start_epoch > args.n_epoch:
        logging.error(f'already reach the maximum training epochs. Done!')

    logging.info("training start ... ")
    for ep in range(start_epoch, args.n_epoch + 1):
        reporter.set_epoch(ep)
        # (10.1) train
        with reporter.observe("train") as sub_reporter:
            train_one_epoch(
              args=args,
              model=model,
              epoch = ep,
              train_dl=train_iter,
              optimizer=optimizer,
              scheduler=scheduler,
              reporter=sub_reporter,
              parent_reporter=reporter,
              SQCodec=SQCodec,
              writer=writer if rank == 0 else None
            )
        # (10.3) epoch logging. 
        logging.info(reporter.log_message())
        # (10.4) save checkpoint
        checkpoint_path = args.exp_dir + f"/ep{ep}.checkpoint"
        logging.info(f"Saving checkpoint file {checkpoint_path}")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, reporter)

def train_one_epoch(args, model, epoch, train_dl, optimizer, scheduler, reporter, parent_reporter, SQCodec, writer):
    model = model.train()
    optimizer.zero_grad()
    global_step = 0
    for b_idx, batch in enumerate(reporter.measure_iter_time(train_dl, "iter_time"), 1):
        global_step += 1
        global_step_among_all = (epoch - 1) * len(train_dl) + b_idx
        audios, spectrograms, durations = batch
        audios = audios.to(torch.cuda.current_device())
        audios = audios[:,:,:int(args.segment_duration*24000)] # control the max size
        
        spectrograms = spectrograms.to(torch.cuda.current_device())
        data_stats = {"batch_size": len(audios), "seq_len": audios.size(2)//24000}
        reporter.register(data_stats)
        vae_frame_rate = 25
        with reporter.measure_time("forward_time"):
            with torch.no_grad():
                true_latent = SQCodec.encode(audios).transpose(1,2) # B,1,T --- > B, T/480, d
            true_latent_mask = torch.zeros(true_latent.shape[0], true_latent.shape[1], dtype=torch.int64, device=true_latent.device)
            for binx in range(len(durations)):
                true_latent_mask[binx, 0:int(durations[binx] * vae_frame_rate)] = 2
            loss_flow, phone_rec_loss, commitment_loss = model(input_audios=audios, spectrograms=spectrograms, latents=true_latent, latent_masks=true_latent_mask)
            loss = loss_flow + commitment_loss + 0.5*phone_rec_loss
            metrics = {'loss_flow': loss_flow.clone().detach(), 
                       'phone_rec_loss': phone_rec_loss.clone().detach(), 'commitment_loss': commitment_loss.clone().detach(), 'loss': loss.clone().detach()} 
            for v in metrics.values(): # Cross-GPU statistics
                dist.all_reduce(v, dist.ReduceOp.AVG)
            reporter.register(metrics)

        with reporter.measure_time("backward_time"):
            loss.backward()

        with reporter.measure_time("optim_time"):
            if b_idx % args.grad_accum == 0:
                grad_norm = model.clip_grad_norm_(args.grad_clip)
                if math.isnan(grad_norm):
                    logging.warning(f"grad norm is NaN. Discard this gradient")
                    optimizer.zero_grad()
                optimizer.step() # update the model even with ill gradient - sync the training
                optimizer.zero_grad()
                scheduler.step()

                if args.rank == 0 and writer is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('train/lr', current_lr, global_step_among_all)
                    #writer.add_scalar('train/loss_tot', metrics['loss_tot'], global_step_among_all)
                    for k,v in metrics.items():
                        # if k != 'loss':
                        writer.add_scalar(f'train/{k}', v, global_step_among_all)
                    if grad_norm is not None:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step_among_all)

            reporter.register({f'lr_param_{i}': pg['lr'] for i, pg in enumerate(optimizer.param_groups)})

        # must call this here so that the saved checkpoint is valid for reporter
        reporter.next()

        if b_idx % args.print_freq == 0:
            logging.info(reporter.log_message(-args.print_freq))
            print(reporter.log_message(-args.print_freq))

        if args.save_interval > 0 and b_idx % args.save_interval == 0:
            checkpoint_path = args.exp_dir + f"/ep{reporter.get_epoch()}-iter{b_idx}.checkpoint"
            logging.info(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            print(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, parent_reporter)
        
if __name__ == '__main__':
    main()    



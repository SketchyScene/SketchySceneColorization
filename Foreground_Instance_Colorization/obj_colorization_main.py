import argparse
import importlib
import os
import sys
import shutil
import json
import tensorflow as tf
from time import gmtime, strftime

sys.path.append('obj_lib')
sys.path.append('data_processing')
import main_procedure

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def launch_training(**kwargs):
    # Deal with file and paths
    appendix = kwargs["resume_from"]
    outputs_base_dir = 'outputs'

    if appendix is None or appendix == '':
        cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        log_dir = os.path.join(outputs_base_dir, cur_time, 'log')
        ckpt_dir = os.path.join(outputs_base_dir, cur_time, 'snapshot')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        kwargs["resume_from"] = appendix
        kwargs["iter_from"] = 0
        appendix = cur_time

        # Save parameters
        with open(os.path.join(log_dir, 'param_%d.json' % 0), 'w') as fp:
            json.dump(kwargs, fp, indent=4)

        from config import Config
        Config.set_from_dict(kwargs)

        print("Launching new train: %s" % cur_time)
    else:
        if len(appendix.split('-')) != 6:
            print("Invalid resume folder")
            return

        log_dir = os.path.join(outputs_base_dir, appendix, 'log')
        ckpt_dir = os.path.join(outputs_base_dir, appendix, 'snapshot')

        # Get last parameters (recover entry point module name)
        json_files = [f for f in os.listdir(log_dir) if
                      os.path.isfile(os.path.join(log_dir, f)) and os.path.splitext(f)[1] == '.json']
        iter_starts = max([int(os.path.splitext(filename)[0].split('_')[1]) for filename in json_files])
        with open(os.path.join(log_dir, 'param_%d.json' % iter_starts), 'r') as fp:
            params = json.load(fp)

        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file is None:
            raise RuntimeError
        else:
            iter_from = int(os.path.split(ckpt_file)[1].split('-')[1]) + 1
        kwargs['log_dir'] = log_dir
        kwargs['ckpt_dir'] = ckpt_dir
        kwargs['iter_from'] = iter_from

        # Save new set of parameters
        with open(os.path.join(log_dir, 'param_%d.json' % iter_from), 'w') as fp:
            json.dump(kwargs, fp, indent=4)

        from config import Config
        Config.set_from_dict(kwargs)
        print("Launching training from checkpoint: %s" % appendix)

    # Launch train
    # from train_paired_aug_multi_gpu import train
    status = main_procedure.train(**kwargs)

    return status, appendix


def launch_val(**kwargs):
    # Deal with file and paths
    appendix = kwargs["resume_from"]
    outputs_base_dir = 'outputs'

    if appendix is None or appendix == '' or len(appendix.split('-')) != 6:
        print("Invalid resume folder")
        return

    ckpt_dir = os.path.join(outputs_base_dir, appendix, 'snapshot')
    results_dir = os.path.join(outputs_base_dir, appendix, 'validation_results')

    # Get latest checkpoint filename
    kwargs['ckpt_dir'] = ckpt_dir
    kwargs['results_dir'] = results_dir

    from config import Config
    Config.set_from_dict(kwargs)
    print("Launching validation from checkpoint: %s" % appendix)

    # Launch validation
    main_procedure.validation(**kwargs)


def launch_test(**kwargs):
    appendix = kwargs["resume_from"]
    outputs_base_dir = 'outputs'

    if appendix is None or appendix == '' or len(appendix.split('-')) != 6:
        print("Invalid resume folder")
        return

    log_dir = os.path.join(outputs_base_dir, appendix, 'log')
    ckpt_dir = os.path.join(outputs_base_dir, appendix, 'snapshot')
    results_dir = os.path.join(outputs_base_dir, appendix, 'test_results')

    # Get latest checkpoint filename
    kwargs['log_dir'] = log_dir
    kwargs['ckpt_dir'] = ckpt_dir
    kwargs['results_dir'] = results_dir


    from config import Config
    Config.set_from_dict(kwargs)
    print("Launching testing from checkpoint: %s" % appendix)

    main_procedure.test()


def launch_inference(**kwargs):
    appendix = kwargs["resume_from"]
    img_name = kwargs["infer_name"]
    instruction = kwargs["instruction"]

    outputs_base_dir = 'outputs'

    if appendix is None or appendix == '' or len(appendix.split('-')) != 6:
        print("Invalid resume folder")
        return

    log_dir = os.path.join(outputs_base_dir, appendix, 'log')
    ckpt_dir = os.path.join(outputs_base_dir, appendix, 'snapshot')
    results_dir = os.path.join(outputs_base_dir, appendix, 'inference_results')

    # Get latest checkpoint filename
    kwargs['log_dir'] = log_dir
    kwargs['ckpt_dir'] = ckpt_dir
    kwargs['results_dir'] = results_dir


    from config import Config
    Config.set_from_dict(kwargs)
    print("Launching inference from checkpoint: %s" % appendix)

    main_procedure.inference(img_name, instruction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-md', type=str, choices=['train', 'val', 'test', 'inference'],
                        default='train', help="choose a mode")
    parser.add_argument('--resume_from', '-rf', type=str, default='',
                        help="Whether resume last checkpoint from a past run")
    parser.add_argument('--batch_size', '-bs', type=int,
                        default=2, help="Batch size per gpu")
    parser.add_argument('--max_iter', '-mi', type=int,
                        default=100000, help="Max number of iterations")
    parser.add_argument('--optimizer', '-opt', type=str, choices=["RMSprop", "Adam", "AdaDelta", "AdaGrad"],
                        default='Adam', help="Optimizer for the graph")
    parser.add_argument('--lr_G', '-lrg', type=float,
                        default=2e-4, help="learning rate for the generator")
    parser.add_argument('--lr_D', '-lrd', type=float,
                        default=1e-4, help="learning rate for the discriminator")
    parser.add_argument('--small_img', '-si', type=int, choices=[0, 1],
                        default=0, help="Whether using 64x64 instead of 256x256")
    parser.add_argument('--lstm_hybrid', '-lh', type=int, choices=[0, 1],
                        default=1, help="Whether use text to control color")
    parser.add_argument('--distance_map', '-dm', type=int, choices=[0, 1],
                        default=0, help="Whether using distance maps for sketches")
    parser.add_argument('--block_type', '-bt', type=str, choices=['MRU', 'Pix2Pix', 'Residual'],
                        default='MRU', help="choose a block_type")
    parser.add_argument('--vocab_size', '-vs', type=int,
                        default=58, help="vocab size")
    parser.add_argument('--disc_iterations', '-di', type=int,
                        default=1, help="Number of discriminator iterations")
    parser.add_argument('--ld', '-ld', type=int,
                        default=10, help="Gradient penalty lambda hyperparameter")
    parser.add_argument('--num_gpu', '-gpu', type=int,
                        default=1, help="Number of GPUs to use")
    parser.add_argument('--extra_info', '-ei', type=str, default='',
                        help="Extra information saved for record")

    parser.add_argument('--summary_write_freq', '-swf', type=int,
                        default=100, help="Write summary frequence")
    parser.add_argument('--save_model_freq', '-smf', type=int,
                        default=10000, help="Save model frequence")
    parser.add_argument('--count_left_time_freq', '-clt', type=int,
                        default=100, help="Count left time frequence")
    parser.add_argument('--count_inception_score_freq', '-cis', type=int,
                        default=-1, help="Count inception score frequence. -1 for not counting")

    parser.add_argument('--infer_name', '-in', type=str, default='',
                        help="The image name of inference")
    parser.add_argument('--instruction', '-ins', type=str, default='',
                        help="The image name of inference")

    args = parser.parse_args()

    if args.mode == 'inference':
        assert args.infer_name != '' and args.instruction != ''

    # Set default params
    d_params = {
        "dataset_type": args.mode,
        "resume_from": args.resume_from,
        "batch_size": args.batch_size,
        "max_iter_step": args.max_iter,
        "disc_iterations": args.disc_iterations,
        "optimizer": args.optimizer,
        "lr_G": args.lr_G,
        "lr_D": args.lr_D,
        "num_gpu": args.num_gpu,
        "small_img": args.small_img,
        "distance_map": args.distance_map,
        "LSTM_hybrid": args.lstm_hybrid,
        "block_type": args.block_type,
        "vocab_size": args.vocab_size,
        "ld": args.ld,
        "extra_info": args.extra_info,
        "summary_write_freq": args.summary_write_freq,
        "save_model_freq": args.save_model_freq,
        "count_left_time_freq": args.count_left_time_freq,
        "count_inception_score_freq": args.count_inception_score_freq,

        "infer_name": args.infer_name,
        "instruction": args.instruction,
    }

    if args.mode == 'train':
        # Launch training
        status, appendix = launch_training(**d_params)
        while status == -1:  # NaN during training
            print("Training ended with status -1. Restarting..")
            d_params["resume_from"] = appendix
            status = launch_training(**d_params)

    elif args.mode == 'val':
        launch_val(**d_params)

    elif args.mode == 'test':
        launch_test(**d_params)

    elif args.mode == 'inference':
        launch_inference(**d_params)
    else:
        raise Exception('Unknown args_mode:', args.mode)

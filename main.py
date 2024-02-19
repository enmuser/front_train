import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='/kaggle/working/front_train/results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    # parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='/kaggle/input/movingmnist/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    # parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    # parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    parser.add_argument('--data_train_path', type=str, default='/kaggle/input/movingmnist/')
    parser.add_argument('--data_val_path', type=str, default='/kaggle/input/movingmnist/')
    parser.add_argument('--data_test_path', type=str, default='/kaggle/input/movingmnist/')
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--real_length', type=int, default=20)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--pred_length', type=int, default=10)
    parser.add_argument('--img_height', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=64)
    parser.add_argument('--sr_size', type=int, default=4)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model_name', type=str, default='mau')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--filter_size_one', type=int, default=(7, 7))
    parser.add_argument('--filter_size_two', type=int, default=(5, 5))
    parser.add_argument('--filter_size_three', type=int, default=(3, 3))
    parser.add_argument('--filter_size_four', type=int, default=(1, 1))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--time', type=int, default=2)
    parser.add_argument('--time_stride', type=int, default=1)
    parser.add_argument('--tau_one', type=int, default=7)
    parser.add_argument('--tau_two', type=int, default=5)
    parser.add_argument('--tau_three', type=int, default=3)
    parser.add_argument('--tau_four', type=int, default=1)
    parser.add_argument('--is_training', type=str, default='True')
    parser.add_argument('--cell_mode', type=str, default='normal')
    parser.add_argument('--model_mode', type=str, default='normal')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.90)
    parser.add_argument('--delay_interval', type=float, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_iterations', type=int, default=1500000)
    parser.add_argument('--train_level_base_line', type=int, default=520000)
    parser.add_argument('--max_epoches', type=int, default=1500000)
    parser.add_argument('--display_interval', type=int, default=1)
    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--snapshot_interval', type=int, default=500)
    parser.add_argument('--num_save_samples', type=int, default=5)
    parser.add_argument('--n_gpu', type=int, default=1)
    # /kaggle/input/movingmnist/model.ckpt-168000
    parser.add_argument('--pretrained_model', type=str, default='/kaggle/input/movingmnist/model.ckpt-177500')
    parser.add_argument('--perforamnce_dir', type=str, default='/kaggle/working/front_train/results/mau/')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/front_train/checkpoints/mau/')
    parser.add_argument('--gen_frm_dir', type=str, default='/kaggle/working/front_train/results/mau/')
    parser.add_argument('--scheduled_sampling', type=bool, default=True)
    parser.add_argument('--sampling_stop_iter', type=int, default=50000)
    parser.add_argument('--sampling_start_value', type=float, default=1.0)
    parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)

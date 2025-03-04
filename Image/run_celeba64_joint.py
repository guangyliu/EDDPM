from templates import *
from templates_latent import *

if __name__ == '__main__':
    gpus = [0, 1]
    conf = celeba64d2c_autoenc_joint()
    conf.latent_infer_path = None
    conf.batch_size = 200 # 100
    conf.ldm_weight = 1.0  # w,   2-w
    conf.T_eval = 100
    conf.latent_T_eval = 100
    conf.name = 'celeba64d2c_2gpu_b100_l0_lc_w1_scale_ft_teval100' #'v6_b100_l0_lc_w1_scale_scldm' #4gpu_noscale_noln_d3_w'+str(conf.ldm_weight)+'_b200_ll_l2l2_detach1_ft'  # ckpt name
    # conf.pretrain = PretrainConfig(
    #     name='200M',
    #     path=f'checkpoints/celeba64d2c_autoenc/last.ckpt', # ae + ldm
    #     # path=f'checkpoints/ffhq128_autoenc/last_aeonly.ckpt', # only ae
    #     # path=f'checkpoints/ciai_bk/last.ckpt'
    #     # path=f'checkpoints/diffae_ckpt/v3/celeba64d2c_2gpu_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt'
    # )
    
    conf.latent_model = PretrainConfig(
        name='200M',
        path=f'checkpoints/celeba64d2c_autoenc_latent/last.ckpt', # ae + ldm
        # path=f'checkpoints/ffhq128_autoenc/last_aeonly.ckpt', # only ae
        # path=f'checkpoints/ciai_bk/last.ckpt'
        # path=f'checkpoints/diffae_ckpt/v3/celeba64d2c_2gpu_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt'
    )

    conf.autoenc = PretrainConfig(
        name='200M',
        path=f'checkpoints/celeba64d2c_autoenc/last.ckpt', # ae + ldm
        # path=f'checkpoints/ffhq128_autoenc/last_aeonly.ckpt', # only ae
        # path=f'checkpoints/ciai_bk/last.ckpt'
        # path=f'checkpoints/diffae_ckpt/v3/celeba64d2c_2gpu_b100_l0_lc_w1_scale_ft/fid_ckpt/best_fid_model.ckpt'
    )
    conf.latent_infer_path = 'checkpoints/celeba64d2c_autoenc/latent.pkl'

    
    # conf.pretrain = None
    # print(conf.lr)
    # conf.lr = 0.00005
    conf.sample_every_samples = 500_000  # sample images, 4 gpus, 10 min
    conf.save_every_samples = 500_000 # save ckpt
    conf.base_dir= 'checkpoints/diffae_ckpt/v3'  
    conf.latent_loss_type = LossType.mse   # LossType.mse  LossType.l1 
    conf.eval_ema_every_samples = 500_000  # eval fid
    conf.eval_every_samples = 500_000
    conf.latent_beta_scheduler = 'const0.008' # const0.008,  latent diffusion, default: const0.008
    conf.beta_scheduler = 'linear' # const0.008 decoder diffusion, default: linear
    # conf.fid_cache = conf.fid_cache+conf.name
    train(conf, gpus=gpus)









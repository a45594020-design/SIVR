import os
import argparse
import torch
import scipy.io as sio
parser = argparse.ArgumentParser()

# TRAINING
parser.add_argument('--batch_size', type=int, default=512, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=32, help="dimension of common factor latent space")
# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# DIRECTORIES
# experiments
parser.add_argument('--dir_experiment', type=str, default='./logs', help="directory to save logs in")
parser.add_argument('--results_dir', type=str, default='./results', help="directory for metrics and per-sample outputs")
parser.add_argument('--run_tag', type=str, default='', help="unique run identifier used to isolate checkpoints and outputs")
parser.add_argument('--data_dir_root', type=str, default='',
                    help="optional dataset root; defaults to ./data/<dataname>")
parser.add_argument('--compact_sample_outputs', action='store_true',
                    help="save only labels, SVC probabilities, Top-5 rankings, and metrics")
parser.add_argument('--num_workers', type=int, default=8, help="DataLoader worker count")
parser.add_argument('--checkpoint_interval', type=int, default=5,
                    help="save a recoverable training state every N completed epochs; 0 disables periodic saves")
parser.add_argument('--resume_checkpoint', type=str, default='',
                    help="explicit recoverable training-state checkpoint to load")
parser.add_argument('--auto_resume', action='store_true',
                    help="resume from dir_checkpoints/training_state_latest.pt when it exists")
parser.add_argument('--dataname', type=str, default='DIR-Wiki', help="dataset")
parser.add_argument('--sbj', type=str, default='sub-03', help="fmri subject")
parser.add_argument('--roi', type=str, default='LVC_HVC_IT', help="ROI")
parser.add_argument('--text_model', type=str, default='GPTNeo', help="text embedding model")
parser.add_argument('--image_model', type=str, default='pytorch/repvgg_b3g4', help="image embedding model")
parser.add_argument('--stability_ratio', type=str, default='', help="stability_ratio")
parser.add_argument('--test_type', type=str, default='zsl', help='normal or zsl')
parser.add_argument('--aug_type', type=str, default='image_text', help='no_aug, image_text, image_only, text_only')
#multimodal
parser.add_argument('--method', type=str, default='joint_elbo', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")
parser.add_argument('--seed', type=int, default=2021, help="random seed for initialization and DataLoader order")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=0.0, help="default initial weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")
parser.add_argument('--lambda1', type=float, default=0.001, help="default weight of intra_mi terms")
parser.add_argument('--lambda2', type=float, default=0.001, help="default weight of inter_mi terms")
parser.add_argument('--lambda_spatial', type=float, default=0.0, help="weight of graph Laplacian spatial regularization")
parser.add_argument('--spatial_laplacian_path', type=str, default='', help="path to the channel graph Laplacian .npy file")
parser.add_argument('--spatial_laplacian_norm', type=bool, default=True, help="normalize graph Laplacian by average diagonal")
parser.add_argument('--lambda_topo', type=float, default=-1.0, help="weight of topology regularization; negative falls back to lambda_spatial")
parser.add_argument('--topology_mode', type=str, default='legacy_pca_laplacian',
                    choices=['legacy_pca_laplacian', 'pre_pca_raw_edges'],
                    help="legacy post-PCA Laplacian or valid pre-PCA raw-space edge topology")
parser.add_argument('--pre_pca_laplacian_path', type=str, default='', help="path to laplacian_pre_pca_*.npz from build_pre_pca_topology.py")
parser.add_argument('--pca_model_path', type=str, default='', help="path to brain_pca_model.npz saved by data preparation")
parser.add_argument('--topology_metadata_path', type=str, default='', help="path to topology_metadata_*.json")
parser.add_argument('--topology_gradient_check', type=bool, default=True, help="fail if topology or base gradient diagnostics are invalid")


FLAGS = parser.parse_args()
data_dir_root = FLAGS.data_dir_root or os.path.join('./data', FLAGS.dataname)
brain_dir = os.path.join(data_dir_root, 'brain_feature', FLAGS.roi, FLAGS.sbj)
image_dir_train = os.path.join(data_dir_root, 'visual_feature/ImageNetTraining', FLAGS.image_model+'-PCA', FLAGS.sbj)
text_dir_train = os.path.join(data_dir_root, 'textual_feature/ImageNetTraining/text', FLAGS.text_model, FLAGS.sbj)

train_brain = sio.loadmat(os.path.join(brain_dir, 'fmri_train_data'+FLAGS.stability_ratio+'.mat'))['data'].astype('double')
train_image = sio.loadmat(os.path.join(image_dir_train, 'feat_pca_train.mat'))['data'].astype('double')#[:,0:3000]
train_text = sio.loadmat(os.path.join(text_dir_train, 'text_feat_train.mat'))['data'].astype('double')
train_brain = torch.from_numpy(train_brain)
train_image = torch.from_numpy(train_image)
train_text = torch.from_numpy(train_text)
dim_brain = train_brain.shape[1]
dim_image = train_image.shape[1]
dim_text = train_text.shape[1]

parser.add_argument('--m1_dim', type=int, default=dim_brain, help="dimension of modality brain")
parser.add_argument('--m2_dim', type=int, default=dim_image, help="dimension of modality image")
parser.add_argument('--m3_dim', type=int, default=dim_text, help="dimension of modality text")
parser.set_defaults(data_dir_root=data_dir_root)

FLAGS = parser.parse_args()
print(FLAGS)

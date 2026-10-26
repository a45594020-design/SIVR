import os
import numpy as np
import math
import random
import itertools
import torch
from torch.autograd import Variable
import torch.distributions as dist
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from divergence_measures.kl_div import calc_kl_divergence
from sklearn.svm import SVC
from sklearn.metrics import top_k_accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import csv
import json
from utils import utils
from utils.TBLogger import TBLogger
import scipy.sparse as sp

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
torch.set_default_tensor_type(torch.DoubleTensor)
TINY = 1e-8
CONSTANT = 1e6
# global variables
SEED = 2021
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def m_cubo( x, qz_x, px_zs, zss):
    lpz = dist.Normal(torch.zeros(1,zss.size(2)).cuda(),torch.ones(1,zss.size(2)).cuda()).log_prob(zss).sum(-1)
    lqz_x = qz_x.log_prob(zss).sum(-1)
    if 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        lpx_z1 = px_zs['image'].log_prob(x['image']).sum(-1)
        lpx_z2 = px_zs['text'].log_prob(x['text']).sum(-1)
        lpx_z3 = px_zs['brain'].log_prob(x['brain']).sum(-1)
        cubo = 0.5*log_mean_exp(2*(lpz+lpx_z1+lpx_z2+lpx_z3-lqz_x))
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        lpx_z1 = px_zs['image'].log_prob(x['image']).sum(-1)
        lpx_z2 = px_zs['text'].log_prob(x['text']).sum(-1)
        cubo = 0.5*log_mean_exp(2*(lpz+lpx_z1+lpx_z2-lqz_x))
    return cubo.mean()

def log_li(x_var, dist_info):
    mean   = dist_info[0]
    std    = dist_info[1]
    epsilon = (x_var - mean) / (std + TINY)
    pi = Variable(torch.ones(1) * np.pi).to(x_var.device)
    logli = - 0.5 * torch.log(2 * pi) - torch.log(std + TINY) - 0.5 * torch.pow(epsilon,2)
    return logli.sum(1)

def mutual_info(exp,px_zs,z):
    if 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q2 = exp.Q2(px_zs['image'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        mi1 = log_li(z,q1).mean()
        mi2 = log_li(z,q2).mean()
        mi3 = log_li(z,q3).mean()
        return mi1 + mi2 + mi3
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' in px_zs.keys():
        q2 = exp.Q2(px_zs['image'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        mi2 = log_li(z,q2).mean()
        mi3 = log_li(z,q3).mean()
        return mi2 + mi3
    elif 'brain' in px_zs.keys() and 'image' not in px_zs.keys() and 'text' not in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        mi1 = log_li(z,q1).mean()
        return mi1
    elif 'brain' not in px_zs.keys() and 'image' in px_zs.keys() and 'text' not in px_zs.keys():
        q2 = exp.Q2(px_zs['image'].loc)
        mi2 = log_li(z,q2).mean()
        return mi2
    elif 'brain' not in px_zs.keys() and 'image' not in px_zs.keys() and 'text' in px_zs.keys():
        q3 = exp.Q3(px_zs['text'].loc)
        mi3 = log_li(z,q3).mean()
        return mi3
    elif 'brain' in px_zs.keys() and 'image' in px_zs.keys() and 'text' not in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q2 = exp.Q2(px_zs['image'].loc)
        mi1 = log_li(z,q1).mean()
        mi2 = log_li(z,q2).mean()
        return mi1+mi2
    elif 'brain' in px_zs.keys() and 'image' not in px_zs.keys() and 'text' in px_zs.keys():
        q1 = exp.Q1(px_zs['brain'].loc)
        q3 = exp.Q3(px_zs['text'].loc)
        mi1 = log_li(z,q1).mean()
        mi3 = log_li(z,q3).mean()
        return mi1+mi3


def calc_log_probs(exp, result, batch):
    mods = exp.modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch[0].keys():
            mod = mods[m_key]
            log_probs[mod.name] = -mod.calc_log_prob(result['rec'][mod.name],
                                                     batch[0][mod.name],
                                                     exp.flags.batch_size)
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
        else:
            mod = mods[m_key]
            log_probs[mod.name] = 0
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
    return log_probs, weighted_log_prob


def calc_klds(exp, result):
    latents = result['latents']['subsets']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=exp.flags.batch_size)
    return klds


def calc_klds_style(exp, result):
    latents = result['latents']['modalities']
    klds = dict()
    for m, key in enumerate(latents.keys()):
        if key.endswith('style'):
            mu, logvar = latents[key]
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=exp.flags.batch_size)
    return klds


def calc_style_kld(exp, klds):
    mods = exp.modalities
    style_weights = exp.style_weights
    weighted_klds = 0.0
    for m, m_key in enumerate(mods.keys()):
        weighted_klds += style_weights[m_key]*klds[m_key+'_style']
    return weighted_klds

def get_lambda_topo(exp):
    lambda_topo = getattr(exp.flags, 'lambda_topo', -1.0)
    if lambda_topo is not None and lambda_topo >= 0:
        return lambda_topo
    return getattr(exp.flags, 'lambda_spatial', 0.0)

def get_spatial_laplacian(exp):
    if getattr(exp, 'spatial_laplacian', None) is not None:
        return exp.spatial_laplacian
    lambda_spatial = get_lambda_topo(exp)
    if lambda_spatial <= 0:
        exp.spatial_laplacian = None
        return None

    path = getattr(exp.flags, 'spatial_laplacian_path', '')
    if not path:
        sbj_tag = exp.flags.sbj.replace('-', '')
        path = 'laplacian_' + sbj_tag + '.npy'
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError('Spatial Laplacian file not found: ' + path)

    laplacian = np.load(path).astype('double')
    if laplacian.shape != (exp.flags.m1_dim, exp.flags.m1_dim):
        raise ValueError('Spatial Laplacian shape {} does not match brain dim {}'.format(
            laplacian.shape, exp.flags.m1_dim))
    if not np.isfinite(laplacian).all():
        raise ValueError('Spatial Laplacian contains NaN or Inf: ' + path)
    diag_mean = np.trace(laplacian) / max(laplacian.shape[0], 1)
    if np.count_nonzero(laplacian) == 0:
        raise ValueError('Spatial Laplacian is all zero: ' + path)
    if diag_mean <= 0:
        raise ValueError('Spatial Laplacian diagonal mean must be positive after graph-stage normalization: ' + str(diag_mean))
    exp.spatial_laplacian = torch.from_numpy(laplacian).to(exp.flags.device)
    print('Loaded legacy PCA-space Laplacian:', path, exp.spatial_laplacian.shape, 'diag_mean=', diag_mean)
    return exp.spatial_laplacian

def get_pre_pca_topology(exp):
    if getattr(exp, 'pre_pca_topology', None) is not None:
        return exp.pre_pca_topology
    lambda_topo = get_lambda_topo(exp)
    if lambda_topo <= 0:
        exp.pre_pca_topology = None
        return None

    lap_path = getattr(exp.flags, 'pre_pca_laplacian_path', '')
    pca_path = getattr(exp.flags, 'pca_model_path', '')
    if not lap_path or not pca_path:
        raise ValueError('pre_pca_raw_edges mode requires --pre_pca_laplacian_path and --pca_model_path.')
    if not os.path.isabs(lap_path):
        lap_path = os.path.join(os.getcwd(), lap_path)
    if not os.path.isabs(pca_path):
        pca_path = os.path.join(os.getcwd(), pca_path)
    if not os.path.exists(lap_path):
        raise FileNotFoundError('Pre-PCA topology Laplacian not found: ' + lap_path)
    if not os.path.exists(pca_path):
        raise FileNotFoundError('PCA model not found: ' + pca_path)

    laplacian = sp.load_npz(lap_path).tocsr().astype(np.float64)
    if laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError('Pre-PCA Laplacian must be square: {}'.format(laplacian.shape))
    if laplacian.nnz == 0:
        raise ValueError('Pre-PCA Laplacian is all zero: ' + lap_path)
    diag_mean = float(laplacian.diagonal().mean())
    if not np.isfinite(diag_mean) or diag_mean <= 0:
        raise ValueError('Pre-PCA Laplacian diagonal mean must be positive: {}'.format(diag_mean))

    pca = np.load(pca_path, allow_pickle=False)
    components = pca['components'].astype(np.float64)
    if components.shape[0] != exp.flags.m1_dim:
        raise ValueError('PCA components first dim {} does not match brain PCA dim {}'.format(
            components.shape[0], exp.flags.m1_dim))
    if components.shape[1] != laplacian.shape[0]:
        raise ValueError('PCA raw dim {} does not match Laplacian raw dim {}'.format(
            components.shape[1], laplacian.shape[0]))

    upper_adjacency = sp.triu(-laplacian, k=1).tocoo()
    keep = upper_adjacency.data > 0
    edge_i = upper_adjacency.row[keep].astype(np.int64)
    edge_j = upper_adjacency.col[keep].astype(np.int64)
    edge_w = upper_adjacency.data[keep].astype(np.float64)
    if edge_w.size == 0:
        raise ValueError('Pre-PCA topology graph has no positive edges: ' + lap_path)

    metadata = {}
    metadata_path = getattr(exp.flags, 'topology_metadata_path', '')
    if metadata_path:
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(os.getcwd(), metadata_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
    graph_edges = int(metadata.get('graph_edges', edge_w.size))
    if graph_edges <= 0:
        raise ValueError('Pre-PCA topology graph metadata reports no edges.')

    device = exp.flags.device
    topology = {
        'components': torch.from_numpy(components).to(device),
        'edge_i': torch.from_numpy(edge_i).long().to(device),
        'edge_j': torch.from_numpy(edge_j).long().to(device),
        'edge_w': torch.from_numpy(edge_w).to(device),
        'raw_dim': int(components.shape[1]),
        'graph_edges': graph_edges,
        'diag_mean': diag_mean,
        'laplacian_path': lap_path,
        'pca_model_path': pca_path,
        'metadata': metadata,
    }
    topology['components'].requires_grad_(False)
    exp.pre_pca_topology = topology
    print('Loaded pre-PCA raw topology:', lap_path)
    print('PCA components shape:', tuple(topology['components'].shape))
    print('Graph nodes/edges:', topology['raw_dim'], topology['graph_edges'])
    print('Laplacian diagonal mean:', topology['diag_mean'])
    return exp.pre_pca_topology

def calc_spatial_regularization(exp, results):
    lambda_spatial = get_lambda_topo(exp)
    zero = torch.zeros((), dtype=torch.double, device=exp.flags.device)
    if lambda_spatial <= 0 or 'brain' not in results['rec']:
        return zero, zero

    topology_mode = getattr(exp.flags, 'topology_mode', 'legacy_pca_laplacian')
    brain_rec = results['rec']['brain'].loc
    if topology_mode == 'pre_pca_raw_edges':
        topology = get_pre_pca_topology(exp)
        if topology is None:
            return zero, zero
        brain_rec_raw = torch.matmul(brain_rec, topology['components'])
        diff = brain_rec_raw[:, topology['edge_i']] - brain_rec_raw[:, topology['edge_j']]
        smoothness = (diff.pow(2) * topology['edge_w']).sum(dim=1).mean()
        if not torch.isfinite(smoothness):
            raise ValueError('Topology loss is non-finite.')
        if not getattr(exp, '_printed_topology_shapes', False):
            print('brain_rec_pca shape:', tuple(brain_rec.shape))
            print('PCA components shape:', tuple(topology['components'].shape))
            print('raw reconstructed response shape:', tuple(brain_rec_raw.shape))
            print('graph nodes/edges:', topology['raw_dim'], topology['graph_edges'])
            print('Laplacian diagonal mean:', topology['diag_mean'])
            print('lambda_topo:', lambda_spatial)
            exp._printed_topology_shapes = True
        return lambda_spatial * smoothness, smoothness

    laplacian = get_spatial_laplacian(exp)
    if laplacian is None:
        return zero, zero
    smoothness = torch.einsum('bi,ij,bj->b', brain_rec, laplacian, brain_rec).mean()
    smoothness = smoothness / max(brain_rec.shape[1], 1)
    if not torch.isfinite(smoothness):
        raise ValueError('Topology loss is non-finite.')
    return lambda_spatial * smoothness, smoothness

def is_finite_tensor(value):
    if not torch.is_tensor(value):
        return True
    return bool(torch.isfinite(value.detach()).all().item())

def finite_backward_step(loss, optimizer, parameters, label):
    if not is_finite_tensor(loss):
        print('skip non-finite loss in ' + label)
        return False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=5.0)
    optimizer.step()
    return True

def grad_norm_from_loss(loss, parameters):
    params = [p for p in parameters if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.zeros((), dtype=torch.double, device=loss.device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return torch.sqrt(total)

def topology_diagnostics(exp, base_loss, topology_loss, label):
    if get_lambda_topo(exp) <= 0 or not getattr(exp.flags, 'topology_gradient_check', True):
        return
    if getattr(exp, '_printed_topology_diagnostics', False):
        return
    if not torch.isfinite(base_loss):
        raise ValueError('Base loss is non-finite in ' + label)
    if not torch.isfinite(topology_loss):
        raise ValueError('Topology loss is non-finite in ' + label)
    topo_grad_norm = grad_norm_from_loss(topology_loss, exp.mm_vae.parameters())
    base_grad_norm = grad_norm_from_loss(base_loss, exp.mm_vae.parameters())
    if float(topo_grad_norm.item()) == 0.0:
        raise ValueError('Topology gradient norm is zero in ' + label)
    if float(base_grad_norm.item()) == 0.0:
        raise ValueError('Base gradient norm is zero in ' + label)
    ratio = get_lambda_topo(exp) * topology_loss.detach() / (base_loss.detach().abs() + TINY)
    print('topology loss:', float(topology_loss.detach().cpu().item()))
    print('base loss:', float(base_loss.detach().cpu().item()))
    print('lambda_topo:', get_lambda_topo(exp))
    print('lambda_topo * topology_loss / base_loss:', float(ratio.cpu().item()))
    print('topology gradient norm:', float(topo_grad_norm.cpu().item()))
    print('base gradient norm:', float(base_grad_norm.cpu().item()))
    exp._printed_topology_diagnostics = True

def shuffle(a):
    return a[torch.randperm(a.size()[0])]

def make_loader(exp, dataset, batch_size, shuffle=True, drop_last=True, seed_offset=0):
    generator = torch.Generator()
    generator.manual_seed(int(getattr(exp.flags, 'seed', SEED)) + int(seed_offset))
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=int(getattr(exp.flags, 'num_workers', 8)), drop_last=drop_last,
                      generator=generator)

def true_neg_idx(data, shuffle_data):
    a = data.mean(1)
    b = shuffle_data.mean(1)
    index = torch.arange(0, len(a))
    idx = index[a!=b]
    return idx

def negative_sample_generator(batch_new, batch, case):
    batch_d = batch_new[0]
    data = batch
    if 'brain' in batch_d.keys() and 'image' in batch_d.keys() and 'text' in batch_d.keys():
        if case==1:
            shuffle_data = shuffle(data[0])
            idx = true_neg_idx(data[0], shuffle_data)
            neg_batch = [shuffle_data[idx,:], data[1][idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 2:
            shuffle_data = shuffle(data[1])
            idx = true_neg_idx(data[1], shuffle_data)
            neg_batch = [data[0][idx,:], shuffle_data[idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 3:
            shuffle_data = shuffle(data[2])
            idx = true_neg_idx(data[2], shuffle_data)
            neg_batch = [data[0][idx,:], data[1][idx,:], shuffle_data[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 4:
            shuffle_data0 = shuffle(data[0])
            idx1 = true_neg_idx(data[0], shuffle_data0)
            shuffle_data1 = shuffle(data[1])
            idx2 = true_neg_idx(data[1], shuffle_data1)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [shuffle_data0[idx,:], shuffle_data1[idx,:], data[2][idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 5:
            shuffle_data0 = shuffle(data[0])
            idx1 = true_neg_idx(data[0], shuffle_data0)
            shuffle_data2 = shuffle(data[2])
            idx2 = true_neg_idx(data[2], shuffle_data2)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [shuffle_data0[idx,:], data[1][idx,:], shuffle_data2[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

        elif case == 6:
            shuffle_data1 = shuffle(data[1])
            idx1 = true_neg_idx(data[1], shuffle_data1)
            shuffle_data2 = shuffle(data[2])
            idx2 = true_neg_idx(data[2], shuffle_data2)
            idx = np.unique(np.concatenate((idx1,idx2),axis=0))
            neg_batch = [data[0][idx,:], shuffle_data1[idx,:], shuffle_data2[idx,:]]
            batch = [neg_batch[0], neg_batch[1], neg_batch[2]]
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = data[3]
            return batch_new

    elif 'brain' not in batch_d.keys() and 'image' in batch_d.keys() and 'text' in batch_d.keys():
        if case == 1:
            shuffle_data = shuffle(data[0])
            idx = true_neg_idx(data[0], shuffle_data)
            neg_batch = [shuffle_data[idx,:], data[1][idx,:]]
            batch = [neg_batch[0], neg_batch[1]]
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0], 'text': batch[1]}
            # batch_new[1] = data[3]
            return batch_new
        elif case == 2:
            shuffle_data = shuffle(data[1])
            idx = true_neg_idx(data[1], shuffle_data)
            neg_batch = [data[0][idx,:], shuffle_data[idx,:]]
            batch = [neg_batch[0], neg_batch[1]]
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0], 'text': batch[1]}
            # batch_new[1] = data[3]
            return batch_new

def basic_routine_epoch(exp, batch, epoch):
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta + epoch * 0.01
    if beta>1.0:
        beta = 1.0
    rec_weight = 1.0
    lambda1 = exp.flags.lambda1


    mm_vae = exp.mm_vae
    batch_d = batch[0]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d)

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    z = results['class_embeddings']
    px_zs = results['rec']
    intra_mi = -mutual_info(exp,px_zs,z)
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)

    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        elbo_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()}
        elbos = dict()
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style']
            else:
                kld_style_m = 0.0
            klds_joint['style'][m_key] = kld_style_m
            if exp.flags.poe_unimodal_elbos:
                if m_key=='brain' and 'brain' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='brain' and 'brain' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key=='image' and 'image' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='image' and 'image' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key == 'text' and 'text' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='text' and 'text' not in batch_d.keys():
                    elbos[m_key] = 0
        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint)
        elbos['joint'] = elbo_joint
        elbo_loss = sum(elbos.values())

    base_loss = elbo_loss + lambda1 * intra_mi
    spatial_loss, topology_loss = calc_spatial_regularization(exp, results)
    total_loss = base_loss + spatial_loss

    out_basic_routine = dict()
    out_basic_routine['results'] = results
    out_basic_routine['log_probs'] = log_probs
    out_basic_routine['total_loss'] = total_loss
    out_basic_routine['base_loss'] = base_loss
    out_basic_routine['klds'] = klds
    out_basic_routine['intra_mi'] = intra_mi
    out_basic_routine['elbo_loss'] = elbo_loss
    out_basic_routine['spatial_loss'] = spatial_loss
    out_basic_routine['topology_loss'] = topology_loss
    return out_basic_routine


def elbo_contrast(exp, batch, epoch):
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    beta = exp.flags.beta + epoch * 0.01
    if beta>1.0:
        beta = 1.0
    rec_weight = 1.0

    mm_vae = exp.mm_vae
    batch_d = batch[0]
    mods = exp.modalities
    for k, m_key in enumerate(batch_d.keys()):
        batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
    results = mm_vae(batch_d, K=30)
    cubo = m_cubo(batch_d, results['qz_x'], results['px_zs'], results['zss'])

    log_probs, weighted_log_prob = calc_log_probs(exp, results, batch)
    group_divergence = results['joint_divergence']

    klds = calc_klds(exp, results)
    z = results['class_embeddings']

    neg_batch_size = z.shape[0]
    if exp.flags.factorized_representation:
        klds_style = calc_klds_style(exp, results)

    if (exp.flags.modality_jsd or exp.flags.modality_moe
        or exp.flags.joint_elbo):
        if exp.flags.factorized_representation:
            kld_style = calc_style_kld(exp, klds_style)
        else:
            kld_style = 0.0
        kld_content = group_divergence
        kld_weighted = beta_style * kld_style + beta_content * kld_content
        elbo_loss = rec_weight * weighted_log_prob + beta * kld_weighted
    elif exp.flags.modality_poe:
        klds_joint = {'content': group_divergence,
                      'style': dict()}
        elbos = dict()
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            if exp.flags.factorized_representation:
                kld_style_m = klds_style[m_key + '_style']
            else:
                kld_style_m = 0.0
            klds_joint['style'][m_key] = kld_style_m
            if exp.flags.poe_unimodal_elbos:
                if m_key=='brain' and 'brain' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='brain' and 'brain' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key=='image' and 'image' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='image' and 'image' not in batch_d.keys():
                    elbos[m_key] = 0
                elif m_key == 'text' and 'text' in batch_d.keys():
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      exp.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(exp, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
                elif m_key=='text' and 'text' not in batch_d.keys():
                    elbos[m_key] = 0

        elbo_joint = utils.calc_elbo(exp, 'joint', log_probs, klds_joint)
        elbos['joint'] = elbo_joint
        elbo_loss = sum(elbos.values())

    # elbo_scale = - elbo_loss / CONSTANT
    elbo_scale = cubo / CONSTANT
    out_basic_routine = dict()
    out_basic_routine['elbo_nega_sample_loss'] = torch.log(elbo_scale.exp().sum() * exp.flags.batch_size / neg_batch_size + TINY) * CONSTANT
    return out_basic_routine

def update_Qnet(exp, batch):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        batch_d = batch[0]
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = Variable(batch_d[m_key]).to(exp.flags.device)
        results = mm_vae(batch_d)
        z = results['class_embeddings']
        px_zs = results['rec']
    intra_mi = -mutual_info(exp, px_zs, z)
    return intra_mi

def train_aug(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae
    lambda2 = exp.flags.lambda2

    if exp.flags.aug_type == 'image_text':
        print('aug type: image_text')
        aug_loader = make_loader(exp, exp.dataset_aug, exp.flags.batch_size,
                                 shuffle=True, drop_last=True, seed_offset=1000 + epoch)
        for iteration, batch in enumerate(aug_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'image':batch[0],'text':batch[1]}
            # batch_new[1] = batch[0]
            batch = [batch[0], batch[1]]
            # Stage 1
            intra_mi = update_Qnet(exp, batch_new)
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            finite_backward_step(intra_mi, exp.optimizer['Qnet'],
                                 itertools.chain(exp.Q1.parameters(), exp.Q2.parameters(), exp.Q3.parameters()),
                                 'Qnet train_aug image_text')

            # Stage 2
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']
            elbo_loss = basic_routine['elbo_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
            basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
            basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

            elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2
            inter_mi_loss = elbo_loss + elbo_nega_sample_loss/2.0

            total_loss = total_loss + lambda2 * inter_mi_loss
            finite_backward_step(total_loss, exp.optimizer['mvae'], exp.mm_vae.parameters(),
                                 'mvae train_aug image_text')
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'text_only':
        print('aug type: text_only')
        aug_loader = make_loader(exp, exp.dataset_aug, exp.flags.batch_size,
                                 shuffle=True, drop_last=True, seed_offset=2000 + epoch)
        for iteration, batch in enumerate(aug_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'text': batch[0]}
            # batch_new[1] = batch[0]

            # Stage 1
            intra_mi = update_Qnet(exp, batch_new)
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            finite_backward_step(intra_mi, exp.optimizer['Qnet'],
                                 itertools.chain(exp.Q1.parameters(), exp.Q2.parameters(), exp.Q3.parameters()),
                                 'Qnet train_aug text_only')

            # Stage 2
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            finite_backward_step(total_loss, exp.optimizer['mvae'], exp.mm_vae.parameters(),
                                 'mvae train_aug text_only')
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'image_only':
        print('aug type: image_only')
        aug_loader = make_loader(exp, exp.dataset_aug, exp.flags.batch_size,
                                 shuffle=True, drop_last=True, seed_offset=3000 + epoch)
        for iteration, batch in enumerate(aug_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'image': batch[0]}
            # batch_new[1] = batch[0]

            # Stage 1
            intra_mi = update_Qnet(exp, batch_new)
            exp.optimizer['Qnet'].zero_grad()
            exp.optimizer['mvae'].zero_grad()
            finite_backward_step(intra_mi, exp.optimizer['Qnet'],
                                 itertools.chain(exp.Q1.parameters(), exp.Q2.parameters(), exp.Q3.parameters()),
                                 'Qnet train_aug image_only')

            # Stage 2
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            total_loss = basic_routine['total_loss']

            exp.optimizer['mvae'].zero_grad()
            exp.optimizer['Qnet'].zero_grad()

            finite_backward_step(total_loss, exp.optimizer['mvae'], exp.mm_vae.parameters(),
                                 'mvae train_aug image_only')
            print('====> Epoch: {:03d} Train loss: {:.4f}'.format(epoch, total_loss))
    elif exp.flags.aug_type == 'no_aug':
        print('aug type: no augmentation')

def train(epoch, exp, tb_logger):
    mm_vae = exp.mm_vae
    mm_vae.train()
    exp.mm_vae = mm_vae
    lambda2 = exp.flags.lambda2

    test_loader = make_loader(exp, exp.dataset_test, exp.flags.batch_size,
                              shuffle=True, drop_last=True, seed_offset=4000 + epoch)

    for iteration, batch in enumerate(test_loader):
        batch_new = {}
        batch_new[0] = dict()
        batch_new[0] = {'image':batch[1],'text':batch[2]}
        # batch_new[1] = batch[3]
        batch = [batch[1],batch[2]]

        # Stage 1
        intra_mi = update_Qnet(exp, batch_new)
        exp.optimizer['Qnet'].zero_grad()
        exp.optimizer['mvae'].zero_grad()
        finite_backward_step(intra_mi, exp.optimizer['Qnet'],
                             itertools.chain(exp.Q1.parameters(), exp.Q2.parameters(), exp.Q3.parameters()),
                             'Qnet train image_text_only')

        # Stage 2
        basic_routine = basic_routine_epoch(exp, batch_new, epoch)
        total_loss = basic_routine['total_loss']
        elbo_loss = basic_routine['elbo_loss']

        exp.optimizer['mvae'].zero_grad()
        exp.optimizer['Qnet'].zero_grad()

        neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

        elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2
        inter_mi_loss = elbo_loss + elbo_nega_sample_loss/2.0

        total_loss = total_loss + lambda2 * inter_mi_loss
        finite_backward_step(total_loss, exp.optimizer['mvae'], exp.mm_vae.parameters(),
                             'mvae train image_text_only')

    d_loader = make_loader(exp, exp.dataset_train, exp.flags.batch_size,
                           shuffle=True, drop_last=True, seed_offset=5000 + epoch)

    for iteration, batch in enumerate(d_loader):
        batch_new = {}
        batch_new[0] = dict()
        batch_new[0] = {'brain':batch[0],'image':batch[1],'text':batch[2]}
        # batch_new[1] = batch[3]

        # Stage 1
        intra_mi = update_Qnet(exp, batch_new)
        exp.optimizer['Qnet'].zero_grad()
        exp.optimizer['mvae'].zero_grad()
        finite_backward_step(intra_mi, exp.optimizer['Qnet'],
                             itertools.chain(exp.Q1.parameters(), exp.Q2.parameters(), exp.Q3.parameters()),
                             'Qnet train trimodal')

        # Stage 2
        basic_routine = basic_routine_epoch(exp, batch_new, epoch)
        results = basic_routine['results']
        total_loss = basic_routine['total_loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']
        intra_mi = basic_routine['intra_mi']
        elbo_loss = basic_routine['elbo_loss']

        exp.optimizer['mvae'].zero_grad()
        exp.optimizer['Qnet'].zero_grad()

        neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=3)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_3 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=4)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_4 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=5)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_5 = basic_routine_contrast['elbo_nega_sample_loss']

        neg_batch_new = negative_sample_generator(batch_new, batch, case=6)
        basic_routine_contrast=elbo_contrast(exp, neg_batch_new, epoch)
        elbo_nega_sample_loss_case_6 = basic_routine_contrast['elbo_nega_sample_loss']

        elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2 + elbo_nega_sample_loss_case_3 + elbo_nega_sample_loss_case_4 + elbo_nega_sample_loss_case_5 + elbo_nega_sample_loss_case_6
        inter_mi_loss = elbo_loss + elbo_nega_sample_loss/6.0
        total_loss = total_loss + lambda2 * inter_mi_loss
        base_loss = basic_routine['base_loss'] + lambda2 * inter_mi_loss
        topology_diagnostics(exp, base_loss, basic_routine['topology_loss'], 'mvae train trimodal')
        finite_backward_step(total_loss, exp.optimizer['mvae'], exp.mm_vae.parameters(),
                             'mvae train trimodal')

        tb_logger.write_training_logs(results, total_loss, log_probs, klds, -inter_mi_loss)
        print('====> Epoch: {:03d} Train loss: {:.4f} ELBO: {:.4f} IntraMI: {:.4f} InterMI: {:.4f}'.format(epoch,
                                                                                                           total_loss,
                                                                                                           -elbo_loss,
                                                                                                           -intra_mi,
                                                                                                           -inter_mi_loss))

def test(epoch, exp, tb_logger):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        lambda2 = exp.flags.lambda2

        d_loader = make_loader(exp, exp.dataset_test, 200,
                               shuffle=True, drop_last=False, seed_offset=6000 + epoch)

        for iteration, batch in enumerate(d_loader):
            batch_new = {}
            batch_new[0] = dict()
            batch_new[0] = {'brain': batch[0], 'image': batch[1], 'text': batch[2]}
            # batch_new[1] = batch[3]
            basic_routine = basic_routine_epoch(exp, batch_new, epoch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            intra_mi = basic_routine['intra_mi']
            elbo_loss = basic_routine['elbo_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=1)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_1 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=2)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_2 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=3)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_3 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=4)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_4 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=5)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_5 = basic_routine_contrast['elbo_nega_sample_loss']

            neg_batch_new = negative_sample_generator(batch_new, batch, case=6)
            basic_routine_contrast = elbo_contrast(exp, neg_batch_new, epoch)
            elbo_nega_sample_loss_case_6 = basic_routine_contrast['elbo_nega_sample_loss']

            elbo_nega_sample_loss = elbo_nega_sample_loss_case_1 + elbo_nega_sample_loss_case_2 + elbo_nega_sample_loss_case_3 + elbo_nega_sample_loss_case_4 + elbo_nega_sample_loss_case_5 + elbo_nega_sample_loss_case_6
            inter_mi_loss = elbo_loss + elbo_nega_sample_loss / 6.0
            total_loss = total_loss + lambda2 * inter_mi_loss
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds, -inter_mi_loss)
            print('====> Epoch: {:03d} Test loss: {:.4f} ELBO: {:.4f} IntraMI: {:.4f} InterMI: {:.4f}'.format(epoch,
                                                                                                           total_loss,
                                                                                                           -elbo_loss,
                                                                                                           -intra_mi,
                                                                                                           -inter_mi_loss))

def image_text_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type=='zsl':
            image_text_data = {'image': exp.dataset_test.tensors[1].cuda(),'text': exp.dataset_test.tensors[2].cuda()}
            label = exp.dataset_test.tensors[3]
            brain = exp.dataset_test.tensors[0]
        elif type=='normal':
            image_text_data = {'image': exp.dataset_val.tensors[1].cuda(),'text': exp.dataset_val.tensors[2].cuda()}
            label = exp.dataset_val.tensors[3]
            brain = exp.dataset_test.tensors[0]
        results = mm_vae(image_text_data)
        z = results['class_embeddings']
        brain_rec = mm_vae.lhoods['brain'](*mm_vae.decoders['brain'](None, z))
        return z.cpu().numpy(), label.cpu().numpy(), brain_rec.loc.cpu().numpy(), brain.cpu().numpy()

def brain_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'brain':exp.dataset_test.tensors[0].cuda()}
            label = exp.dataset_test.tensors[3]
            image = exp.dataset_test.tensors[1]
            text = exp.dataset_test.tensors[2]
        elif type == 'normal':
            data = {'brain':exp.dataset_val.tensors[0].cuda()}
            label = exp.dataset_val.tensors[3]
            image = exp.dataset_test.tensors[1]
            text = exp.dataset_test.tensors[2]
        results = mm_vae(data)
        z = results['class_embeddings']
        image_rec = mm_vae.lhoods['image'](*mm_vae.decoders['image'](None, z))
        text_rec = mm_vae.lhoods['text'](*mm_vae.decoders['text'](None, z))
        return z.cpu().numpy(), label.cpu().numpy(), image_rec.loc.cpu().numpy(), text_rec.loc.cpu().numpy(), image.cpu().numpy(), text.cpu().numpy()

def image_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'image':exp.dataset_test.tensors[1].cuda()}
            label = exp.dataset_test.tensors[3]
        elif type == 'normal':
            data = {'image':exp.dataset_val.tensors[1].cuda()}
            label = exp.dataset_val.tensors[3]
        results = mm_vae(data)
        z = results['class_embeddings']
        return z.cpu().numpy(), label.cpu().numpy()

def text_inference(exp, type):
    with torch.no_grad():
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae
        if type == 'zsl':
            data = {'text':exp.dataset_test.tensors[2].cuda()}
            label = exp.dataset_test.tensors[3]
        elif type == 'normal':
            data = {'text':exp.dataset_val.tensors[2].cuda()}
            label = exp.dataset_val.tensors[3]
        results = mm_vae(data)
        z = results['class_embeddings']
        return z.cpu().numpy(), label.cpu().numpy()

def run_classification_test(exp,observation, type):
    if observation=='image_text':
        z_train = []
        train_label = []
        for i in range(5):
            z, label,brain_rec,brain= image_text_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.squeeze(np.vstack(train_label))

        X = np.concatenate((brain, brain_rec),axis=0)
        sgn=np.concatenate((np.ones_like(np.squeeze(label)),np.zeros_like(np.squeeze(label))),axis=0)

        # tsne = TSNE()
        # X_embedded = tsne.fit_transform(X)
        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=sgn, style=sgn, legend='brief', palette=palette)
        # path = './results/' + exp.flags.dataname + '_' + exp.flags.sbj + '_' + exp.flags.roi + '_' + exp.flags.aug_type + '_' + exp.flags.text_model + '_' + exp.flags.image_model.split('/')[-1] + '_' + str(exp.flags.lambda1) + '_' + str(exp.flags.lambda2) + '_' + str(exp.flags.class_dim) + '_' + exp.flags.method
        # plt.savefig(path+'_brain_vs_brain_rec.pdf',dpi=500)

    elif observation == 'image':
        z_train = []
        train_label = []
        for i in range(5):
            z, label= image_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.squeeze(np.vstack(train_label))
    elif observation == 'text':
        z_train = []
        train_label = []
        for i in range(5):
            z, label= text_inference(exp,type)
            z_train.append(z)
            train_label.append(label)
        z_train = np.vstack(z_train)
        train_label = np.squeeze(np.vstack(train_label))

    z_test, test_label, image_rec, text_rec, image, text = brain_inference(exp,type)
    test_label = np.squeeze(test_label)

    classifiers = [
        SVC(
            gamma=0.00001,
            C=1.0,
            probability=True,
            random_state=int(getattr(exp.flags, 'seed', SEED)),
        ),
        ]
    for clf in classifiers:
        clf.fit(z_train, train_label)
        top1_labels = clf.predict(z_test)
        score = np.mean(top1_labels == test_label)
        print(f"{observation}\n"
              f"Classification report for classifier {clf}:\n"
              f"{score}\n")
        probas = clf.predict_proba(z_test)
        top_acc = top_k_accuracy_score(test_label, probas, k=5)
        print(f"{observation}\n"
              f"Classification Top 5 Acc for classifier {clf}:\n"
              f"{top_acc}\n")
        save_sample_level_outputs(exp, observation, type, clf, probas, top1_labels, z_train,
                                  train_label, z_test, test_label, image_rec,
                                  text_rec, image, text)

    return score, top_acc

def sample_level_npz_path(exp, observation):
    spatial_tag = ''
    if getattr(exp.flags, 'lambda_spatial', 0.0) > 0:
        spatial_tag = '_spatial_' + str(exp.flags.lambda_spatial)
    filename = exp.flags.dataname+'_'+exp.flags.sbj+'_'+exp.flags.roi+'_'+exp.flags.aug_type+'_'+exp.flags.text_model+'_'+exp.flags.image_model.split('/')[-1]+'_'+str(exp.flags.lambda1)+'_'+str(exp.flags.lambda2)+'_'+'_'+str(exp.flags.class_dim)+'_'+exp.flags.method+spatial_tag+'_'+observation+'_sample_level.npz'
    return os.path.join(exp.flags.results_dir, 'sample_level', filename)

def save_sample_level_outputs(exp, observation, test_type, clf, probas, top1_labels, z_train,
                              train_label, z_test, test_label, image_rec,
                              text_rec, image, text):
    out_path = sample_level_npz_path(exp, observation)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    class_labels = np.asarray(clf.classes_)
    top5_indices = np.argsort(-probas, axis=1)[:, :5]
    top5_labels = class_labels[top5_indices]

    if getattr(exp.flags, 'compact_sample_outputs', False):
        compact_names = {'image': 'V', 'text': 'T', 'image_text': 'VT'}
        compact_name = compact_names[observation]
        os.makedirs(exp.flags.results_dir, exist_ok=True)
        labels_path = os.path.join(exp.flags.results_dir, 'y_true.npy')
        class_order_path = os.path.join(exp.flags.results_dir, 'class_order.npy')
        if os.path.exists(labels_path):
            if not np.array_equal(np.load(labels_path), test_label):
                raise ValueError('y_true differs across reference spaces in the same run.')
        else:
            np.save(labels_path, test_label)
        if os.path.exists(class_order_path):
            if not np.array_equal(np.load(class_order_path), class_labels):
                raise ValueError('SVC class order differs across reference spaces in the same run.')
        else:
            np.save(class_order_path, class_labels)
        np.savez_compressed(
            os.path.join(exp.flags.results_dir, 'svc_probabilities_{}.npz'.format(compact_name)),
            scores=probas,
            class_labels=class_labels,
            sample_ids=np.arange(len(test_label), dtype=np.int64),
        )
        np.save(os.path.join(exp.flags.results_dir, 'top1_{}.npy'.format(compact_name)), top1_labels)
        np.save(os.path.join(exp.flags.results_dir, 'top5_{}.npy'.format(compact_name)), top5_labels)
        print('Saved compact sample-level outputs:', exp.flags.results_dir, compact_name)
        return

    prototype_labels = np.unique(train_label)
    prototype_latents = []
    for label in prototype_labels:
        prototype_latents.append(z_train[train_label == label].mean(axis=0))
    prototype_latents = np.vstack(prototype_latents)

    np.savez_compressed(
        out_path,
        sample_ids=np.arange(len(test_label), dtype=np.int64),
        labels=test_label,
        class_labels=class_labels,
        scores=probas,
        top1_labels=top1_labels,
        top5_indices=top5_indices,
        top5_labels=top5_labels,
        brain_latent=z_test,
        reference_latents=z_train,
        reference_labels=train_label,
        prototype_latents=prototype_latents,
        prototype_labels=prototype_labels,
        image_rec=image_rec,
        text_rec=text_rec,
        test_image_reference=image,
        test_text_reference=text,
        observation=np.array([observation]),
        test_type=np.array([test_type]),
        dataname=np.array([exp.flags.dataname]),
        subject=np.array([exp.flags.sbj]),
        roi=np.array([exp.flags.roi]),
        method=np.array([exp.flags.method]),
        lambda1=np.array([exp.flags.lambda1]),
        lambda2=np.array([exp.flags.lambda2]),
        lambda_spatial=np.array([getattr(exp.flags, 'lambda_spatial', 0.0)]),
        class_dim=np.array([exp.flags.class_dim]),
        seed=np.array([int(getattr(exp.flags, 'seed', SEED))]),
    )
    print('Saved sample-level retrieval outputs:', out_path)

def create_csv(path,top1,top5):
    results_dir = os.path.dirname(path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(path,'w') as f:
        csv_writer = csv.writer(f)
        head = ["top1","top5"]
        csv_writer.writerow(head)

def write_csv(path,top1,top5):
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        row = []
        row.append(top1)
        row.append(top5)
        csv_writer.writerow(row)

def result_csv_path(exp, observation):
    spatial_tag = ''
    if getattr(exp.flags, 'lambda_spatial', 0.0) > 0:
        spatial_tag = '_spatial_' + str(exp.flags.lambda_spatial)
    filename = exp.flags.dataname+'_'+exp.flags.sbj+'_'+exp.flags.roi+'_'+exp.flags.aug_type+'_'+exp.flags.text_model+'_'+exp.flags.image_model.split('/')[-1]+'_'+str(exp.flags.lambda1)+'_'+str(exp.flags.lambda2)+'_'+'_'+str(exp.flags.class_dim)+'_'+exp.flags.method+spatial_tag+'_'+observation+'.csv'
    return os.path.join(exp.flags.results_dir, filename)


def training_state_path(exp):
    explicit = getattr(exp.flags, 'resume_checkpoint', '')
    if explicit:
        return os.path.abspath(explicit)
    return os.path.join(exp.flags.dir_checkpoints, 'training_state_latest.pt')


def atomic_save_training_state(exp, completed_epoch):
    checkpoint_path = training_state_path(exp)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'format_version': 1,
        'completed_epoch': int(completed_epoch),
        'next_epoch': int(completed_epoch) + 1,
        'dataset': exp.flags.dataname,
        'subject': exp.flags.sbj,
        'roi': exp.flags.roi,
        'seed': int(exp.flags.seed),
        'run_tag': getattr(exp.flags, 'run_tag', ''),
        'mm_vae': exp.mm_vae.state_dict(),
        'q1': exp.Q1.state_dict(),
        'q2': exp.Q2.state_dict(),
        'q3': exp.Q3.state_dict(),
        'optimizers': {name: optimizer.state_dict()
                       for name, optimizer in exp.optimizer.items()},
        'schedulers': {name: scheduler.state_dict()
                       for name, scheduler in exp.scheduler.items()},
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
    }
    temporary_path = checkpoint_path + '.tmp'
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    metadata_path = checkpoint_path + '.json'
    temporary_metadata_path = metadata_path + '.tmp'
    with open(temporary_metadata_path, 'w', encoding='utf-8') as stream:
        json.dump({
            'format_version': 1,
            'completed_epoch': int(completed_epoch),
            'next_epoch': int(completed_epoch) + 1,
            'checkpoint_path': checkpoint_path,
        }, stream, indent=2)
    os.replace(temporary_metadata_path, metadata_path)
    print('Saved recoverable training state:', checkpoint_path,
          'next_epoch=', int(completed_epoch) + 1)


def load_training_state_if_requested(exp):
    checkpoint_path = training_state_path(exp)
    should_load = bool(getattr(exp.flags, 'resume_checkpoint', ''))
    should_load = should_load or bool(getattr(exp.flags, 'auto_resume', False))
    if not should_load or not os.path.exists(checkpoint_path):
        return int(exp.flags.start_epoch)

    try:
        state = torch.load(
            checkpoint_path, map_location=exp.flags.device, weights_only=False
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location=exp.flags.device)
    expected = {
        'dataset': exp.flags.dataname,
        'subject': exp.flags.sbj,
        'roi': exp.flags.roi,
        'seed': int(exp.flags.seed),
        'run_tag': getattr(exp.flags, 'run_tag', ''),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError('Checkpoint identity mismatch: {}'.format(mismatches))

    exp.mm_vae.load_state_dict(state['mm_vae'])
    exp.Q1.load_state_dict(state['q1'])
    exp.Q2.load_state_dict(state['q2'])
    exp.Q3.load_state_dict(state['q3'])
    for name, optimizer_state in state['optimizers'].items():
        exp.optimizer[name].load_state_dict(optimizer_state)
    for name, scheduler_state in state['schedulers'].items():
        exp.scheduler[name].load_state_dict(scheduler_state)
    random.setstate(state['python_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    torch.set_rng_state(state['torch_rng_state'].cpu())
    if torch.cuda.is_available() and state.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state['cuda_rng_state_all']]
        )

    next_epoch = int(state['next_epoch'])
    if next_epoch < int(exp.flags.start_epoch) or next_epoch > int(exp.flags.end_epoch):
        raise ValueError('Checkpoint next_epoch {} is outside requested range [{}, {}]'.format(
            next_epoch, exp.flags.start_epoch, exp.flags.end_epoch))
    print('Resumed recoverable training state:', checkpoint_path,
          'completed_epoch=', int(state['completed_epoch']),
          'next_epoch=', next_epoch)
    return next_epoch

def run_epochs_trimodal(exp):
    seed = int(getattr(exp.flags, 'seed', SEED))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # initialize summary writer
    writer = SummaryWriter(exp.flags.dir_logs)
    tb_logger = TBLogger(exp.flags.str_experiment, writer)
    str_flags = utils.save_and_log_flags(exp.flags)
    tb_logger.writer.add_text('FLAGS', str_flags, 0)
    lr_list = []
    first_epoch = load_training_state_if_requested(exp)
    print('training epochs progress:')
    for epoch in range(first_epoch, exp.flags.end_epoch):
        utils.printProgressBar(epoch, exp.flags.end_epoch)
        # one epoch of training and testing
        exp.scheduler['Qnet'].step()
        exp.scheduler['mvae'].step()
        lr_list.append(exp.optimizer['Qnet'].state_dict()['param_groups'][0]['lr'])
        train_aug(epoch, exp, tb_logger)
        train(epoch, exp, tb_logger)
        test(epoch, exp, tb_logger)
        # save checkpoints after every 1 epochs
        if (epoch + 1) % 100 == 0 or (epoch + 1) == exp.flags.end_epoch:
            dir_network_epoch = os.path.join(exp.flags.dir_checkpoints, str(epoch).zfill(4))
            if not os.path.exists(dir_network_epoch):
                os.makedirs(dir_network_epoch)
            exp.mm_vae.save_networks()
            torch.save(exp.mm_vae.state_dict(),
                       os.path.join(dir_network_epoch, exp.flags.mm_vae_save))

        checkpoint_interval = int(getattr(exp.flags, 'checkpoint_interval', 5))
        if ((checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0)
                or (epoch + 1) == exp.flags.end_epoch):
            atomic_save_training_state(exp, epoch)

        print('lr = ',lr_list[-1])
    # plt.plot(range(exp.flags.end_epoch), lr_list, color='r')
    # plt.show()

    metrics = {}
    if exp.flags.test_type=='normal':
        top1, top5 = run_classification_test(exp,'image_text', 'normal')
        metrics['VT'] = {'top1': float(top1), 'top5': float(top5)}
    elif exp.flags.test_type=='zsl':
        for observation in ['image', 'text', 'image_text']:
            top1, top5 = run_classification_test(exp, observation, 'zsl')
            metric_name = {'image': 'V', 'text': 'T', 'image_text': 'VT'}[observation]
            metrics[metric_name] = {'top1': float(top1), 'top5': float(top5)}
            path = result_csv_path(exp, observation)
            create_csv(path, top1, top5)
            write_csv(path, top1, top5)

    os.makedirs(exp.flags.results_dir, exist_ok=True)
    metrics_payload = {
        'dataset': exp.flags.dataname,
        'subject': exp.flags.sbj,
        'roi': exp.flags.roi,
        'seed': seed,
        'run_tag': getattr(exp.flags, 'run_tag', ''),
        'topology_mode': getattr(exp.flags, 'topology_mode', ''),
        'lambda_topo': float(get_lambda_topo(exp)),
        'metrics': metrics,
    }
    with open(os.path.join(exp.flags.results_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
    print('Saved run metrics:', os.path.join(exp.flags.results_dir, 'metrics.json'))

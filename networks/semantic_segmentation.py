import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_scatter
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# import spconv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv
from utils.lovasz_losses import lovasz_softmax
from utils.ssc_loss import sem_scal_loss
from networks.point_mamba import Encoder
from networks.point_mamba import Group
from networks.point_mamba import PointMamba

class BasicBlock(spconv.SparseModule):
    def __init__(self, C_in, C_out, indice_key):
        super(BasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(C_in, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(C_out, C_out, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(C_out)
        )
        self.relu2 = spconv.SparseSequential(
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        identity = self.layers_in(x)
        out = self.layers(x)
        output = spconv.SparseConvTensor(sum([i.features for i in [identity, out]]),
                                         out.indices, out.spatial_shape, out.batch_size)
        output.indice_dict = out.indice_dict
        output.grid = out.grid
        return self.relu2(output)


def make_layers_sp(C_in, C_out, blocks, indice_key):
    layers = []
    layers.append(BasicBlock(C_in, C_out, indice_key))
    for _ in range(1, blocks):
        layers.append(BasicBlock(C_out, C_out, indice_key))
    return spconv.SparseSequential(*layers)


def scatter(x, idx, method, dim=0):
    if method == "max":
        return torch_scatter.scatter_max(x, idx, dim=dim)[0]
    elif method == "mean":
        return torch_scatter.scatter_mean(x, idx, dim=dim)
    elif method == "sum":
        return torch_scatter.scatter_add(x, idx, dim=dim)
    else:
        print("unknown method")
        exit(-1)


def gather(x, idx):
    """
    :param x: voxelwise features
    :param idx:
    :return: pointwise features
    """
    return x[idx]

def voxel_sem_target(feature_voxel_coords, point_voxel_coords, point_labels):
    """
    Make sparse voxel tensor of semantic labels using feature voxel coordinates.
    
    Args:
        feature_voxel_coords (M, bxyz): feature-wise voxel coordinates (e.g., vw1_coord)
        point_voxel_coords (N, bxyz): point-wise voxel coordinates
        point_labels (N, ): point-wise semantic label
        
    Return:
        feature_voxel_labels (M, ): voxel-wise semantic label for feature_voxel_coords
    """
    # Create a tensor of point-wise voxel coordinates and labels
    voxel_sem = torch.cat([point_voxel_coords, point_labels.unsqueeze(-1)], dim=-1)
    unq_voxel_sem, unq_sem_count = torch.unique(voxel_sem, return_counts=True, dim=0)
    unq_voxel, unq_ind = torch.unique(unq_voxel_sem[:, :4], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_voxel_sem[:, -1][label_max_ind]
    
    # Now, find the corresponding labels for feature_voxel_coords
    feature_voxel_labels = []
    for coord in feature_voxel_coords:
        # Find the index of the matching unq_voxel
        match_idx = (unq_voxel == coord).all(dim=1).nonzero(as_tuple=True)[0]
        
        # Get the corresponding label and append it to the list
        # Use the first match if there are multiple matches
        if match_idx.size(0) > 0:
            feature_voxel_labels.append(unq_sem[match_idx[0]])
        else:
            feature_voxel_labels.append(0)  # Special value for no matching label
   
    # Convert the list of labels to a tensor
    feature_voxel_labels = torch.tensor(feature_voxel_labels, dtype=torch.long)
    
    return feature_voxel_labels


class SemanticBranch(nn.Module):
    def __init__(self, sizes=[256, 256, 32], nbr_class=19, init_size=32, class_frequencies=None, phase='trainval'):
        super().__init__()
        self.class_frequencies = class_frequencies
        self.sizes = sizes
        self.nbr_class = nbr_class
        # self.conv1_block = SFE(init_size, init_size, "svpfe_0")
        # self.conv2_block = SFE(64, 64, "svpfe_1")
        # self.conv3_block = SFE(128, 128, "svpfe_2")
        
        
        self.mamba1 = PointMamba(trans_dim=init_size, encoder_dims=init_size, output_channels=64, num_group=512)
        self.mamba2 = PointMamba(trans_dim=64, encoder_dims=64, output_channels=128, num_group=256)
        self.mamba3 = PointMamba(trans_dim=128, encoder_dims=128, output_channels=256, num_group=128)
        
        
        # self.proj1_block = SGFE(input_channels=init_size, output_channels=64,\
        #                         reduce_channels=init_size, name="proj1")
        # self.proj2_block = SGFE(input_channels=64, output_channels=128,\
        #                         reduce_channels=64, name="proj2")
        # self.proj3_block = SGFE(input_channels=128, output_channels=256,\
        #                         reduce_channels=128, name="proj3")

        self.phase = phase
        if phase == 'trainval':
            num_class = self.nbr_class  # SemanticKITTI: 19
            self.out2 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.BatchNorm1d(64, ),
                nn.LeakyReLU(0.1),
                nn.Linear(64, num_class)
            )
            self.out4 = nn.Sequential(
                nn.Linear(128, 64, bias=False),
                nn.BatchNorm1d(64, ),
                nn.LeakyReLU(0.1),
                nn.Linear(64, num_class)
            )
            self.out8 = nn.Sequential(
                nn.Linear(256, 64, bias=False),
                nn.BatchNorm1d(64, ),
                nn.LeakyReLU(0.1),
                nn.Linear(64, num_class)
            )
    def reshape_coord(self, coord):
        # 获取batch size
        B = coord[:, 0].max().item() + 1

        # 初始化一个列表来保存每个batch的结果
        coord_list = []

        # 计算所有batch中最大的体素数量
        max_N = 0
        for b in range(B):
            coord_b = coord[coord[:, 0] == b]
            max_N = max(max_N, coord_b.shape[0])

        # 遍历每个batch
        for b in range(B):
            # 获取当前batch的体素
            coord_b = coord[coord[:, 0] == b]

            # 删除第一列（即batch index），使其形状变为(N, 3)
            coord_b = coord_b[:, 1:]

            # 计算需要填充的体素数量
            padding_size = max_N - coord_b.shape[0]

            # 创建一个填充张量，形状为(padding_size, 3)
            padding = torch.full((padding_size, 3), 0, device=coord.device, dtype=coord.dtype)

            # 将填充张量添加到coord_b
            coord_b = torch.cat([coord_b, padding], dim=0)

            # 将结果添加到列表中
            coord_list.append(coord_b)

        # 将结果堆叠在一起，形状为(B, max_N, 3)
        coord = torch.stack(coord_list)

        return coord

    def bev_projection(self, vw_features, vw_coord, sizes, batch_size):
        unq, unq_inv = torch.unique(torch.cat([vw_coord[:, 0].reshape(-1, 1), vw_coord[:, -2:]], dim=-1).int(), return_inverse=True, dim=0)
        bev_fea = scatter(vw_features, unq_inv, method='max')
        #print("unq min:", vw_coord.min(dim=0))
        # print("unq max:", vw_coord.max(dim=0))
        # print("sizes:", sizes)
        bev_dense = spconv.SparseConvTensor(bev_fea, unq.int(), sizes[-2:], batch_size).dense()  # B, C, H, W

        return bev_dense

    def forward_once(self, vw_features, coord_ind, full_coord, pw_label, info):
            batch_size = info['batch']
            if pw_label is not None:
                pw_label = torch.cat(pw_label, dim=0)

            coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1) # N 4
            # vw1_coord : N 4; proj1_vw: N 64
            proj1_vw, vw1_coord = self.mamba1(coord, 128)  
            vw1_coord = vw1_coord.to(dtype=torch.int32)      
            proj1_bev = self.bev_projection(proj1_vw, vw1_coord, (np.array(self.sizes, np.int32) // 2)[::-1], batch_size) #  16, 128, 128
            
            # vw1_coord : N 4; proj1_vw: N 128
            proj2_vw, vw2_coord = self.mamba2(coord, 64)
            vw2_coord = vw2_coord.to(dtype=torch.int32)
            proj2_bev = self.bev_projection(proj2_vw, vw2_coord, (np.array(self.sizes, np.int32) // 4)[::-1], batch_size) # 8, 64, 64

            # vw1_coord : N 4; proj1_vw: N 256
            proj3_vw, vw3_coord = self.mamba3(coord,32)
            vw3_coord = vw3_coord.to(dtype=torch.int32)
            proj3_bev = self.bev_projection(proj3_vw, vw3_coord, (np.array(self.sizes, np.int32) // 8)[::-1], batch_size) # 4, 32, 32
            
            if self.phase == 'trainval':
                index_02 = torch.cat([info[2]['bxyz_indx'][:, 0].unsqueeze(-1),torch.flip(info[2]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
                index_04 = torch.cat([info[4]['bxyz_indx'][:, 0].unsqueeze(-1),torch.flip(info[4]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
                index_08 = torch.cat([info[8]['bxyz_indx'][:, 0].unsqueeze(-1),torch.flip(info[8]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
                
                
                vw_label_02 = voxel_sem_target(vw1_coord, index_02.int(), pw_label.int())
                vw_label_04 = voxel_sem_target(vw2_coord, index_04.int(), pw_label.int())
                vw_label_08 = voxel_sem_target(vw3_coord, index_08.int(), pw_label.int())
            
                # import pdb
                # pdb.set_trace()
                return dict(
                    mss_bev_dense = [proj1_bev, proj2_bev, proj3_bev],
                    mss_logits_list = [
                        [vw_label_02.clone(), self.out2(proj1_vw)],
                        [vw_label_04.clone(), self.out4(proj2_vw)],
                        [vw_label_08.clone(), self.out8(proj3_vw)]]
                )
            

            return dict(
                mss_bev_dense = [proj1_bev, proj2_bev, proj3_bev]
            )


    def forward(self, data_dict, example):
        if self.phase == 'trainval':
            
            out_dict = self.forward_once(data_dict['vw_features'],data_dict['coord_ind'], data_dict['full_coord'], example['points_label'], data_dict['info'])
            all_teach_pair = out_dict['mss_logits_list']

            class_weights = self.get_class_weights().to(device=data_dict['vw_features'].device, dtype=data_dict['vw_features'].dtype)
            loss_dict = {}
            for i in range(len(all_teach_pair)):
                teach_pair = all_teach_pair[i]
                voxel_labels_copy = teach_pair[0].long().clone()
                voxel_labels_copy[voxel_labels_copy == 0] = 256
                voxel_labels_copy = voxel_labels_copy - 1
               
                res04_loss = lovasz_softmax(F.softmax(teach_pair[1], dim=1), voxel_labels_copy.to(device=data_dict['vw_features'].device), ignore=255)
                res04_ssc_loss = sem_scal_loss(teach_pair[1], voxel_labels_copy.to(device=data_dict['vw_features'].device))
                res04_loss2 = F.cross_entropy(teach_pair[1], voxel_labels_copy.to(device=data_dict['vw_features'].device), weight=class_weights, ignore_index=255)
                loss_dict["vw_" + str(i) + "lovasz_loss"] = res04_loss
                loss_dict["vw_" + str(i) + "_ssc_loss"] = res04_ssc_loss
                loss_dict["vw_" + str(i) + "ce_loss"] = res04_loss2
                
                
            return dict(
                mss_bev_dense=out_dict['mss_bev_dense'],
                loss=loss_dict
            )
        else:
            out_dict = self.forward_once(data_dict['vw_features'], data_dict['coord_ind'], data_dict['full_coord'], None, data_dict['info'])
            return out_dict

    def get_class_weights(self):
        '''
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        '''
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(np.array(self.class_frequencies) + epsilon_w))

        return weights

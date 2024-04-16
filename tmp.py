import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_scatter

# import spconv
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.lovasz_losses import lovasz_softmax

import ocnn
from ocnn.octree import Octree
from ocnn.octree import Points

from typing import Optional, List, Dict
from networks.PointMamba import PointMamba

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


def voxel_sem_target(point_voxel_coors, sem_label):
    """make sparse voxel tensor of semantic labels
    Args:
        point_voxel_coors(N, bxyz): point-wise voxel coors
        sem_label(N, ): point-wise semantic label
    Return:
        unq_sem(M, ): voxel-wise semantic label
        unq_voxel(M, bxyz): voxel-wise voxel coors
    """
    voxel_sem = torch.cat([point_voxel_coors, sem_label.reshape(-1, 1)], dim=-1)
    unq_voxel_sem, unq_sem_count = torch.unique(voxel_sem, return_counts=True, dim=0)
    unq_voxel, unq_ind = torch.unique(unq_voxel_sem[:, :4], return_inverse=True, dim=0)
    label_max_ind = torch_scatter.scatter_max(unq_sem_count, unq_ind)[1]
    unq_sem = unq_voxel_sem[:, -1][label_max_ind]
    return unq_sem, unq_voxel


class SFE(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, layer_name, layer_num=1):
        super().__init__()
        self.spconv_layers = make_layers_sp(in_channels, out_channels, layer_num, layer_name)

    def forward(self, inputs):
        conv_features = self.spconv_layers(inputs)
        return conv_features


class SGFE(nn.Module):
    def __init__(self, input_channels, output_channels, reduce_channels, name, p_scale=[2, 4, 6, 8]):
        super().__init__()
        self.inplanes = input_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name

        self.feature_reduce = nn.Linear(input_channels, reduce_channels)
        self.pooling_scale = p_scale
        self.fc_list = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for _, _ in enumerate(self.pooling_scale):
            self.fc_list.append(nn.Sequential(
            nn.Linear(reduce_channels, reduce_channels//2),
            nn.ReLU(),
            ))
            self.fcs.append(nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2)))
        self.scale_selection = nn.Sequential(
            nn.Linear(len(self.pooling_scale) * reduce_channels//2,
                                       reduce_channels),nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(reduce_channels//2, reduce_channels//2, bias=False),
                                nn.ReLU(inplace=False))
        self.out_fc = nn.Linear(reduce_channels//2, reduce_channels, bias=False)
        self.linear_output = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduce_channels, output_channels),
        )

    def forward(self, coords_info, input_data, output_scale, input_coords=None, input_coords_inv=None):

        reduced_feature = F.relu(self.feature_reduce(input_data))
        output_list = [reduced_feature]
        for j, ps in enumerate(self.pooling_scale):
            index = torch.cat([input_coords[:, 0].unsqueeze(-1),
                              (input_coords[:, 1:] // ps).int()], dim=1)
            unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
            fkm = scatter(reduced_feature, unq_inv, method="mean", dim=0)
            att = self.fc_list[j](fkm)[unq_inv]
            out = ( att)
            output_list.append(out)
        scale_features = torch.stack(output_list[1:], dim=1)
        feat_S = scale_features.sum(1)
        feat_Z = self.fc(feat_S)
        attention_vectors = [fc(feat_Z) for fc in self.fcs]
        attention_vectors = torch.sigmoid(torch.stack(attention_vectors, dim=1))
        scale_features = self.out_fc(torch.sum(scale_features * attention_vectors, dim=1))

        output_f = torch.cat([reduced_feature, scale_features], dim=1)
        proj = self.linear_output(output_f)
        proj = proj[input_coords_inv]
        
        index = torch.cat([coords_info[output_scale]['bxyz_indx'][:, 0].unsqueeze(-1),
                           torch.flip(coords_info[output_scale]['bxyz_indx'], dims=[1])[:, :3]], dim=1)

        unq, unq_inv = torch.unique(index, return_inverse=True, dim=0)
        tv_fmap = scatter(proj, unq_inv, method="max", dim=0)
        return tv_fmap, unq, unq_inv

class Points:
    def __init__(self, points, batch_size=1, normals=None, features=None):
        self.points = points  # 点云数据
        self.batch_size = batch_size  # 批次大小
        self.device = points.device  # 设备信息
        self.normals = normals
        self.features = features

    @property
    def batch_id(self):
        # 如果批次大小为1，batch_id可以是None或者一个全零的张量
        # 根据您的具体需求调整
        return torch.zeros(self.points.shape[0], dtype=torch.int64, device=self.device)
    

class SemanticBranch(nn.Module):
    def __init__(self, sizes=[256, 256, 32], nbr_class=19, init_size=32, class_frequencies=None, phase='trainval'):
        super().__init__()
        self.class_frequencies = class_frequencies
        self.sizes = sizes
        self.nbr_class = nbr_class
        
        # self.conv1_block = SFE(init_size, init_size, "svpfe_0")
        # self.conv2_block = SFE(64, 64, "svpfe_1")
        # self.conv3_block = SFE(128, 128, "svpfe_2")

        # self.proj1_block = SGFE(input_channels=init_size, output_channels=64,\
        #                         reduce_channels=init_size, name="proj1")
        # self.proj2_block = SGFE(input_channels=64, output_channels=128,\
        #                         reduce_channels=64, name="proj2")
        # self.proj3_block = SGFE(input_channels=128, output_channels=256,\
        #                         reduce_channels=128, name="proj3")
        
        # 使用 PointMamba 提取特征
        self.conv1_block = PointMamba(in_channels=init_size, channels=[32,64,128], num_blocks=[2,2,18],drop_path=0.5, nempty=True, stem_down=2, \
            fpn_channel=168, head_drop=[0.5, 0.5])
        self.conv2_block = PointMamba(in_channels=64, channels=[32,64,128], num_blocks=[2,2,18],drop_path=0.5, nempty=True, stem_down=2, \
            fpn_channel=168, head_drop=[0.5, 0.5])
        self.conv2_block = PointMamba(in_channels=128, channels=[32,64,128], num_blocks=[2,2,18],drop_path=0.5, nempty=True, stem_down=2, \
            fpn_channel=168, head_drop=[0.5, 0.5])
  
        # 修改 SGFE 模块，接收 PointMamba 提取的特征
        self.proj1_block = SGFE(input_channels=init_size, output_channels=64, reduce_channels=init_size, name="proj1")
        self.proj2_block = SGFE(input_channels=64, output_channels=128, reduce_channels=64, name="proj2")
        self.proj3_block = SGFE(input_channels=128, output_channels=256, reduce_channels=128, name="proj3")

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

    def bev_projection(self, vw_features, vw_coord, sizes, batch_size):
        unq, unq_inv = torch.unique(
            torch.cat([vw_coord[:, 0].reshape(-1, 1), vw_coord[:, -2:]], dim=-1).int(), return_inverse=True, dim=0)
        bev_fea = scatter(vw_features, unq_inv, method='max')
        bev_dense = spconv.SparseConvTensor(bev_fea, unq.int(), sizes[-2:], batch_size).dense() # B, C, H, W

        return bev_dense
    
    def get_input_feature(self, octree):
        # Check if the 'normals' attribute exists and is not None
        if hasattr(self, 'normals') and self.normals is not None:
            octree_feature = ocnn.modules.InputFeature(feature="NDF", nempty=True)
            data = octree_feature(octree)
            return data
        else:
            # Handle the case when 'normals' attribute does not exist or is None
            # Return an empty tensor or a default value
            return torch.tensor([])
         

    def process_batch(self, coord, depth, full_depth):
        def points2octree(points_obj):
            # 确保已经有了ocnn库的正确导入和使用
            octree = ocnn.octree.Octree(depth, full_depth)
            octree.build_octree(points_obj)
            return octree

        # 确保 coord 是在正确的设备上（例如 CUDA 设备）
        coord = coord.cuda(non_blocking=True)

        # 封装 coord 为 Points 类的实例
        points_obj = Points(points=coord, batch_size=1)
        #points_obj.points = points_obj.points.to('cuda:0')

        # 生成八叉树
        octree = points2octree(points_obj)
        #octree.construct_all_neigh()

        # 返回八叉树
        return octree
        # def process_batch(self, batch):
        # def points2octree(points):
            
        #     octree = ocnn.octree.Octree(8, 2)
        #     octree.build_octree(points)
        #     return octree
        
        # if 'octree' in batch:
        #     batch['octree'] = batch['octree'].cuda(non_blocking=True)
        #     batch['coord_ind'] = batch['coord_ind'].cuda(non_blocking=True)
        # else:
        #     #points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        #     # 假设 batch['coord_ind'] 是一个形状为 [N, 4] 的张量
        #     coord_ind = batch['coord_ind']

        #     # 按批次索引对点云数据进行分组
        #     points_by_batch = defaultdict(list)
        #     for i in range(coord_ind.shape[0]):
        #         batch_idx = coord_ind[i, 0].item()
        #         point = coord_ind[i, 1:4]
        #         points_by_batch[batch_idx].append(point)

        #     # 为每个分组创建一个 Points 对象
        #     points_list = []
        #     for batch_idx, points in points_by_batch.items():
        #         points_tensor = torch.stack(points)  # 将点列表转换为张量
        #         points_obj = Points(points=points_tensor, batch_size=1)
        #         points_list.append(points_obj)
            
            
        #     import pdb
        #     pdb.set_trace()
        #     octrees = [points2octree(pts) for pts in points_list]
        #     octree = ocnn.octree.merge_octrees(octrees)
        #     octree.construct_all_neigh()
        #     batch['coord_ind'] = ocnn.octree.merge_points(points_list)
        #     batch['octree'] = octree
        # return batch
    def forward_once(self, vw_features, coord_ind, full_coord, pw_label, info):
        batch_size = info['batch']
        if pw_label is not None:
            pw_label = torch.cat(pw_label, dim=0)

       
       # Convert voxel coordinates to octree
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        coord = coord[:, 1:]
        
        depth = 6
        octree = self.process_batch(coord, depth=6, full_depth=2)        
        data = self.get_input_feature(octree)
        import pdb
        pdb.set_trace()
        mamba_features1 = self.conv1_block(vw_features, octree, depth)
        mamba_features2 = self.conv2_block(vw_features, octree, depth)
        mamba_features3 = self.conv3_block(vw_features, octree, depth)
        
   

        # Use PointMamba features in the rest of the forward pass
        proj1_vw, vw1_coord, pw1_coord = self.proj1_block(info, mamba_features1, output_scale=2, input_coords=coord.int(),
            input_coords_inv=full_coord)
        proj1_bev = self.bev_projection(proj1_vw, vw1_coord, (np.array(self.sizes, np.int32) // 2)[::-1], batch_size)

        proj2_vw, vw2_coord, pw2_coord = self.proj2_block(info, mamba_features2, output_scale=4, input_coords=vw1_coord.int(),
            input_coords_inv=pw1_coord)
        proj2_bev = self.bev_projection(proj2_vw, vw2_coord, (np.array(self.sizes, np.int32) // 4)[::-1], batch_size)

        proj3_vw, vw3_coord, _ = self.proj3_block(info, mamba_features3, output_scale=8, input_coords=vw2_coord.int(),
            input_coords_inv=pw2_coord)
        proj3_bev = self.bev_projection(proj3_vw, vw3_coord, (np.array(self.sizes, np.int32) // 8)[::-1], batch_size)

        
        # input_tensor = spconv.SparseConvTensor(
        #     vw_features, coord.int(), np.array(self.sizes, np.int32)[::-1], batch_size
        # )
        # conv1_output = self.conv1_block(input_tensor)
        # proj1_vw, vw1_coord, pw1_coord = self.proj1_block(info, conv1_output.features, output_scale=2, input_coords=coord.int(),
        #     input_coords_inv=full_coord)
        # proj1_bev = self.bev_projection(proj1_vw, vw1_coord, (np.array(self.sizes, np.int32) // 2)[::-1], batch_size)

        # conv2_input_tensor = spconv.SparseConvTensor(
        #     proj1_vw, vw1_coord.int(), (np.array(self.sizes, np.int32) // 2)[::-1], batch_size
        # )
        # conv2_output = self.conv2_block(conv2_input_tensor)
        # proj2_vw, vw2_coord, pw2_coord = self.proj2_block(info, conv2_output.features, output_scale=4, input_coords=vw1_coord.int(),
        #     input_coords_inv=pw1_coord)
        # proj2_bev = self.bev_projection(proj2_vw, vw2_coord, (np.array(self.sizes, np.int32) // 4)[::-1], batch_size)

        # conv3_input_tensor = spconv.SparseConvTensor(
        #     proj2_vw, vw2_coord.int(), (np.array(self.sizes, np.int32) // 4)[::-1], batch_size
        # )
        # conv3_output = self.conv3_block(conv3_input_tensor)
        # proj3_vw, vw3_coord, _ = self.proj3_block(info, conv3_output.features, output_scale=8, input_coords=vw2_coord.int(),
        #     input_coords_inv=pw2_coord)
        # proj3_bev = self.bev_projection(proj3_vw, vw3_coord, (np.array(self.sizes, np.int32) // 8)[::-1], batch_size)


        if self.phase == 'trainval':
            index_02 = torch.cat([info[2]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(info[2]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            index_04 = torch.cat([info[4]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(info[4]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            index_08 = torch.cat([info[8]['bxyz_indx'][:, 0].unsqueeze(-1),
                               torch.flip(info[8]['bxyz_indx'], dims=[1])[:, :3]], dim=1)
            vw_label_02 = voxel_sem_target(index_02.int(), pw_label.int())[0]
            vw_label_04 = voxel_sem_target(index_04.int(), pw_label.int())[0]
            vw_label_08 = voxel_sem_target(index_08.int(), pw_label.int())[0]
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
            out_dict = self.forward_once(data_dict['vw_features'], 
                data_dict['coord_ind'], data_dict['full_coord'], example['points_label'], data_dict['info'])
            all_teach_pair = out_dict['mss_logits_list']

            class_weights = self.get_class_weights().to(device=data_dict['vw_features'].device, dtype=data_dict['vw_features'].dtype)
            loss_dict = {}
            for i in range(len(all_teach_pair)):
                teach_pair = all_teach_pair[i]
                voxel_labels_copy = teach_pair[0].long().clone()
                voxel_labels_copy[voxel_labels_copy == 0] = 256
                voxel_labels_copy = voxel_labels_copy - 1

                res04_loss = lovasz_softmax(F.softmax(teach_pair[1], dim=1), voxel_labels_copy, ignore=255)
                res04_loss2 = F.cross_entropy(teach_pair[1], voxel_labels_copy, weight=class_weights, ignore_index=255)
                loss_dict["vw_" + str(i) + "lovasz_loss"] = res04_loss
                loss_dict["vw_" + str(i) + "ce_loss"] = res04_loss2
            return dict(
                mss_bev_dense=out_dict['mss_bev_dense'],
                loss=loss_dict
            )
        else:
            out_dict = self.forward_once(data_dict['vw_features'],
                data_dict['coord_ind'], data_dict['full_coord'], None, data_dict['info'])
            return out_dict

    def get_class_weights(self):
        '''
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        '''
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(np.array(self.class_frequencies) + epsilon_w))

        return weights

"""Adapted from https://arxiv.org/abs/2406.11944 for Mamba feature tracking"""

import torch
import torch.nn as nn
from torch import einsum
from typing import Optional, Union, List, Dict
from functools import partial
import numpy as np
import copy
import enum
from dataclasses import dataclass
from mambacoder import MambaCoder
import torch.nn.functional as F

# Adapted from https://github.com/AmeenAli/HiddenMambaAttn/blob/main/mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py#L741
def compute_attn_matrix_fn(dA: torch.Tensor, dB: torch.Tensor, C: torch.Tensor):
    B, L, ED, N = dA.shape
    attn_matrices = torch.zeros((B, ED, L, L), device=dA.device) # B, ED, L, L
    for r in range(L):
        # This is the output idx
        # C is equal for each head, it depends solely on the output idx
        curr_C = C[:, r, :] # (B, N)
        for c in range(r+1):
            # Now store the As, we are going from c to r (c, c+1, c+2, ..., r)
            curr_A = torch.ones((B, ED, N), device=dA.device)
            if c < r:
                for i in range(r-c):
                    curr_A = curr_A * dA[:, r-i, :, :]
            curr_B = dB[:, c, :, :]
            attn_matrices[:, :, r, c] = torch.sum(curr_C * curr_A * curr_B, axis=-1)
    return attn_matrices

# Adapted from https://github.com/Itamarzimm/UnifiedImplicitAttnRepr/blob/main/MambaVision/mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py#L828
def conv2mat(conv1d_weight: torch.Tensor, L: int):
    """
    Given a convolutional weight matrix return a Toeplitz matrix T
    such that:
     - T * x = conv(x)
    """  
    H, D = conv1d_weight.shape
    conv1d_weight = torch.flip(conv1d_weight,[-1])
    # Initialize the matrix with zeros
    M = torch.zeros(H, L, L).to(conv1d_weight.device)
    
    # Fill the matrix with the kernel weights
    for h in range(H):
        for i in range(L):
            # Set the diagonal and the next (D-1) positions, respecting the input length L
            M[h, i, i:i+D] = conv1d_weight[h, :max(0, D - (i + D - L))]
    return M.flip(dims=[-1])

def get_rmsnorm_constant(cache, l: int, t: int):

    # Take inputs before and after normalization
    pre_norm = cache[l]["pre_rms"][0, t, :]
    post_norm = cache[l]["inputs"][0, t, :]

    # Calculate rmsnorm constant
    return post_norm / pre_norm

class ComponentType(enum.Enum):
    # Components for which we track features
    # We are tracking features from mixer and from the gate
    MIXER = 'mixer'
    BLOCK = 'block'
    EMBED = 'embed'

    def __str__(self):
        return self.value

class FeatureType(enum.Enum):
    # In our case we have only TC but it can be expanded
    NONE = 'none'
    MAMBACODER = 'mc'

    def __str__(self):
        return self.value

@dataclass
class Component:
    """This is the Component type that contains the feature vector."""
    layer: int 
    component_type: ComponentType
    token: Optional[int] = None
    attn_head: Optional[int] = None # this will be replaced by Mamba head
    feature_idx: Optional[int] = None # the index of the activated feature in layer
    feature_type: Optional[FeatureType] = None # TC or something else

    def __str__(self, show_token=True):
        base_str = f"{self.component_type}{self.layer}" # like mixer3
        attn_str = "" if self.component_type != ComponentType.MIXER else f"[{self.attn_head}]" # like mixer3[2]

        feature_str = ""
        if self.feature_type is not None and self.feature_idx is not None:
            feature_str = f"{self.feature_type}[{self.feature_idx}]" # Here we have MC[124]
        
        token_str = ""
        if self.token is not None and show_token:
            token_str = f"@{self.token}"

        ret_str = base_str + attn_str + feature_str + token_str
        return ret_str
    
    def __repr__(self):
        return f"<Component object {str(self)}>"

# Helper class for tracking causal chains: which gates caused the vector to activate?
@dataclass
class FeatureVector:
    component_path: List[Component]
    vector: torch.Tensor
    layer: int
    sublayer: str # e.g resid_pre
    token: Optional[int] = None
    contrib: Optional[float] = None # contribution calculated with TC
    error: float = 0.0

    def __str__(self, show_full=True, show_contrib=True, show_last_token=True):
        retstr = ""
        token_str = '' if self.token is None or not show_last_token else f"@{self.token}"
        if len(self.component_path) > 0:
            if show_full:
                retstr = "->".join(x.__str__(show_token=True) for x in self.component_path[:-1])
            retstr = ''.join([retstr, self.component_path[-1].__str__(show_token=True)])
        else:
            retstr = f'*{self.sublayer}{self.layer}{token_str}'
        if show_contrib and self.contrib is not None:
            retstr = ''.join([retstr, f": {self.contrib:.2}"])
        return retstr

    def __repr__(self):
        return f"<FeatureVector object {str(self)}, sublayer={self.sublayer}>"

@torch.no_grad()
def get_top_mambacoder_features(transcoder: MambaCoder, f: FeatureVector, layer: int, t: int, k: int):
    """
    Calculate top mambacoder features from layer l.
    This computes, for every feature j in layer l:
        f' = f_enc(l, j) * f_dec(l, j)
    """
    # Push feature f to device
    feature_vector = f.vector

    # This is f_dec times feature vector
    pulledback_features = transcoder.decoders[layer].weight.T @ feature_vector # (nf, out) x (out)

    # Now we need to calculate acts from MC
    x = transcoder.base_model.cache[layer]["inputs"].squeeze()[t, :] # (D)
    acts = transcoder.encoders[layer](x) # (num_features)

    # Calculate contributions
    contribs = acts * pulledback_features

    # Take top k contribs
    topk_vals, topk_idxs = torch.topk(contribs, k=k)

    top_contrib_list = []

    for contrib, idx in zip(topk_vals, topk_idxs):
        
        # Reconstruct feature vector from idx
        vector = transcoder.encoders[layer].weight[idx, :] # (D)
        vector = vector * (transcoder.decoders[layer].weight.T @ feature_vector)[idx] # (is D dimensional)
        
        new_component = Component(
            layer=layer,
            component_type=ComponentType.BLOCK,
            token=t,
            feature_type=FeatureType.MAMBACODER,
            feature_idx=idx.item() # New component from index idx in layer l
        )

        top_contrib_list.append(FeatureVector(
            component_path=f.component_path + [new_component],
            vector=vector,
            layer=layer,
            sublayer="resid_pre",
            contrib=contrib.item(),
            token=t
        ))
    
    return top_contrib_list

@torch.no_grad()
def get_top_mamba_features(transcoder: MambaCoder, cache: Dict, f: FeatureVector, layer: int, t: int, k: int = 5):

    # Extract feature vector
    feature_vector = f.vector

    # First we need to push back feature before out_proj
    W3 = transcoder.base_model.backbone.layers[layer].mixer.out_proj.weight
    feature_vector = W3.T @ feature_vector # (ED)

    # Push back through gate channel
    feature_vector = feature_vector * cache[layer]["outputs"][0, t] # (ED)
    
    # Calculate attention matrix for current layer
    attn_matrices = compute_attn_matrix_fn(
        transcoder.base_model.backbone.layers[layer].mixer.deltaA,
        transcoder.base_model.backbone.layers[layer].mixer.deltaB,
        transcoder.base_model.backbone.layers[layer].mixer.C
    ) # (B, H, L, L)

    # Calculate inputs to SSM
    block_input = cache[layer]['inputs'] # (B, L, D)

    B, L, D = block_input.shape
    input_after_W2 = transcoder.base_model.backbone.layers[layer].mixer.in_proj(block_input) # (B, L, 2*ED)

    # Store W2 for SSM and gate
    W2_SSM, W2_gate = transcoder.base_model.backbone.layers[layer].mixer.in_proj.weight.data.chunk(2, dim=-2)

    # Take chunk of SSM path
    SSM_path_input = input_after_W2.chunk(2, dim=-1)[0] # the other goes into the gate path

    # Apply convolution
    in_conv = SSM_path_input.transpose(-1, -2)
    out_conv = transcoder.base_model.backbone.layers[layer].mixer.conv1d(in_conv)
    out_conv = out_conv[:, :, :L].transpose(-1, -2)

    # Calculate conv scaling
    conv_scaling_factor = out_conv / SSM_path_input # (B, L, ED)

    SSM_inputs = F.silu(out_conv) # Apply Silu (B, L, ED)
    SSM_input_tok = SSM_inputs[:, t, :] # (B, ED)

    contribs_list = []
    cur_idx = t if t > 0 else L+t

    for query_tok in range(cur_idx):
        
        # Take attention scores across all ED heads between key and query token
        attn_scores = attn_matrices[:, :, t, query_tok] # (B, ED, 1, 1)

        # Calculate contribution across all heads
        contribs = attn_scores * feature_vector * SSM_input_tok # (B, ED)

        # Extract most activated head
        val, most_act_head = torch.max(contribs.squeeze(), dim=-1)

        # Sum over heads to obtain contribution
        contrib = torch.sum(contribs, dim=-1).squeeze() # (1)

        # Now calculate new feature vector
        # Push back through S6, SiLU and Conv+W2
        new_feat_vector = attn_scores * feature_vector # (B, ED)
        new_feat_vector = new_feat_vector * F.sigmoid(out_conv)[:, query_tok, :] # This is for SILU(x) = Sigmoid(x) * x (ED)
        
        # Here we are considering the convolution operation just to scale our vector
        # So we are "skipping" the real conv operation
        # However, we assume that the mixing done in the conv operation is negligible
        # compared to the one done in the S6 layer
        new_feat_vector = new_feat_vector * conv_scaling_factor[:, query_tok, :] # (B, ED)
        new_feat_vector = new_feat_vector @ W2_SSM # (B, D)

        # And before the input projection, the model scales the input through an RMSNorm layer
        # We apply that scaling by calculating the scaling factor
        new_feat_vector = new_feat_vector * get_rmsnorm_constant(cache, layer, t)

        # Push into final list
        contribs_list.append(FeatureVector(
                component_path=f.component_path + [Component(layer=layer, component_type=ComponentType.MIXER, token=query_tok, attn_head=most_act_head)],
                vector=new_feat_vector.squeeze(),
                layer=layer,
                sublayer="resid-pre",
                token=query_tok,
                contrib=contrib.item()
            )
        )

    return contribs_list


@torch.no_grad()
def get_top_embs_features(cache: Dict, feature_vector: FeatureVector, t: int):
    
    # Take inputs after embedding
    emb_vec = cache[0]['inputs'][0, t] # Extract from the token (D dimensional)
    emb_score = torch.dot(emb_vec, feature_vector.vector.squeeze())

    # Here you don't need to recalculate any feature vector
    embs_contrib = []

    embs_contrib.append(FeatureVector(
        component_path=feature_vector.component_path + [Component(layer=0, component_type=ComponentType.EMBED, token=t)],
        vector=feature_vector,
        layer=0,
        sublayer="resid_pre",
        contrib=emb_score
    ))

    return embs_contrib

@torch.no_grad()
def get_top_contribs(transcoder: MambaCoder, cache: Dict, feature_vector: FeatureVector, k=5):
    """
    This function returns the top k contributions starting from a feature vector
    """

    # Put feature vector on device
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device) # (D)

    # Where to store contributions
    all_contribs = []

    # Cycle over previous layers
    for l in range(feature_vector.layer):

        # Calculate attributions coming from the other MCs
        all_contribs += get_top_mambacoder_features(transcoder, feature_vector, l, feature_vector.token, k=k)

        # Calculate attributions from Mamba heads
        all_contribs += get_top_mamba_features(transcoder, cache, feature_vector, l, feature_vector.token, k=k)

    # Calculate attributions from embeddings
    all_contribs += get_top_embs_features(cache, feature_vector, feature_vector.token)
    
    # Now return topk contribs
    _, top_idxs = torch.topk(torch.tensor([x.contrib for x in all_contribs]), k=min(k, len(all_contribs)))
    return [all_contribs[i] for i in top_idxs]


@torch.no_grad()
def greedy_get_top_paths(transcoder, cache, feature_vector, num_iters=2, num_branches=5):

    # Put feature vector into device
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device)

    all_paths = []
    root = copy.deepcopy(feature_vector)
    cur_paths = [[root]]

    for _ in range(num_iters):
        new_paths = []
        for path in cur_paths:
            last_feat = path[-1]
            if last_feat.layer == 0 and last_feat.sublayer == "resid_pre":
                continue
            top_contribs = get_top_contribs(transcoder, cache, last_feat, k=num_branches)
            for contrib in top_contribs:
                new_paths.append(path + [contrib])

        scores = torch.tensor([p[-1].contrib for p in new_paths])
        _, top_idx = torch.topk(scores, k=min(num_branches, len(new_paths)))
        cur_paths = [new_paths[i] for i in top_idx]
        all_paths.append(cur_paths)

    return all_paths

@torch.no_grad()
def greedy_get_top_paths_normalized(
    transcoder,
    cache,
    feature_vector,
    num_iters=2,
    num_branches=5,
    novelty_weight=0.5,   # bigger weight => more exploration
):
    feature_vector.vector = feature_vector.vector.to(transcoder.cfg.device)

    all_paths = []
    root = copy.deepcopy(feature_vector)
    cur_paths = [[root]]
    # counts per feature id
    seen = {}  # key: (layer, idx, sublayer) -> count

    for _ in range(num_iters):
        new_paths = []
        # gather all raw contribs for normalization per parent
        parent_to_cands = []
        for path in cur_paths:
            last_feat = path[-1]
            if last_feat.layer == 0 and last_feat.sublayer == "resid_pre":
                continue
            top_contribs = get_top_contribs(transcoder, cache, last_feat, k=num_branches*2)
            parent_to_cands.append((path, top_contribs))

        # process each parent's candidates (normalize then penalty)
        for path, top_contribs in parent_to_cands:
            raw = torch.tensor([float(c.contrib) for c in top_contribs])
            if raw.numel() == 0:
                continue
            mean = raw.mean()
            std = raw.std(unbiased=False).clamp(min=1e-6)
            normalized = (raw - mean) / std  # z-score
            for c, z in zip(top_contribs, normalized):
                key = (c.layer, getattr(c, "idx", None), c.sublayer)
                count = seen.get(key, 0)
                adjusted = float(z) - novelty_weight * (count)
                # store adjusted into a separate field to avoid overwriting original
                c._adjusted = adjusted
                new_paths.append(path + [c])
                # do NOT bump seen here — bump when the candidate is actually selected

        # select top branches while updating seen
        if len(new_paths) == 0:
            break
        scores = torch.tensor([p[-1]._adjusted for p in new_paths])
        _, top_idx = torch.topk(scores, k=min(num_branches, len(new_paths)))
        selected = [new_paths[i] for i in top_idx]

        # update seen counts for selected features
        for p in selected:
            c = p[-1]
            key = (c.layer, getattr(c, "idx", None), c.sublayer)
            seen[key] = seen.get(key, 0) + 1

        cur_paths = selected
        all_paths.append(cur_paths)

    return all_paths

def print_all_paths(paths):

    if len(paths) == 0: return

    if type(paths[0][0]) is list:
        for i, cur_paths in enumerate(paths):
            try:
                print(f"--- Paths of size {len(cur_paths[0])} ---")
            except:
                continue

            for j, cur_path in enumerate(cur_paths):
                print(f"Path [{i}][{j}]: ", end="")
                print("->".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))

    else:
        for j, cur_path in enumerate(paths):
            print(f"Path [{j}]: ", end="")
            print("->".join(map(lambda x: x.__str__(show_full=False, show_last_token=True), cur_path)))
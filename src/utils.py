import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from mamba_py.mambapy.mamba import MambaForLM
from mambacoder import MambaCoder
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Tuple, Dict
import torch.nn.functional as F
from tqdm import tqdm

def search_string_in_tokens(pattern: str, text: Union[str, List[str]], tokenizer: AutoTokenizer):

    if isinstance(text, str):
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        char_token_position = [
            idx for idx, token_id in enumerate(input_ids)
            if pattern in tokenizer.convert_ids_to_tokens(token_id)
        ]
    elif isinstance(text, list):
        input_ids = tokenizer(text, padding=True)['input_ids']
        char_token_position = [
            (b, idx) for b in range(len(input_ids)) for idx, token_id in enumerate(input_ids[b])
            if pattern in tokenizer.convert_ids_to_tokens(token_id)
        ]
    else:
        raise ValueError("Text must be one of str or List[str]")
    
    return char_token_position

def extract_most_act_feat(char_token_pos: Dict[int, int], outs: torch.Tensor, layer: int, k: int = 5, report: bool = False):

    batch_indices, token_indices = zip(*char_token_pos)
    selected_activations = outs['feature_activations'][layer][batch_indices, token_indices]

    # Sum activations over tokens
    selected_activations = selected_activations.sum(dim=-2)

    # Now search for most activated features
    vals, idxs = torch.topk(selected_activations, k=k)

    # Print the report of most act features
    if report:
        for val, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            print(f"Feature n.{idx}: {val}")

    return idxs

def get_k_prompts(tokenizer: AutoTokenizer, dataset: Dataset, tc: MambaCoder, k: int = 30):

    dataset = dataset.select(range(k))
    input_ids = tokenizer(list(dataset['text']), return_tensors='pt')['input_ids']
    return input_ids

def which_most_act(text: List[str], tokenizer: AutoTokenizer, mc: MambaCoder, device: torch.device, tok: int = -1, k: int=5, report: bool=True):

    input_ids = tokenizer(text, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Calculate outs from Mambacoder
    outs = mc(input_ids)

    for layer_idx in range(mc.model_cfg.n_layers):

        activations_from_layer = outs['feature_activations'][layer_idx]
        act_last_token = activations_from_layer[0, tok, :]
        topk_vals, topk_feats = torch.topk(act_last_token, k=k)
        topk_feats = topk_feats.cpu().tolist()

        if report:
            print(f"--- Layer {layer_idx} ---")
            for i, feat in enumerate(topk_feats):
                print(f"Feat #{i+1}: {feat}")

    return outs['last_hidden_state']

import torch
import torch.nn.functional as F

@torch.no_grad()
def correlate_features_with_gates(m, mc, tokenizer, text, device, layer_idx: int):
    """
    Compute correlations between transcoder features and true gate activations (z) in a Mamba layer.
    
    Returns:
        corr_matrix: (num_features, ED) tensor of correlations.
        mean_corr_per_feature: (num_features,) mean absolute correlation.
    """
    # Tokenize and run Mamba + MambaCoder
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)

    _ = m(input_ids)  # to populate m.backbone.layers[layer].mixer.cache
    outs = mc(input_ids)

    # Extract activations
    z_true = m.backbone.layers[layer_idx].mixer.cache["outputs"]  # (B, L, ED)
    feat_act = outs["feature_activations"][layer_idx]             # (B, L, num_features)

    # Flatten across batch and time
    B, L, ED = z_true.shape
    _, _, Fdim = feat_act.shape

    z_flat = z_true.reshape(B * L, ED)
    f_flat = feat_act.reshape(B * L, Fdim)

    # Normalize to zero mean / unit variance along samples
    z_flat = F.layer_norm(z_flat, (ED,))
    f_flat = F.layer_norm(f_flat, (Fdim,))

    # Compute correlation matrix (F x ED)
    corr = (f_flat.T @ z_flat) / (B * L)
    mean_corr = corr.abs().mean(dim=1)

    return corr, mean_corr


def most_act_diff(text1: List[str], text2: List[str], tokenizer: AutoTokenizer, mc: MambaCoder, device: torch.device, tok: int = -1, k: int = 5):

    # Tokenize input texts
    input_ids1 = tokenizer(text1, return_tensors='pt', padding=True)['input_ids'].to(device)
    input_ids2 = tokenizer(text2, return_tensors='pt', padding=True)['input_ids'].to(device)
    
    # Calculate outs from Mambacoder
    outs1 = mc(input_ids1)
    outs2 = mc(input_ids2)

    diff_act_dict = {}

    for layer_idx in range(mc.model_cfg.n_layers):

        acts_prompt1 = outs1['feature_activations'][layer_idx]
        acts_prompt2 = outs2['feature_activations'][layer_idx]
        acts_token1 = acts_prompt1[0, tok, :]
        acts_token2 = acts_prompt2[0, tok, :]

        # Express difference between the two
        acts_diff = acts_token2 - acts_token1

        _, topk_feats = torch.topk(acts_diff, k=k)

        diff_dict = {
            feat.item(): {"P1": acts_token1[feat].item(), "P2": acts_token2[feat].item()}
            for feat in topk_feats
        }

        diff_act_dict[layer_idx] = diff_dict

    return diff_act_dict

def suppress_features_across_layers(
    text_prompt: str,
    feature_suppression: Dict[int, list[int]],
    m: MambaForLM,
    mc: MambaCoder,
    tok: AutoTokenizer,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Suppresses selected sparse features at arbitrary layers and
    returns the modified logits for the next token prediction
    """

    # Tokenize text
    input_ids = tok(text_prompt, return_tensors='pt')['input_ids'].to(device)

    # Run model and capture the cache
    logits = m(input_ids)
    cache = m.cache

    # See whether we have to modify acts
    # If not return last token predictions
    if not feature_suppression:
        return logits[0, -1]

    # Save original logits for comparison
    original_logits = logits[0, -1]

    # Start modifying acts from min layer in dict
    min_layer = min(feature_suppression.keys())
    resid = cache[min_layer]['inputs'] # (B, L, D)

    B, L, D = resid.shape

    # From min_layer to end, run manually
    for layer in range(min_layer, m.config.n_layers):
        block = m.backbone.layers[layer]

        # If this layer is in the suppression dict
        if layer in feature_suppression:
            # 1. Enter mixing block
            W2_out = block.mixer.in_proj(resid) # (B, L, 2*ED)
            in_mixer, in_gate = W2_out.chunk(2, dim=-1) # (B, L, ED) each

            # 2. Run conv
            in_conv = in_mixer.transpose(-1, -2) # (B, ED, L)
            out_conv = block.mixer.conv1d(in_conv) # (B, ED, O)
            out_conv = out_conv[:, :, :L].transpose(-1, -2) # (B, L, ED)

            # 3. Apply SiLU
            out_silu = F.silu(out_conv) # (B, L, ED)

            # 4. Apply S6
            out_S6 = block.mixer.ssm(out_silu, in_gate) # (B, L, ED)

            # 5. Reconstruct gate only for last token
            last_tok_acts = resid[0, -1] # (D)
            feats = mc.encoders[layer](last_tok_acts) # (num_features)
            feats = mc.activation_functions[layer](feats) # (num_features)
            
            # Suppress features
            feats[feature_suppression[layer]] = 0

            # 5.1 Decode to get modified output
            modified_gate_out = mc.decoders[layer](feats) # (ED)
            gate_out = cache[layer]['outputs'].clone() # (B, L, ED)
            gate_out[0, -1] = modified_gate_out # (B, L, ED)

            # 6. Update resid
            out_block = gate_out * out_S6

            # 7. Apply W3 proj
            out_block = block.mixer.out_proj(out_block)

            # Apply final norm
            resid = block.norm(out_block)

        else:
            # Apply normal block
            resid = block(resid)

    # Final layer norm and logits
    pre_logits = m.backbone.norm_f(resid)
    modified_logits = m.lm_head(pre_logits) # (B, L, num_tokens)

    # Print report
    new_token = original_logits.softmax(dim=-1).argmax()
    print(f"Original logits. Best token {new_token} with value {original_logits[new_token]}. Token: {tok.decode(new_token)} with p={original_logits.softmax(dim=-1).max()}")

    new_token_modified = modified_logits[0, -1].softmax(dim=-1).argmax()
    print(f"Modified logits. Best token {new_token_modified} with value {modified_logits[0, -1][new_token_modified]}. Token: {tok.decode(new_token_modified)} with p={modified_logits[0, -1].softmax(dim=-1).max()}")

    print(f"Probabilities change from {original_logits.softmax(dim=-1)[new_token]} to {modified_logits[0, -1].softmax(dim=-1)[new_token]}")
    
    return original_logits, modified_logits[0, -1]

@torch.no_grad()
def activate_features_across_layers(
    text_prompt: str,
    feature_activation: Dict[int, List[int]],
    m: MambaForLM,
    tok: AutoTokenizer,
    mc: MambaCoder,
    device: torch.device,
    activation_value: float = 10.0,
    mitigation_weight: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Activates selected sparse features across Mamba layers.
    """

    input_ids = tok(text_prompt, return_tensors='pt')['input_ids'].to(device)
    logits = m(input_ids)
    cache = m.cache

    original_logits = logits[0, -1]

    if not feature_activation:
        return original_logits.softmax(dim=-1), original_logits.softmax(dim=-1)
    
    min_layer = min(feature_activation.keys())
    resid = cache[min_layer]['out_block'].clone() # this is the Residual's Block ouptut

    for layer in range(min_layer, m.config.n_layers):
        block = m.backbone.layers[layer]
        if layer in feature_activation:
            
            x = resid.clone()
            feats = mc.encoders[layer](x[0, -1])
            feats = mc.activation_functions[layer](feats)
            feats[feature_activation[layer]] = activation_value
            modified = mc.decoders[layer](feats) # this is the output of the res. block

            # Update out block
            resid[0, -1] = mitigation_weight * resid[0, -1] + (1 - mitigation_weight) * modified # This is already the output of the res block, so we 
            
        else:
            resid = block(resid)


    pre_logits = m.backbone.norm_f(resid)
    modified_logits = m.lm_head(pre_logits)

    # Report
    orig_probs = original_logits.softmax(dim=-1)
    mod_probs = modified_logits[0, -1].softmax(dim=-1)
    new_token_orig = orig_probs.argmax()
    new_token_mod = mod_probs.argmax()

    print(f"Original: {tok.decode(new_token_orig)} (p={orig_probs[new_token_orig]:.4f})")
    print(f"Modified: {tok.decode(new_token_mod)} (p={mod_probs[new_token_mod]:.4f})")
    print(f"Δp[{tok.decode(new_token_mod)}] = {mod_probs[new_token_mod] - orig_probs[new_token_mod]:.4f}")

    probs_diff = mod_probs - orig_probs # Relative difference
    _, most_impr_idxs = torch.topk(probs_diff, k=5)

    most_impr_tokens = [tok.decode(idx.item()) for idx in most_impr_idxs]
    
    print(f"Max DeltaP: {most_impr_tokens}")

    return original_logits, modified_logits[0, -1]

@torch.no_grad()
def generate_when_activating(
    text_prompt: str,
    feature_activation: Dict[int, List[int]],
    m: MambaForLM,
    tok: AutoTokenizer,
    mc: MambaCoder,
    device: torch.device,
    max_tokens: int,
    activation_value: float = 10.0,
    mitigation_weight: float = 0.8
) -> Tuple[str, str]:
    
    # Initialize new strings (original and modified)
    orig, mod = text_prompt, text_prompt

    # Tokenize the prompt to get input_ids
    original_input_ids = tok(text_prompt, return_tensors="pt")["input_ids"].to(device)
    modified_input_ids = tok(text_prompt, return_tensors="pt")["input_ids"].to(device)

    # TODO: Add for cycle
    for _ in tqdm(range(max_tokens), total=max_tokens):
        # Compute normal forward pass
        original_logits = m(original_input_ids)
        original_probs = original_logits[0, -1].softmax(dim=-1)

        # Compute forward pass to be modified
        _ = m(modified_input_ids)
        cache = m.cache

        min_layer = min(feature_activation.keys())
        resid = cache[min_layer]["out_block"].clone()

        # Now start modifying the model
        for layer in range(min_layer, m.config.n_layers):
            block = m.backbone.layers[layer]

            if layer in feature_activation:
                # x = resid.clone()
                # feats = mc.encoders[layer](x[0, -1])
                # feats = mc.activation_functions[layer](feats)
                # feats[feature_activation[layer]] = activation_value
                # modified = mc.decoders[layer](feats) # this is the output of the res. block

                # # Update out block
                # out_block = cache[layer]["out_block"].clone() # taken from the residual stream

                # # Extract layernorm constant
                # ln_constant = out_block[0, -1] / F.layer_norm(out_block, out_block.shape[-1:])[0, -1]
                # out_block[0, -1] = modified * ln_constant # MC out is already in the "residual" space
                # resid = out_block + x
                x = resid.clone()
                feats = mc.encoders[layer](x[0, -1])
                feats = mc.activation_functions[layer](feats)
                feats[feature_activation[layer]] = activation_value
                modified = mc.decoders[layer](feats) # this is the output of the res. block

                # Update out block
                resid[0, -1] = mitigation_weight * resid[0, -1] + (1 - mitigation_weight) * modified # This is already the output of the res block, so we 
                
            else:
                resid = block(resid) # here residual connection is implemented
        
        # Apply final layer norm and head
        modified_logits = m.lm_head(m.backbone.norm_f(resid))
        modified_probs = modified_logits[0, -1].softmax(dim=-1)

        # Sample from both distribs
        orig_token = torch.multinomial(original_probs, num_samples=1)
        mod_token = torch.multinomial(modified_probs, num_samples=1)

        # Concat to input_ids
        original_input_ids = torch.cat([original_input_ids, orig_token.unsqueeze(0)], dim=-1)
        modified_input_ids = torch.cat([modified_input_ids, mod_token.unsqueeze(0)], dim=-1)

        # Concat to string
        orig += tok.decode(orig_token)
        mod += tok.decode(mod_token)

    return orig, mod

def plot_top_tokens_comparison(original_logits, modified_logits, tokenizer, top_k=10):
    """
    Plot a barplot comparing top-k most probable tokens from original and modified logits.

    Args:
        original_logits (torch.Tensor): Logits before intervention (1D tensor of shape [vocab_size])
        modified_logits (torch.Tensor): Logits after intervention (same shape)
        tokenizer (transformers.PreTrainedTokenizer): Hugging Face tokenizer to decode token ids
        top_k (int): Number of top tokens to show
    """
    # Convert logits to probabilities
    original_probs = torch.softmax(original_logits, dim=-1)
    modified_probs = torch.softmax(modified_logits, dim=-1)

    # Get top-k token indices for both
    orig_topk_vals, orig_topk_idxs = torch.topk(original_probs, top_k)
    mod_topk_vals, mod_topk_idxs = torch.topk(modified_probs, top_k)

    # Decode tokens
    orig_tokens = [tokenizer.decode(idx) for idx in orig_topk_idxs.tolist()]
    mod_tokens = [tokenizer.decode(idx) for idx in mod_topk_idxs.tolist()]

    # Build bar plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Original
    axs[0].barh(range(top_k), orig_topk_vals.tolist(), color='skyblue')
    axs[0].set_yticks(range(top_k))
    axs[0].set_yticklabels(orig_tokens)
    axs[0].invert_yaxis()
    axs[0].set_title("Original logits: Top {} tokens".format(top_k))
    axs[0].set_xlabel("Probability")
    axs[0].set_xlim([0, 1])

    # Modified
    axs[1].barh(range(top_k), mod_topk_vals.tolist(), color='salmon')
    axs[1].set_yticks(range(top_k))
    axs[1].set_yticklabels(mod_tokens)
    axs[1].invert_yaxis()
    axs[1].set_title("Modified logits: Top {} tokens".format(top_k))
    axs[1].set_xlabel("Probability")
    axs[1].set_xlim([0, 1])

    plt.tight_layout()
    plt.show()

def logit_lens(m: MambaForLM, mc: MambaCoder, text_prompt: str, tok: AutoTokenizer, device: torch.device, use_mc: bool = False) -> Dict[int, List[Dict[str, float]]]:
    """
    Extract most valuable token from Mamba layers
    """

    input_ids = tok(text_prompt, return_tensors='pt')['input_ids'].to(device)

    # Define return dict
    most_prob_tokens_dict = {}

    # Run model and extract cache
    _ = m(input_ids)
    cache = m.cache

    if use_mc:
        out_mc = mc(input_ids)['reconstructed_activations']
    
    # Extract most prob tokens from each layer
    for layer in range(m.config.n_layers):

        if not use_mc:
            layer_out = cache[layer]['out_block']
        else:
            inputs = cache[layer]['inputs']
            layer_out = out_mc[layer] # + inputs
            # z = out_mc[layer]
            # y = cache[layer]["y"]
            # out_block = z * y
            # layer_out = m.backbone.layers[layer].mixer.out_proj(out_block) + inputs

        # Apply final norm and lm head
        pre_logits = m.backbone.norm_f(layer_out)
        layer_logits = m.lm_head(pre_logits)
        probs = layer_logits.softmax(dim=-1)
        probs_last_tok = probs[0, -1] # (num_tokens)

        # Extract most probable token
        most_prob_vals, most_prob_idxs = torch.topk(probs_last_tok, k=10)
        most_prob_idxs = [idx.item() for idx in most_prob_idxs]
        most_prob_vals = [val.item() for val in most_prob_vals]
        most_prob_tokens = [tok.decode(idx) for idx in most_prob_idxs]

        # Put into return dict
        layer_dict = dict(zip(most_prob_tokens, most_prob_vals))

        most_prob_tokens_dict[layer] = layer_dict

    return most_prob_tokens_dict

def generate(m: MambaForLM, max_new_tokens: int, input_ids: torch.Tensor):

    for _ in range(max_new_tokens): 
   
        # Run model on input_ids
        logits = m(input_ids)
        last_tok_probs = logits.softmax(dim=-1)[0, -1] # (num_tokens)

        # Sample from the distribution
        new_tok = torch.multinomial(last_tok_probs, num_samples=1)

        # Concatenate to original
        input_ids = torch.cat((input_ids, new_tok.unsqueeze(dim=0)), dim=-1)

    # Return the tokenized string
    return input_ids

@torch.no_grad()
def top_pred_tokens(mc: MambaCoder, m: MambaForLM, layer: int, num_feat: int, num_prompts: int, tokenizer: AutoTokenizer, device: torch.device):

    assert num_feat < mc.cfg.num_features, f"MambaCoder layer has only {mc.cfg.num_features} features!"

    # Obtain k prompts from dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    prompts = dataset.select(range(num_prompts))['text']

    # Get tokens
    batch_tokens = tokenizer(list(prompts), padding=True, return_tensors="pt")['input_ids'].to(device)

    # Run model on prompts to collect data
    _ = m(batch_tokens) # now m.cache is populated
    cache = m.cache

    # Get average y vector
    avg_y_vec = cache[layer]['y'].mean(dim=0).mean(dim=0) # (ED)

    # Take out decoder vector from mc
    dec_vec = mc.decoders[layer].weight.data[:, num_feat] # (ED)

    # Consider average out_block
    average_out_block = avg_y_vec * dec_vec # (ED)

    # Now project into vocabulary space
    out_block = m.backbone.layers[layer].mixer.out_proj(average_out_block) # (D)
    average_tokens = m.lm_head(m.backbone.norm_f(out_block)) # (num_vocab)

    # Compute top k pred tokens
    _, top_idxs = torch.topk(average_tokens, k=10)
    top_idxs = [idx.item() for idx in top_idxs]

    return [tokenizer.convert_ids_to_tokens(idx) for idx in top_idxs]

@torch.no_grad()
def top_dec_tokens(mc: MambaCoder, m: MambaForLM, layer: int, num_feat: int, tokenizer: AutoTokenizer):

    assert num_feat < mc.cfg.num_features, f"MambaCoder layer has only {mc.cfg.num_features} features!"

    # Take out decoder vector from mc
    dec_vec = mc.decoders[layer].weight.data[:, num_feat] # (D)

    # Normalize and multiply by WU
    norm_dec = m.backbone.norm_f(dec_vec)
    out_vec = m.lm_head(norm_dec)

    # Extract most probable tokens
    probs = out_vec.softmax(dim=-1)
    _, top_idxs = torch.topk(probs, k=10)

    return [(tokenizer.decode(idx.item()), idx.item()) for idx in top_idxs]

@torch.no_grad()
def top_embeds_tokens(mc: MambaCoder, layer: int, num_feat: int, tokenizer: AutoTokenizer):

    assert num_feat < mc.cfg.num_features, f"MambaCoder layer has only {mc.cfg.num_features} features!"

    # Extract feature vector from encoder
    feat_vec = mc.encoders[layer].weight.data[num_feat, :] # (D)

    # Calculate attribution with embeddings
    embedding_table = mc.base_model.backbone.embeddings.weight.data
    act_tokens = embedding_table @ feat_vec

    # Retrieve top k tokens
    _, top_idxs = torch.topk(act_tokens, k=10)
    top_idxs = [idx.item() for idx in top_idxs]

    return [(tokenizer.decode(idx), idx) for idx in top_idxs]

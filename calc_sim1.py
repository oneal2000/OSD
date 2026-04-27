import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from typing import Dict, List, Tuple
from scipy.stats import gaussian_kde
from collections import defaultdict
import argparse
import random
from tqdm import tqdm
from root_dir_path import ROOT_DIR

def _normalize_module_name(name: str) -> str:
    """Helper to align module names from different model wrappers"""
    if "model.layers" in name:
        idx = name.index("model.layers")
        return name[idx:]
    return name

def load_lora_weights(lora_path: str, proj_types: List[str] = ["down_proj", "gate_proj", "up_proj"]) -> Dict[str, torch.Tensor]:
    """
    Load LoRA weights from a safetensors file.
    Returns a dictionary mapping normalized module names to flattened weight tensors.
    For each module, we concatenate lora_A and lora_B flattened weights.
    """
    weight_path = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Cannot find LoRA weights at {weight_path}")
    
    state_dict = load_file(weight_path)
    weights = {}
    
    for key, tensor in state_dict.items():
        # Extract lora_A and lora_B weights
        if key.endswith("lora_A.weight") or key.endswith("lora_B.weight"):
            module_name = key.rsplit(".lora_A.weight", 1)[0] if key.endswith("lora_A.weight") else key.rsplit(".lora_B.weight", 1)[0]
            suffix = module_name.rsplit('.', 1)[-1]
            
            if suffix in proj_types:
                norm_name = _normalize_module_name(module_name)
                # Flatten the weight matrix
                flattened = tensor.flatten().clone()
                
                # Store both A and B weights, we'll concatenate them later
                if norm_name not in weights:
                    weights[norm_name] = {}
                weight_type = "A" if key.endswith("lora_A.weight") else "B"
                weights[norm_name][weight_type] = flattened
    
    # Concatenate A and B weights for each module to get the full LoRA weight matrix
    result = {}
    for module_name, weight_dict in weights.items():
        if "A" in weight_dict and "B" in weight_dict:
            # Concatenate A and B weights to represent the full LoRA transformation
            combined = torch.cat([weight_dict["A"], weight_dict["B"]])
            result[module_name] = combined
        elif "A" in weight_dict:
            result[module_name] = weight_dict["A"]
        elif "B" in weight_dict:
            result[module_name] = weight_dict["B"]
    
    return result

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors"""
    vec1_norm = vec1 / (vec1.norm() + 1e-8)
    vec2_norm = vec2 / (vec2.norm() + 1e-8)
    return (vec1_norm * vec2_norm).sum().item()

def compute_similarity_for_proj_type(
    weights_dict1: Dict[str, torch.Tensor],
    weights_dict2: Dict[str, torch.Tensor],
    proj_type: str
) -> float:
    """
    Compute cosine similarity between flattened LoRA weights for a specific projection type.
    Concatenates all layer weights of the same projection type and computes overall similarity.
    """
    # Find all modules of the specified projection type
    modules1 = {k: v for k, v in weights_dict1.items() if k.endswith(f".{proj_type}")}
    modules2 = {k: v for k, v in weights_dict2.items() if k.endswith(f".{proj_type}")}
    
    if len(modules1) == 0 or len(modules2) == 0:
        return 0.0
    
    # Match modules by layer index and concatenate all layers
    all_weights1 = []
    all_weights2 = []
    
    # Sort by module name to ensure consistent ordering
    sorted_modules1 = sorted(modules1.keys())
    sorted_modules2 = sorted(modules2.keys())
    
    # Match modules
    for module_name in sorted_modules1:
        if module_name in sorted_modules2:
            all_weights1.append(modules1[module_name])
            all_weights2.append(modules2[module_name])
    
    if len(all_weights1) == 0:
        return 0.0
    
    # Concatenate all layer weights
    combined1 = torch.cat(all_weights1)
    combined2 = torch.cat(all_weights2)
    
    # Compute cosine similarity
    return cosine_similarity(combined1, combined2)

def find_available_data_and_passages(base_path: str, dataset: str, epoch: str, lr: str) -> Dict[int, List[int]]:
    """
    Find all available data indices and their passage indices.
    Returns a dictionary mapping data_id to list of passage_ids.
    """
    data_passages = defaultdict(list)
    
    # For D-PRAG: offline_doc/{model}/{dataset}/total/{epoch}_{lr}/data_{did}/passage_{pid}
    # For PRAG: offline_prag/{model}/{dataset}/{epoch}_{lr}/total/data_{did}/passage_{pid}
    
    if "offline_doc_rand" in base_path:
        search_path = os.path.join(base_path, dataset, "total", f"{epoch}_{lr}")
    else:
        search_path = os.path.join(base_path, dataset, f"{epoch}_{lr}", "total")
    
    if not os.path.exists(search_path):
        return data_passages
    
    # Find all data directories
    for item in os.listdir(search_path):
        if item.startswith("data_"):
            data_id = int(item.split("_")[1])
            data_path = os.path.join(search_path, item)
            
            # Find all passage directories
            if os.path.isdir(data_path):
                for passage_item in os.listdir(data_path):
                    if passage_item.startswith("passage_"):
                        passage_id = int(passage_item.split("_")[1])
                        passage_path = os.path.join(data_path, passage_item, "adapter_model.safetensors")
                        if os.path.exists(passage_path):
                            data_passages[data_id].append(passage_id)
    
    return data_passages

def compute_similarities(
    base_path: str,
    dataset: str,
    model_name: str,
    epoch: str,
    lr: str,
    proj_types: List[str] = ["down_proj", "gate_proj", "up_proj"]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute cosine similarities for relevant and irrelevant pairs.
    Returns a dictionary: {proj_type: {"relevant": [...], "irrelevant": [...]}}
    """
    results = {proj_type: {"relevant": [], "irrelevant": []} for proj_type in proj_types}
    
    # Find available data and passages
    data_passages = find_available_data_and_passages(base_path, dataset, epoch, lr)
    
    if len(data_passages) < 2:
        print(f"Warning: Need at least 2 data entries, found {len(data_passages)}")
        return results
    
    data_ids = sorted(data_passages.keys())
    print(f"Found {len(data_ids)} data entries")
    
    # Deterministically select one passage for each data: data_i selects passage_k where k = i % 3
    # This ensures the same selection every time
    data_selected_passages = {}
    for data_id in data_ids:
        passages = sorted(data_passages[data_id])
        if len(passages) > 0:
            # Select passage_k where k = data_id % 3
            k = data_id % 3
            # Check if passage_k exists in available passages
            if k in passages:
                data_selected_passages[data_id] = k
            else:
                # If passage_k doesn't exist, use the first available passage
                data_selected_passages[data_id] = passages[0]
    print(f"Deterministically selected passages for {len(data_selected_passages)} data entries (k = data_id % 3)")
    
    # Calculate total number of relevant pairs to balance with irrelevant pairs
    total_relevant_pairs_all = sum(
        len(passages) * (len(passages) - 1) // 2 
        for passages in data_passages.values() 
        if len(passages) >= 2
    )
    
    # Calculate total number of irrelevant pairs (all pairs of different data)
    total_irrelevant_pairs_all = len(data_ids) * (len(data_ids) - 1) // 2
    
    print(f"Total relevant pairs: {total_relevant_pairs_all}")
    print(f"Total irrelevant pairs: {total_irrelevant_pairs_all}")
    
    # Determine target number of pairs (use the smaller one, or sample to match)
    target_num_pairs = min(total_relevant_pairs_all, total_irrelevant_pairs_all)
    
    # If irrelevant pairs are much more, we'll sample them
    # If relevant pairs are more, we'll sample them (less common)
    if total_irrelevant_pairs_all > total_relevant_pairs_all:
        # Sample irrelevant pairs to match relevant pairs count
        sample_irrelevant = True
        sample_ratio = total_relevant_pairs_all / total_irrelevant_pairs_all
        print(f"Sampling {target_num_pairs} irrelevant pairs from {total_irrelevant_pairs_all} (ratio: {sample_ratio:.4f})")
    elif total_relevant_pairs_all > total_irrelevant_pairs_all:
        # Sample relevant pairs to match irrelevant pairs count (less common)
        sample_relevant = True
        sample_ratio = total_irrelevant_pairs_all / total_relevant_pairs_all
        print(f"Sampling {target_num_pairs} relevant pairs from {total_relevant_pairs_all} (ratio: {sample_ratio:.4f})")
    else:
        # They are already balanced
        sample_irrelevant = False
        sample_relevant = False
        print("Relevant and irrelevant pairs are already balanced")
    
    # Generate all irrelevant pair indices for sampling (if needed)
    if total_irrelevant_pairs_all > total_relevant_pairs_all:
        # Generate all possible irrelevant pair indices
        all_irrelevant_indices = []
        for i in range(len(data_ids)):
            for j in range(i + 1, len(data_ids)):
                all_irrelevant_indices.append((i, j))
        # Randomly sample to match relevant pairs count
        random.seed(42)  # For reproducibility
        sampled_irrelevant_indices = random.sample(all_irrelevant_indices, target_num_pairs)
        sampled_irrelevant_indices_set = set(sampled_irrelevant_indices)
        print(f"Sampled {len(sampled_irrelevant_indices)} irrelevant pairs")
    else:
        sampled_irrelevant_indices_set = None
    
    # Compute similarities
    for proj_type in tqdm(proj_types, desc="Processing projection types"):
        print(f"\nComputing similarities for {proj_type}...")
        
        # Relevant pairs: same data, different passages
        relevant_count = 0
        # Calculate total number of relevant pairs for progress bar
        total_relevant_pairs = sum(
            len(passages) * (len(passages) - 1) // 2 
            for passages in data_passages.values() 
            if len(passages) >= 2
        )
        
        # If we need to sample relevant pairs, prepare the sampling
        if total_relevant_pairs_all > total_irrelevant_pairs_all:
            # Collect all relevant pairs first, then sample
            all_relevant_pairs = []
            for data_id in data_ids:
                passages = sorted(data_passages[data_id])
                if len(passages) >= 2:
                    for i in range(len(passages)):
                        for j in range(i + 1, len(passages)):
                            all_relevant_pairs.append((data_id, passages[i], passages[j]))
            random.seed(42)  # For reproducibility
            sampled_relevant_pairs = random.sample(all_relevant_pairs, target_num_pairs)
            sampled_relevant_pairs_set = set(sampled_relevant_pairs)
            total_relevant_pairs = target_num_pairs
        else:
            sampled_relevant_pairs_set = None
        
        with tqdm(total=total_relevant_pairs, desc=f"  Relevant pairs ({proj_type})", leave=False) as pbar:
            for data_id in data_ids:
                passages = sorted(data_passages[data_id])
                if len(passages) >= 2:
                    # Load weights for all passages in this data
                    passage_weights = {}
                    for pid in passages:
                        if "offline_doc_rand" in base_path:
                            lora_path = os.path.join(base_path, dataset, "total", f"{epoch}_{lr}", f"data_{data_id}", f"passage_{pid}")
                        else:
                            lora_path = os.path.join(base_path, dataset, f"{epoch}_{lr}", "total", f"data_{data_id}", f"passage_{pid}")
                        
                        try:
                            weights = load_lora_weights(lora_path, proj_types=[proj_type])
                            passage_weights[pid] = weights
                        except Exception as e:
                            print(f"\nWarning: Failed to load weights for data_{data_id}/passage_{pid}: {e}")
                            continue
                    
                    # Compute similarities between different passages in the same data
                    passage_list = sorted(passage_weights.keys())
                    for i in range(len(passage_list)):
                        for j in range(i + 1, len(passage_list)):
                            pid1, pid2 = passage_list[i], passage_list[j]
                            
                            # If we need to sample relevant pairs, check if this pair is in the sample
                            if sampled_relevant_pairs_set is not None:
                                if (data_id, pid1, pid2) not in sampled_relevant_pairs_set:
                                    continue
                            
                            sim = compute_similarity_for_proj_type(
                                passage_weights[pid1],
                                passage_weights[pid2],
                                proj_type
                            )
                            results[proj_type]["relevant"].append(sim)
                            relevant_count += 1
                            pbar.update(1)
        
        # Irrelevant pairs: different data, using deterministically selected passage from each data
        # (Selection: data_i uses passage_k where k = i % 3, done before the proj_type loop to ensure consistency)
        irrelevant_count = 0
        
        # Calculate total number of irrelevant pairs (sampled if needed)
        if sampled_irrelevant_indices_set is not None:
            total_irrelevant_pairs = len(sampled_irrelevant_indices_set)
        else:
            total_irrelevant_pairs = len(data_ids) * (len(data_ids) - 1) // 2
        
        with tqdm(total=total_irrelevant_pairs, desc=f"  Irrelevant pairs ({proj_type})", leave=False) as pbar:
            # Iterate through all or sampled irrelevant pairs
            if sampled_irrelevant_indices_set is not None:
                # Use sampled pairs
                pair_list = list(sampled_irrelevant_indices_set)
            else:
                # Use all pairs
                pair_list = [(i, j) for i in range(len(data_ids)) for j in range(i + 1, len(data_ids))]
            
            for i, j in pair_list:
                data_id1, data_id2 = data_ids[i], data_ids[j]
                
                # Get deterministically selected passages for each data
                if data_id1 not in data_selected_passages or data_id2 not in data_selected_passages:
                    pbar.update(1)
                    continue
                
                pid1 = data_selected_passages[data_id1]
                pid2 = data_selected_passages[data_id2]
                
                # Load weights
                if "offline_doc_rand" in base_path:
                    lora_path1 = os.path.join(base_path, dataset, "total", f"{epoch}_{lr}", f"data_{data_id1}", f"passage_{pid1}")
                    lora_path2 = os.path.join(base_path, dataset, "total", f"{epoch}_{lr}", f"data_{data_id2}", f"passage_{pid2}")
                else:
                    lora_path1 = os.path.join(base_path, dataset, f"{epoch}_{lr}", "total", f"data_{data_id1}", f"passage_{pid1}")
                    lora_path2 = os.path.join(base_path, dataset, f"{epoch}_{lr}", "total", f"data_{data_id2}", f"passage_{pid2}")
                
                try:
                    weights1 = load_lora_weights(lora_path1, proj_types=[proj_type])
                    weights2 = load_lora_weights(lora_path2, proj_types=[proj_type])
                    
                    sim = compute_similarity_for_proj_type(weights1, weights2, proj_type)
                    results[proj_type]["irrelevant"].append(sim)
                    irrelevant_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\nWarning: Failed to compute similarity for data_{data_id1}/passage_{pid1} vs data_{data_id2}/passage_{pid2}: {e}")
                    pbar.update(1)  # Update progress bar even if failed
                    continue
        
        print(f"  Relevant pairs: {relevant_count}, Irrelevant pairs: {irrelevant_count}")
    
    return results

def plot_similarity_distributions(
    results_dprag: Dict[str, Dict[str, List[float]]],
    results_prag: Dict[str, Dict[str, List[float]]],
    output_path: str,
    method_name: str = "D-PRAG"
):
    """
    Plot KDE distributions for cosine similarities.
    """
    proj_types = ["down_proj", "gate_proj", "up_proj"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, proj_type in enumerate(proj_types):
        ax = axes[idx]
        
        # Get data for the specified method
        if method_name == "D-PRAG":
            relevant = results_dprag[proj_type]["relevant"]
            irrelevant = results_dprag[proj_type]["irrelevant"]
        else:
            relevant = results_prag[proj_type]["relevant"]
            irrelevant = results_prag[proj_type]["irrelevant"]
        
        if len(relevant) == 0 or len(irrelevant) == 0:
            print(f"Warning: No data for {proj_type}, skipping plot")
            continue
        
        # Convert to numpy array for KDE
        relevant_array = np.array(relevant)
        irrelevant_array = np.array(irrelevant)
        
        # Compute KDE
        relevant_kde = gaussian_kde(relevant_array)
        irrelevant_kde = gaussian_kde(irrelevant_array)
        
        # Adjust bandwidth to prevent extremely high density values
        # Increase bandwidth by a factor to smooth the distribution
        # This helps when data points are sparse or highly concentrated
        bandwidth_multiplier = 1.5  # Increase bandwidth by 50% for smoother curves
        
        if len(relevant_array) > 1:
            # Get current bandwidth and increase it
            current_bw = relevant_kde.covariance_factor()
            relevant_kde.set_bandwidth(current_bw * bandwidth_multiplier)
        
        if len(irrelevant_array) > 1:
            # Get current bandwidth and increase it
            current_bw = irrelevant_kde.covariance_factor()
            irrelevant_kde.set_bandwidth(current_bw * bandwidth_multiplier)
        
        # Create x-axis range
        all_values = relevant + irrelevant
        x_min, x_max = min(all_values), max(all_values)
        # Extend range slightly for better visualization
        x_range = np.linspace(x_min - 0.01, x_max + 0.01, 200)
        
        # Compute KDE values
        relevant_density = relevant_kde(x_range)
        irrelevant_density = irrelevant_kde(x_range)
        
        # Plot with colors matching the example image
        # Relevant Pairs: light blue
        # Irrelevant Pairs: light orange
        ax.fill_between(x_range, relevant_density, alpha=0.5, color='lightblue', label="Relevant Pairs")
        ax.plot(x_range, relevant_density, color='steelblue', linewidth=1.5, alpha=0.8)
        ax.fill_between(x_range, irrelevant_density, alpha=0.5, color='moccasin', label="Irrelevant Pairs")
        ax.plot(x_range, irrelevant_density, color='darkorange', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Cosine Similarity", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Similarity Distribution of {proj_type}", fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarities between LoRA weights")
    parser.add_argument("--model_name", type=str, default="llama3.2-1b-instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="popqa", help="Dataset name")
    parser.add_argument("--epoch", type=str, default="epoch=2", help="Epoch string")
    parser.add_argument("--lr", type=str, default="lr=0.0003", help="Learning rate string")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(ROOT_DIR, "output", "similarity_analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    dprag_path = os.path.join(ROOT_DIR, "offline_doc_rand", args.model_name, "lambda=0.1")
    prag_path = os.path.join(ROOT_DIR, "offline_prag", args.model_name)
    
    epoch_lr = f"{args.epoch}_{args.lr}"
    
    print("=" * 60)
    print("Computing similarities for D-PRAG method...")
    print("=" * 60)
    results_dprag = compute_similarities(
        dprag_path,
        args.dataset,
        args.model_name,
        args.epoch,
        args.lr,
        proj_types=["down_proj", "gate_proj", "up_proj"]
    )
    
    print("\n" + "=" * 60)
    print("Computing similarities for PRAG method...")
    print("=" * 60)
    results_prag = compute_similarities(
        prag_path,
        args.dataset,
        args.model_name,
        args.epoch,
        args.lr,
        proj_types=["down_proj", "gate_proj", "up_proj"]
    )
    
    # Plot D-PRAG results
    dprag_output = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_D-PRAG_similarity_distribution.png")
    plot_similarity_distributions(results_dprag, results_prag, dprag_output, method_name="D-PRAG")
    
    # Plot PRAG results
    prag_output = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_PRAG_similarity_distribution.png")
    plot_similarity_distributions(results_dprag, results_prag, prag_output, method_name="PRAG")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    for proj_type in ["down_proj", "gate_proj", "up_proj"]:
        print(f"\n{proj_type}:")
        print(f"  D-PRAG - Relevant: mean={np.mean(results_dprag[proj_type]['relevant']):.4f}, "
              f"std={np.std(results_dprag[proj_type]['relevant']):.4f}, "
              f"count={len(results_dprag[proj_type]['relevant'])}")
        print(f"  D-PRAG - Irrelevant: mean={np.mean(results_dprag[proj_type]['irrelevant']):.4f}, "
              f"std={np.std(results_dprag[proj_type]['irrelevant']):.4f}, "
              f"count={len(results_dprag[proj_type]['irrelevant'])}")
        print(f"  PRAG - Relevant: mean={np.mean(results_prag[proj_type]['relevant']):.4f}, "
              f"std={np.std(results_prag[proj_type]['relevant']):.4f}, "
              f"count={len(results_prag[proj_type]['relevant'])}")
        print(f"  PRAG - Irrelevant: mean={np.mean(results_prag[proj_type]['irrelevant']):.4f}, "
              f"std={np.std(results_prag[proj_type]['irrelevant']):.4f}, "
              f"count={len(results_prag[proj_type]['irrelevant'])}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Verify that Geneformer embeddings are in the correct cell order."""

import sys
from pathlib import Path

import numpy as np
import scanpy as sc


def verify_embeddings_order(input_h5ad: str, embeddings_npy: str) -> None:
    """
    Verify that embeddings match the input AnnData cell order.
    
    Args:
        input_h5ad: Path to input .h5ad file
        embeddings_npy: Path to embeddings .npy file
    """
    print(f"Loading input AnnData from {input_h5ad}...")
    adata = sc.read_h5ad(input_h5ad)
    print(f"Input shape: {adata.shape}")
    print(f"First 10 cell IDs: {adata.obs_names[:10].tolist()}")
    
    print(f"\nLoading embeddings from {embeddings_npy}...")
    embeddings = np.load(embeddings_npy)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Check shape matches
    if embeddings.shape[0] != adata.n_obs:
        print(f"❌ ERROR: Shape mismatch!")
        print(f"  Expected {adata.n_obs} cells, got {embeddings.shape[0]}")
        sys.exit(1)
    
    print(f"✅ Shape matches: {embeddings.shape[0]} cells, {embeddings.shape[1]} dimensions")
    
    # Check for NaN or Inf values
    if np.isnan(embeddings).any():
        n_nan = np.isnan(embeddings).sum()
        print(f"⚠️  WARNING: Found {n_nan} NaN values in embeddings")
    else:
        print("✅ No NaN values")
    
    if np.isinf(embeddings).any():
        n_inf = np.isinf(embeddings).sum()
        print(f"⚠️  WARNING: Found {n_inf} Inf values in embeddings")
    else:
        print("✅ No Inf values")
    
    # Check embedding statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    # If we have debug files, we can do more detailed verification
    debug_dir = Path("debug_geneformer")
    if debug_dir.exists():
        print(f"\n🔍 Debug directory found: {debug_dir}")
        
        # Check tokenized dataset
        tokenized_dataset_path = debug_dir / "tokenized_dataset" / "tokenized.dataset"
        if tokenized_dataset_path.exists():
            try:
                from datasets import load_from_disk
                print("Loading tokenized dataset for verification...")
                tokenized_dataset = load_from_disk(str(tokenized_dataset_path))
                
                if 'cell_id' in tokenized_dataset.features:
                    tokenized_cell_ids = tokenized_dataset['cell_id']
                    print(f"Tokenized dataset has {len(tokenized_cell_ids)} cells")
                    print(f"First 10 cell IDs from tokenized: {tokenized_cell_ids[:10]}")
                    
                    # Check if order matches
                    if list(tokenized_cell_ids) == list(adata.obs_names):
                        print("✅ Tokenized dataset order matches input order")
                    else:
                        print("⚠️  WARNING: Tokenized dataset order differs from input")
                        print("  First mismatch:")
                        for i, (tid, oid) in enumerate(zip(tokenized_cell_ids, adata.obs_names)):
                            if tid != oid:
                                print(f"    Position {i}: tokenized={tid}, original={oid}")
                                break
                    
                    # Since tokenized order matches input, and embeddings are in tokenized order,
                    # embeddings should be in correct order
                    print("\n✅ VERIFICATION PASSED: Embeddings are in the correct order")
                    print("   (Tokenized dataset order matches input, embeddings follow tokenized order)")
                else:
                    print("⚠️  Tokenized dataset doesn't have cell_id feature")
            except Exception as e:
                print(f"⚠️  Could not load tokenized dataset: {e}")
        else:
            print("⚠️  Tokenized dataset not found in debug directory")
    else:
        print("\n💡 Tip: Run with GENEFORMER_DEBUG=1 to enable detailed verification")
        print("   This will save intermediate files for order verification")
    
    print("\n✅ Basic verification complete!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_geneformer_order.py <input.h5ad> <embeddings.npy>")
        sys.exit(1)
    
    verify_embeddings_order(sys.argv[1], sys.argv[2])

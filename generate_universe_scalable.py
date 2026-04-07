#!/usr/bin/env python3
"""
Scalable Universe Generation for Llama 3.1
Generates full vocabulary universe sorted by frequency for efficient slicing.
Output can be sliced to any size (e.g., top 40k) without regeneration.

Usage:
  python generate_universe_scalable.py [--use-gpu] [--batch-size 512]
"""

import torch
import numpy as np
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import argparse
import time
from pathlib import Path

def generate_scalable_universe(args):
    # Device selection
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    start_time = time.time()
    
    # 1. Load tokenizer from local path
    print("📚 Loading tokenizer...")
    # Check multiple possible paths
    possible_paths = [
        Path("/root/tokenizer.json"),  # Uploaded directly
        Path.home() / "Models" / "llama-3.1-8b-instruct-gguf" / "tokenizer.json",
        Path.home() / "models" / "llama-3.1-8b-instruct-gguf" / "tokenizer.json",
        Path("/root/Models/llama-3.1-8b-instruct-gguf/tokenizer.json"),
        Path("/workspace/Models/llama-3.1-8b-instruct-gguf/tokenizer.json"),
    ]
    
    tokenizer_path = None
    for p in possible_paths:
        if p.exists():
            tokenizer_path = p
            break
    
    if tokenizer_path:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = tokenizer.get_vocab_size()
        vocab = tokenizer.get_vocab()
        print(f"✅ Loaded local tokenizer: {tokenizer_path}")
    else:
        # Fallback to open Qwen2 tokenizer (no auth needed)
        print("Local tokenizer not found, using Qwen2-7B (open)")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
    
    print(f"📊 Vocabulary size: {vocab_size:,} tokens")
    
    # 2. Sort tokens by frequency (using token ID as proxy - lower IDs are more frequent)
    # Llama's tokenizer assigns IDs by frequency during BPE training
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])  # Sort by token_id
    token_strings = [token for token, _ in sorted_tokens]
    token_ids = [tid for _, tid in sorted_tokens]
    
    print(f"✅ Sorted {len(token_strings):,} tokens by frequency")
    
    # 3. Load embedding model
    print("🧠 Loading embedding model...")
    if args.model == "llama":
        # Use Llama's own embeddings (requires full model - expensive!)
        model = AutoModel.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            torch_dtype=torch.float16 if args.use_gpu else torch.float32,
            device_map="auto" if args.use_gpu else None
        )
        embed_dim = 4096
    else:
        # Use sentence transformer (much lighter, still good quality)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        model = model.to(device)
        embed_dim = 768  # Will pad to 4096
    
    # 4. Generate embeddings in batches
    print(f"🚀 Generating {embed_dim}D embeddings (batch size: {args.batch_size})...")
    all_positions = []
    
    for i in tqdm(range(0, vocab_size, args.batch_size), desc="Embedding tokens"):
        batch_tokens = token_strings[i:i + args.batch_size]
        
        if args.model == "llama":
            # Use Llama embeddings
            inputs = tokenizer(batch_tokens, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Use last hidden state, mean pooled
                embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 4096]
        else:
            # Use sentence transformer
            embeddings = model.encode(batch_tokens, convert_to_tensor=True, device=device)
            # Pad to 4096D if needed (using zeros for extra dims)
            if embeddings.shape[1] < 4096:
                padding = torch.zeros((embeddings.shape[0], 4096 - embeddings.shape[1]), 
                                     device=device, dtype=embeddings.dtype)
                embeddings = torch.cat([embeddings, padding], dim=1)
        
        all_positions.append(embeddings.cpu().float())
    
    positions = torch.cat(all_positions, dim=0)
    print(f"✅ Generated positions: {positions.shape}")
    
    # 5. Compute mass using Zipf's law (frequency-based)
    print("⚖️ Computing mass (Zipfian distribution)...")
    mass = torch.zeros(vocab_size)
    for i in range(vocab_size):
        rank = i + 1  # 1-indexed rank
        # Zipfian mass: less frequent = higher mass
        # Top tokens (rank 1-1000): low mass for fast movement
        # Rare tokens (rank 50000+): high mass for stability
        if rank <= 1000:
            mass[i] = 0.1 + (rank / 1000) * 0.9  # 0.1 to 1.0
        elif rank <= 10000:
            mass[i] = 1.0 + np.log10(rank / 1000)  # 1.0 to ~2.0
        else:
            mass[i] = 2.0 + np.log10(rank / 10000) * 3  # 2.0 to ~5.0
    
    # 6. Compute charge (simplified - special tokens get unique charges)
    print("⚡ Computing charge...")
    charge = torch.zeros(vocab_size)
    for i, token in enumerate(token_strings):
        if token.startswith("<") and token.endswith(">"):  # Special tokens
            charge[i] = 2.0
        elif token.startswith("##"):  # Subword tokens
            charge[i] = -0.5
        elif token.isupper():  # Acronyms/proper nouns
            charge[i] = 1.0
        elif any(c.isdigit() for c in token):  # Numbers
            charge[i] = 0.5
        else:
            charge[i] = 0.0  # Neutral
    
    # 7. Add frequency ranks for easy slicing
    frequency_rank = torch.arange(vocab_size, dtype=torch.int32)
    
    # 8. Save universe
    output_file = args.output or f"universe_llama31_{vocab_size}.safetensors"
    print(f"💾 Saving to {output_file}...")
    
    tensors = {
        "positions": positions,        # [vocab_size, 4096]
        "mass": mass,                  # [vocab_size]
        "charge": charge,              # [vocab_size]
        "frequency_rank": frequency_rank,  # [vocab_size] - for reference
    }
    
    save_file(tensors, output_file)
    
    # Save token map
    map_file = output_file.replace(".safetensors", "_token_map.json")
    with open(map_file, "w") as f:
        json.dump(token_strings, f)
    
    elapsed = time.time() - start_time
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    
    print(f"\n✨ Universe generation complete!")
    print(f"📁 Output: {output_file} ({file_size_mb:.1f} MB)")
    print(f"📋 Token map: {map_file}")
    print(f"⏱️ Time: {elapsed/60:.1f} minutes")
    print(f"\n💡 To use only top 40k tokens in Rust:")
    print(f'   positions.narrow(0, 0, 40000)?  // Instant slice!')
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scalable universe for physics simulation")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for embedding")
    parser.add_argument("--model", choices=["llama", "nomic"], default="nomic", 
                        help="Embedding model (llama needs 40GB VRAM)")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    generate_scalable_universe(args)
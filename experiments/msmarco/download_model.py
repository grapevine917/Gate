"""Run from a network-enabled terminal to fetch only the encoder files (~90MB)."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent/'models/all-MiniLM-L6-v2'
    snapshot_download(
        repo_id='sentence-transformers/all-MiniLM-L6-v2',
        revision='c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
        local_dir=str(folder),
        allow_patterns=['*.json','*.txt','model.safetensors','1_Pooling/*'],
    )
    print(f'Model saved: {folder}')
    print('Next: bash experiments/msmarco/run.sh --stage all --offline')

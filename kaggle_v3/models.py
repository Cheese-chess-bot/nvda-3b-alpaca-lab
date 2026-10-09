"""Allowlisted model identities, access preflight and immutable downloads."""
import argparse
import json
import os
import re

REGISTRY = {
    'qwen': {'repo': 'Qwen/Qwen2.5-3B-Instruct', 'adapter': 'qwen'},
    'gemma': {'repo': 'google/gemma-3-4b-it', 'adapter': 'gemma'},
}


def selected_models(value):
    names = value.split(',') if isinstance(value, str) else list(value)
    names = [n.strip() for n in names]
    if not names or len(set(names)) != len(names) or any(n not in REGISTRY for n in names):
        raise ValueError('Choose qwen,gemma or a single qwen/gemma ablation')
    return [n for n in REGISTRY if n in names]


def precision(alias, kind):
    """Gemma never uses FP16: use FP32 compute, including on older T4 GPUs."""
    if alias not in REGISTRY or kind not in ('cpu', 'cuda', 'rocm', 'xpu'):
        raise ValueError('Unknown model or device')
    dtype = 'float32' if alias == 'gemma' or kind == 'cpu' else 'float16'
    return {'dtype': dtype, 'quantization': 'nf4' if kind == 'cuda' else 'none'}


def validate_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Expected an immutable Hugging Face commit SHA')
    return revision


def resolve(names):
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get('HF_TOKEN')
    result = {}
    for alias in selected_models(names):
        repo = REGISTRY[alias]['repo']
        try:
            revision = validate_revision(HfApi(token=token).model_info(repo).sha)
            # Metadata alone can be public while actual model files are gated.
            for filename in ('config.json', 'tokenizer_config.json'):
                hf_hub_download(repo, filename, revision=revision, token=token)
        except Exception as exc:
            raise RuntimeError(f'{alias} access check failed ({type(exc).__name__}). '
                'For Gemma, review/accept its terms on Hugging Face and enable an HF_TOKEN '
                'Kaggle Secret with read access. Check Internet access too. No model was silently dropped.') from None
        result[alias] = dict(REGISTRY[alias], revision=revision)
    return result


def download(config):
    from huggingface_hub import snapshot_download
    for alias in selected_models(config['models']):
        model = config['models'][alias]
        if model['repo'] != REGISTRY[alias]['repo']:
            raise ValueError('Unexpected model repository')
        try:
            snapshot_download(model['repo'], revision=validate_revision(model['revision']),
                token=os.environ.get('HF_TOKEN'),
                allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'LICENSE*'])
        except Exception as exc:
            raise RuntimeError(f'{alias} download failed ({type(exc).__name__}). '
                'Check network, disk space and model access; completed downloads remain cached.') from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['resolve', 'download'])
    parser.add_argument('--models', default='qwen,gemma')
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.mode == 'resolve':
        print(json.dumps(resolve(args.models)))
    else:
        with open(args.config) as handle:
            download(json.load(handle))

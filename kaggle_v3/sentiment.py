"""Sharded, resumable news inference; one model per process and stage."""
import argparse
import os
from pathlib import Path
from core import digest, dump, read, parse_signal
from models import REGISTRY, precision
from model_runtime import PROMPT_VERSION


def cache_tag(config, alias):
    return digest(dict(model=config['models'][alias], prompt=PROMPT_VERSION,
        execution=precision(alias, config['hardware']['kind']), source_hash=config['source_hash'],
        packages=config['packages']))


def cached_signal(path, article_key, tag):
    saved = read(path)
    if saved.get('model_tag') != tag or saved.get('article_key') != article_key:
        raise RuntimeError('News cache provenance mismatch')
    signal = saved['signal']
    if signal is not None:
        import json
        if parse_signal(json.dumps(signal)) is None:
            raise RuntimeError('Invalid cached signal schema')
    return signal


def analyze(alias):
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    work = Path(os.environ['NVDA_WORK']); config = read(work/'config.json')
    if alias not in config['models'] or alias not in REGISTRY:
        raise ValueError('Model is not enabled for this cycle')
    plan = read(work/'news_plan.json'); cache = work/'news_cache'/alias
    cache.mkdir(parents=True, exist_ok=True); tag = cache_tag(config, alias)
    todo = []
    for article in plan['articles'][rank::world]:
        path = cache/(article['key']+'.json')
        if path.exists():
            cached_signal(path, article['key'], tag)
        else:
            todo.append(article)
    if not todo:
        print(alias, 'rank', rank, 'cache complete', flush=True); return
    import torch
    from hardware import resolve_device, clear_cache
    from qwen import QwenAnalyzer
    from gemma import GemmaAnalyzer
    device = resolve_device(rank=rank); analyzer = None
    try:
        cls = {'qwen': QwenAnalyzer, 'gemma': GemmaAnalyzer}[alias]
        analyzer = cls(alias, config['models'][alias], device)
        for index, article in enumerate(todo):
            signal = analyzer.analyze(article)
            dump(cache/(article['key']+'.json'), dict(article_key=article['key'], model_tag=tag,
                signal=signal, error=None if signal is not None else 'invalid_structured_output'))
            if index % 25 == 0:
                print(alias, 'rank', rank, 'completed', index+1, '/', len(todo), flush=True)
    except torch.OutOfMemoryError:
        clear_cache(device)
        raise RuntimeError(alias+' ran out of memory; keep cached signals and rerun on adequate hardware. '
                           'Gemma uses FP32 compute, NF4 on NVIDIA; CPU/AMD/Intel need full-weight memory.') from None
    finally:
        if analyzer is not None:
            analyzer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=REGISTRY)
    analyze(parser.parse_args().model)

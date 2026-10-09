"""CPU-only tests for vendor routing, process count and news-mode behavior."""
import unittest
from hardware import choose_plan
from train_box import effective_news


def host(devices,**backends):
    return dict(devices=devices,distributed=backends,torch_version='2.6.0',cpu='test CPU')


class HardwareTests(unittest.TestCase):
    def test_nvidia_two_gpu(self):
        p=choose_plan(host({'cuda':['T4','T4']},nccl=True))
        self.assertEqual((p['kind'],p['workers'],p['distributed_backend']),('cuda',2,'nccl'))
    def test_amd_uses_cuda_device_namespace(self):
        p=choose_plan(host({'rocm':['AMD 0','AMD 1']},nccl=True))
        self.assertEqual((p['kind'],p['device_type'],p['workers']),('rocm','cuda',2))
    def test_intel_xccl(self):
        p=choose_plan(host({'xpu':['Intel 0','Intel 1']},xccl=True))
        self.assertEqual((p['kind'],p['distributed_backend']),('xpu','xccl'))
    def test_no_collectives_uses_single_gpu(self):
        p=choose_plan(host({'xpu':['Intel 0','Intel 1']},xccl=False))
        self.assertEqual(p['workers'],1);self.assertIsNone(p['distributed_backend'])
    def test_cpu_fallback(self):
        self.assertEqual(choose_plan(host({},gloo=True))['kind'],'cpu')
    def test_explicit_device_does_not_silently_fallback(self):
        with self.assertRaises(RuntimeError): choose_plan(host({}),'xpu')
    def test_gpu_limit_and_override(self):
        info=host({'cuda':['A','B','C','D']},nccl=True)
        self.assertEqual(choose_plan(info,max_devices=1)['workers'],1)
        self.assertEqual(choose_plan(info,'cpu')['kind'],'cpu')
    def test_news_auto_without_credentials(self):
        self.assertFalse(effective_news('auto','',{})['enabled'])
    def test_news_with_credentials(self):
        self.assertTrue(effective_news('auto','',dict(a='test',b='test'))['use_alpaca_news'])
    def test_news_file_overrides_api(self):
        mode=effective_news('auto','news.jsonl',{})
        self.assertTrue(mode['enabled']);self.assertFalse(mode['use_alpaca_news'])
    def test_required_news_cannot_be_dropped(self):
        with self.assertRaises(RuntimeError): effective_news('required','',{})
    def test_off_news_conflict(self):
        with self.assertRaises(ValueError): effective_news('off','news.jsonl',{})


if __name__=='__main__': unittest.main()

import json,struct,pathlib,collections
root=pathlib.Path('/model')
c=json.loads((root/'config.json').read_text());t=c['text_config']
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in index if ('layers.2.' in k or 'layers.20.' in k) and any(s in k for s in ['wq_b.weight','wq_a.weight','wkv.weight','attn_sink','wo_a.weight','wo_b.weight']) and 'scale' not in k]
headers={}
for shard in {index[k] for k in keys}:
 with open(root/shard,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];headers[shard]=json.loads(f.read(n))
result={'architecture':c['architectures'],'text_config':t,'backbone_ratio_counts':dict(collections.Counter(t['compress_ratios'][:t['num_hidden_layers']])),'tp8_local_heads':t['num_attention_heads']//8,'checkpoint_examples':{k:{a:headers[index[k]][k][a] for a in ['dtype','shape']} for k in keys},'scope':'Read config/index/safetensors headers only; no complete weights loaded. Attention function tests exclude projections, indexer, MoE, Engram and full-model evaluation.'}
pathlib.Path('results/model-structure.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

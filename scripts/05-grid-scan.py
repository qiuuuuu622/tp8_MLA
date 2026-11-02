import sys,os,json,math
from pathlib import Path
import torch,torch.distributed as dist
from experiments.antgrid_workspace import FusedWorkspace
from benchmarks.kernels.head_exchange.benchmark import graph_measure,warm
from benchmarks.kernels.head_exchange.baseline import measure
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
configs=[(None,256),(4,128),(8,256),(16,256),(32,256),(64,512)]
result={'scope':'Transport-only grid ablation; unchanged payload/barrier, only host block limit raised from 32 to 64. Ant uses 64x512; not the full Ant implementation.','physical_ranks':8,'cases':[],'status':'running'}
for rows in [8,32,96,128,288,512,1024]:
 for blocks,threads in configs:
  q=torch.zeros(rows,1,64,512,dtype=torch.bfloat16,device='cuda')
  q[:,:,:8]=rank+1
  sink=torch.zeros(64,dtype=torch.float32,device='cuda')
  ws=FusedWorkspace(q,sink,blocks=blocks,threads=threads)
  def move():
   ws.forward_q(q);ws.backward_o()
  def roundtrip():
   ws.forward_q(q);ws.row_o.copy_(ws.row_q);ws.backward_o()
  # Validate both head placement and inverse mapping in eager and graph.
  g=torch.cuda.CUDAGraph()
  roundtrip();torch.cuda.synchronize()
  with torch.cuda.graph(g):roundtrip()
  for i in range(3):
   q[:,:,:8]=rank+1+i*8;g.replay();torch.cuda.synchronize()
   assert torch.equal(ws.output[:,:,:8],q[:,:,:8])
   for h in range(8):
    start=min(rank*ws.chunk,rows);valid=min(ws.chunk,rows-start)
    assert torch.all(ws.row_q[:valid,:,h*8:(h+1)*8]==h+1+i*8).item()
  warm(move,30)
  graph=graph_measure(move,20,30);eager=measure(move,10,20)
  local={'rank':rank,'rows':rows,'requested_blocks':blocks,'blocks':ws.blocks,'threads':threads,'correct':True,'graph':graph,'eager':eager}
  gathered=[None]*8;dist.all_gather_object(gathered,local)
  if rank==0:
   result['cases'].append({'rows':rows,'requested_blocks':blocks,'blocks':ws.blocks,'threads':threads,'ranks':gathered})
   print(json.dumps({'rows':rows,'blocks':blocks,'actual_blocks':ws.blocks,'threads':threads,'graph_us':max(x['graph']['p50_us'] for x in gathered)}),flush=True)
   (Path(os.environ.get('RESULT_DIR', 'results')) / 'tp8-grid-scan.json').write_text(json.dumps(result,indent=2))
result['status']='completed'
if rank==0:(Path(os.environ.get('RESULT_DIR', 'results')) / 'tp8-grid-scan.json').write_text(json.dumps(result,indent=2))
dist.destroy_process_group()

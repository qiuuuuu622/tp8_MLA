import faulthandler,os,runpy
faulthandler.enable();faulthandler.dump_traceback_later(35,repeat=True)
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from vllm_dsv41_head_exchange import workspace
rank=os.environ['LOCAL_RANK']
def wrap(owner,name):
 old=getattr(owner,name)
 def call(*a,**kw):
  print(rank,'BEGIN',name,flush=True)
  result=old(*a,**kw)
  print(rank,'END',name,flush=True)
  return result
 setattr(owner,name,call)
for name in ['all_gather_into_tensor','barrier']:wrap(dist,name)
for name in ['empty','rendezvous']:wrap(symm,name)
wrap(workspace.FusedWorkspace,'forward_q');wrap(workspace.FusedWorkspace,'backward_o')
runpy.run_module('benchmarks.kernels.head_exchange.verify',run_name='__main__')

import sys, argparse
from pathlib import Path
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from types import SimpleNamespace
parser = argparse.ArgumentParser()
parser.add_argument('resize_right_checkout', help='ResizeRight checkout at 510d4d5')
root = Path(parser.parse_args().resize_right_checkout).resolve()
sys.path.insert(0,str(root))
import resize_right
ns=dict(torch=torch,nn=nn,F=F,T=T,TF=TF,resize=resize_right.resize,
        args=SimpleNamespace(animation_mode='None'),padargs={},cutout_debug=False)
source=Path(__file__).with_name('disco.py')
exec(compile(source.read_text(),str(source),'exec'),ns)
values={}
for overview in [4,12]:
 for augment in [False,True]:
  ns['skip_augs']=not augment
  x=torch.linspace(-1.3,1.2,3*40*64).reshape(1,3,40,64).requires_grad_()
  torch.manual_seed(123)
  y=ns['MakeCutoutsDango'](16,Overview=overview,InnerCrop=3,IC_Size_Pow=1,IC_Grey_P=.2)(x.add(1).div(2))
  grad=torch.autograd.grad(y.square().sum(),x)[0]
  key=f'{overview}_{int(augment)}'
  values[key+'_cuts']=y.detach().numpy();values[key+'_grad']=grad.numpy()
np.savez_compressed(source.with_name('cutouts.npz'),**values)
print('Golden cutouts saved from upstream ResizeRight:',resize_right.__file__)

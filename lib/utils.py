import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect

def adapt_model(parent, orig, new, attrs=None, extra_args={}):
    
    # grab necessary arguments
    if not attrs:
        attrs = inspect.getfullargspec(orig).args[1:]
        print(f"changing all {orig} to {new}")
    
    # replace
    for n, m in parent.named_children():
        attr_str = n
        target_attr = m 
        if type(target_attr) == orig:
            # print('replacing:', n, attr_str)
            args = {attr: getattr(target_attr, attr) for attr in attrs}
            args.update(extra_args)
            setattr(parent, attr_str, new(**args))
            continue
            
        # recurse
        adapt_model(m, orig=orig, new=new, attrs=attrs, extra_args=extra_args)

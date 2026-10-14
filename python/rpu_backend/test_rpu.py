import torch
import rpu_backend

torch._C._rename_privateuse1_backend("rpu")
x = torch.tensor([1., 2., 3.,1., 2., 3.,1., 2., 3.,1., 2., 3., 3.,1., 2., 3], device="rpu", dtype=torch.float16)
y = torch.tensor([1., 2., 3.,1., 2., 3.,1., 2., 3.,1., 2., 3., 3.,1., 2., 3], device="rpu", dtype=torch.float16)
print("x:", x.to("cpu"))
print("y:", y.to("cpu"))
z = x + y
print("x + y:", z.to("cpu"))
# print(x)
zz = x.to("cpu") + y.to("cpu")
print(zz)
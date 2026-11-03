import sys,torch,pytorch_ocl
torch.set_num_threads(2)
x=torch.tensor([[0,.1,.2,.3,.4,.5],[1,.6,.7,.8,.9,1.]])
g=x.to('ocl:0')
for j in range(2):
 i=g[:,0]
 _,counts=i.unique(return_counts=True)
 counts=counts.to(torch.int32)
 out=torch.zeros((2,int(counts.max()),5),device='ocl:0')
 m=i==j;n=m.sum()
 if n:
  selected=g[m,1:]
  actual=selected.cpu();expected=x[x[:,0]==j,1:]
  print(j,'actual',actual,'expected',expected,flush=True)
  torch.testing.assert_close(actual,expected)
  out[j,:n]=selected
print('PASS complete target-selection chain',flush=True)

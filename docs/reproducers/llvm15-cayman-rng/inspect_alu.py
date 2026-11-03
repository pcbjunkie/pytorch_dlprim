"""Inspect the single-kernel reproducer ELF; CPU only, no GPU or dependencies."""
import struct,sys
from pathlib import Path
b=Path(sys.argv[1]).read_bytes()
assert b[:6]==b'\x7fELF\x01\x01','expected little-endian ELF32'
off=struct.unpack_from('<I',b,32)[0]
size,count,strings=struct.unpack_from('<HHH',b,46)
headers=[struct.unpack_from('<10I',b,off+i*size)for i in range(count)]
st=headers[strings];names=b[st[4]:st[4]+st[5]]
sections={names[h[0]:].split(b'\0',1)[0].decode():b[h[4]:h[4]+h[5]]for h in headers if h[1]!=8}
text=sections['.text'];clauses=[];limit=len(text)//8;i=0
while i<limit:
 word0,word1=struct.unpack_from('<II',text,i*8)
 op=(word1>>26)&15
 if op>=8:
  address=word0&0x3fffff;count=(word1>>18)&127
  clauses.append((i,address,count));limit=min(limit,address)
 i+=1
bad=False
for j,(pc,address,count)in enumerate(clauses):
 end=clauses[j+1][1]if j+1<len(clauses)else len(text)//8
 span=end-address
 print(f'CF[{pc}] ALU address={address}, encoded_count={count}, encoded_slots={count+1}, physical_span={span}')
 if count+1!=span:bad=True
print('INVALID clause length'if bad else'Clause lengths agree (single-kernel fixture)')
raise SystemExit(1 if bad else 0)

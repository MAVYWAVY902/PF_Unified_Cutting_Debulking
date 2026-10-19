"""Generate a closed needle OBJ (cylinder shaft + conical tip, tip at -y) for the Step-3b point cut
feature (core/tool_features.h): apex angle 18 deg < 30 deg → exactly one sharp point.
Usage: python3 tools/generate_needle_obj.py resources/tools/needle.obj
"""
# Closed, outward-wound needle: cylinder (radius r, along y) + conical tip pointing to -y.
import math, sys
r, L, lt, n = 0.0004, 0.015, 0.0025, 16
ytop, ybase = L/2, -L/2 + lt          # cylinder from ybase to ytop, tip apex at -L/2
V=[(0,ytop,0)]                        # 0: top cap centre
V+=[(r*math.cos(2*math.pi*k/n), ytop, r*math.sin(2*math.pi*k/n)) for k in range(n)]   # 1..n top ring
V+=[(r*math.cos(2*math.pi*k/n), ybase, r*math.sin(2*math.pi*k/n)) for k in range(n)]  # n+1..2n base ring
V.append((0,-L/2,0))                  # 2n+1: tip apex
F=[]
T=lambda k:1+k%n; B=lambda k:1+n+k%n; tip=2*n+1
for k in range(n):
    F.append((0,T(k+1),T(k)))                     # top cap, normal +y
    F.append((T(k),T(k+1),B(k+1))); F.append((T(k),B(k+1),B(k)))   # side
    F.append((B(k),B(k+1),tip))                   # cone
# fix winding: make every face normal point away from the axis/centroid
def sub(a,b): return [a[i]-b[i] for i in range(3)]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
out=[]
for f in F:
    a,b,c=[V[i] for i in f]; nrm=cross(sub(b,a),sub(c,a)); cen=[(a[i]+b[i]+c[i])/3 for i in range(3)]
    ref=[cen[0],0 if f[0]!=0 else cen[1],cen[2]] if f[0]!=0 else [0,1,0]
    d=[cen[0],cen[1]-0.0,cen[2]]; out_dir=[cen[0],0,cen[2]] if f[0]!=0 else [0,1,0]
    if f[0]!=0 and abs(out_dir[0])+abs(out_dir[2])<1e-12: out_dir=[0,-1,0]
    if sum(nrm[i]*out_dir[i] for i in range(3))<0: f=(f[0],f[2],f[1])
    out.append(f)
with open(sys.argv[1],'w') as fo:
    fo.write('# needle r=%.4g L=%.4g tip=%.4g n=%d (apex angle %.1f deg)\n'%(r,L,lt,n,2*math.degrees(math.atan(r/lt))))
    for v in V: fo.write('v %.9g %.9g %.9g\n'%v)
    for f in out: fo.write('f %d %d %d\n'%(f[0]+1,f[1]+1,f[2]+1))

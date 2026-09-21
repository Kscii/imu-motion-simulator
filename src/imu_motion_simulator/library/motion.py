"""Source-preserving BVH research adapter; not a production contract.

Supported profile: root XYZ translations followed by three distinct rotation
channels; descendants have three rotation channels. Units are never guessed.
Quaternion convention is wxyz, active rotations, column vectors.
"""
import hashlib
import re
from fractions import Fraction
import numpy as np


def multiply(a,b):
    a,b=np.broadcast_arrays(a,b)
    return np.concatenate((a[...,:1]*b[...,:1]-np.sum(a[...,1:]*b[...,1:],axis=-1,keepdims=True),
        a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)


def matrix(q):
    q=np.asarray(q);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(q.shape[:-1]+(3,3))


def fk(q,p,parents):
    local=matrix(q);world=np.empty_like(local);xyz=np.empty_like(p)
    for i,parent in enumerate(parents):
        if parent<0:world[...,i,:,:]=local[...,i,:,:];xyz[...,i,:]=p[...,i,:]
        else:
            if parent>=i:raise ValueError('Parent must precede child')
            world[...,i,:,:]=world[...,parent,:,:]@local[...,i,:,:]
            xyz[...,i,:]=xyz[...,parent,:]+np.einsum('...ij,...j->...i',world[...,parent,:,:],p[...,i,:])
    return world,xyz


def parse(raw):
    header,motion=raw.decode('utf-8').split('MOTION',1);tokens=re.findall(r'[^\s{}]+|[{}]',header);cursor=0
    names=[];parents=[];offsets=[];channels=[];ends=[]
    def take(wanted=None):
        nonlocal cursor
        if cursor>=len(tokens):raise ValueError('Truncated hierarchy')
        value=tokens[cursor];cursor+=1
        if wanted is not None and value!=wanted:raise ValueError('Expected '+wanted+', found '+value)
        return value
    def offset():
        take('OFFSET');return [float(take()) for _ in range(3)]
    def node(parent,kind):
        take(kind);name=take()
        if name in names:raise ValueError('Duplicate joint name')
        idx=len(names);names.append(name);parents.append(parent);take('{');offsets.append(offset());take('CHANNELS')
        width=int(take());ch=[take() for _ in range(width)];channels.append(ch)
        while tokens[cursor]!='}':
            if tokens[cursor]=='JOINT':node(idx,'JOINT')
            elif tokens[cursor]=='End':
                take('End');take('Site');take('{');ends.append({'parent':idx,'offset':offset()});take('}')
            else:raise ValueError('Unsupported hierarchy token '+tokens[cursor])
        take('}')
    take('HIERARCHY');node(-1,'ROOT')
    if cursor!=len(tokens):raise ValueError('Trailing hierarchy tokens')
    lines=[line.strip() for line in motion.splitlines() if line.strip()]
    count=re.fullmatch(r'Frames:\s*(\d+)',lines[0]);clock=re.fullmatch(r'Frame Time:\s*(\S+)',lines[1])
    if not count or not clock:raise ValueError('Missing frame count/time')
    n=int(count[1]);dt_text=clock[1];dt=Fraction(dt_text)
    if dt<=0 or len(lines)-2!=n:raise ValueError('Bad time or frame count')
    widths=[len(ch) for ch in channels];width=sum(widths)
    if any(len(line.split())!=width for line in lines[2:]):raise ValueError('Frame channel count mismatch')
    values=np.fromstring('\n'.join(lines[2:]),sep=' ').reshape(n,width)
    if not np.isfinite(values).all() or not np.isfinite(offsets).all():raise ValueError('Nonfinite source')
    q=np.zeros((n,len(names),4));q[:,:,0]=1.;pos=np.broadcast_to(offsets,(n,len(names),3)).copy();start=0
    for j,ch in enumerate(channels):
        rot=ch[3:] if j==0 else ch
        if (j==0 and ch[:3]!=['Xposition','Yposition','Zposition']) or len(rot)!=3 or set(rot)!={'Xrotation','Yrotation','Zrotation'}:raise ValueError('Unsupported BVH channel profile')
        if j==0:pos[:,0]=values[:,:3] # explicit LAFAN/ACCAD BVH profile, not OFFSET + translation
        for k,name in enumerate(rot):
            angle=np.deg2rad(values[:,start+(3 if j==0 else 0)+k]);axis='XYZ'.index(name[0]);one=np.zeros((n,4));one[:,0]=np.cos(angle/2);one[:,axis+1]=np.sin(angle/2);q[:,j]=multiply(q[:,j],one)
        start+=len(ch)
    # Sign continuity preserves represented rotations and does not smooth motion.
    sign=np.where(np.sum(q[1:]*q[:-1],axis=-1)<0,-1.,1.)
    q[1:]*=np.cumprod(sign,axis=0)[...,None]
    meta={'names':names,'parents':parents,'rest_offsets':offsets,'end_sites':ends,'channels':channels,'frames':n,
        'frame_time_text':dt_text,'frame_time_fraction':[dt.numerator,dt.denominator],'unit':'unknown','world_axes':'unknown',
        'source_sha256':hashlib.sha256(raw).hexdigest(),'quaternion':'wxyz active local-to-parent',
        'root_translation_semantics':'root channels replace root OFFSET for supported source profile; rest OFFSET retained',
        'execution_ready':False,'production_contract':False,'event_labels':'unknown; filename theme is not interval truth'}
    return meta,{'local_quaternion':q,'local_translation':pos,'frame_id':np.arange(n,dtype=np.int64),'source_time':np.arange(n)*float(dt),
        'valid_source_numeric':np.ones((n,len(names)),dtype=bool)}


def windows(n,width=150,stride=5):
    if n<width:raise ValueError('Source too short for declared windows')
    starts=[0,(n-width)//2,n-width]
    return [{'start':s,'stop_exclusive':s+width,'frame_ids':np.arange(s,s+width),
        'sample_ids':np.unique(np.r_[np.arange(s,s+width,stride),s+width-1])} for s in starts]


def project_two_axes(q,plane):
    """Global closest Rx(a)Ry(b) etc via a rank-one quaternion 2x2 SVD.

    Maximizing absolute quaternion inner product minimizes SO(3) geodesic
    distance. The quaternion of two fixed-axis rotations is an outer product
    of two unit 2-vectors; largest singular vectors solve that maximization.
    This projects source-coordinate planes, NOT anatomical shoulder axes.
    """
    q=np.asarray(q,float);q=q/np.linalg.norm(q);i,j=['xyz'.index(x) for x in plane];k=3-i-j
    sign=np.cross(np.eye(3)[i],np.eye(3)[j])[k]
    target=np.array([[q[0],q[j+1]],[q[i+1],sign*q[k+1]]]);u,s,vh=np.linalg.svd(target)
    best=np.outer(u[:,0],vh[0]);out=np.zeros(4);out[0]=best[0,0];out[j+1]=best[0,1];out[i+1]=best[1,0];out[k+1]=sign*best[1,1]
    if np.dot(out,q)<0:out=-out
    residual=float(2*np.arccos(np.clip(abs(np.dot(out,q)),0,1)))
    return out,residual,{'singular_values':s.tolist(),'ambiguous_optimum':bool(abs(s[0]-s[1])<1e-10)}


def topology(internal):
    if internal not in (37,41):raise ValueError('Only declared candidates')
    rows=[];cursor=0
    def add(name,parent,dof,source,missing):
        nonlocal cursor
        rows.append({'id':name,'parent':parent,'dof':dof,'tangent_slice':[cursor,cursor+dof],
            'source_nodes':source,'unresolved':missing,'physical_axes':None,'mass_properties':None,'ability':None,'damping':None})
        cursor+=dof
    add('pelvis',None,6,['Hips'],['world axes','SI units','root pose has 7 components; no root actuator'])
    add('waist','pelvis',3,['Spine','Spine1'],['spine allocation','rest anatomy'])
    add('chest','waist',3,['Spine1','Spine2'],['spine allocation','chest mount'])
    add('head','chest',3,['Neck','Head'],['two-segment merging','collision location'])
    for side,src in [('left','Left'),('right','Right')]:
        add(side+'_shoulder_girdle','chest',2 if internal==41 else 0,[src+'Shoulder'],['anatomical axes','girdle mass split','coupling'])
        add(side+'_upper_arm',side+'_shoulder_girdle',3,[src+'Arm'],['glenohumeral rest frame'])
        add(side+'_elbow',side+'_upper_arm',1,[src+'ForeArm'],['hinge axis','off-axis residual'])
        add(side+'_forearm_twist',side+'_elbow',1,[src+'ForeArm',src+'Hand'],['twist/wrist identifiability'])
        add(side+'_hand',side+'_forearm_twist',2,[src+'Hand'],['wrist decomposition','hand contact geometry'])
        add(side+'_thigh','pelvis',3,[src+'UpLeg'],['hip rest frame'])
        add(side+'_shin',side+'_thigh',1,[src+'Leg'],['knee axis','off-axis residual'])
        add(side+'_foot',side+'_shin',2,[src+'Foot'],['ankle axes','sole contact'])
        add(side+'_toe',side+'_foot',1,[src+'Toe'],['toe hinge and contact'])
    if cursor!=internal+6:raise AssertionError('DOF sum')
    return {'candidate':str(internal)+'+6','internal_dof':internal,'tangent_dof':cursor,'root_pose_components':7,
        'rows':rows,'slice_scope':'tangent velocity/effort bookkeeping only; generalized position encoding unfrozen',
        'source_mapping_scope':'functional coverage; overlapping sources are unresolved, not duplicate assigned rotations',
        'execution_ready':False,'human_parameters_validated':False,'full_body_q_ref_created':False}

"""Mech Royale v1: survival, local obstacle steering and weapon-aware aiming.
c1
"""
import array
import json
import math
import os
import sys
import mechfight as mf


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def inside(x, y, ring):
    result = False
    j = len(ring) - 2
    for i in range(0, len(ring), 2):
        ax, ay, bx, by = ring[i], ring[i+1], ring[j], ring[j+1]
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            result = not result
        j = i
    return result


class World:
    def __init__(self, root):
        with open(os.path.join(root, 'v2/world.json'), encoding='utf-8') as f:
            w = json.load(f)
        self.bounds, self.grid = w['bounds'], w['terrain']
        self.heights = array.array('h')
        with open(os.path.join(root, 'v2/terrain.i16'), 'rb') as f:
            self.heights.frombytes(f.read())
        if sys.byteorder != 'little':
            self.heights.byteswap()
        self.cells = {}
        for p in w['prisms']:
            r = p['ring']
            for cx in range(math.floor(min(r[::2])/32), math.floor(max(r[::2])/32)+1):
                for cy in range(math.floor(min(r[1::2])/32), math.floor(max(r[1::2])/32)+1):
                    self.cells.setdefault((cx, cy), []).append(p)

    def ground(self, x, y):
        g = self.grid
        u = max(0, min(g['nx']-1.001, (x-g['minX'])/g['step']))
        v = max(0, min(g['ny']-1.001, (y-g['minY'])/g['step']))
        i, j = int(u), int(v)
        fx, fy = u-i, v-j
        n = j*g['nx']+i
        h = self.heights
        return ((h[n]*(1-fx)+h[n+1]*fx)*(1-fy)+(h[n+g['nx']]*(1-fx)+h[n+g['nx']+1]*fx)*fy)*g.get('scale', .01)

    def solid(self, x, y, lo, hi):
        for p in self.cells.get((math.floor(x/32), math.floor(y/32)), ()):
            if p['top'] > lo and p['base'] < hi and inside(x, y, p['ring']):
                if not any(inside(x, y, h) for h in p.get('holes', ())):
                    return True
        return False

    def clear_shot(self, a, b):
        # Conservative sampled line of sight, including tracked enemies behind cover.
        n = max(2, math.ceil(math.dist(a, b)/3))
        for i in range(1, n):
            t = i/n
            x, y, z = [a[k]+(b[k]-a[k])*t for k in range(3)]
            if self.ground(x, y) > z or self.solid(x, y, z-.05, z+.05):
                return False
        return True

    def walkable(self, pos, direction, radius, height, step, slope):
        x, y, z = pos
        dx, dy = direction
        previous = z
        for length in (2., 4., 7., 10.):
            nx, ny = x+dx*length, y+dy*length
            b = self.bounds
            if not (b['minX']+radius < nx < b['maxX']-radius and b['minY']+radius < ny < b['maxY']-radius):
                return False
            floor = self.ground(nx, ny)
            # Preserve current roof if the feet remain over a supporting prism.
            if self.solid(nx, ny, z-.3, z+.1):
                floor = max(floor, z)
            if floor-previous > max(step, math.tan(math.radians(slope))*2):
                return False
            if floor < previous-8:
                return False
            for ox, oy in ((0,0), (radius,0), (-radius,0), (0,radius), (0,-radius)):
                if self.solid(nx+ox, ny+oy, floor+step, floor+height):
                    return False
            previous = floor
        return True


class Bot:
    name = 'Charlie-v1'

    def __init__(self, hello):
        self.cfg = hello['config']
        self.slot = hello['slot']
        self.world = None
        try:
            self.world = World(hello['map']['dir'])
        except (OSError, ValueError, KeyError) as e:
            mf.log('Map unavailable; using reactive movement:', e)
        self.last_pos = None
        self.last_time = 0
        self.stuck = 0
        self.last_heading = None
        self.recover_until = 0
        self.retreat_until = 0

    def steer(self, obs, goal, threats):
        me = obs['self']; pos = me['pos']; t = obs['t']; stats = me['stats']
        base = math.atan2(goal[1]-pos[1], goal[0]-pos[0])
        if self.last_pos is not None and t-self.last_time >= .6:
            self.stuck = self.stuck+1 if dist(pos, self.last_pos) < 1.2 else 0
            self.last_pos, self.last_time = pos[:], t
        elif self.last_pos is None:
            self.last_pos, self.last_time = pos[:], t
        if self.stuck >= 3:
            self.recover_until = t+1.5
            self.stuck = 0
        recover = t < self.recover_until
        if recover:
            base += (1 if self.slot % 2 else -1)*math.pi/2
        choices = []
        size = 1+.05*me['cubes']
        for offset in (0, .35, -.35, .7, -.7, 1.15, -1.15, 1.6, -1.6, 2.2, -2.2, math.pi):
            a = base+offset; direction = (math.cos(a), math.sin(a))
            if self.world and not self.world.walkable(pos, direction, self.cfg['mech']['radius']*size,
                    self.cfg['mech']['height']*size, stats.get('step_up',1.2), stats.get('max_slope_deg',50)):
                continue
            endpoint = [pos[0]+direction[0]*10, pos[1]+direction[1]*10]
            score = (dist(pos, goal)-dist(endpoint, goal))*1.5
            if recover:
                score += 12*math.cos(offset)
            zone = obs['zone']
            score -= max(0, dist(endpoint, zone['center'])-zone['radius']+12)*4
            if self.last_heading is not None:
                score += 2*math.cos(a-self.last_heading)
            for enemy in threats:
                d = dist(endpoint, enemy['pos'])
                score -= max(0, 65-d)*(.6 if t < self.retreat_until else .12)
            choices.append((score, a, direction))
        if not choices:
            return [math.cos(base), math.sin(base)], True
        _, self.last_heading, direction = max(choices)
        return list(direction), recover

    def act(self, obs):
        me = obs['self']; pos = me['pos']; t = obs['t']; zone = obs['zone']
        hp = me['hp']/me['max_hp']; energy = me['energy']
        if hp < .42:
            self.retreat_until = t+6
        retreat = t < self.retreat_until and hp < .85
        if not retreat:
            self.retreat_until = 0
        enemies = [m for m in obs['mechs'] if m['kind']=='mech']
        nearest = min(enemies, key=lambda e: dist(pos,e['pos']), default=None)
        danger = nearest is not None and dist(pos,nearest['pos']) < 95
        urgent = not me['in_zone'] or dist(pos,zone['center']) > zone['radius']-25
        goal = list(zone['next_center'])
        mode = 'zone'
        # Seek easy cubes and weapons only when safe; do not chase through the storm.
        rank = {'gun':0, 'mortar':1, 'lightning':2, 'laser':3}
        held = [w['kind'] for w in me['weapons'] if w]
        secondary = next((w for w in me['weapons'] if w and w['kind']!='gun'), None)
        loot = [p for p in obs['pickups'] if p['kind'] not in held and rank[p['kind']] > rank.get(secondary['kind'] if secondary else 'gun',0)]
        if not urgent and not danger:
            targets = [(dist(pos,c['pos'])-15, c['pos'], 'cube') for c in obs['cubes'] if abs(c['pos'][2]-pos[2])<4]
            targets += [(dist(pos,p['pos'])-8, p['pos'], 'weapon') for p in loot if abs(p['pos'][2]-pos[2])<4]
            targets = [v for v in targets if v[0]<65 and dist(v[1],zone['center']) < zone['radius']-15]
            if targets:
                _, goal, mode = min(targets, key=lambda v:v[0])
        if retreat and nearest and not urgent:
            dx,dy = pos[0]-nearest['pos'][0],pos[1]-nearest['pos'][1]
            length = max(1,math.hypot(dx,dy))
            goal = [pos[0]+dx/length*65, pos[1]+dy/length*65]; mode='retreat'
        move, recover = self.steer(obs,goal,enemies)
        if dist(pos,goal)<5 and not danger:
            move=[0.,0.]
        act = {'tick':obs['tick'], 'move':move,
               'boost':bool(energy>30 and (urgent or (retreat and danger))),
               'jump':bool(recover and me['grounded'] and energy>25), 'fire':[False,False]}
        if obs['offer']:
            scores={'armor':100,'nanites':95,'servos':90,'plating':85,'gun_velocity':78,'gun_rate':75,'gun_damage':73,
                    'laser_damage':80,'laser_charge':79,'lightning_range':76,'reactor':65,'gyros':64,'gun_mag':60,'jets':58,'capacitor':50,'sensors':40}
            act['pick']=max(range(len(obs['offer']['choices'])),key=lambda i:scores.get(obs['offer']['choices'][i]['id'],30))
        for p in sorted(loot,key=lambda p:dist(pos,p['pos'])):
            if dist(pos,p['pos']) < self.cfg['loot']['pickup_radius'] and abs(pos[2]-p['pos'][2])<4:
                act['take']=p['id']
                if all(me['weapons']):
                    slot = next((i for i,w in enumerate(me['weapons']) if w['kind']!='gun'),1)
                    if not me['weapons'][slot].get('charge',0):
                        act['drop']=slot
                    else:
                        act.pop('take',None)
                break
        muzzle = [pos[0],pos[1],pos[2]+me['stats'].get('muzzle_height',7)]
        candidates = list(enemies)
        if not danger and not retreat:
            candidates += list(obs['crates'])
        candidates = sorted(candidates,key=lambda e: dist(pos,e['pos'])+(25 if e.get('kind')=='mech' else 0))
        target = None
        for e in candidates[:8]:
            ep = e['pos']; center = ep[2]+(self.cfg['mech']['height']*(1+.05*e.get('cubes',0))/2 if e.get('kind')=='mech' else 1.5)
            point=[ep[0],ep[1],center]
            if dist(pos,ep)>230 or (retreat and e.get('kind')=='mech' and dist(pos,ep)>45):
                continue
            if not self.world or self.world.clear_shot(muzzle,point):
                target=(e,point); break
        act['aim']=[math.atan2(move[1],move[0])+.5*math.sin(t*.8),0.]
        if retreat and nearest:
            act['aim']=[math.atan2(nearest['pos'][1]-pos[1],nearest['pos'][0]-pos[0]),0.]
        if target:
            e, point=target
            distance=dist(pos,e['pos']); vel=e.get('vel',[0,0,0])
            # Share torso aim: prioritize close lightning, otherwise laser, then gun.
            slots=[i for i,w in enumerate(me['weapons']) if w]
            def priority(i):
                w=me['weapons'][i]; kind=w['kind']
                return 4 if kind=='lightning' and distance<w['stats']['range'] and energy>20 else 3 if kind=='laser' and (w.get('charge',0)>0 or w['cooldown']<=.1) and not retreat else 2 if kind=='gun' else 1
            primary=max(slots,key=priority)
            weapon=me['weapons'][primary]; kind=weapon['kind']; s=weapon['stats']
            aimpoint=point[:]
            if kind=='gun':
                # Solve interception for a straight projectile, using observed world velocity.
                r=[point[k]-muzzle[k] for k in range(3)]
                a=sum(v*v for v in vel)-s['speed']**2
                b=2*sum(r[k]*vel[k] for k in range(3)); c=sum(v*v for v in r)
                disc=b*b-4*a*c
                roots=[v for v in ((-b+math.sqrt(max(0,disc)))/(2*a),(-b-math.sqrt(max(0,disc)))/(2*a)) if 0<v<2] if abs(a)>1e-6 and disc>=0 else []
                lead=min(roots) if roots else 0
                aimpoint=[point[k]+vel[k]*lead for k in range(3)]
            yaw=math.atan2(aimpoint[1]-muzzle[1],aimpoint[0]-muzzle[0])
            pitch=math.atan2(aimpoint[2]-muzzle[2],dist(muzzle,aimpoint))
            mortar_ok=False
            if kind=='mortar':
                speed=s['speed']; gravity=s['gravity']; d=distance
                dz=point[2]-(pos[2]+self.cfg['mech']['height']*(1+.05*me['cubes'])+.5)
                disc=speed**4-gravity*(gravity*d*d+2*dz*speed*speed)
                if disc>=0 and d>25:
                    pitch=math.atan((speed*speed-math.sqrt(disc))/(gravity*d)); mortar_ok=True
            act['aim']=[yaw,pitch]
            aligned=abs(angle(yaw-me['aim'][0]))<max(.025,math.atan2(2.2,max(1,distance))) and abs(pitch-me['aim'][1])<.065
            for i,w in enumerate(me['weapons']):
                if not w: continue
                k=w['kind']; st=w['stats']; ready=w['cooldown']<=.1
                if k=='gun':
                    act['fire'][i]=bool(kind=='gun' and aligned and ready and distance<st['range'] and not w.get('reload',0))
                elif k=='lightning':
                    act['fire'][i]=bool(ready and distance<st['range']-2 and energy>st['energy']+15 and abs(angle(yaw-me['aim'][0]))<math.radians(st['cone_deg'])*.8)
                elif k=='laser':
                    charge=w.get('charge',0)
                    act['fire'][i]=bool(kind=='laser' and distance<st['range'] and not retreat and (charge>0 or ready) and (charge<st['charge'] or not aligned))
                elif k=='mortar':
                    act['fire'][i]=bool(kind=='mortar' and mortar_ok and aligned and ready)
            mode='retreat' if retreat else 'aim'
        for w in me['weapons']:
            if w and w['kind']=='gun' and not w.get('reload',0):
                if w.get('ammo',0)==0 or (target is None and w.get('ammo',0)<w['stats']['mag']*.7):
                    act['reload']=True
        mf.draw.label(f'{mode} hp={me["hp"]:.0f}')
        return act


if __name__ == '__main__':
    mf.run(Bot)

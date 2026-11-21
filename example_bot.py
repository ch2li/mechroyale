"""Mech Royale v1.1: survival, local obstacle steering and weapon-aware aiming.
Standard library only. Map planning is approximate, not a full simulator.
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
    name = 'Charlie-v1.1'

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
        self.last_move = [0., 0.]
        self.npc_hostile_until = {}

    def steer(self, obs, goal, threats, moving=True):
        me = obs['self']; pos = me['pos']; t = obs['t']; stats = me['stats']
        base = math.atan2(goal[1]-pos[1], goal[0]-pos[0])
        if not moving:
            # Waiting at a goal is intentional, not evidence of a blocked route.
            self.last_pos, self.last_time = pos[:], t
            self.stuck = 0
            self.recover_until = 0
            self.last_move = [0., 0.]
            return [0., 0.], False
        if self.last_pos is not None and t-self.last_time >= .6:
            commanded = math.hypot(*self.last_move) > .1
            self.stuck = self.stuck+1 if commanded and dist(pos, self.last_pos) < 1.2 else 0
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
                avoid_range = self.cfg['npc'].get('aggro_range',45)+8 if enemy.get('kind')=='npc' else 65
                score -= max(0, avoid_range-d)*(.6 if t < self.retreat_until else .12)
            choices.append((score, a, direction))
        if not choices:
            self.last_move = [math.cos(base), math.sin(base)]
            return self.last_move[:], True
        _, self.last_heading, direction = max(choices)
        self.last_move = list(direction)
        return self.last_move[:], recover

    def act(self, obs):
        me = obs['self']; pos = me['pos']; t = obs['t']; zone = obs['zone']
        hp = me['hp']/me['max_hp']; energy = me['energy']
        if hp < .42:
            self.retreat_until = t+6
        retreat = t < self.retreat_until and hp < .85
        if not retreat:
            self.retreat_until = 0
        enemies = [m for m in obs['mechs'] if m['kind']=='mech']
        npcs = [m for m in obs['mechs'] if m['kind']=='npc']
        npc_ids = {m['id'] for m in npcs}
        for event in obs.get('events', []):
            attacker = event.get('by')
            if event.get('type')=='damaged' and event.get('weapon')=='npc' and attacker is not None:
                self.npc_hostile_until[attacker] = t+8
            if event.get('type')=='hit' and event.get('target') in npc_ids:
                self.npc_hostile_until[event['target']] = t+8
        self.npc_hostile_until = {i:until for i,until in self.npc_hostile_until.items() if until>t}
        npc_threats = [m for m in npcs if dist(pos,m['pos']) < self.cfg['npc'].get('aggro_range',45)+5
                       or self.npc_hostile_until.get(m['id'],0)>t]
        threats = enemies+npc_threats
        nearest = min(threats, key=lambda e: dist(pos,e['pos']), default=None)
        danger = any(dist(pos,m['pos'])<95 for m in threats)
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
        stop_distance = 5.
        if mode=='cube':
            stop_distance = min(2., self.cfg['cubes']['pickup_radius']*.6)
        elif mode=='weapon':
            stop_distance = min(2., self.cfg['loot']['pickup_radius']*.6)
        moving = dist(pos,goal)>stop_distance or danger
        move, recover = self.steer(obs,goal,threats,moving=moving)
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
        candidates = list(enemies)+npc_threats
        if not danger and not retreat:
            candidates += list(obs['crates'])
        candidates = sorted(candidates,key=lambda e: dist(pos,e['pos'])+(25 if e.get('kind')=='mech' else 0))
        target = None
        for e in candidates[:8]:
            ep = e['pos']
            if e.get('kind')=='mech':
                center = ep[2]+self.cfg['mech']['height']*(1+.05*e.get('cubes',0))/2
            elif e.get('kind')=='npc':
                center = ep[2]+self.cfg['npc'].get('height',3.5)/2
            else:
                center = ep[2]+1.5
            point=[ep[0],ep[1],center]
            if dist(pos,ep)>230 or (retreat and e.get('kind') in ('mech','npc') and dist(pos,ep)>45):
                continue
            if not self.world or self.world.clear_shot(muzzle,point):
                target=(e,point); break
        act['aim']=[math.atan2(move[1],move[0])+.5*math.sin(t*.8),0.]
        if retreat and nearest:
            act['aim']=[math.atan2(nearest['pos'][1]-pos[1],nearest['pos'][0]-pos[0]),0.]
        if target:
            e, point=target
            distance=dist(pos,e['pos']); vel=e.get('vel',[0,0,0])
            # Compute a separate solution for each weapon. Only the torso aim is shared.
            # Trigger compatibility is checked against each weapon's own solution.
            plans = {}
            target_radius = (self.cfg['npc'].get('radius',1.8) if e.get('kind')=='npc' else
                             self.cfg['mech']['radius']*(1+.05*e.get('cubes',0)) if e.get('kind')=='mech' else 1.5)
            for i,w in enumerate(me['weapons']):
                if not w:
                    continue
                k=w['kind']; st=w['stats']; ready=w['cooldown']<=.1
                point_for_weapon=point[:]
                valid=True
                priority=0
                shot_distance=math.dist(muzzle,point)
                if k=='gun':
                    r=[point[j]-muzzle[j] for j in range(3)]
                    a=sum(v*v for v in vel)-st['speed']**2
                    b=2*sum(r[j]*vel[j] for j in range(3)); c=sum(v*v for v in r)
                    disc=b*b-4*a*c
                    roots=[v for v in ((-b+math.sqrt(max(0,disc)))/(2*a),(-b-math.sqrt(max(0,disc)))/(2*a))
                           if 0<v<2] if abs(a)>1e-6 and disc>=0 else []
                    lead=min(roots) if roots else 0
                    point_for_weapon=[point[j]+vel[j]*lead for j in range(3)]
                    valid=shot_distance<st['range'] and w.get('ammo',0)>0 and not w.get('reload',0)
                    priority=20 if ready else 5
                elif k=='laser':
                    valid=shot_distance<st['range'] and not retreat and (w.get('charge',0)>0 or ready)
                    priority=50 if w.get('charge',0)>0 else 35
                elif k=='lightning':
                    valid=shot_distance<st['range']-2 and energy>st['energy']+15 and ready
                    priority=45
                elif k=='mortar':
                    # Low-angle solution to a stationary target. Full arc navigation is later work.
                    origin_z=pos[2]+self.cfg['mech']['height']*(1+.05*me['cubes'])+.5
                    dz=point[2]-origin_z
                    speed=st['speed']; gravity=st['gravity']
                    discriminant=speed**4-gravity*(gravity*distance*distance+2*dz*speed*speed)
                    valid=discriminant>=0 and distance>max(25,st.get('splash_radius',7)*2) and ready
                    priority=30
                else:
                    valid=False
                yaw=math.atan2(point_for_weapon[1]-muzzle[1],point_for_weapon[0]-muzzle[0])
                pitch=math.atan2(point_for_weapon[2]-muzzle[2],dist(muzzle,point_for_weapon))
                if k=='mortar' and valid:
                    pitch=math.atan((speed*speed-math.sqrt(discriminant))/(gravity*distance))
                valid=bool(valid and me['stats'].get('min_pitch',-.7)<=pitch<=me['stats'].get('max_pitch',1.3))
                plans[i]={'aim':[yaw,pitch], 'valid':valid, 'priority':priority}
            usable=[i for i,p in plans.items() if p['valid']]
            if usable:
                primary=max(usable,key=lambda i:plans[i]['priority'])
                act['aim']=plans[primary]['aim']
                primary_kind=me['weapons'][primary]['kind']
                for i,w in enumerate(me['weapons']):
                    if not w:
                        continue
                    k=w['kind']; st=w['stats']; plan=plans[i]
                    if not plan['valid']:
                        continue
                    yaw,pitch=plan['aim']
                    yaw_error=abs(angle(yaw-me['aim'][0]))
                    pitch_error=abs(pitch-me['aim'][1])
                    aligned=yaw_error<max(.025,math.atan2(target_radius*.85,max(1,distance))) and pitch_error<.065
                    ready=w['cooldown']<=.1
                    if k=='gun':
                        # Fire alongside a charging laser whenever the actual aim fits the gun too.
                        # Do not fire down a mortar arc or while rotating toward one.
                        next_aim=act['aim']
                        compatible=(primary_kind!='mortar' and
                                    abs(angle(next_aim[0]-yaw))<max(.025,math.atan2(target_radius*.85,max(1,distance))) and
                                    abs(next_aim[1]-pitch)<.065)
                        act['fire'][i]=bool(aligned and ready and compatible)
                    elif k=='lightning':
                        act['fire'][i]=bool(yaw_error<math.radians(st['cone_deg'])*.8)
                    elif k=='laser':
                        charge=w.get('charge',0)
                        act['fire'][i]=bool(i==primary and (charge<st['charge'] or not aligned))
                    elif k=='mortar':
                        act['fire'][i]=bool(i==primary and ready and aligned)
            mode='retreat' if retreat else 'aim'
        for w in me['weapons']:
            if w and w['kind']=='gun' and not w.get('reload',0):
                if w.get('ammo',0)==0 or (target is None and w.get('ammo',0)<w['stats']['mag']*.7):
                    act['reload']=True
        mf.draw.label(f'{mode} hp={me["hp"]:.0f}')
        return act


if __name__ == '__main__':
    mf.run(Bot)

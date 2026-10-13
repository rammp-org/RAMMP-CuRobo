"""The live view of /mebot_sim: a web page that shows the sim and drives it.

A headless Jetson has no screen, so the sim renders on EGL and this streams
the picture to a browser as MJPEG at 20 fps: http://127.0.0.1:8765 unless
host:= and port:= say otherwise. On the page: drag to orbit, shift- or
right-drag to pan, scroll to zoom; send a tool goal to either planner and
watch the plan flown; push the joints by hand; switch the stand-in's
posture; see contacts and their forces. No test depends on it.
"""

import collections
import io
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from PIL import Image

from mebot_sim import sim_arm, stack

FPS = 20
SIZE = (960, 720)
# The camera's start: the arm and the chair from the front, on the arm's side.
HOME = ([-0.10, 0.14, 0.22], 2.85, 141.0, -27.0)  # lookat, distance, azimuth, elevation
SPEEDS = (0.25, 0.5, 1.0)
GOAL_RGBA = np.array([0.1, 0.9, 0.3, 0.9], dtype=np.float32)


class View:
    """The page, its stream and its commands. render() runs in the thread
    that steps the sim, which owns the EGL context; the page's requests
    arrive on server threads, and a plan on a thread of its own.
    `planners` are stack.Planner clients by node name."""

    FPS = FPS

    def __init__(self, sim, planners):
        self.sim = sim
        self.planners = planners
        self.worlds = stack.worlds()
        self.camera = mujoco.MjvCamera()
        self._home()
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.renderer = self.model = None
        self.frame = None
        self.fresh = threading.Condition()
        self.camera_moves = queue.Queue()
        self.lock = threading.Lock()
        self.log = collections.deque(maxlen=12)
        self.goal = None
        self.planning = False
        self.speed = 0.5
        self.servers = []
        self.ready = (0.0, False)  # when the planners were last asked, and their answer

    def _home(self):
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:], self.camera.distance = HOME[0], HOME[1]
        self.camera.azimuth, self.camera.elevation = HOME[2], HOME[3]

    def say(self, message):
        with self.lock:
            self.log.appendleft(time.strftime("%H:%M:%S ") + message)

    # ------------------------------------------------------------ the picture
    def render(self, sim):
        """A frame of `sim` for the stream (in the sim's thread)."""
        if sim.model is not self.model:  # built anew for another posture
            if self.renderer is not None:
                self.renderer.close()
            self.model = sim.model
            self.renderer = mujoco.Renderer(self.model, height=SIZE[1], width=SIZE[0])
        while True:
            try:
                move = self.camera_moves.get_nowait()
            except queue.Empty:
                break
            if move is None:
                self._home()
            else:
                action, dx, dy = move
                mujoco.mjv_moveCamera(
                    self.model, action, dx, dy, self.renderer.scene, self.camera
                )
        self.renderer.update_scene(
            sim.data, camera=self.camera, scene_option=self.option
        )
        scene = self.renderer.scene
        if self.goal is not None and scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.025, 0.0, 0.0]),
                np.asarray(self.goal, dtype=float),
                np.eye(3).flatten(),
                GOAL_RGBA,
            )
            scene.ngeom += 1
        jpeg = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(jpeg, "JPEG", quality=78)
        with self.fresh:
            self.frame = jpeg.getvalue()
            self.fresh.notify_all()

    # -------------------------------------------------------------- commands
    def command(self, c):
        """Carry out one of the page's commands (a decoded JSON object)."""
        kind = c["cmd"]
        if kind == "camera":
            action = {
                "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
                "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
            }[c["mode"]]
            self.camera_moves.put((action, _number(c["dx"]), _number(c["dy"])))
        elif kind == "home":
            self.camera_moves.put(None)
        elif kind == "plan":
            pos = [_number(v) for v in c["pos"]]
            rpy_deg = [_number(v) for v in c["rpy_deg"]]
            if len(pos) != 3 or len(rpy_deg) != 3:
                raise ValueError("a goal is three numbers and three angles")
            planner, world, sees = self.worlds[c["world"]]
            with self.lock:
                busy = self.planning or self.sim.snapshot()["moving"]
                if not busy:
                    self.planning = True
            if busy:
                self.say("busy: wait for the plan or the motion under way")
                return
            self.goal = pos
            threading.Thread(
                target=self._plan,
                args=(planner, world, sees, pos, rpy_deg),
                daemon=True,
            ).start()
        elif kind == "jog":
            joint = int(c["joint"])
            if not 0 <= joint < len(stack.JOINTS):
                raise ValueError("no joint %d" % joint)
            self.sim.post("jog", joint, _number(c["value"]))
        elif kind == "posture":
            name = c["name"]
            if name != "none" and name not in self.sim.assets.postures["postures"]:
                raise ValueError("no posture %r" % name)
            self.sim.post("posture", None if name == "none" else name)
            self.say("seated body: %s" % name.replace("_", " "))
        elif kind == "speed":
            speed = _number(c["value"])
            if speed not in SPEEDS:
                raise ValueError("speed is one of %s" % (SPEEDS,))
            self.speed = speed
        elif kind == "stop":
            self.sim.post("stop", None)
            self.say("stopped")
        elif kind == "reset":
            self.sim.post("reset")
            self.say("the arm is back at its start")
        else:
            raise ValueError("unknown command %r" % kind)

    def _plan(self, planner, world, sees, pos, rpy_deg):
        name = self.planners[planner].name
        try:
            self.say(
                "planning to (%.2f, %.2f, %.2f) on %s (%s)..." % (*pos, name, sees)
            )
            pushed, message = self.planners[planner].set_world(world)
            if not pushed:
                self.say("%s refused the world: %s" % (name, message))
                return
            start = self.sim.snapshot()["q"]
            result = self.planners[planner].plan(start, pos=pos, rpy_deg=rpy_deg)
            if result is None or not result.success:
                why = "goal rejected" if result is None else result.message
                self.say("NO PLAN on %s: %s" % (name, why))
                return
            times, rows = stack.trajectory_arrays(result.trajectory)
            # the plan is at full speed; flying it slower is the client's call
            speed = self.speed
            self.sim.post("fly", sim_arm.Flight(np.divide(times, speed), rows))
            self.say(
                "planned on %s in %.2f s; flying %d points at speed %.2f (%.1f s)"
                % (name, result.planning_time, len(rows), speed, times[-1] / speed)
            )
        except Exception as exc:  # the page shows it; the sim goes on
            self.say("planning failed: %r" % (exc,))
        finally:
            with self.lock:
                self.planning = False

    def state(self):
        """What the page shows: the sim's snapshot and the view's own."""
        now = time.monotonic()
        if now - self.ready[0] > 1.0:
            self.ready = (now, all(p.ready() for p in self.planners.values()))
        m = self.sim.model
        limits = []
        for name in stack.JOINTS:
            joint = m.joint(name).id
            limited = m.jnt_limited[joint]
            limits.append(
                m.jnt_range[joint].tolist() if limited else [-2 * math.pi, 2 * math.pi]
            )
        with self.lock:
            log, planning = list(self.log), self.planning
        return dict(
            self.sim.snapshot(),
            planners="ready" if self.ready[1] else "starting",
            planning=planning,
            speed=self.speed,
            log=log,
            limits=limits,
        )

    # ------------------------------------------------------------ the server
    def serve(self, hosts, port):
        """Serve the page on each of `hosts` at `port`; returns their URLs."""
        urls = []
        for host in hosts:
            server = ThreadingHTTPServer((host, port), _handler(self))
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
            urls.append("http://%s:%d" % server.server_address[:2])
        return urls

    def close(self):
        """Stop serving, open streams included, and free the EGL context (in
        the thread that rendered)."""
        servers, self.servers = self.servers, []
        for server in servers:
            server.shutdown()
            server.server_close()
        with self.fresh:
            self.fresh.notify_all()
        if self.renderer is not None:
            self.renderer.close()


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("not a finite number: %r" % value)
    return value


def _handler(view):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, kind, body):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/state":
                self._send(200, "application/json", json.dumps(view.state()).encode())
            elif self.path == "/stream":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def _stream(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sent = None
            try:
                while view.servers:
                    with view.fresh:
                        view.fresh.wait(timeout=1.0)
                        frame = view.frame
                    if frame is None or frame is sent:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: %d\r\n\r\n" % len(frame)
                    )
                    self.wfile.write(frame + b"\r\n")
                    sent = frame
            except OSError:  # the browser went away
                pass

        def do_POST(self):
            if self.path != "/cmd":
                self._send(404, "text/plain", b"not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                view.command(json.loads(self.rfile.read(length) or b"{}"))
            except (KeyError, TypeError, ValueError) as exc:
                self._send(400, "text/plain", ("bad command: %r" % (exc,)).encode())
                return
            self._send(200, "application/json", b"{}")

    return Handler


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEBot sim</title><style>
:root{--bg:#12161b;--panel:#1b222b;--fg:#e7ecf1;--mut:#97a4b1;--line:#2d3743;--acc:#6fa0ff;--good:#6fd39b;--bad:#ff8a80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px;max-width:1500px;margin:auto}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
#view{width:100%;display:block;background:#000;cursor:grab;user-select:none;touch-action:none}
.panel{background:var(--panel);border:1px solid var(--line);padding:12px;display:flex;flex-direction:column;gap:10px}
h1{font-size:17px;margin:0}h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:6px 0 0}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
input[type=number]{width:74px;background:#0e1217;color:var(--fg);border:1px solid var(--line);padding:5px}
button,select{background:#263140;color:var(--fg);border:1px solid var(--line);padding:6px 9px;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}button.main{background:#27467d}
#badge{font-weight:600;padding:6px 9px;border:1px solid var(--line)}
#badge.ok{color:var(--good)}#badge.hit{color:#12161b;background:var(--bad)}
#log{font:12px/1.4 ui-monospace,monospace;color:var(--mut);white-space:pre-wrap;max-height:230px;overflow:auto}
#log b{color:var(--fg);font-weight:400}.mut{color:var(--mut);font-size:12px}
.j{display:grid;grid-template-columns:18px 1fr 48px;gap:6px;align-items:center;font:12px ui-monospace,monospace}
.j input{width:100%}
</style></head><body><div class="wrap">
<div><img id="view" src="/stream" alt="the MuJoCo sim, live" draggable="false">
<div class="mut">Drag to orbit, shift-drag or right-drag to pan, scroll to zoom. The translucent boxes are what the planner sees, padded: blue the wheelchair, orange the person, grey the floor. The green ball is the goal; red marks are contacts.</div></div>
<div class="panel">
<h1>MEBot sim</h1>
<div class="row"><span id="badge" class="ok">no contact</span><span class="mut" id="stat"></span></div>
<h2>Send the tool to a point</h2>
<div class="mut">Metres from the arm's base: x forward, y left, z up. The person sits at y = +0.27.</div>
<div class="row">x <input id="gx" type="number" step="0.05" value="0.50"> y <input id="gy" type="number" step="0.05" value="-0.40"> z <input id="gz" type="number" step="0.05" value="0.30"></div>
<div class="row">roll <input id="gr" type="number" step="15" value="90"> pitch <input id="gp" type="number" step="15" value="0"> yaw <input id="gw" type="number" step="15" value="90"></div>
<div class="row"><button class="main" onclick="plan('all')">Plan and fly: every layer</button></div>
<div class="row"><button onclick="plan('no_person')">Person layer off</button><button onclick="plan('floor_only')">Floor only</button></div>
<div class="row mut">Try: <button onclick="preset(0.50,-0.40,0.30,90,0,90)">outboard</button><button onclick="preset(0.45,0.20,0.25,90,0,90)">over the lap</button><button onclick="preset(0.35,-0.20,-0.30,180,0,0)">low by the wheel</button></div>
<div class="row"><button onclick="cmd({cmd:'stop'})">Stop</button><button onclick="cmd({cmd:'reset'})">Back to the start</button><button onclick="cmd({cmd:'home'})">Reset the view</button></div>
<h2>Sim</h2>
<div class="row">Body <select id="po" onchange="cmd({cmd:'posture',name:this.value})"><option value="upright_lap">upright, hands on the lap</option><option value="lean_lap">leaning forward</option><option value="forearms_forward">forearms forward</option><option value="none">empty chair</option></select>
Speed <select id="sp" onchange="cmd({cmd:'speed',value:+this.value})"><option>0.25</option><option selected>0.5</option><option>1.0</option></select></div>
<h2>Push the joints by hand</h2><div class="mut">The sliders move the joints directly, with no planner: the arm can hit things.</div>
<div id="joints"></div>
<h2>Log</h2><div id="log"></div>
</div></div><script>
let drag=null,built=false;
function cmd(o){return fetch('/cmd',{method:'POST',body:JSON.stringify(o)})}
function v(id){return +document.getElementById(id).value}
function plan(w){cmd({cmd:'plan',world:w,pos:[v('gx'),v('gy'),v('gz')],rpy_deg:[v('gr'),v('gp'),v('gw')]})}
function preset(x,y,z,r,p,w){gx.value=x;gy.value=y;gz.value=z;gr.value=r;gp.value=p;gw.value=w}
function joints(limits){const J=document.getElementById('joints');limits.forEach((l,i)=>{const d=document.createElement('div');d.className='j';
d.innerHTML=`<span>${i+1}</span><input type="range" id="j${i}" min="${l[0]}" max="${l[1]}" step="0.01"><span id="jv${i}"></span>`;J.appendChild(d);
const s=d.querySelector('input');s.addEventListener('input',()=>{s.dataset.hold=Date.now();cmd({cmd:'jog',joint:i,value:+s.value})})});built=true}
const im=document.getElementById('view');
im.addEventListener('contextmenu',e=>e.preventDefault());
im.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,mode:(e.shiftKey||e.button===2)?'pan':'rotate'};im.setPointerCapture(e.pointerId);e.preventDefault()});
im.addEventListener('pointermove',e=>{if(!drag)return;const h=im.clientHeight;cmd({cmd:'camera',mode:drag.mode,dx:(e.clientX-drag.x)/h,dy:(e.clientY-drag.y)/h});drag.x=e.clientX;drag.y=e.clientY});
im.addEventListener('pointerup',()=>{drag=null});
im.addEventListener('wheel',e=>{e.preventDefault();cmd({cmd:'camera',mode:'zoom',dx:0,dy:-e.deltaY*0.0006})},{passive:false});
const who={standin:'the person',mebot:'the wheelchair',floor:'the floor'};
async function tick(){try{const s=await (await fetch('/state')).json();if(!built)joints(s.limits);
const b=document.getElementById('badge');if(s.now.length){b.className='hit';
b.textContent='CONTACT: '+s.now.map(c=>`${c.link} on ${who[c.group]} (${c.part}) ${c.force_n} N`).join('; ')}
else{b.className='ok';b.textContent='no contact'}
document.getElementById('stat').textContent=`planners ${s.planners}${s.planning?', planning':''}${s.moving?', moving':''} · t ${s.t.toFixed(1)} s · body ${s.posture}`;
document.getElementById('log').innerHTML=s.log.map((l,i)=>{l=l.replace(/&/g,'&amp;').replace(/</g,'&lt;');return i?l:'<b>'+l+'</b>'}).join('\n');
for(let i=0;i<7;i++){const el=document.getElementById('j'+i);if(Date.now()-(+el.dataset.hold||0)>700)el.value=s.ref[i];document.getElementById('jv'+i).textContent=s.q[i].toFixed(2)}
}catch(e){document.getElementById('stat').textContent='sim not reachable'}setTimeout(tick,250)}tick();
</script></body></html>"""

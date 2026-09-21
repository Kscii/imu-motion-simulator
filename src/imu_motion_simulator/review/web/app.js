import * as THREE from './three.module.js';

const manifest = await (await fetch('./manifest.json')).json();
const load = async spec => {
  const buffer = await (await fetch(spec.path)).arrayBuffer();
  const TypedArray = spec.dtype.includes('f4') ? Float32Array
    : spec.dtype.includes('u4') ? Uint32Array : Uint16Array;
  return new TypedArray(buffer);
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111827);
const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, .01, 100);
camera.position.set(4, -6, 2.3);
camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.querySelector('#view').append(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x374151, 2));
const sun = new THREE.DirectionalLight(0xffffff, 2);
sun.position.set(3, -3, 5);
scene.add(sun);
const floor = new THREE.GridHelper(12, 24, 0x64748b, 0x334155);
floor.rotation.x = Math.PI / 2;
scene.add(floor);

const specs = [manifest.model.vertices, manifest.model.faces,
  manifest.model.skin_indices, manifest.model.skin_weights,
  manifest.model.rest_joints, manifest.motion.root_position_m,
  manifest.motion.joint_local_quaternion_wxyz,
  manifest.sensors.specific_force_m_s2,
  manifest.sensors.angular_velocity_rad_s];
const [vertices, faces, indices, weights, joints, rootPos, quats, force, gyro]
  = await Promise.all(specs.map(load));
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(indices, 4));
geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
geometry.setIndex(new THREE.BufferAttribute(faces, 1));
geometry.computeVertexNormals();
const bones = manifest.joint_names.map(() => new THREE.Bone());
for (let i = 0; i < bones.length; i++) {
  const parent = manifest.model.parents[i];
  const x = joints[i * 3], y = joints[i * 3 + 1], z = joints[i * 3 + 2];
  if (parent < 0) bones[i].position.set(x, y, z);
  else {
    bones[parent].add(bones[i]);
    bones[i].position.set(x - joints[parent * 3], y - joints[parent * 3 + 1],
                          z - joints[parent * 3 + 2]);
  }
}
const material = new THREE.MeshStandardMaterial({
  color: 0x60a5fa, roughness: .7, metalness: 0, side: THREE.DoubleSide});
const mesh = new THREE.SkinnedMesh(geometry, material);
mesh.add(bones[0]);
mesh.bind(new THREE.Skeleton(bones));
scene.add(mesh);
for (const mount of manifest.layout.mounts) {
  const marker = new THREE.Mesh(
    new THREE.BoxGeometry(.055, .035, .015),
    new THREE.MeshStandardMaterial({color: 0xfacc15}));
  marker.position.fromArray(mount.position_joint_m);
  const q = mount.quaternion_joint_from_sensor_wxyz;
  marker.quaternion.set(q[1], q[2], q[3], q[0]);
  bones[manifest.joint_names.indexOf(mount.joint)].add(marker);
}

const slider = document.querySelector('#time');
const clock = document.querySelector('#clock');
const status = document.querySelector('#status');
const playButton = document.querySelector('#play');
slider.max = manifest.frame_count - 1;
const dmpl = manifest.dynamic_shape?.source_available === false
  ? 'DMPL unavailable' : 'DMPL source';
status.textContent = `${manifest.frame_count} frames · ${manifest.layout.layout_id}`
  + ` · ${dmpl} · QA ${manifest.qa.passed ? 'pass' : 'review'}`;
document.querySelector('#labels').value = JSON.stringify(
  manifest.selection?.label_candidates ?? [], null, 2);
// Static sample servers have no review API.  Keep decision controls inert
// unless the single-bundle review server explicitly advertises write access.
try {
  const response = await fetch('./api/capabilities', {cache: 'no-store'});
  const capabilities = response.ok ? await response.json() : {};
  if (capabilities.review_write === true) {
    document.querySelectorAll('[data-decision]').forEach(button => button.disabled = false);
    document.querySelector('#review-mode').textContent = '拖动画面旋转，滚轮缩放';
  }
} catch (_) {
  // An offline or static preview remains read-only.
}

let frame = 0;
let playing = false;
let playStartedAt = 0;
let playStartFrame = 0;
function pose(index) {
  frame = Math.max(0, Math.min(manifest.frame_count - 1, index));
  slider.value = frame;
  const sourceFrame = manifest.source_start_frame + frame;
  const sourceSeconds = sourceFrame * manifest.frame_period_s;
  clock.textContent = `source ${sourceFrame} · ${sourceSeconds.toFixed(3)}s · 1×`;
  bones[0].position.fromArray(rootPos, frame * 3);
  for (let joint = 0; joint < bones.length; joint++) {
    const offset = (frame * bones.length + joint) * 4;
    bones[joint].quaternion.set(
      quats[offset + 1], quats[offset + 2], quats[offset + 3], quats[offset]);
  }
}
function pause() {
  playing = false;
  playButton.textContent = '▶';
}
function start(now = performance.now()) {
  if (frame === manifest.frame_count - 1) pose(0);
  playing = true;
  playStartedAt = now;
  playStartFrame = frame;
  playButton.textContent = '❚❚';
}
slider.oninput = () => { pause(); pose(+slider.value); };
playButton.onclick = () => playing ? pause() : start();
document.querySelector('#previous').onclick = () => { pause(); pose(frame - 1); };
document.querySelector('#next').onclick = () => { pause(); pose(frame + 1); };
pose(0);
mesh.updateMatrixWorld(true);

const pivot = new THREE.Vector3(0, 0, 1);
const angles = {x: .9, z: -.4};
let radius = 4.0;
window.__imuReviewViewerVersion = 2;
function cameraPose() {
  camera.position.set(
    pivot.x + radius * Math.cos(angles.z) * Math.cos(angles.x),
    pivot.y + radius * Math.sin(angles.z) * Math.cos(angles.x),
    pivot.z + radius * Math.sin(angles.x));
  camera.lookAt(pivot);
}
cameraPose();
let drag = null;
renderer.domElement.onpointerdown = event => drag = [event.clientX, event.clientY];
renderer.domElement.onpointerup = () => drag = null;
renderer.domElement.onpointermove = event => {
  if (!drag) return;
  angles.z -= (event.clientX - drag[0]) * .006;
  angles.x = Math.max(-.1, Math.min(
    1.45, angles.x + (event.clientY - drag[1]) * .006));
  drag = [event.clientX, event.clientY];
  cameraPose();
};
renderer.domElement.addEventListener('wheel', event => {
  event.preventDefault();
  radius = Math.max(2, Math.min(15, radius * (1 + event.deltaY * .001)));
  cameraPose();
}, {passive: false});

addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey
      || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target?.tagName || '')) return;
  if (parent !== window) return;
  if (event.key === ' ') {
    event.preventDefault();
    playing ? pause() : start();
  } else if (event.key.toLowerCase() === 'r') {
    pause(); pose(0); start();
  }
});

const canvas = document.querySelector('#plot');
const context = canvas.getContext('2d');
function plot() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  const samples = manifest.sensors.specific_force_m_s2.shape[0];
  const mounts = manifest.sensors.specific_force_m_s2.shape[1];
  const sensorFrame = Math.min(samples - 1, Math.round(
    frame * manifest.frame_period_s * manifest.sensor_rate_hz));
  for (const [data, color, scale] of [[force, '#38bdf8', 5], [gyro, '#f472b6', 1]]) {
    context.strokeStyle = color;
    context.beginPath();
    for (let i = 0; i < samples; i++) {
      let magnitude = 0;
      for (let axis = 0; axis < 3; axis++) {
        magnitude += data[(i * mounts) * 3 + axis] ** 2;
      }
      magnitude = Math.sqrt(magnitude);
      const x = i / (samples - 1) * canvas.width;
      const y = canvas.height * .8 - magnitude / scale * canvas.height * .12;
      i ? context.lineTo(x, y) : context.moveTo(x, y);
    }
    context.stroke();
  }
  context.strokeStyle = '#facc15';
  context.beginPath();
  const x = sensorFrame / (samples - 1) * canvas.width;
  context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
}
async function save(decision) {
  try {
    const reviewer = document.querySelector('#reviewer').value;
    const reason = document.querySelector('#reason').value;
    const labels = JSON.parse(document.querySelector('#labels').value);
    const response = await fetch('/api/review', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({decision, reviewer, reason, labels})});
    const result = await response.json();
    status.textContent = result.saved ? `review r${result.revision} saved`
      : `review failed: ${result.error}`;
  } catch (error) {
    status.textContent = `review failed: ${error.message}`;
  }
}
document.querySelectorAll('[data-decision]').forEach(
  button => button.onclick = () => save(button.dataset.decision));
function animate(now) {
  requestAnimationFrame(animate);
  if (playing) {
    const elapsedFrames = Math.floor(
      (now - playStartedAt) / (manifest.frame_period_s * 1000));
    const requested = Math.min(manifest.frame_count - 1,
      playStartFrame + elapsedFrames);
    if (requested !== frame) pose(requested);
    if (requested === manifest.frame_count - 1) pause();
  }
  plot();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

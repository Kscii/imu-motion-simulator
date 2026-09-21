"""Blender-side exact surface renderer. Arguments follow a standalone ``--``."""
import math
from pathlib import Path
import sys

import bpy
import numpy as np


def material(name, color, metallic=0.0, roughness=.65):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1)
    value.metallic = metallic; value.roughness = roughness
    return value


cache_path, output_path, fps, width, height = sys.argv[sys.argv.index('--') + 1:]
fps, width, height = int(fps), int(width), int(height)
with np.load(cache_path, allow_pickle=False) as data:
    vertices = data['vertices']; faces = data['faces']
    sensors = data['sensor_position_m']; sensor_quaternion = data['sensor_quaternion_wxyz']

bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('smplh-surface')
mesh.from_pydata(vertices[0].tolist(), [], faces.tolist()); mesh.update()
body = bpy.data.objects.new('SMPL+H', mesh); bpy.context.collection.objects.link(body)
body.data.materials.append(material('body', (.18, .48, .86)))

bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.object; floor.name = 'floor'
floor.data.materials.append(material('floor', (.12, .15, .20), roughness=.9))

markers = []
for index in range(sensors.shape[1]):
    bpy.ops.mesh.primitive_cube_add(size=.045)
    marker = bpy.context.object; marker.name = f'sensor-{index}'
    marker.scale = (1.4, 1., .35)
    marker.data.materials.append(material(f'sensor-material-{index}', (.95, .65, .05),
                                          metallic=.2, roughness=.35))
    markers.append(marker)

bpy.ops.object.camera_add(location=(4.2, -6.0, 2.4))
camera = bpy.context.object; bpy.context.scene.camera = camera
target = np.mean(vertices[:, :, :2], axis=(0, 1)); center = (float(target[0]), float(target[1]), 1.0)
direction = np.asarray(center) - np.asarray(camera.location)
camera.rotation_euler = __import__('mathutils').Vector(direction).to_track_quat('-Z', 'Y').to_euler()

bpy.ops.object.light_add(type='AREA', location=(2, -3, 6)); bpy.context.object.data.energy = 1200; bpy.context.object.data.shape = 'DISK'; bpy.context.object.data.size = 5
bpy.ops.object.light_add(type='AREA', location=(-3, 1, 3)); bpy.context.object.data.energy = 700; bpy.context.object.data.size = 4

scene = bpy.context.scene; scene.frame_start = 0; scene.frame_end = len(vertices) - 1
scene.render.engine = 'BLENDER_EEVEE_NEXT'; scene.render.resolution_x = width; scene.render.resolution_y = height; scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'FFMPEG'; scene.render.ffmpeg.format = 'MPEG4'; scene.render.ffmpeg.codec = 'H264'; scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'; scene.render.fps = fps; scene.render.filepath = str(Path(output_path).resolve())
scene.world.color = (.025, .03, .045)

def update(frame):
    mesh.vertices.foreach_set('co', vertices[frame].reshape(-1)); mesh.update()
    for index, marker in enumerate(markers):
        marker.location = sensors[frame, index]
        q = sensor_quaternion[frame, index]
        marker.rotation_mode = 'QUATERNION'; marker.rotation_quaternion = (q[0], q[1], q[2], q[3])

def frame_change(scene):
    update(scene.frame_current)

bpy.app.handlers.frame_change_pre.clear(); bpy.app.handlers.frame_change_pre.append(frame_change)
update(0); bpy.ops.wm.save_as_mainfile(filepath=str(Path(output_path).with_suffix('.blend')))
bpy.ops.render.render(animation=True)

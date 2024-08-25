import sys
import os
import time
import argparse
import math
import subprocess
import numpy as np
import json
import mysql.connector as sql

from pathlib import Path
from tqdm import tqdm
from random import choice
from typing import Type
from scipy.interpolate import CubicSpline

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print(' -- ERROR: Set up CoppeliaSim ZeroMQ client as described here: '\
          'https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm')
    sys.exit()


# NOTE: make sure you define the path to where the planners are located on your machine:
# -- paths to the different planners on my PC:
path_to_planners = {}
if os.name == 'nt':
    # -- this is the path to the planners on the Windows side:
    path_to_planners['PDDL4J'] = 'D:/PDDL4J/pddl4j-3.8.3.jar'
    path_to_planners['fast-downward'] = 'C:/Users/david/fast-downward-23.06/fast-downward.py'
else:
    # -- this is the path to the planners on the Ubuntu side:
    path_to_planners['fast-downward'] = '/home/david/fast-downward-22.06/fast-downward.py'

planner_to_use = 'fast-downward'

# NOTE: algorithm is a key in the configs dictionary, while
#   heuristic is an index for the list of configs per algorithm:
configs = {
    'astar': ['astar(lmcut())', 'astar(ff())'],
    'eager': ['eager(lmcut())', 'eager(ff())'],
    'lazy': ['lazy(lmcut())', 'lazy(ff())']
}
algorithm = 'astar'
heuristic = 0


class Interfacer():
    def __init__(
            self,
            scene_file_name: str,
            robot_name: str = "Panda",
            port_number: int = None,
        ):
        self.robot_name = robot_name

        self.client = None
        if port_number:
            # -- specify another port number for a separate client:
            self.client = RemoteAPIClient(host='localhost', port=port_number)
        else:
            # -- just use the default port number:
            self.client = RemoteAPIClient(host='localhost')

        self.sim = self.client.require('sim')
        status = self.load_scenario(scene_file_name)
        if not bool(status):
            print("ERROR: something went wrong with scene load function!")

        # -- loading required modules for simulation:
        self.simIK = self.client.require('simIK')
        self.simOMPL = self.client.require('simOMPL')

        self.objects_in_sim = self.get_sim_objects(self.sim)

        # -- self-contained dictionaries containing object properties:
        self.object_positions, self.object_orientations, self.object_bb_dims = {}, {}, {}

        self.start_pose = self.sim.getObjectPose(self.sim.getObject(f'/{self.robot_name}/target'), -1)

        self.start_time = time.time()

    def get_elapsed_time(self):
        return float(time.time() - self.start_time)

    def sim_print(self, string: str):
        self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f'[UTAMP-driver] : {string}')

    def reset(self):
        self.objects_in_sim = self.get_sim_objects(self.sim)
        self.start_time = time.time()

    def pause(self):
        self.sim.pauseSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_paused:
            time.sleep(0.001)

    def start(self):
        self.sim.startSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_advancing_running:
            time.sleep(0.001)

    def stop(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.001)

    def load_scenario(self, scene_file_name: str) -> int:
        # -- if any simulation is currently running, stop it before closing the scene:
        self.stop(); self.sim.closeScene()

        # -- finally, get the full path to the scene file and load it:
        complete_scene_path = os.path.abspath(scene_file_name)
        self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), '[UTAMP-driver] : Loading scene "' + complete_scene_path + '" into CoppeliaSim...')

        if not bool(self.sim.loadScene(complete_scene_path)):
            return False

        ideal_positions = [
            [-2.10, 0, 1.39],
            [-1.75, 0, 2.5],
        ]
        ideal_orientations = [
            [-math.pi, 1.40, -(math.pi/2.0)],
            [-math.pi, 0.7, -(math.pi/2.0)],
        ]

        # -- set the camera pose to an ideal view of the table-top scene:
        self.sim.setObjectPosition(self.sim.getObject('/DefaultCamera'), -1, ideal_positions[-1])
        self.sim.setObjectOrientation(self.sim.getObject('/DefaultCamera'), -1, ideal_orientations[-1])

        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        # -- make sure that the gripper is opened (i.e., set the signal to value of 1):
        self.sim.setInt32Signal('close_gripper', 0)

        return True

    def get_sim_objects(self, sim) -> list[str]:
        # NOTE: this function should be run at the very beginning of robot execution:
        objects_list = []

        obj_index = 0
        while True:
            obj_handle = sim.getObjects(obj_index, sim.handle_all)
            if obj_handle == -1:
                break

            obj_index += 1

            # -- check if the object has no parent:
            if sim.getObjectParent(obj_handle) != -1:
                continue

            # -- check if the object is non-static and respondable:
            if bool(sim.getObjectInt32Param(obj_handle, sim.shapeintparam_static)) and not bool(sim.getObjectInt32Param(obj_handle, sim.shapeintparam_respondable)):
                continue

            # -- check if the object is not a valid "shape" type object (to exclude anything like the camera, dummy objects, etc.):
            if sim.getObjectType(obj_handle) != sim.object_shape_type:
                continue

            # -- add the name of the object to the list of all objects of interest in the scene:
            obj_name = sim.getObjectAlias(obj_handle)
            if self.robot_name == obj_name:
                continue

            objects_list.append(obj_name)

        return objects_list

    def get_sim_object_params(self, verbose: bool = True) -> tuple[dict,dict,dict]:

        sim = self.sim

        self.object_positions, self.object_orientations, self.object_bb_dims = {}, {}, {}

        # NOTE: if object parameters are being read, use tqdm library if verbose:
        verbose_loop = "tqdm(self.objects_in_sim, desc='  -- Checking object properties in CoppeliaSim scene...')"

        for obj in (eval(verbose_loop) if verbose else self.objects_in_sim):
            # -- get an object's handle:
            obj_handle = sim.getObject(f'/{obj}')

            if obj_handle == -1:
                print("Warning: '" + obj + "' does not exist!")
                # -- if an object that exists in the scene was not found, then it possibly is attached to the robot's gripper:
                obj_handle = sim.getObject(f'/{self.robot_name}_gripper_attachPoint/{obj}')

            if obj_handle == -1:
                print("Warning: '" + obj + "' does not exist!")

                # -- if an object does not exist in the scene, just add it as a null object:
                self.object_positions[obj] = None
                self.object_orientations[obj] = None
                self.object_bb_dims[obj] = None

            else:
                # -- get the position and orientation of an object in the simulation environment:
                self.object_positions[obj] = sim.getObjectPosition(obj_handle, -1)
                self.object_orientations[obj] = sim.getObjectOrientation(obj_handle, -1)

                # -- get coordinates of the bounding box for the object in the simulation environment:
                bb_min_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
                bb_min_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
                bb_min_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_z)
                bb_max_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
                bb_max_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)
                bb_max_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_z)

                self.object_bb_dims[obj] = [bb_min_x, bb_max_x, bb_min_y,
                                            bb_max_y, bb_min_z, bb_max_z]

        return self.object_positions, self.object_orientations, self.object_bb_dims

    def is_collision(
            self,
            obj1: str,
            obj2: str,
        ) -> bool:

        collision = self.sim.checkCollision(
            self.sim.getObject(f'/{obj1}') if not obj1.isnumeric() else obj1,
            self.sim.getObject(f'/{obj2}') if not obj2.isnumeric() else obj2,
        )
        if not isinstance(collision, tuple) or collision[0] == 0:
            return False

        return True

    def is_on_object(
            self,
            obj_above: str,
            obj_below: str,
            check_collision: bool = True,
            verbose: bool = False,
        ) -> bool:

        sim = self.sim

        try:
            obj1_position, obj1_bbox = self.object_positions[obj_above], self.object_bb_dims[obj_above]
        except KeyError:
            # -- some kind of object that is not in the object-level plan (e.g., table spot dummy)
            obj1_position = sim.getObjectPosition(obj_above, -1)
            obj1_bbox = [
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_min_x),
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_max_x),
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_min_y),
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_max_y),
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_min_z),
                sim.getObjectFloatParam(obj_above, sim.objfloatparam_objbbox_max_z),
            ]
            obj_above = sim.getObjectAlias(obj_above)

        try:
            obj2_position, obj2_bbox = self.object_positions[obj_below], self.object_bb_dims[obj_below]
        except KeyError:
            # -- some kind of object that is not in the object-level plan (e.g., table spot dummy)
            obj2_position = sim.getObjectPosition(obj_below, -1)
            obj2_bbox = [
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_min_x),
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_max_x),
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_min_y),
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_max_y),
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_min_z),
                sim.getObjectFloatParam(obj_below, sim.objfloatparam_objbbox_max_z),
            ]
            obj_below = sim.getObjectAlias(obj_below)

        # NOTE: these dimensions are relative to the object, so the position needs to be added:

        # -- get the min-max z-position for obj1 (obj_above)
        #   just to determine on/under positions:
        obj1_min_z = obj1_position[2] + obj1_bbox[4]
        obj1_max_z = obj1_position[2] + obj1_bbox[5]

        # -- get the min-max x-y-z positions for obj2 (obj_below):
        obj2_min_x = obj2_position[0] + obj2_bbox[0]
        obj2_max_x = obj2_position[0] + obj2_bbox[1]
        obj2_min_y = obj2_position[1] + obj2_bbox[2]
        obj2_max_y = obj2_position[1] + obj2_bbox[3]
        obj2_min_z = obj2_position[2] + obj2_bbox[4]
        obj2_max_z = obj2_position[2] + obj2_bbox[5]

        # -- then, we need to see if there is any object that lies within the boundary
        #       of the grid-space and/or object below it:
        is_within_bb_x, is_within_bb_y = False, False
        if obj1_position[0] < obj2_max_x and obj1_position[0] >= obj2_min_x:
            # -- this object lies within the bounding box in the x-axis:
            is_within_bb_x = True

        if obj1_position[1] < obj2_max_y and obj1_position[1] >= obj2_min_y:
            # -- this object lies within the bounding box in the y-axis:
            is_within_bb_y = True

        is_top_bb_z = True
        if obj1_position[2] <= obj2_position[2]:
            # -- if obj1 is actually below obj2, then just continue:
            is_top_bb_z = False

        elif (obj1_min_z - obj2_max_z) > 0.001:
            is_top_bb_z = False

        is_touching = True
        if check_collision:
            # -- checking if there is any contact between the object pair:
            is_touching = self.is_collision(obj_above, obj_below)

        if verbose:
            print(f"{obj_above} is within x-bb {obj_below}? -> {is_within_bb_x}")
            print(f"{obj_above} is within y-bb {obj_below}? -> {is_within_bb_y}")
            print(f"{obj_above} is on top of {obj_below}? -> {is_top_bb_z}")
            print(f"{obj_above} is touching {obj_below}? -> {is_touching}")

        return bool(is_within_bb_x and is_within_bb_y and is_top_bb_z and is_touching)

    def take_snapshot(self, file_name: str, render_mode: str = 'opengl'):
        # NOTE: a lot of this code was pulled directly from the "Screenshot tool" provided by CoppeliaSim:
        sim = self.sim
        cam = sim.getObject('/DefaultCamera')
        pose = sim.getObjectPose(cam)

        config = {
            'res': [1700, 1200],
            'renderMode': render_mode,
            'povray': {
                'available': False,
                'focalBlur': False,
                'focalDistance': 2,
                'aperture': 0.05,
                'blurSamples': 10,
            },
            'nearClipping': 0.01,
            'farClipping': 250,
            'viewAngle': 60 * math.pi / 180
        }

        visSens = sim.createVisionSensor(
            3,
            [config['res'][0], config['res'][1], 0, 0],
            [config['nearClipping'], config['farClipping'], config['viewAngle'], 0.1, 0.1, 0.1, 0, 0, 0, 0, 0]
        )
        sim.setObjectPose(visSens, pose)
        sim.setObjectInt32Param(
            visSens,
            sim.visionintparam_pov_focal_blur,
            config['povray']['focalBlur'] and 1 or 0
        )
        sim.setObjectFloatParam(
            visSens,
            sim.visionfloatparam_pov_blur_distance,
            config['povray']['focalDistance']
        )
        sim.setObjectFloatParam(visSens, sim.visionfloatparam_pov_aperture, config['povray']['aperture'])
        sim.setObjectInt32Param(visSens, sim.visionintparam_pov_blur_sampled, config['povray']['blurSamples'])
        if config['renderMode'] == 'opengl':
            sim.setObjectInt32Param(visSens, sim.visionintparam_render_mode, 0)
        elif config['renderMode'] == 'opengl3':
            sim.setObjectInt32Param(visSens, sim.visionintparam_render_mode, 7)

        savedVisibilityMask = sim.getObjectInt32Param(cam, sim.objintparam_visibility_layer)
        sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0)
        newAttr = sim.displayattribute_renderpass + sim.displayattribute_forvisionsensor + sim.displayattribute_ignorerenderableflag
        sim.setObjectInt32Param(visSens, sim.visionintparam_rendering_attributes, newAttr)
        sim.handleVisionSensor(visSens)
        sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, savedVisibilityMask)

        image, res = sim.getVisionSensorImg(visSens, 0, 0, [0, 0], [0, 0])
        sim.removeObjects([visSens])
        sim.saveImage(image, res, 0, file_name, -1)


    def perform_sensing(
            self,
            method: int = 1,
            check_collision: bool = False,
            verbose: bool = False,
        ) -> dict:
        """"
        This function is used for all perception. There are 3 variations of sensing:
        1. **PDDL-based sensing** (returns a dictionary of in-on-under relations);
        2. **UTAMP sensing** (returns a string giving object poses and dimensions);
        3. **LLM sensing** (returns a textual description of method #1)

        :param method: 1 (default: PDDL-based sensing for in/on/under relations) | 2 (UTAMP sensing) | 3 (LLM sensing for state as sentences)
        :param check_collision: checks whether objects are "colliding" (i.e., touching) in sim (default: False -- due to inconsistencies)
        """

        sim = self.sim
        goal_objects = self.objects_in_sim

        # -- read the current state of the world:
        self.get_sim_object_params(verbose)

        def sensing_for_utamp():
            str_objects = []
            robot_handle = sim.getObject(f'/{self.robot_name}')

            for x in [f'/{self.robot_name}', f'/{self.robot_name}/target', '/worksurface']:
                obj_handle = sim.getObject(x)
                obj_position, obj_orientation = sim.getObjectPosition(obj_handle, robot_handle), sim.getObjectOrientation(obj_handle, robot_handle)

                # -- getting bounding box dimensions of objects in the scene:
                if 'target' in x.lower():
                    # NOTE: the motion planning should consider the dimensions of the robot's gripper:
                    obj_handle = sim.getObject(f'/{self.robot_name}_gripper')
                elif self.robot_name in x:
                    obj_handle = sim.getObject(f'/{self.robot_name}_link0_visual')

                obj_bb_min_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
                obj_bb_max_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)

                if 'target' in x.lower():
                    # NOTE: the motion planning only considers the height of each finger in the Panda's gripper:
                    obj_handle = sim.getObject(f'/{self.robot_name}_leftfinger_visible')

                obj_bb_min_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
                obj_bb_max_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
                obj_bb_min_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_z)
                obj_bb_max_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_z)

                # NOTE: forcing the robot's bounding box to 0:
                if x == f'/{self.robot_name}':
                    obj_bb_max_x = obj_bb_min_x = 0.0
                    obj_bb_max_y = obj_bb_min_y = 0.0
                    obj_bb_max_z = obj_bb_min_z = 0.0

                str_objects.append(
                    f'{"hand" if "target" in x.lower() else "robot" if self.robot_name in x else "worksurface"}:'\
                    f'x,{obj_position[0]}:'\
                    f'y,{obj_position[1]}:'\
                    f'z,{obj_position[2]}:'\
                    f'roll,{obj_orientation[0]}:'\
                    f'pitch,{obj_orientation[1]}:'\
                    f'yaw,{obj_orientation[2]}:'\
                    f'dx,{obj_bb_max_x-obj_bb_min_x}:'\
                    f'dy,{obj_bb_max_y-obj_bb_min_y}:'\
                    f'dz,{obj_bb_max_z-obj_bb_min_z};'
                )

            obj_index = 0
            while True:
                obj_handle = sim.getObjects(obj_index, sim.handle_all)
                if obj_handle == -1:
                    break

                obj_name = sim.getObjectAlias(obj_handle)
                if obj_name in goal_objects:
                    # print(obj_name)
                    # -- get position and orientation of objects with respect to robot:
                    obj_position = sim.getObjectPosition(obj_handle, robot_handle)
                    obj_orientation = sim.getObjectOrientation(obj_handle, robot_handle)

                    obj_bb_min_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
                    obj_bb_min_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
                    obj_bb_min_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_z)

                    obj_bb_max_x = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
                    obj_bb_max_y = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)
                    obj_bb_max_z = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_z)

                    str_objects.append(
                        f'{obj_name}:'\
                        f'x,{obj_position[0]}:'\
                        f'y,{obj_position[1]}:'\
                        f'z,{obj_position[2]}:'\
                        f'roll,{obj_orientation[0]}:'\
                        f'pitch,{obj_orientation[1]}:'\
                        f'yaw,{obj_orientation[2]}:'\
                        f'dx,{obj_bb_max_x-obj_bb_min_x}:'\
                        f'dy,{obj_bb_max_y-obj_bb_min_y}:'\
                        f'dz,{obj_bb_max_z-obj_bb_min_z};'
                    )

                obj_index += 1

            return "".join(str_objects)
        #enddef

        def sensing_for_pddl():

            in_object_layout, on_object_layout, under_object_layout = {}, {}, {}

            # -- by default, all objects have "air" on and under them:
            for obj in goal_objects:
                on_object_layout[obj] = 'air'
                under_object_layout[obj] = 'table'

            sim.addLog(sim.getInt32Param(sim.intparam_verbosity), '[UTAMP-driver] : Checking geometric configuration of objects...')

            # -- we will only request the positions and bounding box details only once:
            object_positions, object_orientations = self.object_positions, self.object_positions

            # NOTE: we will also check from perception whether the hand is empty or not:
            gripper_sensor = sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachProxSensor')
            gripper_attachPoint = sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachPoint')

            is_grasping = False

            for obj in goal_objects:
                obj_handle = sim.getObject(f'/{obj}')

                # -- check if an object is within the range of the end-effector's proximity sensor:
                is_in_hand, _, _, _, _ = sim.checkProximitySensor(gripper_sensor, obj_handle)

                # -- check if there is an object "attached" to the gripper:
                obj_in_hand = self.sim.getObjectChild(gripper_attachPoint, 0)

                if bool(is_in_hand) and obj_in_hand > -1:
                    is_grasping = True

                    # -- checking to see if the hand is truly *on top* of the object:
                    obj_position, obj_orientation = self.object_positions[obj_in_hand], self.object_orientations[obj_in_hand]
                    gripper_position = sim.getObjectPosition(sim.getObject(f'/{self.robot_name}/target'), -1)
                    gripper_orientation =  sim.getObjectOrientation(sim.getObject(f'/{self.robot_name}/target'), -1)

                    is_similar_position = True
                    for x in range(len(obj_position)):
                        if abs(obj_position[x] - gripper_position[x]) > 0.1:
                            is_similar_position = False

                    is_similar_orientation = True
                    for x in range(len(obj_orientation)):
                        if abs(math.cos(obj_orientation[x] - gripper_orientation[x])) < 0.85:
                            is_similar_orientation = False

                    if is_similar_position and is_similar_orientation:
                        # -- we can now say that the hand is on top of the object:
                        on_object_layout[obj_in_hand] = 'hand'

                    under_object_layout['hand'] = obj_in_hand
                    in_object_layout['hand'] = obj_in_hand

            if not is_grasping:
                # -- if there's nothing in the hand, just make it contain "air":
                under_object_layout['hand'] = 'air'
                in_object_layout['hand'] = 'air'

            # --  we can check if there are any empty spots on the table to say that there's "air" on it:
            if bool(self.find_empty_spots()):
                on_object_layout['table'] = 'air'

            # -- since we have already pre-loaded all the positions and coordinates of the bounding boxes for each object,
            #       we can save some time in determining the current geometric layout of the environment.
            for obj1 in goal_objects:

                # -- get the position of an object in the simulation environment:
                obj1_position = object_positions[obj1]
                if obj1_position is None:
                    continue

                # -- first, let's try to see if the object_1 is inside of a specific grid-space or cell:
                for obj2 in goal_objects:

                    obj2_position = object_positions[obj2]
                    if obj2_position is None:
                        continue

                    if obj1 == obj2:
                        continue

                    if self.is_on_object(obj1, obj2, check_collision=check_collision):
                        if obj2 in on_object_layout:
                            # -- this means that there is also another object that was found to be on top of this object:
                            if on_object_layout[obj2] in object_positions:
                                existing_obj_position = object_positions[on_object_layout[obj2]]
                            else:
                                existing_obj_position = [sys.maxsize, sys.maxsize, sys.maxsize]

                            # -- we shall check if this other object is lower than the present obj2 so that we can consider it as the object on top:
                            if existing_obj_position[2] > obj1_position[2]:
                                on_object_layout[obj2] = obj1

                        else:
                            on_object_layout[obj2] = obj1

                        # -- checking to see if an object is rotated (about x- or y-axis):
                        obj1_orientation = [x / math.pi *
                                            180.0 for x in object_orientations[obj1]]
                        if abs(obj1_orientation[0]) >= 179 or abs(obj1_orientation[1]) >= 179:
                            # -- we will say that the obj2 is "on" obj1, since obj1 is upside-down:
                            on_object_layout[obj1] = obj2

                            # -- if something is on top of the upside-down object, then the object is "under" it:
                            if obj2 in under_object_layout:
                                under_object_layout[obj2] = obj1
                        else:
                            under_object_layout[obj1] = obj2

                        if obj2 in ['mixing_bowl']:
                            if obj2 not in in_object_layout or in_object_layout[obj2] == 'air':
                                in_object_layout[obj2] = [obj1]
                            else:
                                in_object_layout[obj2].append(obj1)

                        #endif
                    #endif
                #endfor
            #endfor

            if in_object_layout:
                return {'on': on_object_layout, 'under': under_object_layout, 'in': in_object_layout}

            return {'on': on_object_layout, 'under': under_object_layout}
        #enddef

        def sensing_for_text():
            # -- this sub-function will read the object-centered state dict and transform into sentences:
            state, state_ments = sensing_for_pddl(), []

            for rel in state:
                for obj1 in state[rel]:
                    obj2 = state[rel][obj1]
                    state_ments.append(f"- {obj2} is {rel} {obj1}")

                    # NOTE: when it comes to the "under" relations, we also need to put the opposite "on" relation:
                    if rel == 'under':
                        state_ments.append(f"- {obj1} is on {obj2}")

            return "\n".join(list(set(state_ments)))
        #enddef

        if method == 2: return sensing_for_utamp()
        if method == 3: return sensing_for_text()

        return sensing_for_pddl()

    def find_empty_spots(self) -> list[str]:
        # NOTE: the purpose of this function is to find empty places on the table for place actions:

        # -- get the current state and parameters of objects in the scene:
        _ = self.get_sim_object_params()

        table, empty_spots = [], []

        while True:
            try:
                dummy = self.sim.getObject('/Dummy', {'index': len(table)})
            except Exception:
                break

            is_empty = True
            for obj in self.objects_in_sim:
                if self.is_on_object(
                    obj_above=obj,
                    obj_below=dummy,
                    check_collision=False,
                    verbose=False,
                ):
                    is_empty = False
                    break

            if is_empty:
                # print(f'/Dummy[{len(table)}] has nothing on it')
                empty_spots.append({
                    'index': len(table),
                    'handle': dummy
                })

            table.append(dummy)


        return empty_spots

    def find_pose_for_ompl(self, target_object: str) -> list[float]:

        robot = self.sim.getObject(f"/{self.robot_name}")
        target = self.sim.getObject(f"/{self.robot_name}/target")

        index = 0
        # -- check if the target object refers to the table, as we will have to find an empty spot:
        if target_object in ["table", "worksurface"]:
            empty_spot = choice(self.find_empty_spots())
            index, target_object = empty_spot['index'], self.sim.getObjectAlias(empty_spot['handle'])
            self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")
            print(f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")

        goal = self.sim.getObject(f"/{target_object}", {"index": index})

        # -- Find a collision-free config that matches a specific pose:
        goal_pose = self.sim.getObjectPose(goal, robot)

        # -- determine the height based on whether there is an object in hand or not:
        obj_in_hand = self.sim.getObjectChild(self.sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachPoint'), 0)
        if obj_in_hand != -1:
            # -- first, we find a spot that sits RIGHT ON TOP of the surface...
            goal_pose[2] += self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_z)
            # ... then we will find a spot that considers the height of the object:
            goal_pose[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z) * 1.75
        else:
            # -- find a spot that lies in a very safe spot above the object:
            goal_pose[2] += self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_z) * 2

        # -- keep the same orientation of the hand for pick-from-top actions:
        gripper_pose = self.sim.getObjectPose(target, robot)
        goal_pose[3:] = gripper_pose[3:]

        return goal_pose

    def path_planning(
            self,
            target_object: str,
            goal_pose: list[float],
            algorithm: int,
            num_ompl_attempts: int,
            max_compute: int,
            max_simplify: int,
        ) -> bool:

        # -- formatting the string name for printing a cool message:
        if target_object:
            self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f'[OMPLement]: finding a plan to object "{target_object}"...')

        # -- create a dummy object that will represent the target goal:
        target_goal = self.sim.createDummy(0.025)
        self.sim.setObjectPose(target_goal, goal_pose, self.sim.getObject(f'/{self.robot_name}'))
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_ambient_diffuse, [0.0, 1.0, 0.0])
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_emission, [0.6, 0.6, 0.6])
        self.sim.setObjectAlias(target_goal, 'OMPL_target')

        # -- we sleep for a bit so we can see this object appear in the sim:
        time.sleep(0.0001)

        ompl_script = self.sim.getScript(self.sim.scripttype_simulation, self.sim.getObject('/OMPLement'))
        path = self.sim.callScriptFunction(
            'path_planning',
            ompl_script,
            {
                "robot": self.robot_name,
                "goal": target_goal,
                "algorithm": algorithm,
                "num_attempts": num_ompl_attempts,
                "max_compute": max_compute,
                "max_simplify": max_simplify,
            },
        )

        if path:
            self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f'[OMPLement]: plan found!')

            # -- we need to disable the IK following done by the "target" dummy of the robot:
            target = self.sim.getObject(f'/{self.robot_name}')
            self.sim.setModelProperty(target, self.sim.modelproperty_scripts_inactive)

            self.start()

            # -- with the computed path, we will gradually change the configuration of the robot:
            for P in path:
                self.sim.callScriptFunction('setConfig', ompl_script, P)
                time.sleep(0.0001)

            # -- we need to re-enable the IK following done by the "target" dummy of the robot:
            self.sim.setObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
            self.sim.setObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
            self.sim.setModelProperty(target, 0)

        else:
            self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f'[OMPLement]: plan not found!')

        # -- remove the OMPL target object:
        self.sim.removeObjects([self.sim.getObject('/OMPL_target')])

        return bool(path)

    def execute(
            self,
            target_object: str,
            gripper_action: int,
            algorithm: str = 'RRTstar',
            num_ompl_attempts: int = 20,
            max_compute: int = 5,
            max_simplify: int = 5,
        ) -> bool:

        try:
            algorithm = eval(f'self.simOMPL.Algorithm.{algorithm}')
        except AttributeError:
            print(f'WARNING: path planning algorithm "{algorithm}" does not exist!')
            algorithm = eval('self.simOMPL.Algorithm.RRTstar')

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        for _ in range(3 if target_object in ['table', 'worksurface'] else 1):
            # -- we will be generous and allow three attempts at finding an empty place on the table:
            goal_pose = self.find_pose_for_ompl(target_object)

            success = self.path_planning(target_object, goal_pose, algorithm, num_ompl_attempts, max_compute, max_simplify)

            if success: break

        # -- if for whatever reason the system failed, then we just
        if not success: return False

        # -- now that we've done obstacle avoidance to get to a good position to pick/place an object,
        #   we will now perform the rest of the action.

        # NOTE: gripper_action == 1 :- pick, gripper_action == 0 :- place
        # -- use the gripper actions to decide whether to open or close gripper (-1 and 1 respectively, otherwise 0 -- no change):
        if gripper_action == 1:
            # -- this means we want to pick up an object:
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            # -- we want to position the gripper on the upper part of the object such that the proximity sensor detects an object:
            end = self.sim.getObjectPosition(self.sim.getObject(f'/{target_object}'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            end[2] += 0.5 * self.sim.getObjectFloatParam(self.sim.getObject(f'/{target_object}'), self.sim.objfloatparam_objbbox_max_z)

            traj_move_to_obj = self.generate_trajectory(
                {'time': [0, 1], 'trajectory': [start, end]},
                ntraj=25
            )

            # -- execute the trajectory and open the gripper:
            self.execute_trajectory(traj_move_to_obj)

        self.end_effector(close=gripper_action)

        time.sleep(0.1)

        obj_in_hand = self.sim.getObjectChild(self.sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachPoint'), 0)
        if gripper_action == 1:
            if obj_in_hand > -1:
                # -- this means that we want to move the gripper up to remove the object from the top of the below object's surface:
                start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
                # -- we want to move up by half the height of the object:
                end = list(start)
                end[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z)

                traj_move_up = self.generate_trajectory(
                    {'time': [0, 1], 'trajectory': [start, end]},
                    ntraj=25
                )

                # -- execute the trajectory but do not change the state of the gripper:
                self.execute_trajectory(traj_move_up)

                if target_object in self.sim.getObjectAlias(obj_in_hand):
                # -- this means that we have successfully grasped the intended object:
                    return True
        else:
            if obj_in_hand == -1:
                # -- this means that we have successfully placed the object (or at least there is nothing in the gripper):
                return True

        return False

    def execute_utamp(
            self,
            goal_poses: list[float],
            gripper_action: int,
            algorithm: str = 'RRTstar',
            num_ompl_attempts: int = 5,
            max_compute: int = 5,
            max_simplify: int = 5,
        ) -> bool:

        # -- extract the pre-grasping/placing and post-grasping/placing poses:
        pre_grasp, post_grasp = goal_poses[0], goal_poses[1]

        # -- we need to convert coordinates from Euler angles to quaternions:
        pre_grasp = self.sim.buildPose(pre_grasp[:3], pre_grasp[3:])
        post_grasp = self.sim.buildPose(post_grasp[:3], post_grasp[3:])

        try:
            algorithm = eval(f'self.simOMPL.Algorithm.{algorithm}')
        except AttributeError:
            print(f'WARNING: path planning algorithm "{algorithm}" does not exist!')
            algorithm = eval('self.simOMPL.Algorithm.RRTstar')

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        if gripper_action == 0:
            # -- if we just want to have an object placed, then we use the post-grasp given by UTAMP:
            success = self.path_planning(None, post_grasp, algorithm, num_ompl_attempts, max_compute, max_simplify)
        else:
            # -- if we just want to pick an object, then we use the pre-grasp given by UTAMP:
            success = self.path_planning(None, pre_grasp, algorithm, num_ompl_attempts, max_compute, max_simplify)

        # -- if for whatever reason the system failed, then we just
        if not success: return False

        # -- now that we've done obstacle avoidance to get to a good position to pick/place an object,
        #   we will now perform the rest of the action.

        if gripper_action == 1:
            # -- this means that UTAMP will tell us to move down to either release the object (0) or grasp the object (1):
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            end = post_grasp[:3] + list(start[3:])

            traj_move_down = self.generate_trajectory(
                {'time': [0, 1], 'trajectory': [start, end]},
                ntraj=25
            )

            self.execute_trajectory(traj_move_down)

        self.end_effector(gripper_action)

        time.sleep(0.01)

        obj_in_hand = self.sim.getObjectChild(self.sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachPoint'), 0)
        if obj_in_hand > -1:
            # -- this means that we want to move the gripper up to remove the object from the top of the below object's surface:
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            # -- we want to move up by half the height of the object:
            end = list(start)
            end[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z)

            traj_move_up = self.generate_trajectory(
                {'time': [0, 1], 'trajectory': [start, end]},
                ntraj=25
            )

            # -- execute the trajectory but do not change the state of the gripper:
            self.execute_trajectory(traj_move_up)

        return True

    def end_effector(self, close: int = -1):
        message = "maintaining gripper state..."
        if close == 1:
            message = 'closing gripper...'
        elif close == 0:
            message = 'opening gripper...'

        self.sim.addLog(self.sim.getInt32Param(self.sim.intparam_verbosity), f'[OMPLement]: {message}')
        if close != -1:
            self.sim.setInt32Signal('close_gripper', close)

    def generate_trajectory(
            self,
            keypoints: list[float],
            ntraj: int = 150,
        ):

        # -- use cubic spline interpolation:
        cs = CubicSpline(sorted(list(set(keypoints['time']))), keypoints['trajectory'])
        xs = np.arange(0, 1+1/ntraj, 1/ntraj)

        return cs(xs)

    def execute_trajectory(
            self,
            traj: list[float],
            gripper_action: int = -1,
        ):
        robot_handle, gripper_handle = self.sim.getObject(f'/{self.robot_name}'), self.sim.getObject(f'/{self.robot_name}/target')

        self.start()
        for T in traj:
            # -- set the gripper's position and orientation according to the recorded trajectory:
            self.sim.setObjectPosition(gripper_handle, robot_handle, list(T[0:3]))
            self.sim.setObjectOrientation(gripper_handle, robot_handle, list(T[3:6]))

            # -- pause for dramatic effect (just kidding, only for ensuring instructions are sent correctly)
            time.sleep(0.001)
        #endfor

        # -- use the gripper actions to decide whether to open or close gripper (0 and 1 respectively, otherwise -1 -- no change):
        if gripper_action != -1: self.end_effector(close=gripper_action)


class UTAMP:
    def __init__(
            self,
            config_fpath: str = 'utamp.config.json',
        ):
        try:
            self.cnx = sql.connect(**json.load(open(config_fpath, 'r')))
        except FileNotFoundError:
            print(f'[UTAMP] : Error with config file for UTAMP connection (no file found at "{config_fpath}")!')
            sys.exit()
        self.cnx.autocommit = True
        self.cursor = self.cnx.cursor(buffered=True)

    def parse_server_output(
            self,
            str_actions: str,
        ):

        # -- split the string based on semi-colons:
        str_actions = str_actions.split(':')
        time, traj = [], []
        gripper_action = None
        for t in range(3):
            coordinates = []
            for S in range(len(str_actions)):
                if f't_{t+1}' in str_actions[S]:
                    time.append( eval(str(str_actions[S]).split(',')[1]) )
                elif f'_{t+1}' in str_actions[S]:
                    coordinates.append( eval(str(str_actions[S]).split(',')[1]) )
                elif not gripper_action and 'gripper' in str(str_actions[S]):
                    gripper_action = eval(str_actions[S].split(',')[1][:-1])

            if coordinates:
                traj.append(coordinates)

        keypoints = {
            'time': time,
            'trajectory': traj
        }

        return keypoints, gripper_action

    def plan_and_execute(
            self,
            sim_interfacer: Type[Interfacer],
            goals_for_utamp: list = [],
            verbose: bool = False,
            pick_from_top: bool = True,
            num_ompl_attempts: int = 3,
            algorithm: str = 'RRTstar',
            max_compute: int = 5,
            max_simplify: int = 5,
        ) -> bool:

        cnx, cursor = self.cnx, self.cursor

        # -- parse through the goal predicates and format it as a string needed for UTAMP:
        obj_to_goals = {}

        for P in goals_for_utamp:
            # -- we want to get strings in the required format for the UTAMP system:
            pred_parts = P[1:-1].split(" ")
            if len(pred_parts) < 3:
                continue

            if bool(set(pred_parts[1:]).intersection(set(["hand", "table"]))):
                # -- ignore any references to the hand or the table -- this is being handled by UTAMP.
                continue

            if pred_parts[1] not in obj_to_goals:
                obj_to_goals[pred_parts[1]] = []

            # -- format all the predicates in the string format indicated by Alejandro:
            obj_to_goals[pred_parts[1]].append(f'{pred_parts[0]},{pred_parts[2]}')

        # -- post-process the goal states to identify any objects that should have "air" on it:
        for O in obj_to_goals:
            on_found = False
            for P in obj_to_goals[O]:
                if 'on,' in P:
                    on_found = True

            if not on_found:
                obj_to_goals[O].append('on,air')

        utamp_goal_string = str()
        for O in obj_to_goals:
            utamp_goal_string += f'{O}:{":".join(obj_to_goals[O])};'

        # -- this adds constraint to only pick objects from the top:
        if pick_from_top:
            utamp_goal_string += "constraints:placefromabove,1;"

        vars_to_insert = {
            'user_id': 51,
            'goal': utamp_goal_string,
            'objects': 'n/a',
            'actions': 'n/a',
            'status': 0,
        }

        sql_command = f'UPDATE utamp_data SET status = -1 WHERE user_id = {vars_to_insert["user_id"]}'
        cursor.execute(sql_command, vars_to_insert)
        cnx.commit()

        sql_command = f'UPDATE utamp_data SET status = 0, goal = "{vars_to_insert["goal"]}" WHERE user_id = {vars_to_insert["user_id"]}'
        cursor.execute(sql_command, vars_to_insert)
        cnx.commit()

        sql_command = ("SELECT * from utamp_data")
        cursor.execute(sql_command)

        if verbose:
            for user_id, goal, objects, actions, _, status, msg in cursor:
                print('user', user_id)
                print(goal)
                print(objects)
                print(actions)
                print(status)
                print(msg)
                print()

        # NOTE: status numbers:
        # Status	Meaning
        # -1	    Waiting for goal (updated by Server).
        #  0	    Goal/Sensing/Execution request concluded (updated by Client).
        #  1	    Execution request (updated by Server).
        #  2	    Execution in progress (updated by Client).
        #  3	    Execution failed (updated by Client).
        # 10	    Sensing request (updated by Server).
        # 20	    Sensing in progress (updated by Client).
        # 30	    Sensing failed (updated by Client).
        status = 0

        print(f"\n{'*' * 10} UTAMP CLIENT/SERVER INTERACTION {'*' * 10}\n")

        print(f" -- goals sent to utamp:\t{utamp_goal_string}\n")

        print('UTAMP interactions:')

        last_status = None

        while status > -1:
            sql_command = ("SELECT status, msg from utamp_data where user_id = 51")
            cursor.execute(sql_command)
            status, msg = cursor.fetchone()

            sql_command = ("SELECT status from utamp_data where user_id = 51")

            if last_status != (status, msg):
                print(f'status {status}:\t{msg}')
                last_status = (status, msg)

            if status == 1:
                # -- execution request:
                status = 2
                sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
                cursor.execute(sql_command)
                # print('performing execution...')

                sql_command = ("SELECT actions from utamp_data where user_id = 51")
                cursor.execute(sql_command)
                str_actions = cursor.fetchone()[0]

                # -- we will parse the output given by the UTAMP server...
                keypoints, end_effector = self.parse_server_output(str_actions)
                # ... and then we will perform path planning with OMPL:
                sim_interfacer.execute_utamp(
                    keypoints['trajectory'][1:],
                    end_effector,
                    algorithm,
                    num_ompl_attempts,
                    max_compute,
                    max_simplify,
                )

                status = 0
                sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
                cursor.execute(sql_command)

            if status == 10:
                # -- sensing request:
                status = 20

            if status == 20:
                sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
                cursor.execute(sql_command)

                str_objects = sim_interfacer.perform_sensing(method=2, verbose=verbose)
                if verbose:
                    print(str_objects)

                for obj in sim_interfacer.objects_in_sim:
                    if obj not in str_objects:
                        print(obj)

                status = 0
                sql_command = f'UPDATE utamp_data SET objects = "{str_objects}", status = {status} WHERE user_id = 51'
                cursor.execute(sql_command)

        if status == -1:
            return True

        return False

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    # YZX
    # qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    # qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    # qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # XYZ
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) - np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


def create_domain_file(
        objects_in_sim: list,
        micro_fpath: str,
        template_domain_fpath: str,
        domain_name: str = 'olp-2024',
        typing: list = None,
    ) -> str:

    # -- we are going to inject new objects into the
    micro_domain_fpath = os.path.join(micro_fpath, template_domain_fpath)

    with open(micro_domain_fpath, 'w') as pddl_file:

        # -- the template file contains all of the default skills for our robot:
        with open(template_domain_fpath) as f:
            domain_file_lines = f.readlines()
            for L in domain_file_lines:
                if '(:constants' in L:
                    pddl_file.write(f'{L}')
                    for O in objects_in_sim:
                        # -- check if an object in the scene has been classified as special PDDL type:
                        object_type = 'object'
                        for T in typing:
                            if O in typing[T]:
                                object_type = T

                        pddl_file.write(f'\t\t{O} - {object_type}\n')

                elif '(domain' in L:
                    pddl_file.write(f'(define (domain {domain_name})')

                else:
                    pddl_file.write(f'{L}')

    return micro_domain_fpath


def create_problem_file(
        preconditions: list,
        effects: list,
        micro_fpath: str,
        action_name: str,
        state: dict,
        domain_name: str = 'olp-2024',
    ) -> str:

    on_object_layout = state['on']
    in_object_layout = state['in']
    under_object_layout = state['under']

    micro_problem_fpath = os.path.join(micro_fpath, f"{action_name}.pddl",)

    with open(micro_problem_fpath, 'w') as pddl_file:

        pddl_file.write(f'(define (problem {Path(micro_fpath).stem})\n')
        pddl_file.write(f'(:domain {domain_name})\n')
        pddl_file.write('(:init' + '\n')

        # -- dictionary that will ground object names from conceptual (domain-independent) space to
        # 	simulated (domain-specific) space:
        grounded_objects = {}

        if 'hand' in in_object_layout:
            if in_object_layout['hand'] == 'air':
                # -- hand should be empty (i.e., contains 'air')
                pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
            else:
                # -- hand is not empty for whatever reason:
                pddl_file.write('\t; hand/end-effector contains some type of object:\n')

            pddl_file.write('\t' + '(in hand ' + str(in_object_layout['hand']) +')' + '\n')
        else:
            # -- hand will be empty by default:
            pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
            pddl_file.write('\t' + '(in hand air)' + '\n')

        pddl_file.write('\t' + '(not (is-mobile))' + '\n')

        # -- identifying all objects that are being referenced as preconditions:
        all_objects, on_objects = set(), set()
        for pred in preconditions:
            predicate_parts = pred[1:-1].split(' ')
            # -- we are just reviewing all of the objects referenced by predicates...
            all_objects.add(predicate_parts[1])
            if len(predicate_parts) > 2:
                all_objects.add(predicate_parts[2])

            # ... but we also need to note all that have already been referenced by the "on" predicate:
            if predicate_parts[0] == 'on' and predicate_parts[1] != 'cutting_board':
                on_objects.add(predicate_parts[1])
        # endfor

        # -- we will write the current object layout as obtained from the "perception" step.
        written_predicates = []

        for obj in list(on_object_layout.keys()):
            # -- writing the "on" relation predicate:
            on_pred = str('(on ' + obj + ' ' + on_object_layout[obj] + ')')
            written_predicates.append(on_pred)

            if obj in under_object_layout:
                # -- writing the "under" relation predicate:
                under_pred = str('(under ' + obj + ' ' + under_object_layout[obj] + ')')

                # -- write the predicate to the micro-problem file:
                written_predicates.append(under_pred)

        for obj in list(under_object_layout.keys()):
            # if under_object_layout[obj] not in on_object_layout and under_object_layout[obj] not in ['air']:
            if under_object_layout[obj] not in ['air']:
                on_pred = str('(on ' + under_object_layout[obj] + ' ' + obj + ')')
                written_predicates.append(on_pred)

        if written_predicates:
            # -- remove any duplicates:
            written_predicates = sorted(list(set(written_predicates)))

            pddl_file.write('\n\t; current state of environment (object layout from perception):\n')
            for pred in written_predicates:
                pddl_file.write('\t' + pred + '\n')

        written_predicates = []

        # NOTE: some objects in the simulation space have names that will combine two words together:
        #	e.g., we could have:	1. (in bottle vodka) -- bottle_vodka
        #				2. (in cup lemon_juice) -- cup_lemon_juice
        # -- because of this, we need to review those predicates that remain for "grounding"
        # 	to simulation instances.

        for pred in preconditions:
            # -- if we use the perception-based method to acquire initial state,
            #	then we will need to change required FOON states to current state:
            pred_parts = pred[1:-1].split(' ')

            # -- when using the perception-based method, we will acquire "on" or "under" conditions for the tables;
            #	therefore, we could ignore these states in particular.

            if len(pred_parts) < 3:
                written_predicates.append(str('\t' + pred + '\n'))

            elif 'table' in [pred_parts[1], pred_parts[2]]:
                continue

            else:
                # -- in this case, we should only be considering predicates
                #       of relation "in":
                if pred_parts[0] != 'in':
                    continue

                # -- we would have to consider grounding the objects to their simulated counter-parts.
                obj1, obj2 = pred_parts[1], pred_parts[2]
                if str(pred_parts[1] + '_' + pred_parts[2]) in on_object_layout:
                    obj1 = pred_parts[1] + '_' + pred_parts[2]
                if str(pred_parts[2] + '_' + pred_parts[1]) in on_object_layout:
                    obj2 = pred_parts[2] + '_' + pred_parts[1]

                # -- we will preserve a mapping between domain-independent and domain-specific objects:
                grounded_objects[pred_parts[1]] = obj1
                grounded_objects[pred_parts[2]] = obj2

                grounded_pred = str('(' + pred_parts[0] + ' ' + obj1 + ' ' + obj2 + ')')
                if obj1 != 'air' and grounded_pred not in written_predicates:
                    written_predicates.append(grounded_pred)

                    # -- we will also put the reverse fact that obj1 is under obj2 if it is in obj1
                    reversed_pred = str('(' + 'under' + ' ' + obj2 + ' ' + obj1 + ')')
                    written_predicates.append(reversed_pred)

        if written_predicates:
            # -- remove any duplicates:
            written_predicates = list(set(sorted(written_predicates)))

            pddl_file.write('\n\t; precondition predicates obtained directly from macro PO:\n')
            for pred in written_predicates:
                pddl_file.write('\t' + pred + '\n')

        pddl_file.write(')\n\n')

        pddl_file.write('(:goal (and\n')

    # -- hand must be empty after executing action to free it for next macro-PO:
        pddl_file.write('\t; hand/end-effector must be also be empty after execution (i.e. contains "air"):\n')
        pddl_file.write('\t(in hand air)\n\n')

        pddl_file.write('\t; effect predicates obtained directly from macro PO:\n')

        # -- the effects (post-conditions) for a macro PO will form the goals:

        grounded_goals = {'reg': [], 'neg': []}

        for pred in effects:
            predicate_parts = pred[1:-1].split(' ')

            # -- we need to treat negation predicates special so we print them correctly:
            is_negation = False
            if predicate_parts[0] == 'not':
                is_negation = True
                predicate_parts = list(filter(None, predicate_parts[1:]))

                for x in range(len(predicate_parts)):
                    # -- remove any trailing parentheses characters:
                    predicate_parts[x] = predicate_parts[x].replace('(', '')
                    predicate_parts[x] = predicate_parts[x].replace(')', '')

            if 'mix' in Path(micro_fpath).stem and (len(set(predicate_parts[1:]) - set(all_objects)) > 0 or is_negation):
                # -- if we are mixing ingredients, we need to omit all negations of individual ingredients;
                #	this is because, on the micro level, only the container is viewed as being "mixed".

                # -- here, we omit the following cases of predicates when mixing:
                # 	1. individual ingredients making up will be negated after the process of mixing
                #	2. when an object is mixed, we form a new object that was not referenced as an input or precondition
                #		(e.g. when mixing salad ingredients, we create a new object "salad")
                pass

            else:
                # -- first, check if there is any grounding that needs to be done to simulation objects:
                obj1 = grounded_objects[predicate_parts[1]] if str(
                    predicate_parts[1]) in grounded_objects else str(predicate_parts[1])

                obj2 = ''
                if len(predicate_parts) > 2:
                    obj2 = grounded_objects[predicate_parts[2]] if str(
                        predicate_parts[2]) in grounded_objects else str(predicate_parts[2])

                # # -- if any of the referenced objects are "table", then we need to ground the table to the proper grid-space:
                # if 'table' in [obj1, obj2]:
                #     # -- this means that we have an "on" or "under" predicate if something is on the table:
                #     is_table_obj1 = bool(obj1 == 'table')

                #     found_mapping = False
                #     for item in on_object_layout:
                #         if on_object_layout[item] == (obj2 if is_table_obj1 else obj1):
                #             if is_table_obj1:
                #                 obj1 = item
                #             else:
                #                 obj2 = item
                #             found_mapping = True
                #             break

                #     if not found_mapping:
                #         # -- if we could not find a grounding for the table in the scene, skip this predicate:
                #         continue

                if is_negation:
                    grounded_goals['neg'].append(f"\t(not ({predicate_parts[0]} {obj1} {obj2 if obj2 else ''}))\n")
                else:
                    grounded_goals['reg'].append(f"\t({predicate_parts[0]} {obj1} {obj2 if obj2 else ''})\n")
                    pddl_file.write(grounded_goals['reg'][-1])

        # print(f'\n\n{grounded_goals}\n\n')

        # -- checking to see if there are any contradictions:
        for pred1 in grounded_goals['neg']:
            is_contradiction = False
            for pred2 in grounded_goals['reg']:
                is_contradiction = str(pred2).strip() in pred1
                if is_contradiction: break

            if not is_contradiction:
                pddl_file.write(pred1)


        pddl_file.write('))\n')
        pddl_file.write('\n)')

    return micro_problem_fpath
# enddef


def find_plan(
        domain_file: str,
        problem_file: str,
        output_plan_file:str = None,
        verbose: bool = False,
    ) -> str:

    # NOTE: define plan execution function that can be called for different parameters:

    # -- this will store the output that would normally be seen in the terminal
    planner_output = None
    command = None

    # -- based on the planner we use, the required command changes:
    if planner_to_use == 'fast-downward':
        # NOTE: more information on aliases for Fast-Downward can be found here: https://www.fast-downward.org/IpcPlanners
        # -- you can use a different algorithm for more optimal or satisficing results:
        method = configs[algorithm][heuristic]
        command = ['python3', str(path_to_planners[planner_to_use]), '--plan-file', output_plan_file, domain_file, problem_file, '--search', method]

    elif planner_to_use == 'PDDL4J':
        command = ['java', '-jar', str(path_to_planners[planner_to_use]),  '-o', domain_file, '-f', problem_file]

    else:
        return None
    # endif

    if verbose:
        print(f'run this command: {str(" ".join(command))}')

    try:
        planner_output = subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        print(f"  -- [FOON-TAMP] : Planner error (planner:  {planner_to_use}, error code: {e.returncode})!")
        # print(f'     -- Error code: {e.returncode} -- {e.output}')
        pass
    # endtry

    return planner_output
# enddef


def solve(output: str) -> bool:
    # -- using planner-specific string checking to see if a plan was found:
    global planner_to_use
    if planner_to_use == 'fast-downward':
        return True if 'Solution found.' in str(output) else False
    elif planner_to_use == 'PDDL4J':
        # TODO: implement integration with other planners such as PDDL4J
        return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goal",
        type=str, default="None",
        help="This specifies a set of goal predicates as a string."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default='./utamp/scenes/panda_stacking.ttt',
        help="This specifies a set of goal predicates as a string."
    )
    args = parser.parse_args()

    if not eval(args.goal):
        # 'blockc:on,blocka:under,blockb;blocka:on,air:under,blockc;blockb:on,blockc;'
        goals_for_utamp = [
            "(on blockc blocka)",
            "(under blocka blockc)",
            "(on blocka air)",
        ]

    driver = Interfacer(scene_file_name=args.scene)
    utamp_client = UTAMP(); utamp_client.plan_and_execute(driver, goals_for_utamp=eval(args.goal))
import sys
import os
import time
import argparse
import math
import numpy as np
import json
import mysql.connector as sql

from random import choice
from typing import Type
from scipy.interpolate import CubicSpline

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print(' -- ERROR: Set up CoppeliaSim ZeroMQ client as described here: '\
          'https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm')
    sys.exit()


# pip3 install numpy scipy openai tiktoken scikit-learn tqdm pandas coppeliasim-zmqremoteapi-client nltk mysql-connector

class Interfacer():
    def __init__(
            self,
            scene_file_name: str,
            robot_name: str = "Panda",
            robot_gripper: str = "Panda_gripper",
            port_number: int = None,
        ):
        self.robot_name = robot_name
        self.robot_gripper = robot_gripper

        self.client = None
        if port_number:
            # -- specify another port number for a separate client:
            self.client = RemoteAPIClient(host='localhost', port=port_number)
        else:
            # -- just use the default port number:
            self.client = RemoteAPIClient(host='localhost')

        self.sim = self.client.require('sim')
        status = self.load_scene(scene_file_name)
        if not bool(status):
            sys.exit("ERROR: something went wrong with scene load function!")

        # -- loading required modules for simulation:
        self.simIK = self.client.require('simIK')
        self.simOMPL = self.client.require('simOMPL')

        self.objects_in_sim = self.get_sim_objects()

        # -- self-contained dictionaries containing object properties:
        self.object_positions, self.object_orientations, self.object_bb_dims = {}, {}, {}

        self.start_pose = self.sim.getObjectPose(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))

        self.start_time = time.time()

        # -- get the reference to the object containing the OMPLement script in CoppeliaSim:
        self.ompl_script = self.sim.getScript(self.sim.scripttype_simulation, self.sim.getObject('/OMPLement'))


    def load_scene(self, scene_file_name: str) -> int:
        # -- if any simulation is currently running, stop it before closing the scene:
        self.sim_stop(); self.sim.closeScene()

        # -- finally, get the full path to the scene file and load it:
        complete_scene_path = os.path.abspath(scene_file_name)
        self.sim.addLog(self.sim.verbosity_default, '[UTAMP-driver] : Loading scene "' + complete_scene_path + '" into CoppeliaSim...')

        if not bool(self.sim.loadScene(complete_scene_path)):
            return False

        # -- reset camera to some default pose that is centered on the robot:
        self.sim_reset_camera()

        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        # -- make sure that the gripper is opened (i.e., set the signal to value of 1):
        self.sim.setInt32Signal('close_gripper', 0)

        return True

    ############################################################
    # NOTE: simulation control functions:
    ############################################################

    def sim_start(self):
        self.sim_print(string='Starting simulation...')

        self.sim.startSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_advancing_running:
            time.sleep(0.001)

    def sim_stop(self):
        self.sim_print(string='Stopping simulation...')

        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.001)

    def sim_pause(self):
        self.sim_print(string='Pausing simulation...')

        self.sim.pauseSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_paused:
            time.sleep(0.001)

    def sim_reset(self):
        self.objects_in_sim = self.get_sim_objects()
        self.start_time = time.time()

    def sim_print(self, string: str, print_type: str = 'default'):
        # NOTE: print_type can be of the following: ['default', 'warnings', 'errors']
        if print_type not in ['default', 'warnings', 'errors']:
            # -- just default to "default" if anything is wrong:
            print_type = 'default'

        self.sim.addLog(eval(f"self.sim.verbosity_{print_type}"), f'[UTAMP-driver] : {string}')

    def get_elapsed_time(self):
        return float(time.time() - self.start_time)

    ############################################################
    # NOTE: camera visualization stuff:
    ############################################################

    def sim_reset_camera(self):
        ideal_positions = [
            [-2.10, 0, 1.39],
            [-1.7, 0, 2.25],
        ]
        ideal_orientations = [
            [-math.pi, 1.40, -(math.pi/2.0)],
            [-math.pi, 0.8, -(math.pi/2.0)],
        ]

        # -- set the camera pose to an ideal view of the table-top scene:
        self.sim.setObjectPosition(self.sim.getObject('/DefaultCamera'), -1, ideal_positions[-1])
        self.sim.setObjectOrientation(self.sim.getObject('/DefaultCamera'), -1, ideal_orientations[-1])

    def sim_take_snapshot(self, file_name: str, render_mode: str = 'opengl'):
        # NOTE: a lot of this code was pulled directly from the "Screenshot tool" provided by CoppeliaSim:
        sim = self.sim
        cam = sim.getObject('/DefaultCamera')
        pose = sim.getObjectPose(cam)

        config = {
            'res': [1500, 1100],
            'renderMode': render_mode,
            'povray': {
                'available': False,
                'focalBlur': True,
                'focalDistance': 2,
                'aperture': 0.15,
                'blurSamples': 10,
            },
            'nearClipping': 0.01,
            'farClipping': 200,
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

    ############################################################
    # NOTE: getting objects in the scene:
    ############################################################

    def get_sim_objects(self) -> list[str]:
        # NOTE: this function should be run at the very beginning of robot execution:
        objects_list = []

        obj_index = 0
        while True:
            obj_handle = self.sim.getObjects(obj_index, self.sim.handle_all)
            if obj_handle == -1:
                break

            obj_index += 1

            # -- check if the object has no parent (because we only want to consider base objects):
            if self.sim.getObjectParent(obj_handle) != -1:
                continue

            # -- we need to make sure that the object is both non-static/dynamic and respondable:
            if bool(self.sim.getObjectInt32Param(obj_handle, self.sim.shapeintparam_static)) and not bool(self.sim.getObjectInt32Param(obj_handle, self.sim.shapeintparam_respondable)):
                continue

            # -- check if the object is not a valid "shape" type object (to exclude anything like the camera, dummy objects, etc.):
            if self.sim.getObjectType(obj_handle) != self.sim.object_shape_type:
                continue

            # -- add the name of the object to the list of all objects of interest in the scene:
            obj_name = self.sim.getObjectAlias(obj_handle)
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
                obj_handle = sim.getObject(f'/{self.robot_name}/{obj}')

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

    def find_empty_spots(self) -> list[str]:
        # NOTE: the purpose of this function is to find empty places on the table for place actions:

        # -- get the current state and parameters of objects in the scene:
        _ = self.get_sim_object_params(verbose=False)

        table, empty_spots = [], []

        while True:
            try:
                dummy = self.sim.getObject('/Dummy', {'index': len(table)})
            except Exception:
                break

            is_empty = True
            for obj in self.objects_in_sim:
                if self.check_if_on_top(
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

    ############################################################
    # NOTE: object to object checking (for perception/sensing):
    ############################################################

    def check_if_contact(
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

    def check_if_on_top(
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

        elif (obj1_min_z - obj2_max_z) > 0.01:
            is_top_bb_z = False

        is_touching = True
        if check_collision:
            # -- checking if there is any contact between the object pair:
            is_touching = self.check_if_contact(obj_above, obj_below)

        if verbose:
            print(f"{obj_above} is within x-bb {obj_below}? -> {is_within_bb_x}")
            print(f"{obj_above} is within y-bb {obj_below}? -> {is_within_bb_y}")
            print(f"{obj_above} is on top of {obj_below}? -> {is_top_bb_z}")
            print(f"{obj_above} is touching {obj_below}? -> {is_touching}")

        return bool(is_within_bb_x and is_within_bb_y and is_top_bb_z and is_touching)

    def get_object_in_hand(self, skip_proximity_sensor:bool=False):
        # NOTE: this function searches for the gripper's attach point and proximity sensor objects to determine
        #       whether there is an object in the robot's gripper or not.

        # -- search the robot's collection tree for the sensor and attach point handles:
        gripper_attachPoint, gripper_proximitySensor = -1, -1
        for C in self.sim.getObjectsInTree(self.sim.getObject(f"/{self.robot_name}")):
            if "attachPoint" in self.sim.getObjectAlias(C):
                gripper_attachPoint = C
            elif "attachProxSensor" in self.sim.getObjectAlias(C):
                gripper_proximitySensor = C

        if gripper_attachPoint == -1:
            sys.exit("ERROR: robot gripper attach point was not found!")
            self.sim_print("ERROR: robot gripper attach point was not found!", print_type='error')

        elif gripper_proximitySensor == -1:
            sys.exit("ERROR: robot gripper proximity sensor was not found!")
            self.sim_print("ERROR: robot gripper proximity sensor was not found!", print_type='error')

        # NOTE: the object that is in its gripper should be the child of the attach point:
        obj_in_hand = self.sim.getObjectChild(gripper_attachPoint, 0)

        if obj_in_hand == -1: return -1

        # -- check if an object is within the range of the end-effector's proximity sensor:
        is_in_hand, _, _, _, _ = self.sim.checkProximitySensor(gripper_proximitySensor, obj_in_hand)

        if not bool(is_in_hand):
            print("WARNING: there's an object attached but the proximity sensor does not pick it up?")
            self.sim_print("WARNING: there's an object attached but the proximity sensor does not pick it up?", print_type='warning')

        if skip_proximity_sensor: is_in_hand = True

        # -- now that we found the gripper's attach point, we return its child, which should be the object in it:
        return obj_in_hand if bool(is_in_hand) else -1

    def check_if_object_in_hand(self):
        # -- this function merely checks if there is something in the hand or not:
        return self.get_object_in_hand() != -1

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
                    obj_handle = sim.getObject(f'/{self.robot_gripper}')
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
            is_grasping = False

            for obj in goal_objects:
                # -- check if there is an object "attached" to the gripper:
                obj_in_hand = self.get_object_in_hand()

                if self.check_if_object_in_hand():
                    is_grasping = True

                    try:
                        # -- checking to see if the hand is truly *on top* of the object:
                        obj_position, obj_orientation = self.object_positions[obj_in_hand], self.object_orientations[obj_in_hand]
                    except KeyError:
                        obj_position, obj_orientation = self.sim.getObjectPosition(obj_in_hand, -1), self.sim.getObjectOrientation(obj_in_hand, -1)

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
                        on_object_layout[sim.getObjectAlias(obj_in_hand)] = 'hand'
                        under_object_layout[sim.getObjectAlias(obj_in_hand)] = 'air'

                    under_object_layout['hand'] = sim.getObjectAlias(obj_in_hand)
                    in_object_layout['hand'] = sim.getObjectAlias(obj_in_hand)

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

                    if self.check_if_on_top(obj1, obj2, check_collision=check_collision):
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

    ############################################################
    # NOTE: OMPL-related planning:
    ############################################################

    def ompl_get_target_pose(
            self,
            target_object: str,
            affordance: str = 'pick-top',
            verbose: bool = True,
            skip_errors: bool = True,
        ) -> list[float]:

        # -- get robot and end-effector target handles:
        robot = self.sim.getObject(f"/{self.robot_name}")
        target = self.sim.getObject(f"/{self.robot_name}/target")

        index = 0
        # -- check if the target object refers to the table, as we will have to find an empty spot:
        if target_object in ["table", "worksurface"]:
            empty_spot = choice(self.find_empty_spots())
            index, target_object = empty_spot['index'], self.sim.getObjectAlias(empty_spot['handle'])
            self.sim.addLog(self.sim.verbosity_default, f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")
            if verbose:
                print(f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")
                self.sim_print(f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")

        # -- get the pose of the target object:
        goal = self.sim.getObject(f"/{target_object}", {"index": index})

        fingertip_handle = self.sim.getObject(f'/{self.robot_name}_leftfinger_respondable', {"noError": True})

        # -- this will tell us if there is an object in the gripper (obj_in_hand != -1):
        obj_in_hand = self.get_object_in_hand()

        candidate_goal_poses = []

        if "pick" in affordance:
            # -- we are considering different end-effector orientations for picking an object:
            if obj_in_hand == -1:

                for angular_offset in [0.0, (math.pi/2), (math.pi), (3*math.pi/2)]:
                    # -- first, let's get the position set:
                    goal_pose = self.sim.getObjectPose(goal, robot)

                    # NOTE: if there is an object in hand, we are assuming that we want to place it somewhere;
                    #       else, we are assuming that we want to pick up an object instead.

                    if affordance == 'pick-side':
                        goal_pose[0] -= self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_x) * 0.5

                    if affordance == 'pick-top':
                        # -- move the target down by half of the finger so that it is positioned in a more "stable" position:
                        goal_pose[2] += self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_z) \
                            - (self.sim.getObjectFloatParam(fingertip_handle, self.sim.objfloatparam_objbbox_max_z)) if fingertip_handle != -1 else 0.0

                    for rotate in [0.0, math.pi, ]:
                        # -- now we will consider the orientation of the gripper depending on the task:
                        goal_pose_copy = list(goal_pose)

                        orientation = self.sim.getObjectOrientation(goal, robot)
                        if 'side' in affordance:
                            goal_pose_copy[3:] = self.sim.buildPose(goal_pose[:3], [orientation[0], (math.pi/2) + angular_offset, rotate])[3:]
                        elif 'top' in affordance:
                            goal_pose_copy[3:] = self.sim.buildPose(goal_pose[:3], [-(math.pi) + rotate, orientation[1], orientation[2] + angular_offset])[3:]

                        candidate_goal_poses.append(goal_pose_copy)

            elif not skip_errors: raise Exception("ERROR: picking with an object in hand?")

        elif "place" in affordance:
            if obj_in_hand != -1:
                # -- we are considering different end-effector orientations for picking an object:
                for angular_offset in [0.0, (math.pi/2), (math.pi), (math.pi*2)]:
                    # -- sample another empty spot:
                    if target_object in ["table", "worksurface"]:
                        empty_spot = choice(self.find_empty_spots())
                        index, target_object = empty_spot['index'], self.sim.getObjectAlias(empty_spot['handle'])
                        self.sim.addLog(self.sim.verbosity_default, f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")
                        if verbose:
                            print(f"table grounding: found empty spot: /{target_object}[{empty_spot['index']}]")

                    goal_pose = self.sim.getObjectPose(goal, robot)

                    # -- first, we find a spot that sits RIGHT ON TOP of the surface...
                    goal_pose[2] += self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_z)
                    # ... then we will find a spot that considers the height of the object:
                    goal_pose[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z) \
                        + (self.sim.getObjectFloatParam(fingertip_handle, self.sim.objfloatparam_objbbox_max_z)) if fingertip_handle != -1 else 0.0
                    # * (1.5 if target_object not in ["table", "worksurface"] else 1.25)

                    goal_pose[3:] = self.sim.getObjectPose(target, robot)[3:]

                    candidate_goal_poses.append(goal_pose)

            elif skip_errors: raise Exception("ERROR: placing without an object in hand?")

        elif affordance in ['pour']:
            # -- we will go to a position above the target container object at some offset based on the height of the source container object:
            if obj_in_hand == -1 and not skip_errors:
                sys.exit("ERROR: no object in hand for pouring!")

            # -- we are going to keep the same orientation of the robot's gripper, but we will modify the position:
            goal_pose = self.sim.getObjectPose(target, robot)
            # -- let's move the goal position to where the target is as a base:
            goal_pose[:3] = self.sim.getObjectPosition(goal, robot)

            # NOTE: we will consider pouring either along the x-axis or y-axis:
            # -- x-axis increases as we go to  the right!
            pour_direction = 'left' if (self.sim.getObjectPosition(obj_in_hand, robot)[1] > self.sim.getObjectPosition(goal, robot)[1]) else 'right'

            if verbose:
                print(f"{self.sim.getObjectAlias(obj_in_hand)} is {pour_direction} of {self.sim.getObjectAlias(goal)}")
                self.sim_print(f"{self.sim.getObjectAlias(obj_in_hand)} is {pour_direction} of {self.sim.getObjectAlias(goal)}")

            # -- get the heights of the objects:
            src_obj_height = self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z) * 2.0
            # src_obj_width = self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_y) * 2.0

            tgt_obj_height = self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_z) * 2.0
            tgt_obj_width = self.sim.getObjectFloatParam(goal, self.sim.objfloatparam_objbbox_max_y) * 2.0

            # -- we want to position the object somewhere that is some offset to the side of the target container:
            goal_pose[1] += (1.0 if pour_direction == 'left' else -1.0) * ((src_obj_height) + (tgt_obj_width * 0.0))

            # -- we want the object to be placed somewhere above the target container:
            goal_pose[2] += (tgt_obj_height) + (src_obj_height)

            target_goal = self.sim.createDummy(0.025)
            self.sim.setObjectPose(target_goal, goal_pose, self.sim.getObject(f'/{self.robot_name}'))

            candidate_goal_poses.append(goal_pose)

        return candidate_goal_poses

    def ompl_compute(
            self,
            target_pose: list[float],
            target_object: str = None,
            ompl_args: dict = {},
        ) -> list[float]:

        # -- formatting the string name for printing a cool message:
        if target_object:
            self.sim_print(f'finding a plan to object "{target_object}"...')

        # -- create a dummy object that will represent the target goal:
        target_goal = self.sim.createDummy(0.025)
        self.sim.setObjectPose(target_goal, target_pose, self.sim.getObject(f'/{self.robot_name}'))
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_ambient_diffuse, [0.0, 1.0, 0.0])
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_emission, [0.6, 0.6, 0.6])
        self.sim.setObjectAlias(target_goal, 'OMPL_target')

        # -- we sleep for a bit so we can see this object appear in the sim:
        time.sleep(0.001)

        # NOTE: checking if key parameters have been specified for OMPL:
        if not bool(ompl_args): ompl_args = {}
        else: ompl_args = dict(ompl_args)

        if "ompl_algorithm" not in ompl_args:
            ompl_args["ompl_algorithm"] = "RRTConnect"
        try:
            ompl_args['ompl_algorithm'] = eval(f"self.simOMPL.Algorithm.{ompl_args['ompl_algorithm']}")
        except AssertionError:
            print(f"WARNING: {ompl_args['ompl_algorithm']} is not a valid algorithm!\n"
                  " -- Check out this page for list of algorithms: https://manual.coppeliarobotics.com/en/simOMPL.htm#enum:Algorithm")
            # -- just default to using RRTConnect:
            ompl_args['ompl_algorithm'] = eval(f"self.simOMPL.Algorithm.RRTConnect")
        if "ompl_num_attempts" not in ompl_args:
            ompl_args["ompl_num_attempts"] = 5
        if "ompl_max_compute" not in ompl_args:
            ompl_args["ompl_max_compute"] = 15
        if "ompl_max_simplify" not in ompl_args:
            # NOTE: let OMPL do default simplification, signified by -1:
            ompl_args["ompl_max_simplify"] = -1
        if "ompl_len_path" not in ompl_args:
            # NOTE: let OMPL do give default number of configs in solution path, signified by 0:
            ompl_args["ompl_len_path"] = 0
        if "ompl_state_resolution" not in ompl_args:
            ompl_args["ompl_state_resolution"] = float("5.0e-3")
        if "ompl_use_state_validation" not in ompl_args:
            ompl_args["ompl_use_state_validation"] = True
        if "ompl_use_lua" not in ompl_args:
            ompl_args["ompl_use_lua"] = False
        if "ompl_motion_constraint" not in ompl_args:
            ompl_args["ompl_motion_constraint"] = "free"
        if "ompl_limits_6d" not in ompl_args:
            ompl_args["ompl_limits_6d"] = None

        # NOTE: the sim must be started in order for this script function to work:
        self.sim_start()

        path, _ = self.sim.callScriptFunction(
            "ompl_path_planning",
            self.ompl_script,
            {
                "robot": self.robot_name,
                "goal": target_goal,
                "ompl_algorithm": ompl_args["ompl_algorithm"],
                "ompl_max_compute": ompl_args["ompl_max_compute"],
                "ompl_max_simplify": ompl_args["ompl_max_simplify"],
                "ompl_len_path": ompl_args["ompl_len_path"],
                "ompl_state_resolution": ompl_args["ompl_state_resolution"],
                "ompl_motion_constraint": ompl_args["ompl_motion_constraint"],
                "ompl_use_state_validation": ompl_args["ompl_use_state_validation"],
                "ompl_use_lua": ompl_args["ompl_use_lua"],
                "ompl_limits_6d": ompl_args["ompl_limits_6d"],
            },
        )

        time.sleep(0.001)

        # -- remove the OMPL target object:
        self.sim.removeObjects([self.sim.getObject('/OMPL_target')])

        self.sim.addLog(self.sim.verbosity_default, f'[FOON-TAMP]: plan{" not" if not bool(path) else ""} found!')

        return path

    def ompl_execute(
            self,
            target_object: str,
            target_pose: list[float],
            ompl_args: dict = {},
            draw_path: bool = True,
            ignore_dynamics: bool = False,
        ) -> bool:

        path = self.ompl_compute(
            target_pose=target_pose,
            target_object=target_object,
            ompl_args=ompl_args,
        )

        if not bool(path): return False

        time.sleep(0.01)

        self.sim_start()

        # -- if set to true, we will draw the path in the simulation:
        if draw_path: drawn_object = self.sim.callScriptFunction('visualizePath', self.ompl_script, path, [0.0, 1.0, 0.0])

        if ignore_dynamics:
            for obj in self.objects_in_sim:
                obj_handle = self.sim.getObject(f"/{obj}", {"noError": True})
                if obj_handle != -1:
                    self.sim.setObjectInt32Parameter(obj_handle, self.sim.shapeintparam_static, 1)
                    self.sim.setObjectInt32Parameter(obj_handle, self.sim.shapeintparam_respondable, 1)
                    self.sim.resetDynamicObject(obj_handle)

        time.sleep(0.01)

        self.move_to_configs(path)

        time.sleep(0.01)

        if ignore_dynamics:
            for obj in self.objects_in_sim:
                obj_handle = self.sim.getObject(f"/{obj}", {"noError": True})
                if obj_handle != -1:
                    self.sim.setObjectInt32Parameter(obj_handle, self.sim.shapeintparam_static, 0)
                    self.sim.setObjectInt32Parameter(obj_handle, self.sim.shapeintparam_respondable, 1)

        if draw_path: self.sim.removeDrawingObject(drawn_object)

        return True

    def move_to_configs(self, path: list[float]):

        # -- we need to disable the IK following done by the "target" dummy of the robot:
        # self.sim.setModelProperty(target, self.sim.modelproperty_scripts_inactive)
        ik_script = self.sim.getScript(self.sim.scripttype_simulation, self.sim.getObject(f"/{self.robot_name}"))
        if ik_script == -1:
            ik_script = self.sim.getScript(self.sim.scripttype_customization, self.sim.getObject(f"/{self.robot_name}"))

        self.sim.setObjectInt32Param(ik_script, self.sim.scriptintparam_enabled, 0)

        # NOTE: check the sim.moveToConfig() docs here: https://manual.coppeliarobotics.com/en/regularApi/simMoveToConfig.htm
        vel = 120
        accel = 120
        jerk = 120

        maxVel = [vel*math.pi/180, vel*math.pi/180, vel*math.pi/180, vel*math.pi/180, vel*math.pi/180, vel*math.pi/180, vel*math.pi/180]
        maxAccel = [accel*math.pi/180, accel*math.pi/180, accel*math.pi/180, accel*math.pi/180, accel*math.pi/180, accel*math.pi/180, accel*math.pi/180]
        maxJerk = [jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180, jerk*math.pi/180]

        # -- get all the joint angles of the robot:
        joint_handles = []
        num_joints = 1

        while True:
            # -- using "noError" so default handle is -1 (if not found);
            #    read more here: https://manual.coppeliarobotics.com/en/regularApi/simGetObject.htm
            obj_handle = self.sim.getObject(f"/{self.robot_name}/joint", {"noError": True, "index":(num_joints-1)})

            if obj_handle == -1: break

            joint_handles.append(obj_handle)
            num_joints += 1

        robot_collection = self.sim.createCollection()
        self.sim.addItemToCollection(robot_collection, self.sim.handle_tree, self.sim.getObject(f'/{self.robot_name}'), 0)

        # -- change the simulation setting to stepping mode for threaded/non-blocking execution:
        self.sim.setStepping(True)
        self.sim.step()

        # -- iterate through the entire plan of robot configurations:
        for P in range(len(path)):
            params = {
                'joints': joint_handles,
                'targetPos': path[P],
                # 'maxVel': maxVel,
                'targetVel': [0.55 * x for x in maxVel],
                # 'maxAccel': maxAccel,
                # 'maxJerk': maxJerk,
            }
            self.sim.moveToConfig(params)

            # is_collision, handles = self.sim.checkCollision(robot_collection,robot_collection)
            # if bool(is_collision) and handles[0] != handles[1]:
            #     print(f"TEST: there is a collision between {self.sim.getObjectAlias(handles[0])} and {self.sim.getObjectAlias(handles[1])}!")

            self.sim.step()

        # -- turn off stepping mode since we don't need it beyond this point:
        self.sim.setStepping(False)

        # -- we need to re-enable the IK following done by the "target" dummy of the robot:
        self.sim.setObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
        self.sim.setObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
        self.sim.setObjectInt32Param(ik_script, self.sim.scriptintparam_enabled, 1)


    def return_home(self, method: int = 1):
        # -- return back to starting position:
        if method == 1:
            self.ompl_execute(target_object=None, target_pose=self.start_pose, )
        else:
            self.spline_path_planning(self.start_pose)

    def spline_adjust_trajectory(self, T: list[float]):
        ang_min = -0.7245
        ang_max = -1.2130

        trajectory = list(T)

        for T in range(len(trajectory)):
            if trajectory[1][-1] < ang_max:
                trajectory[T][-1] += 2*math.pi;
            elif trajectory[1][-1] > ang_min:
                trajectory[T][-1] -= 2*math.pi;

        return trajectory

    def spline_path_planning(
            self,
            goal_pose: list[float],
            num_ompl_attempts: int  = 5,
        ) -> bool:

        ompl_script = self.sim.getScript(self.sim.scripttype_simulation, self.sim.getObject('/OMPLement'))

        target_goal = self.sim.createDummy(0.025)
        self.sim.setObjectPose(target_goal, goal_pose, self.sim.getObject(f'/{self.robot_name}'))
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_ambient_diffuse, [0.0, 0.0, 1.0])
        self.sim.setObjectColor(target_goal, 0, self.sim.colorcomponent_emission, [0.6, 0.6, 0.6])
        self.sim.setObjectAlias(target_goal, 'OMPL_target')

        config_found = self.sim.callScriptFunction(
            "find_ik_config",
            ompl_script,
            {
                "robot": self.robot_name,
                "goal": target_goal,
                "num_attempts": num_ompl_attempts,
            },
        )

        if not bool(config_found): return False

        self.get_sim_object_params(verbose=False)

        # -- get the initial 6D-pose of the hand(target):
        hand_pos = self.sim.getObjectPosition(self.sim.getObject(':/target'), self.sim.getObject(f'/{self.robot_name}'))
        hand_ori = self.sim.getObjectOrientation(self.sim.getObject(':/target'), self.sim.getObject(f'/{self.robot_name}'))
        hand_6d = hand_pos + hand_ori

        # -- get the goal 6D-pose from the target object goal:
        goal_6d = self.sim.getObjectPosition(target_goal, self.sim.getObject(f'/{self.robot_name}')) + hand_6d[3:]

        # -- we want to move up to some height that is based on the highest object in the scene:
        # -- find the highest obstacle to make sure obstacles are cleared:
        highest_object = None
        for name, pos in self.object_positions.items():
            if not highest_object or pos[-1] > self.object_positions[highest_object][-1]:
                highest_object = name

        # -- absolute position of hand:
        hand_6d_abs = self.sim.getObjectPosition(self.sim.getObject(':/target'), -1)

        post_hand_6d = list(hand_6d)
        pre_goal_6d = list(goal_6d)

        # -- check to see if the highest object is higher than the hand:
        if self.object_positions[highest_object][2] > (hand_6d_abs[2] + (self.object_bb_dims[highest_object][2] * 4)):
            highest_obj_pos = self.sim.getObjectPosition(self.sim.getObject(f"/{highest_object}"), self.sim.getObject(f'/{self.robot_name}')) + hand_6d[3:]
            post_hand_6d[2] = highest_obj_pos[2] - (self.object_bb_dims[highest_object][2] * 4)
            pre_goal_6d[2] = highest_obj_pos[2] - (self.object_bb_dims[highest_object][2] * 4)

            keypoints = [
                hand_6d,
                post_hand_6d,
                pre_goal_6d,
                goal_6d,
            ]

            # -- use cubic spline function to create trajectory:
            actual_grasp = self.spline_adjust_trajectory(
                self.generate_trajectory({'trajectory': keypoints, 'time': [0.0, 0.1, 0.9, 1.0]}, ntraj=500,)
            )
        else:
            pre_goal_6d[2] = post_hand_6d[2]

            keypoints = [
                hand_6d,
                pre_goal_6d,
                goal_6d,
            ]

            # -- use cubic spline function to create trajectory:
            actual_grasp = self.spline_adjust_trajectory(
                self.generate_trajectory({'trajectory': keypoints, 'time': [0.0, 0.5, 1.0]}, ntraj=500,)
            )

        for T in range(len(actual_grasp)):
            # -- we will cap at the height for clearance:
            # if actual_grasp[T][2] > post_hand_6d[2] * 2:
            #     actual_grasp[T][2] = post_hand_6d[2] * 2
            new_trajectory = self.sim.buildPose(list(actual_grasp[T][:3]), list(actual_grasp[T][3:]))
            self.sim.setObjectPose(self.sim.getObject(f':/target'), new_trajectory, self.sim.getObject(f'/{self.robot_name}'))
            time.sleep(0.0001)

        self.sim.removeObjects([target_goal])

        return True

    def execute(
            self,
            target_object: str,
            ompl_args: dict,
            gripper_action: int,
            method: int = 1, # 1 :- OMPL, not(1) :- spline interpolation
        ) -> bool:

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        for _ in range(3 if target_object in ['table', 'worksurface'] else 1):
            # -- we will be generous and allow three attempts at finding an empty place on the table:
            goal_poses = self.ompl_get_target_pose(target_object)

            goal_achieved = False
            for goal in goal_poses:
                if method == 1:
                    # -- use the OMPL-based path planning method:
                    success = self.ompl_compute(
                        target_object,
                        goal,
                        ompl_args,
                    )
                else:
                    # -- use a simpler spline path planning method:
                    success = self.spline_path_planning(goal, ompl_args["ompl_num_attempts"])

                # -- check if we succeeded with this particular pose:
                if success:
                    goal_achieved = success
                    break

            if goal_achieved: break

        # -- if for whatever reason the system failed, then we just
        if not success: return False

        # -- now that we've done obstacle avoidance to get to a good position to pick/place an object,
        #   we will now perform the rest of the action.

        # NOTE: pre-grasping check below:
        if gripper_action == 1:
            # -- this means we want to pick up an object:
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            # -- we want to position the gripper on the upper part of the object such that the proximity sensor detects an object:
            end = self.sim.getObjectPosition(self.sim.getObject(f'/{target_object}'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))

            # -- placing the target to the very top of the object...
            end[2] += (0.5 * self.sim.getObjectFloatParam(self.sim.getObject(f'/{target_object}'), self.sim.objfloatparam_objbbox_max_z))
            # ... then we will lower it based on the length of the gripper's fingertip (if the finger element exists):
            fingertip_handle = self.sim.getObject(f'/{target_object}_leftfinger_respondable', {"noError": True})
            if fingertip_handle != -1:
                # -- move the target down by half of the finger so that it is positioned in a more "stable" position:
                end[2] -= (0.5 * self.sim.getObjectFloatParam(fingertip_handle, self.sim.objfloatparam_objbbox_max_z))

            traj_move_to_obj = self.generate_trajectory(
                {'time': [0, 1], 'trajectory': [start, end]},
                ntraj=25
            )

            # -- execute the trajectory and open the gripper:
            self.execute_trajectory(traj_move_to_obj)

        self.act_end_effector(action=gripper_action)

        time.sleep(0.1)

        # NOTE: post-grasping check below:
        obj_in_hand = self.get_object_in_hand()
        if gripper_action == 1:
            if obj_in_hand > -1:
                # -- this means that we want to move the gripper up to remove the object from the top of the below object's surface:
                start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
                end = list(start)
                # -- we want to move up by half the height of the object:
                end[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z)
                # -- we also want to pick up the object and align it with the base of the robot:

                traj_move_up = self.spline_adjust_trajectory(
                    self.generate_trajectory({'time': [0, 1], 'trajectory': [start, end]}, ntraj=25)
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

        # -- if the robot fails to grip something, open the gripper back up:
        if gripper_action == 1:
            self.act_end_effector(close=0)

        return False

    def execute_utamp(
            self,
            goal_poses: list[float],
            gripper_action: int,
            algorithm: str = 'RRTConnect',
            num_ompl_attempts: int = 5,
            max_compute: int = 5,
            max_simplify: int = 5,
            len_path: int = 0,
            method: int = 1,
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
            algorithm = eval('self.simOMPL.Algorithm.RRTConnect')

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        success = False
        if gripper_action == 0:
            for _ in range(3):
                if method == 1:
                    # NOTE: here's how we decide on the goal pose:
                    # -- if we want to place, then we use the *post-grasp* given by UTAMP
                    # -- if we want to pick , then we use the *pre-grasp* given by UTAMP
                    success = self.ompl_compute(
                        None,
                        (post_grasp if gripper_action == 0 else pre_grasp),
                        algorithm,
                        num_ompl_attempts,
                        max_compute,
                        max_simplify,
                        len_path
                    )
                else:
                    success = self.spline_path_planning((post_grasp if gripper_action == 0 else pre_grasp), num_ompl_attempts)

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
                ntraj=50
            )

            self.execute_trajectory(traj_move_down)

        self.act_end_effector(gripper_action)

        time.sleep(0.01)

        # -- get the object handles for the gripper's attach point:
        obj_in_hand = self.get_object_in_hand()

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

    def act_end_effector(self, action: int = -1):
        # NOTE: action :- 1 - close, 0 - open, -1 - no action
        message = "maintaining gripper state..."
        if action == 1:
            message = 'closing gripper...'
        elif action == 0:
            message = 'opening gripper...'

        self.sim_print(message)
        if action != -1:
            self.sim.setInt32Signal('close_gripper', action)

    def generate_trajectory(
            self,
            keypoints: dict,
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

        self.sim_start()
        for T in traj:
            # -- set the gripper's position and orientation according to the recorded trajectory:
            self.sim.setObjectPosition(gripper_handle, robot_handle, list(T[0:3]))
            self.sim.setObjectOrientation(gripper_handle, robot_handle, list(T[3:6]))

            # -- pause for dramatic effect (just kidding, only for ensuring instructions are sent correctly)
            time.sleep(0.001)
        #endfor

        # -- use the gripper actions to decide whether to open or close gripper (0 and 1 respectively, otherwise -1 -- no change):
        if gripper_action != -1: self.act_end_effector(close=gripper_action)

    ############################################################
    # NOTE: skill definitions:
    ############################################################

    def pick(
            self,
            target_object: str,
            ompl_args: dict = {},
            affordance: str = 'pick-top',
        ):

        print("-- executing 'pick' action...")

        start_time = time.time()

        goal_achieved = False

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        for _ in range(3 if target_object in ['table', 'worksurface'] else 1):
            # -- we will be generous and allow three attempts at finding an empty place on the table:
            goal_poses = self.ompl_get_target_pose(target_object, affordance)

            for G in range(len(goal_poses)):
                # -- use the OMPL-based path planning method:
                success = self.ompl_execute(
                    target_object,
                    goal_poses[G],
                    dict(ompl_args),
                )

                # -- check if we succeeded with this particular pose:
                if success:
                    goal_achieved = goal_poses[G]
                    break

            if bool(goal_achieved): break

        # -- if we could not find a plan, then we just return:
        if not goal_achieved: return False

        # -- close the gripper
        self.act_end_effector(action=1)

        time.sleep(0.5)

        # -- move the end-effector up:
        goal_achieved[2] += 0.1

        # NOTE: post-grasping check below:
        # -- get the object handles for the gripper's attach point:
        obj_in_hand = self.get_object_in_hand()

        if obj_in_hand > -1:
            # -- this means that we want to move the gripper up to remove the object from the top of the below object's surface:
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            end = list(start)
            # -- we want to move up by half the height of the object:
            # end[2] += self.sim.getObjectFloatParam(obj_in_hand, self.sim.objfloatparam_objbbox_max_z)
            end[2] += 0.02

            # -- we also want to pick up the object and align it with the base of the robot:

            traj_move_up = self.spline_adjust_trajectory(
                self.generate_trajectory({'time': [0, 1], 'trajectory': [start, end]}, ntraj=25)
            )

            # -- execute the trajectory but do not change the state of the gripper:
            self.execute_trajectory(traj_move_up)

            if target_object in self.sim.getObjectAlias(obj_in_hand):
                # -- this means that we have successfully grasped the intended object:
                success = True

        else: success = False

        print(f'\t-- total execution time: {time.time() - start_time}')

        return success

    def place(
            self,
            target_object: str,
            ompl_args: dict = {},
            affordance: str = 'place-top',
        ):

        print("-- executing 'place' action...")

        start_time = time.time()

        goal_achieved = False

        if not self.check_if_object_in_hand():
            return False

        # -- we will try to find configs with OMPL and find a collision-avoiding plan up to a certain number of times
        for _ in range(3 if target_object in ['table', 'worksurface'] else 1):
            # -- we will be generous and allow three attempts at finding an empty place on the table:
            goal_poses = self.ompl_get_target_pose(target_object, affordance)

            for goal in goal_poses:
                # -- use the OMPL-based path planning method:
                success = self.ompl_execute(
                    target_object,
                    goal,
                    ompl_args,
                )

                # -- check if we succeeded with this particular pose:
                if success:
                    goal_achieved = success
                    break

            if goal_achieved: break

        # -- if we could not find a plan, then we just return:
        if not goal_achieved: return False

        # -- close the gripper
        self.act_end_effector(action=0)

        if self.check_if_object_in_hand():
            # -- an object should not be in the hand at this point, so just a sanity check:
            return False

        time.sleep(1)

        print(f'\t-- total time: {time.time() - start_time}')

        return True

    def pour(
            self,
            source_container: str,
            target_container: str,
            ompl_args: dict = {},
        ):

        start_time = time.time()

        if self.get_object_in_hand() != self.sim.getObject(f"/{source_container}"):
            print("WARNING: object in hand is not equal to the source container (is this intended?)")

        # NOTE: pouring comprises of the following subactions:
        #   1. pre-pouring: positioning a source container to a target container
        #   3. rotate the source container to transfer contents
        #   4. place the source container back somewhere

        # -- now, we need to find a path for pre-pouring:
        while True:
            result = self.ompl_execute(
                target_object=target_container,
                goal_pose=self.ompl_get_target_pose(
                    target_object=target_container,
                    affordance='pour',
                )[0],
                ompl_args=ompl_args,
            )

            if result: break

        robot = self.sim.getObject(f'/{self.robot_name}')
        target = self.sim.getObject(f'/{self.robot_name}/target')

        src_object_pos = self.sim.getObjectPosition(target, robot)
        tgt_object_pos = self.sim.getObjectPosition(self.sim.getObject(f"/{target_container}"), robot)

        # -- let's determine the direction of pouring:
        pour_direction = 'left' if src_object_pos[1] > tgt_object_pos[1] else 'right'

        # -- this is the maximum angle we will perform rotation:
        max_rotation = 120

        # -- get the initial pose of the gripper:
        start = self.sim.getObjectPose(target, robot)

        trajectory = [start]

        # -- rotate forward...
        for angle in range(max_rotation):
            end = self.sim.rotateAroundAxis(start, [1, 0, 0], start[:3], (angle * math.pi/180) * (-1.0 if pour_direction == "right" else 1.0))
            self.sim.setObjectPose(target, end, robot)
            trajectory.append(list(end))
            time.sleep(0.001)

        time.sleep(1)
        trajectory.reverse()

        # ... and back!
        for pose in trajectory:
            self.sim.setObjectPose(target, pose, robot)
            time.sleep(0.001)

        print(f'\t-- total time: {time.time() - start_time}')

        return True


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
            algorithm: str = 'RRTConnect',
            max_compute: int = 5,
            max_simplify: int = 5,
            path_plan_method: int = 1,
            len_path: int = 0,
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
                    len_path,
                    path_plan_method,
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

                str_objects = self.perform_sensing(method=2, verbose=verbose)
                if verbose:
                    print(str_objects)

                for obj in self.objects_in_sim:
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
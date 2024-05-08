import sys
import os
import time
import numpy as np
import mysql.connector as sql
import argparse
import math

from tqdm import tqdm
from scipy.interpolate import CubicSpline

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print(' -- ERROR: Set up CoppeliaSim ZeroMQ client as described here: '\
          'https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm')
    sys.exit()


class UTAMP():
    def __init__(self, scene_file_name, robot_name="Panda", port_number=None):
        self.robot_name = robot_name

        self.client = None
        if port_number:
            # -- specify another port number for a separate client:
            self.client = RemoteAPIClient(host='localhost', port=port_number)
        else:
            # -- just use the default port number:
            self.client = RemoteAPIClient(host='localhost')

        self.sim, status = self.load_scenario(scene_file_name)
        if not bool(status):
            print("ERROR: something went wrong with scene load function!")

        # -- loading required modules for simulation:
        self.simIK = self.client.require('simIK')
        self.simOMPL = self.client.require('simOMPL')

        self.objects_in_sim = self.get_sim_objects(self.sim)

    def reset(self):
        self.objects_in_sim = self.get_sim_objects(self.sim)

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

    def load_scenario(self, scene_file_name):
        # -- get a remote object:
        sim = self.client.require('sim')

        # defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
        sim.setInt32Param(sim.intparam_idle_fps, 0)

        sim.setObjectPosition(sim.getObject('/DefaultCamera'), -1, [-1.9200871561284523, 0.3133626462867929, 2.008099646928897])
        sim.setObjectOrientation(sim.getObject('/DefaultCamera'), -1, [-3.139155626000006, 0.931177318100004, -1.5738340619999989])

        sim.closeScene()

        complete_scene_path = os.path.abspath(scene_file_name)

        sim.addLog(sim.getInt32Param(sim.intparam_verbosity), '[UTAMP-driver] : Loading scene "' + complete_scene_path + '" into CoppeliaSim...')

        # -- make sure that the gripper is opened (i.e., set the signal to value of 1):
        sim.setInt32Signal('close_gripper', 0)

        return sim, sim.loadScene(complete_scene_path)
    #enddef


    def get_sim_objects(self, sim):
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
    #enddef


    def perform_sensing(self, method=1, check_collision=True):

        sim = self.sim
        goal_objects = self.objects_in_sim

        def get_object_poses():
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

                _, obj_bb_min_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_y)
                _, obj_bb_max_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_y)

                if 'target' in x.lower():
                    # NOTE: the motion planning only considers the height of each finger in the Panda's gripper:
                    obj_handle = sim.getObject(f'/{self.robot_name}_leftfinger_visible')

                _, obj_bb_min_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_x)
                _, obj_bb_max_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_x)

                _, obj_bb_min_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_z)
                _, obj_bb_max_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_z)

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

                    _, obj_bb_min_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_x)
                    _, obj_bb_min_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_y)
                    _, obj_bb_min_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_z)

                    _, obj_bb_max_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_x)
                    _, obj_bb_max_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_y)
                    _, obj_bb_max_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_z)

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

        def get_world_state(check_collision=True):

            in_object_layout, on_object_layout, under_object_layout = {}, {}, {}

            # -- by default, all objects have "air" on and under them:
            for obj in goal_objects:
                on_object_layout[obj] = 'air'
                under_object_layout[obj] = 'table'

            sim.addLog(sim.getInt32Param(sim.intparam_verbosity), '[UTAMP-driver] : Checking geometric configuration of objects...')

            # -- we will only request the positions and bounding box details only once:
            object_positions, object_bounding_boxes, object_orientations = {}, {}, {}

            for obj in tqdm(goal_objects, desc='  -- Acquiring object properties from CoppeliaSim...'):
                # -- get an object's handle:
                obj_handle = sim.getObject(f'/{obj}')

                if obj_handle == -1:
                    print("Warning: '" + obj + "' does not exist!")
                    # -- if an object that exists in the scene was not found, then it possibly is attached to the robot's gripper:
                    obj_handle = sim.getObject(f'/{self.robot_name}_gripper_attachPoint/{obj}')

                if obj_handle == -1:
                    print("Warning: '" + obj + "' does not exist!")

                    # -- if an object does not exist in the scene, just add it as a null object:
                    object_positions[obj] = None
                    object_orientations[obj] = None
                    object_bounding_boxes[obj] = None

                else:
                    # -- get the position and orientation of an object in the simulation environment:
                    object_positions[obj] = sim.getObjectPosition(obj_handle, -1)
                    object_orientations[obj] = sim.getObjectOrientation(obj_handle, -1)

                    # -- get coordinates of the bounding box for the object in the simulation environment:
                    _, bb_min_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_x)
                    _, bb_min_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_y)
                    _, bb_min_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_min_z)
                    _, bb_max_x = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_x)
                    _, bb_max_y = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_y)
                    _, bb_max_z = sim.getObjectFloatParameter(obj_handle, sim.objfloatparam_objbbox_max_z)

                    object_bounding_boxes[obj] = [bb_min_x, bb_max_x, bb_min_y,
                                                bb_max_y, bb_min_z, bb_max_z]
                #endtry
            #endfor

            # NOTE: we will also check from perception whether the hand is empty or not:
            gripper_sensor = sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachProxSensor')
            is_grasping = False

            for obj in goal_objects:
                obj_handle = sim.getObject(f'/{obj}')
                is_in_hand, _, _, _, _ = sim.checkProximitySensor(gripper_sensor, obj_handle)

                if bool(is_in_hand):
                    is_grasping = True
                    object_in_hand = obj

                    # -- checking to see if the hand is truly *on top* of the object:
                    obj_position, obj_orientation = object_positions[object_in_hand], object_orientations[object_in_hand]
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
                        on_object_layout[object_in_hand] = 'hand'

                    under_object_layout['hand'] = object_in_hand
                    in_object_layout['hand'] = object_in_hand

            if not is_grasping:
                # -- if there's nothing in the hand, just make it contain "air":
                under_object_layout['hand'] = 'air'
                in_object_layout['hand'] = 'air'

            # -- since we have already pre-loaded all the positions and coordinates of the bounding boxes for each object,
            #       we can save some time in determining the current geometric layout of the environment.
            for obj1 in goal_objects:

                # -- get the position of an object in the simulation environment:
                obj1_position = object_positions[obj1]
                if obj1_position is None:
                    continue

                # -- first, let's try to see if the object_1 is inside of a specific grid-space or cell:
                for obj2 in goal_objects:

                    # -- get position of the object:
                    obj2_position = object_positions[obj2]
                    if obj2_position is None:
                        continue

                    if obj1 == obj2:
                        continue

                    # -- get the dimensions of objects:
                    obj1_bbox, obj2_bbox = object_bounding_boxes[obj1], object_bounding_boxes[obj2]

                    # NOTE: these dimensions are relative to the object, so the position needs to be added:

                    # -- get the min-max z-position for obj1 (just to determine on/under positions):
                    obj1_min_z, obj1_max_z = obj1_position[2] + \
                        obj1_bbox[4], obj1_position[2] + obj1_bbox[5]

                    # -- get the min-max x-y-z positions for obj2:
                    obj2_min_x, obj2_max_x = obj2_position[0] + \
                        obj2_bbox[0], obj2_position[0] + obj2_bbox[1]
                    obj2_min_y, obj2_max_y = obj2_position[1] + \
                        obj2_bbox[2], obj2_position[1] + obj2_bbox[3]
                    obj2_min_z, obj2_max_z = obj2_position[2] + \
                        obj2_bbox[4], obj2_position[2] + obj2_bbox[5]

                    # -- then, we need to see if there is any object that lies within the boundary
                    #       of the grid-space and/or object below it:
                    is_within_bb_x, is_within_bb_y = False, False
                    if obj1_position[0] < obj2_max_x and obj1_position[0] >= obj2_min_x:
                        # -- this object lies within the bounding box in the x-axis:
                        is_within_bb_x = True
                        #     print(obj1, obj2, 'x')

                    if obj1_position[1] < obj2_max_y and obj1_position[1] >= obj2_min_y:
                        # -- this object lies within the bounding box in the y-axis:
                        is_within_bb_y = True
                        #     print(obj1, obj2, 'y')

                    if is_within_bb_x and is_within_bb_y:


                        if obj1_position[2] <= obj2_position[2]:
                            # -- if obj1 is actually below obj2, then just continue:
                            continue

                        if (obj1_min_z - obj2_max_z) > 0.001:
                            continue

                        # -- checking if there is any contact between the object pair:
                        if check_collision:
                            is_touching = sim.checkCollision(sim.getObjectHandle(obj1), sim.getObjectHandle(obj2))
                            if not isinstance(is_touching, tuple) or is_touching[0] == 0:
                                continue


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

        return get_object_poses() if method == 1 else get_world_state(check_collision)
    #enddef


    def generate_trajectory(self, str_actions, ntraj=150):

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

        # -- use cubic spline interpolation:
        cs = CubicSpline(sorted(list(set(time))), traj)

        xs = np.arange(0, 1+1/ntraj, 1/ntraj)

        traj = cs(xs)

        return traj, gripper_action


    def execute_trajectory(self, traj, gripper_action):
        robot_handle, gripper_handle = self.sim.getObject(f'/{self.robot_name}'), self.sim.getObject(f'/{self.robot_name}/target')

        self.start()
        for T in traj:
            # -- set the gripper's position and orientation according to the recorded trajectory:
            self.sim.setObjectPosition(gripper_handle, robot_handle, list(T[0:3]))
            self.sim.setObjectOrientation(gripper_handle, robot_handle, list(T[3:6]))

            # -- pause for dramatic effect (just kidding, only for ensuring instructions are sent correctly)
            time.sleep(0.001)
        #endfor

        # -- use the gripper actions to decide whether to open or close gripper (-1 and 1 respectively, otherwise 0 -- no change):
        if gripper_action > -1:
            if gripper_action == 1:
                print('close')
            else:
                print('open')
            self.sim.setInt32Signal('close_gripper', gripper_action)
        #endif

    #enddef


    def path_planning(self, target_object, gripper_action, algorithm='PRMstar'):
        # ompl_script = self.sim.getScript(self.sim.scripttype_addonscript , -1, 'OMPLement')
        try:
            algorithm = eval(f'self.simOMPL.Algorithm.{algorithm}')
        except AttributeError:
            print(f'WARNING: path planning algorithm "{algorithm}" does not exist!')
            algorithm = eval('self.simOMPL.Algorithm.RRTstar')

        ompl_script = self.sim.getScript(self.sim.scripttype_childscript , self.sim.getObject('/OMPLement'))
        path = self.sim.callScriptFunction(
            'path_planning',
            ompl_script,
            target_object,
            algorithm,
            self.robot_name)

        if path:
            target = self.sim.getObject(f'/{self.robot_name}')
            self.sim.setModelProperty(target, self.sim.modelproperty_scripts_inactive)

            self.start()
            for P in path:
                self.sim.callScriptFunction('setConfig', ompl_script, P)
                time.sleep(0.0001)

            # -- use the gripper actions to decide whether to open or close gripper (-1 and 1 respectively, otherwise 0 -- no change):
            if gripper_action > -1:
                if gripper_action == 1:
                    # -- this means we want to pick up an object:
                    self.sim.setObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
                    self.sim.setModelProperty(target, 0)

                    start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/tip'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/tip'), self.sim.getObject(f'/{self.robot_name}'))
                    end = self.sim.getObjectPosition(self.sim.getObject(f'/{target_object}'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/tip'), self.sim.getObject(f'/{self.robot_name}'))

                    str_actions = 'hand:'\
                        f't_1,0:x_1,{start[0]}:y_1,{start[1]}:z_1,{start[2]}:roll_1,{start[3]}:pitch_1,{start[4]}:yaw_1,{start[5]}:'\
                        f't_2,1:x_2,{end[0]}:y_2,{end[1]}:z_2,{end[2]}:roll_2,{end[3]}:pitch_2,{end[4]}:yaw_2,{end[5]}:gripper,-1;'

                    traj_move_to_obj, _ = self.generate_trajectory(str_actions, ntraj=50)

                    self.execute_trajectory(traj_move_to_obj, gripper_action=-1)

                else:
                    # -- this means we want to place an object on top of another:
                    self.sim.setObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/tip')), -1)
                    self.sim.setModelProperty(target, 0)

                self.sim.setInt32Signal('close_gripper', gripper_action)
            #endif

            # -- move the gripper up to a fixed point:
            start = self.sim.getObjectPosition(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}')) + self.sim.getObjectOrientation(self.sim.getObject(f'/{self.robot_name}/target'), self.sim.getObject(f'/{self.robot_name}'))
            end = list(start)
            end[2] = 0.25

            str_actions = 'hand:'\
                f't_1,0:x_1,{start[0]}:y_1,{start[1]}:z_1,{start[2]}:roll_1,{start[3]}:pitch_1,{start[4]}:yaw_1,{start[5]}:'\
                f't_2,1:x_2,{end[0]}:y_2,{end[1]}:z_2,{end[2]}:roll_2,{end[3]}:pitch_2,{end[4]}:yaw_2,{end[5]}:gripper,-1;'

            traj_move_up, _ = self.generate_trajectory(str_actions)

            self.execute_trajectory(traj_move_up, gripper_action=-1)

            obj_in_hand = self.sim.getObjectChild(self.sim.getObject(f'/{self.robot_name}/{self.robot_name}_gripper_attachPoint'), 0)
            if gripper_action == 1:
                if obj_in_hand > -1 and target_object in self.sim.getObjectAlias(obj_in_hand):
                    # -- this means that we have successfully grasped the intended object:
                    return True
            else:
                if obj_in_hand == -1:
                    # -- this means that we have successfully placed the object (or at least there is nothing in the gripper):
                    return True

        return False

#enddef

def connect_to_DB():
    config = {
        'user': 'david',
        'password': 'chevere',
        'host': '93.240.31.32',
        'database': 'uniphyed',
        'raise_on_warnings': True
    }

    cnx = sql.connect(**config)
    cnx.autocommit = True
    cursor = cnx.cursor(buffered=True)

    return cnx, cursor
#enddef


def main(utamp_obj, goal_preds=None):
    # -- parse through the goal predicates and format it as a string needed for UTAMP:
    obj_to_goals = {}
    for P in goal_preds:
        # -- we want to get strings in the required format for the UTAMP system:
        pred_parts = P[1:-1].split(" ")
        if len(pred_parts) < 3:
            continue

        if pred_parts[1] in ["hand", "table"] or pred_parts[2] in ["hand", "table"]:
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

    utamp_goal = str()
    for O in obj_to_goals:
        utamp_goal += f'{O}:{":".join(obj_to_goals[O])};'

    vars_to_insert = {
        'user_id': 51,
        'goal': utamp_goal,
        'objects': 'n/a',
        'actions': 'n/a',
        'status': 0,
    }

    cnx, cursor = connect_to_DB()

    sql_command = f'UPDATE utamp_data SET status = -1 WHERE user_id = {vars_to_insert["user_id"]}'
    cursor.execute(sql_command, vars_to_insert)
    cnx.commit()

    sql_command = f'UPDATE utamp_data SET status = 0, goal = "{vars_to_insert["goal"]}" WHERE user_id = {vars_to_insert["user_id"]}'
    cursor.execute(sql_command, vars_to_insert)
    cnx.commit()

    sql_command = ("SELECT * from utamp_data")
    cursor.execute(sql_command)

    for user_id, goal, objects, actions, status, msg in cursor:
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

    while status > -1:
        sql_command = ("SELECT status, msg from utamp_data where user_id = 51")
        cursor.execute(sql_command)
        status, msg = cursor.fetchone()

        sql_command = ("SELECT status from utamp_data where user_id = 51")

        print(f'status {status}:\t{msg}')

        if status == 1:
            # -- execution request:
            print('performing execution...')
            status = 2
            sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
            cursor.execute(sql_command)

            sql_command = ("SELECT actions from utamp_data where user_id = 51")
            cursor.execute(sql_command)
            str_actions = cursor.fetchone()[0]

            trajectory, gripper_action = utamp_obj.generate_trajectory(str_actions)
            utamp_obj.execute_trajectory(trajectory, gripper_action)

            status = 0
            sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
            cursor.execute(sql_command)

        elif status == 10:
            print('performing sensing...')
            # -- sensing request:
            status = 20
            sql_command = f'UPDATE utamp_data SET status = {status} WHERE user_id = 51'
            cursor.execute(sql_command)
            # str_objects = perform_sensing(sim, list(obj_to_goals.keys()))
            str_objects = utamp_obj.perform_sensing()
            print(str_objects)
            for obj in utamp_obj.objects_in_sim:
                if obj not in str_objects:
                    print(obj)

            # -- now that sensing is done, we switch to the
            status = 0
            sql_command = f'UPDATE utamp_data SET objects = "{str_objects}", status = {status} WHERE user_id = 51'
            cursor.execute(sql_command)

    if status == -1:
        return True

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
        goal_preds = [
            "(on blockc blocka)",
            "(under blocka blockc)",
            "(on blocka air)",
        ]

    driver = UTAMP(scene_file_name=args.scene)
    main(driver, goal_preds=eval(args.goal))
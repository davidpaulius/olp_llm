'''
FOON_TAMP (FOON-TAMP Framework: From Planning to Execution):
-------------------------------------------------------------
-- Written and maintained by: 
    * David Paulius (dpaulius@cs.brown.edu / davidpaulius@tum.de)

-- Special thanks to Alejandro Agostini (alejandro.agostini@uibk.ac.at) for references and help in legacy code,
    as well as guiding me in this project.

NOTE: If using this program and/or annotations provided by our lab, please kindly cite our papers
	so that others may find our work:
* Paulius and Agostini 2022 - https://arxiv.org/abs/2207.05800 
* Paulius et al. 2016 - https://arxiv.org/abs/1902.01537
* Paulius et al. 2018 - https://arxiv.org/abs/1807.02189
'''

''' License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
'''

# %% -- initialization of several parameters or variables:
import sys
import os
import subprocess
import getopt
import json
import numpy as np
import ast
import random
from pathlib import Path

last_updated = '31st March, 2023'

verbose = False

print('\n< FOON_TAMP: converting FOON graph to PDDL planning problem (last updated: ' + last_updated + ')>\n')

path_to_FOON_code = './foon_to_pddl/'
# NOTE: we need to import some files from the other FOON directory:
if path_to_FOON_code not in sys.path:
    sys.path.append(path_to_FOON_code)
    import FOON_to_PDDL as ftp

# -- define the file name and path to the subgraph file that is going to be converted into PDDL:
FOON_subgraph_file = None

# -- name of the files with final plan at the: 1) macro-level, and 2) micro-level:
macro_plan_file = None
micro_plan_file = None

# -- definition of micro planning operator file name:
micro_domain_file = './FOON_micro_domain.pddl'
micro_problems_dir = None

# -- keeping track of history of micro-problems used in this pipeline:
micro_problem_history = None

json_scene_file = None
json_content = None

# NOTE: make sure you define the path to where the planners are located on your machine:
# -- paths to the different planners on my PC:
path_to_planners = {}
if os.name == 'nt':
    # -- this is the path to the planners on the Windows side:
    path_to_planners['PDDL4J'] = 'D:/PDDL4J/pddl4j-3.8.3.jar'
    path_to_planners['fast-downward'] = 'D:/fast-downward-22.06/fast-downward.py'
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

# NOTE: the following variables refer to dictionaries that map objects to objects
#       based on an object-centered relation type (in, on, under):
on_object_layout, under_object_layout, in_object_layout, table_layout = None, None, None, None

# NOTE: these variables refer to dictionaries that will map PO names to their respective
#       definitions (as "PlanningOperator" objects -- see class for definition):
# -- there will be a dictionary for each level of planning (i.e., macro- and micro-level)
all_macro_POs, all_micro_POs = None, None

####################################################################################################

# NOTE: using MATLAB's MatlabEngine module for interfacing with Python (using DMP code from MATLAB):
# -- read more about it here: https://www.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine.html
matlab_engine = None

####################################################################################################
# NOTE: these are various flags to trigger certain actions:

# -- flag_perception :- this is to activate the perception module of detecting the position 
#                       and orientation of objects in the scene
flag_perception = False

# -- flag_randomize_scene :- this is to activate randomizing the configuration of objects in the scene;
#       this is done by a function defined in the FOON_to_CoppeliaSim.py script
flag_randomize_scene = False

# -- flag_dropout :- this is to trigger random dropout of ingredients in FOON macro-planning;
#                   think of this as making certain FOON nodes skippable.
flag_dropout = False        

# -- flag_action_contexts :- this is to trigger use of action contexts (loading them and applying them to simulation)
flag_action_contexts = False
action_contexts = None
AC_location = './dmp_code/data/actioncontexts.mat'

# -- flag_skip_execution :- flag used to indicate that ACs will be assumed to be executed.
flag_skip_execution = False
flag_skip_demo = False

# -- AC_history :- list of all action contexts seen throughout pipeline
AC_history = []

# -- state_history :- dictionary of states (on/under) from each time-step:
state_history = {}

# -- AC_requiring_demo :- list of all action contexts that need some demonstration
#           (to be used in conjunction with "flag_skip_execution" -- this will skip any Coppelia stuff)
AC_requiring_demo = []

flag_ignore_start = False

flag_collision = False

port_nums = [None]

####################################################################################################

# %% -- definitions of several useful functions or operations:

class PlanningOperator(object):

    def __init__(self, PO_name=None):
        self.name = PO_name
        self.preconditions = []
        self.effects = []
        self.parameters = {}
        self.args = []

    def addParameter(self, variable, type):
        self.parameters[variable] = type
        self.args.append(variable)

    def addPrecondition(self, pred):
        self.preconditions.append(pred)

    def addEffect(self, pred):
        self.effects.append(pred)

    def getPreconditions(self):
        return self.preconditions

    def getEffects(self):
        return self.effects

    def getParameters(self):
        return self.parameters, self.args

    def getActionName(self):
        return self.name

    def printPlanningOperator(self):
        if self.name:
            print('(:action ' + self.name)

            # -- print parameters for the planning operator:
            if bool(len(self.args)):
                print('\t:parameters (')
                for P in self.args:
                    print('\t\t' + str(P) + ' - ' + str(self.parameters[P]))
                print('\t)')
            else:
                print('\t:parameters ( )')

            # -- print preconditions..
            print('\t:precondition (and ')
            for pred in self.preconditions:
                if isinstance(pred, list):
                    # -- this is an *or* statement:
                    print('\t\t(or')
                    for subpred in pred:
                        print('\t\t\t' + subpred)
                    print('\t\t)')
                else:
                    print('\t\t' + pred)
            print('\t)')

            # -- print effects..
            print('\t:effect (and')
            for pred in self.effects:
                print('\t\t' + pred)
            print('\t)')
#enddef


def _preliminaries():
    global flag_action_contexts, flag_perception, flag_randomize_scene, flag_ignore_start

    global json_scene_file, json_content

    # -- get the root name of the JSON file with all scene properties:
    scenario = Path(json_scene_file).stem

    try:
        json_content = json.load( open(json_scene_file, 'r') )
    except FileNotFoundError:
        json_content = None
        sys.exit()

    global FOON_subgraph_file

    if not FOON_subgraph_file:
        # -- check the JSON file for the name of the subgraph file if it was not given as command-line argument:
        FOON_subgraph_file = json_content['FOON_subgraph_file']

    global macro_plan_file, micro_plan_file, micro_problems_dir

    # -- definition of macro and micro plan file names:
    macro_plan_file = os.path.splitext(FOON_subgraph_file)[0] + '_macro.plan'
    micro_plan_file = os.path.splitext(FOON_subgraph_file)[0] + '_micro.plan'

    # -- create a new folder for the generated problem files and their corresponding plans:
    micro_problems_dir = './micro_problems-' + Path(FOON_subgraph_file).stem + '-scenario=' + scenario 
    if not os.path.exists(micro_problems_dir):
        os.makedirs(micro_problems_dir)

    # -- perform conversion of the FOON subgraph file to PDDL:
    ftp.FOON_subgraph_file = FOON_subgraph_file
    if flag_dropout and "ingredient_names" in json_content:
        ftp.ingredients_to_ignore = json_content["ingredient_names"]

    if flag_dropout:
        if not flag_skip_execution:
            # -- randomly drop no more than half of all ingredients:
            ftp._convert_to_PDDL('OCP', ingredient_dropout=1)
        else:
            # -- for the simulation "simulation", we will perform a different type of dropout:
            # ftp._convert_to_PDDL('OCP', ingredient_dropout=2)
            ftp._convert_to_PDDL('OCP', ingredient_dropout=1)
    else:
        # -- no dropout of ingredients necessary:
        ftp._convert_to_PDDL('OCP', ingredient_dropout=0)

    print(' -- [FOON-TAMP] : Translated FOON into macro-level domain and problem files!')
    
    # -- try to load the FOON-to-CoppeliaSim module before attemptingt to run any simulation operations:
    if flag_perception:
        try:
            import FOON_to_CoppeliaSim as ftc
        except ImportError:
            # -- if the module could not be loaded, then just skip execution entirely:
            flag_perception = False

    if flag_perception:
        # -- load scenario in CoppeliaSim and other important details from the JSON file:
        global port_number
        return_code = ftc._loadScenario(json_content, port_number=port_nums[0])

        # -- if we did not load any scene, then we will just assume the base micro-problem formation:
        if return_code == -1:
            # -- this means that the scene was not loaded by CoppeliaSim,
            #       so avoid using perception and action contexts if that's the case:
            flag_perception = False
            flag_action_contexts = False
        else:
            flag_perception = True
            ftc._initializeIngredients()

        if flag_randomize_scene:
            # -- randomize placement of robot gripper and objects in the scene:
            ftc._randomizeScene()

            # -- reposition the robot's gripper location 
            #       (if "no-start" is given, we are using DMPs fixed at the location "start_4")
            ftc._repositionRobot(target_location=(None if not flag_ignore_start else 'start_4'))
            
            # NOTE: we will only use fixed starting locations!
            flag_ignore_start = False

        # -- adjust the camera angle to the "sweet spot":
        ftc._adjustCamera()

        # -- play the scene for a few seconds to properly initialize position of objects:
        ftc._initializeScene()

        if verbose:
            # -- no matter what (execution or not), we need to initialize the scene and read the state of the environment:
            current_state = _runPerception(connect_to_sim=True)
            for key in current_state.keys():
                print(current_state[key])

        # input('\n -- [FOON-TAMP] : Kitchen has been initialized! Press ENTER to continue...')
        print('\n -- [FOON-TAMP] : Kitchen has been initialized!')
    #endif

    if flag_action_contexts:
        global matlab_engine, action_contexts

        globals()['matlab'] = __import__('matlab')

        import matlab.engine

        if action_contexts is None:
            # -- loading action contexts file from Alejandro's code section:
            try:
                action_contexts = ftc.loadmat(AC_location, squeeze_me=True, struct_as_record=False)
            except FileNotFoundError:
                print("\n -- [FOON-TAMP] : No action contexts have been loaded! If this is a mistake, check the 'AC_location' variable...")
                action_contexts = []
                # flag_action_contexts = False
            else:
                action_contexts = action_contexts['actioncontexts']
                print('\n -- [FOON-TAMP] : ' + str(len(action_contexts)) + ' action contexts have been loaded!')

        print('\n -- [FOON-TAMP] : Using action contexts! Loading MATLAB engine...', end='')

        if not matlab_engine:
            matlab_engine = matlab.engine.start_matlab()
        print(' done!')
        
        # -- setting the path of the MATLAB engine to the location of the action contexts dataset
        #       and motion planning code:
        matlab_engine.addpath('./dmp_code/coppelia_remoteAPI', nargout=0)
        matlab_engine.addpath('./dmp_code/functions/', nargout=0)
        matlab_engine.addpath('./dmp_code/data/', nargout=0)

        # -- we will either use a predefined port number (for multiple sims) or use default (-1):
        matlab_engine.connect_to_remote_API(port_nums[0] if port_nums[0] else -1, nargout=0)
    #endif

    # -- resetting the micro-problem history list:
    global micro_problem_history

    micro_problem_history = []

    # -- resetting the AC history list as well as list of all unseen action contexts:
    global AC_history, AC_requiring_demo

    AC_history = []
    AC_requiring_demo = []
#enddef


def _parseDomainPDDL(file_name):
    # NOTE: this function is used to parse a domain file with any type of planning operator.

    # -- this function adds them all to a list, and then return them to the main function.
    domain_file = list(open(file_name, 'r'))

    # -- a list of planning operators will be returned:
    planning_operators = []

    # -- variable name for newly creating PO instance:
    new_planning_operator = None

    for x in range(len(domain_file)):

        line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()

        if line.startswith(';'):
            # -- this is a comment; just ignore:
            pass

        elif line.startswith('(:action'):
            # -- this is the beginning of a planning operator (action) from PDDL file:
            macro_PO_name = line.split(' ')[1]

            # -- create new temp planning operator instance and add to list:
            new_planning_operator = PlanningOperator(macro_PO_name)
            planning_operators.append(new_planning_operator)

        elif line.startswith(':parameters') and not line.endswith(')'):
            # -- this is the beginning of the definition of parameters for this PO:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- splitting each line of a parameter based on variable name and type:
                parameter_parts = [X for X in line.split(' ') if X != '-']

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and len(parameter_parts) == 2:
                    # -- save the parameter to the new planning operator object:
                    new_planning_operator.addParameter(variable=parameter_parts[0], type=parameter_parts[1])

                x += 1

        elif line.startswith(':precondition'):
            # -- this is where we start reading the preconditions of the planning operator:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- remember that a predicate will be of the form: <relation> <obj1> <obj2>
                new_predicate = line

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and line.startswith('('):
                    # -- we need to check for potential *OR* operators in the preconditions:
                    if line.startswith('(or'):
                        # -- we will treat an *or* structure as a single precondition list with its arguments:
                        or_predicates = []

                        x += 1
                        while True:
                            # -- properly tokenize and clean up the line:
                            line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                            if line.startswith(')'):
                                break
                            x += 1

                            or_predicates.append(line)

                        # -- add the list of or-predicates to the list of preconditions:
                        new_planning_operator.addPrecondition(or_predicates)
                    else:
                        # -- just add this predicate as normal:
                        new_planning_operator.addPrecondition(new_predicate)

                # -- move onto the next line of the file:
                x += 1

        elif line.startswith(':effect'):
            # -- this is where we start reading the effects (post-conditions) of the planning operator:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- remember that a predicate will be of the form: <relation> <obj1> <obj2>
                new_predicate = line

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and line.startswith('('):
                    new_planning_operator.addEffect(new_predicate)

                # -- move onto the next line of the file:
                x += 1

        else:
            pass

    #endfor

    return planning_operators
#enddef


def _defineMicroProblem(macro_PO_name, preconditions, effects, is_hand_empty=True):
    global flag_perception
    if flag_perception: _runPerception()

    global on_object_layout, under_object_layout

    # -- create the file we are going to write to:
    global micro_problems_dir 
    file_name = str(micro_problems_dir) + '/' + str(macro_PO_name) + '_problem.pddl'
    pddl_file = open(file_name, 'w')

    pddl_file.write('(define (problem ' + macro_PO_name + ')\n')
    pddl_file.write('(:domain FOON_micro)\n')
    pddl_file.write('(:init' + '\n')

    # -- dictionary that will ground object names from conceptual (domain-independent) space to
    # 	simulated (domain-specific) space:
    grounded_objects = {}

    # NOTE: there are some predicates that only exist on the micro-PO level;
    #	these need to be added manually here.
    if is_hand_empty and not flag_perception:
        pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
        # -- hand should be empty (i.e., contains 'air')
        pddl_file.write('\t' + '(in hand air)' + '\n')
    elif 'hand' in in_object_layout:
        if in_object_layout['hand'] == 'air':
            # -- hand should be empty (i.e., contains 'air')
            pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
        else:
            # -- hand is not empty for whatever reason:
            pddl_file.write('\t; hand/end-effector contains some type of object:\n')

        pddl_file.write('\t' + '(in hand ' + str(in_object_layout['hand']) +')' + '\n')
    #end

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

    free_objects = all_objects - on_objects

    if flag_perception:
        # -- we will write the current object layout as obtained from the "perception" step.

        written_predicates = []

        pddl_file.write(
            '\n\t; object layout from current state of environment (from perception):\n')
        
        sorted_objects = list(on_object_layout.keys())
        for obj in sorted_objects:
            # -- writing the "on" relation predicate:
            if obj in ['mixing_bowl']:
                # -- if there is an object on top of a container('s base),
                #       we will consider it as being "contained" inside of the object:
                on_pred = str('(in ' + obj + ' ' + on_object_layout[obj] + ')')
                pddl_file.write('\t' + on_pred + '\n')
        
                written_predicates.append(on_pred)

                on_pred = str('(on ' + obj + ' ' + 'air' + ')')
            else:
                if obj in ['cutting_board']:
                    # -- we need to determine if we should put "air" on the cutting board
                    #       so it can be picked up for specific actions:
                    air_on_top = False
                    for effect in all_macro_POs[macro_PO_name].getEffects():
                        pred_parts = effect.strip()[1:-1].split(' ')
                        if pred_parts[0] == 'on' and pred_parts[1] == 'cutting_board' and pred_parts[2] == 'air':
                            air_on_top = True

                    if air_on_top:
                        on_pred = str('(on ' + obj + ' ' + 'air' + ')')
                        pddl_file.write('\t' + on_pred + '\n')
                        written_predicates.append(on_pred)

                on_pred = str('(on ' + obj + ' ' + on_object_layout[obj] + ')')
            #endif

            # -- write the predicate to the micro-problem file:
            pddl_file.write('\t' + on_pred + '\n')
            written_predicates.append(on_pred)

            if obj in under_object_layout:
                # -- writing the "under" relation predicate:
                under_pred = str('(under ' + obj + ' ' + under_object_layout[obj] + ')')

                # -- write the predicate to the micro-problem file:
                pddl_file.write('\t' + under_pred +'\n')
                written_predicates.append(under_pred)
            #endif
        # endfor

        # -- considering sizes of tiles and objects?
        pddl_file.write(
            '\n\t; precondition predicates based on tile and object sizes:\n')

        for obj in ftc.object_sizes:
            if ftc.object_sizes[obj] == 1:
                # -- this is a short object:
                pddl_file.write('\t' + '(is-short ' + obj + ')\n')
            elif ftc.object_sizes[obj] == 2:
                # -- this is a long/large (2x1 of short) object:
                pddl_file.write('\t' + '(is-long ' + obj + ')\n')
            else:
                # -- this is a wide (2x2 of short) object:
                pddl_file.write('\t' + '(is-wide ' + obj + ')\n')

        pddl_file.write(
            '\n\t; precondition predicates obtained directly from macro PO:\n')

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
                pddl_file.write('\t' + pred + '\n')

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
                    pddl_file.write('\t' + grounded_pred + '\n')
                    written_predicates.append(grounded_pred)

                    # -- we will also put the reverse fact that obj1 is under obj2 if it is in obj1
                    reversed_pred = str('(' + 'under' + ' ' + obj2 + ' ' + obj1 + ')')
                    pddl_file.write('\t' + reversed_pred + '\n')
                    written_predicates.append(reversed_pred)


            # endif
        # endfor

        # -- making some objects that are within containers (not referenced by Coppelia) free (putting "air" on them):
        # for obj in grounded_objects:
        #    if grounded_objects[obj] not in object_layout:
        #        pddl_file.write(
        #            '\t' + '(' + 'on' + ' ' + grounded_objects[obj] + ' ' + 'air' + ')\n')

    else:
        # -- also, complete the relations of the objects that have not been declared:
        pddl_file.write('\n\t' + '; precondition predicates obtained directly from macro PO:' + '\n')

        # -- the preconditions for a macro PO will form the initial states:
        for pred in preconditions:
            # -- without the perception part, we will just assume that all of the required states
            #	from the macro-PO are all that we have and need.
            pddl_file.write('\t' + pred + '\n')
        # endfor

        # -- we should want objects that are on the table to be free from collision (i.e., 'air' is on them)
        pddl_file.write(
            '\n\t; some objects that are on the table for manipulation should be free from collision (i.e. "air" on them):\n')

        for obj in free_objects:
            if obj in ['hand', 'air']:
                continue

            # -- we need to make a check to ensure that we do not include any conflicting predicates about
            #       air on top of an object that is stated to have something else on it based on macro-PO:
            found_similar_predicate = False
            for pred in preconditions:
                if pred.startswith(str('(on ' + obj)):
                    found_similar_predicate = True
                    break

            if found_similar_predicate:
                continue

            pddl_file.write('\t' + '(on ' + obj + ' ' + 'air' + ')\n')

        #endfor

        # -- just to allow POs that do not require table grid sizes:
        pddl_file.write('\n\t(no-perception)\n')

    # endif

    pddl_file.write(')\n\n')

    pddl_file.write('(:goal (and\n')

    # -- hand must be empty after executing action to free it for next macro-PO:
    if 'scoop' not in macro_PO_name:
        # -- however, if it is a scoop action, then the hand will container some tool (like a spoon)
        pddl_file.write('\t' + '; hand/end-effector must be also be empty after execution (i.e. contains "air"):' + '\n')
        pddl_file.write('\t' + '(in hand air)' + '\n\n')

    pddl_file.write('\t; effect predicates obtained directly from macro PO:\n')

    # -- the effects (post-conditions) for a macro PO will form the goals:

    for pred in effects:
        predicate_parts = pred[1:-1].split(' ')

        # -- we need to treat negation predicates special so we print them correctly:
        is_negation = False
        if predicate_parts[0] == 'not':
            is_negation = True
            predicate_parts = [predicate_parts[x]
                               for x in range(1, len(predicate_parts) - 1)]
            for x in range(len(predicate_parts)):
                # -- remove any trailing parentheses characters:
                predicate_parts[x] = predicate_parts[x].replace('(', '')
                predicate_parts[x] = predicate_parts[x].replace(')', '')

        if 'mix' in macro_PO_name and (len(set(predicate_parts[1:]) - set(all_objects)) > 0 or is_negation):
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

            if flag_perception:
                # -- if any of the referenced objects are "table", then we need to ground the table to the proper grid-space:
                if 'table' in [obj1, obj2]:
                    # -- this means that we have an "on" or "under" predicate if something is on the table:
                    is_table_obj1 = bool(obj1 == 'table')

                    found_mapping = False
                    for item in on_object_layout:
                        if on_object_layout[item] == (obj2 if is_table_obj1 else obj1):
                            if is_table_obj1:
                                obj1 = item
                            else:
                                obj2 = item
                            found_mapping = True
                            break
                    # endfor

                    if not found_mapping:
                        # -- if we could not find a grounding for the table in the scene, skip this predicate:
                        continue
                # endif
            # endif

            if is_negation:
                pddl_file.write('\t' + '(not '
                                + '(' + predicate_parts[0] + ' ' + obj1
                                + (str(' ' + obj2) if obj2 else '') + ') )\n')
            else:
                if flag_perception:
                    # -- we want to enforce that objects will be placed on tables:
                    if predicate_parts[0] == 'under' and obj1 in ftc.object_sizes:
                        if 'table' in predicate_parts[2] and 'table' not in obj2:
                            continue
                    elif predicate_parts[0] == 'on' and obj2 in ftc.object_sizes:
                        if 'table' in predicate_parts[1] and 'table' not in obj1:
                            continue

                pddl_file.write('\t' + '('
                                + predicate_parts[0] + ' ' + obj1
                                + (str(' ' + obj2) if obj2 else '') + ')\n')
            # endif
        # endif
    # endfor

    if flag_perception:
        # -- we will check for the last step manually (i.e., find the max):
        state_history[max(state_history.keys())+1] = {  'on' : on_object_layout, 
                                                        'under' : under_object_layout}

    pddl_file.write('))\n')
    pddl_file.write('\n)')

    # -- make sure to close the file after writing to it:
    pddl_file.close()

    return file_name
# enddef


def _runPerception(connect_to_sim=False):
    # -- we will note the layout of objects from CoppeliaSim's side, if available:

    global flag_collision, flag_skip_execution

    def _runPerception_no_sim():
        # -- using the micro-problem history, we will load the problem file used before
        #       to manually update the state of the environment:
        global on_object_layout, under_object_layout, in_object_layout, table_layout

        global micro_problem_history

        # NOTE: we will only randomly configure and load the scene CoppeliaSim at the START!

        try:
            micro_file_name = micro_problem_history[-1]
        except IndexError:
            # -- this means that we did not have an initial load (as referenced in the note above)
            # -- therefore, we will just run the simulation-based perception once:
            _runPerception_with_sim()
        else:
            # -- read the contents of the micro-problem file and parse through the goals:
            micro_problem_file = open(micro_file_name)
            micro_file_contents = micro_problem_file.read().split('(:goal ')[1].split('\n')

            for G in micro_file_contents:
                # -- strip the string of any trailing tab-spaces or white-spaces:
                goal = G.strip()
                if len(goal) > 2 and goal[0] == '(' and goal[-1] == ')':
                    # -- this is a valid predicate:
                    pred_parts = goal.replace('(', '').replace(')', '').split(' ')

                    if 'on' in pred_parts[0]:
                        # -- get the former place where the object was before:
                        old_location = under_object_layout[pred_parts[2]] if pred_parts[2] in under_object_layout else None
                        try:
                            on_object_layout[pred_parts[1]] = pred_parts[2]
                            under_object_layout[pred_parts[2]] = pred_parts[1] if pred_parts[2] != 'air' else under_object_layout[pred_parts[2]]
                        except KeyError:
                            pass

                        if old_location and pred_parts[1] != old_location:
                            on_object_layout[old_location] = 'air'

                        if on_object_layout[pred_parts[2]] == pred_parts[1]:
                            print('la')
                            # -- this is for the drinking glass, which can be upside down:
                            on_object_layout[pred_parts[2]] = 'air'

                    elif 'under' in pred_parts[0]:
                        if pred_parts[1] in under_object_layout and under_object_layout[pred_parts[1]] != pred_parts[2]:
                            under_object_layout[pred_parts[1]] = pred_parts[2]

                    elif 'in' in pred_parts[0]:
                        if pred_parts[1] in in_object_layout:
                            if isinstance(in_object_layout[pred_parts[1]], list):
                                in_object_layout[pred_parts[1]].append(pred_parts[2])
                                in_object_layout[pred_parts[1]] = list(set(in_object_layout[pred_parts[1]]))
                            else:
                                in_object_layout[pred_parts[1]] = [pred_parts[2]] if pred_parts[2] != 'air' else 'air'
                        else:
                            in_object_layout[pred_parts[1]] = [pred_parts[2]] if pred_parts[2] != 'air' else 'air'

                    elif 'not' in pred_parts[0] and 'in' in pred_parts[1]:
                        if pred_parts[2] in in_object_layout:
                            if isinstance(in_object_layout[pred_parts[2]], list):
                                try:
                                    in_object_layout[pred_parts[2]].remove(pred_parts[3])
                                except ValueError:
                                    pass
                                else:
                                    if bool(in_object_layout[pred_parts[2]]):
                                        in_object_layout[pred_parts[2]] = 'air'

                            else:
                                in_object_layout[pred_parts[2]] = 'air'
                    #endif
                #endif
            #endfor

            print('in', in_object_layout)
            print('on', on_object_layout)
            print('under', under_object_layout)

            micro_plan_file = open(str(micro_file_name.split('_problem.')[0] + '_micro.plan'), 'r')
            micro_plan_contents = micro_plan_file.readlines()

            for G in micro_plan_contents:
                if G.startswith('(place'):
                    step_parts = G.strip()[1:-1].split(' ')
                    try:
                        on_object_layout[under_object_layout[step_parts[1]]] = 'air'
                    except KeyError:
                        pass
                    try:
                        on_object_layout[step_parts[2]] = step_parts[1]
                    except KeyError:
                        pass
                    try:
                        under_object_layout[step_parts[1]] = step_parts[2]
                    except KeyError:
                        pass


    def _runPerception_with_sim():    
        # -- get the layout of table cells and objects from scene in CoppeliaSim:
        global on_object_layout, under_object_layout, in_object_layout, table_layout

        # -- get the current state of the simulated environment as dictionary:
        current_state = ftc._getSceneState(check_collision=flag_collision)
        if 'on' in current_state:
            on_object_layout = current_state['on']
        if 'under' in current_state:
            under_object_layout = current_state['under']
        if 'in' in current_state:
            in_object_layout = current_state['in']

        # -- get the layout of the scene based on the grid-map provided in the JSON:
        table_layout = ftc.table_grid_map

    if not flag_skip_execution or connect_to_sim:
        _runPerception_with_sim()
    else:
        # -- in this case, we will manually update the state of the environment 
        #       based on the effects of the last-defined micro-problem.
        _runPerception_no_sim()
    
    return
#endef


def _checkStateSatisfiability(prev_action=None, next_action=None):
    # NOTE: this function is to be used in conjunction with perception to make sure that we have 
    #       satisfied the *effects* of the previous action and the *preconditions* of the following action.

    # -- we have two possible situations:
    #   1. intra-macro execution :- we have executed one micro-action and we are about to execute another;
    #   2. inter-macro execution :- we have finished the last micro-action of a macro-PO,
    #                               and we are moving to another macro-PO

    global all_macro_POs, all_micro_POs

    global on_object_layout, in_object_layout, under_object_layout

    def _check_if_effects_met():
        if prev_action is None:
            return True

        prev_parts = prev_action[1:-1].split(' ')
        prev_name, prev_instances = prev_parts[0], prev_parts[1:]

        prev_PO_definition = all_micro_POs[prev_name]
        _, prev_PO_args = prev_PO_definition.getParameters()

        intended_effects = []

        for pred in prev_PO_definition.getEffects():
            # -- make a copy of the predicate for editing:
            grounded_effect = str(pred)
            for x in range(len(prev_PO_args)):
                if prev_PO_args[x] in grounded_effect:
                    grounded_effect = grounded_effect.replace(prev_PO_args[x], prev_instances[x])

            intended_effects.append(grounded_effect)

        # -- now, we'll check if the preconditions and effects have been met, 
        #       reflected by the current state of the environment:
        unmet_states = []

        for pred in intended_effects:
            # -- check if this current state has been satisfied:
            state_satisfied = True

            pred_parts = pred[1:-1].split(' ')
            if len(pred_parts) < 3:
                # -- this is something non-geometrical, so we can skip this predicate:
                continue

            elif 'in' in pred_parts[0]:
                if pred_parts[1] not in in_object_layout:
                    state_satisfied = False

                # -- we can check if an object is *NOT* listed either in the "in_object_layout" or "under_object_layout" dicts:
                elif pred_parts[2] not in in_object_layout[pred_parts[1]]:
                    if pred_parts[2] != 'air' and pred_parts[1] not in under_object_layout[pred_parts[2]]:
                        state_satisfied = False

            elif 'on' in pred_parts[0]:
                if pred_parts[2] == 'hand':
                    if under_object_layout[pred_parts[2]] != pred_parts[1]:
                        state_satisfied = False
                elif on_object_layout[pred_parts[1]] != pred_parts[2]:
                    state_satisfied = False

            elif 'under' in pred_parts[0] and under_object_layout[pred_parts[1]] != pred_parts[2]:
                state_satisfied = False

            if not state_satisfied:
                unmet_states.append(pred_parts)

        if unmet_states:
            print('unmet effects of ' + prev_action)
            print(unmet_states)

        return not(len(unmet_states))
    #enddef

    def _check_if_preconditions_met():

        grounded_objects = {}

        def _state_satisfiability_precond(pred):
            # NOTE: subroutine for checking if preconditions have been met:
            # -- this was defined since we can have *or* statements, which require sub-iterative checks:

            pred_parts = pred[1:-1].split(' ')

            if len(pred_parts) < 3:
                # -- this is something non-geometrical, so we can skip this predicate:
                return True

            obj1, obj2 = pred_parts[1], pred_parts[2]
            if grounded_objects:
                obj1, obj2 = grounded_objects[pred_parts[1]], grounded_objects[pred_parts[2]]

            if 'table' in pred_parts:
                # -- if we are transitioning to executing a macro-action, we may have ungrounded objects:

                # NOTE: in a macro-PO, objects are stated to be on the table by default; however, in the micro-PO level,
                #       objects can be configured differently in the scene (either on table grids or on other objects)
                # -- therefore, we should be checking if the object exists *ANYWHERE* in the scene (i.e., not in "air")

                if pred_parts[0] == 'under' and 'air' in under_object_layout[obj1]:
                    # -- anywhere but "air":
                    return False
                elif pred_parts[1] == 'on' and obj2 not in on_object_layout:
                    return False

            elif 'in' in pred_parts[0]:
                if obj1 not in in_object_layout:
                    return False
                
                # -- we can check if an object is *NOT* listed either in the "in_object_layout" or "under_object_layout" dicts:
                if obj2 not in in_object_layout[obj1] and obj1 not in under_object_layout[obj2]:
                    return False

            elif 'on' in pred_parts[0]:
                if pred_parts[2] == 'hand':
                    if under_object_layout[obj2] != obj1:
                        return False
                elif on_object_layout[obj1] != obj2:
                    return False

            elif 'under' in pred_parts[0] and under_object_layout[obj1] != obj2:
                return False

            return True
        #enddef

        if next_action is None:
            return True

        next_parts = next_action[1:-1].split(' ')
        next_name, next_instances = next_parts[0], next_parts[1:]

        intended_preconditions = []

        next_PO_definition = all_macro_POs[next_name] if next_name in all_macro_POs else all_micro_POs[next_name]
        _, next_PO_args = next_PO_definition.getParameters()

        # NOTE: object names in macro-POs require some grounding to their corresponding object in the simulation:
        if next_name in all_macro_POs:
            for pred in next_PO_definition.getPreconditions():
                pred_parts = pred[1:-1].split(' ')

                # -- we only look for predicate types with 2 arguments:
                if len(pred_parts) < 3:
                    continue

                grounded_objects[pred_parts[1]] = pred_parts[1]
                grounded_objects[pred_parts[2]] = pred_parts[2]

                # -- we can have labels such as "cup_lemon_juice", which is "cup" + "_" + "lemon_juice":
                if str(pred_parts[1] + '_' + pred_parts[2]) in on_object_layout:
                    grounded_objects[pred_parts[1]] = str(pred_parts[1] + '_' + pred_parts[2])
                elif str(pred_parts[2] + '_' + pred_parts[1]) in on_object_layout:
                    grounded_objects[pred_parts[2]] = str(pred_parts[2] + '_' + pred_parts[1])
        #endif

        for pred in next_PO_definition.getPreconditions():
            # -- make a copy of the predicate for editing:
            grounded_preconds = []

            pred_list = None
            if not isinstance(pred, list):
                pred_list = [str(pred)]
            else:
                pred_list = [str(X) for X in pred]

            for subpred in pred_list:
                grounded_precond = str(subpred)
                for x in range(len(next_PO_args)):
                    if next_name not in all_macro_POs:
                        if next_PO_args[x] in subpred:
                            grounded_precond = grounded_precond.replace(next_PO_args[x], next_instances[x])

                # -- add grounded precondition statements to a list...
                grounded_preconds.append(grounded_precond)

            intended_preconditions.append(grounded_preconds[0] if len(grounded_preconds) == 1 else grounded_preconds)

        # -- now, we'll check if the preconditions and effects have been met, 
        #       reflected by the current state of the environment:
        unmet_states = []

        for pred in intended_preconditions:
            # -- check if this current state has been satisfied:
            state_satisfied = True

            if isinstance(pred, list):
                # -- this means we have an *or* structure with options between predicates:
                any_matches = 0

                for subpred in pred:
                    # -- recall that in an *or* structure, we have a list of predicates:
                    state_satisfied = _state_satisfiability_precond(subpred)
                    if state_satisfied:
                        any_matches += 1

                # -- with an or-statement, we just need to satisfy at least one condition:
                if bool(any_matches):
                    state_satisfied = True
                else:
                    state_satisfied = False
            else:
                state_satisfied = _state_satisfiability_precond(pred)

            if not state_satisfied:
                unmet_states.append((['OR:'] + pred) if isinstance(pred, list) else pred)

        if unmet_states:
            print('unmet preconditions of ' + next_action)
            print(unmet_states)

        return not(len(unmet_states))
    #enddef

    if not (bool(_check_if_effects_met() and _check_if_preconditions_met())) and verbose:
        print(in_object_layout)
        print(on_object_layout)
        print(under_object_layout)

    if not prev_action:
        # -- solely evaluate whether the preconditions of the about-to-be-executed action have been met or not:
        return _check_if_preconditions_met()
    
    if not next_action:
        # -- solely evaluate whether the effects of the last executed action have been met or not:
        return _check_if_effects_met()

    return bool(_check_if_effects_met() and _check_if_preconditions_met())
#enddef


def _generalizeActionContext(table_layout, under_object_layout, action_now, action_prev, action_next):
    # NOTE: the purpose of this function is to facilitate translation of action contexts
    #   to a "generalized" format for finding non-exact but similar action contexts for adaptation and execution.
    # -- each coordinate is in the form of (X, Y)!

    coordinates = {'action_prev': None, 'action_now': (0, 0), 'action_next': None}
    locations = {'action_prev': None, 'action_now': None, 'action_next': None}

    import math

    now_obj, prev_obj, next_obj = None, None, None
 
    now_obj_location, prev_obj_location, next_obj_location = [], [], []

    # -- selecting the appropriate target object:
    action_now_parts = action_now[1:-1].split(' ')
    now_obj = action_now_parts[-1]
    while now_obj in under_object_layout:
        # -- the target object (not surface) is given by the last argument;
        #       use the layout from perception to determine target location:
        now_obj = under_object_layout[now_obj]

    locations['action_now'] = now_obj

    for X in range(len(table_layout)):
        for Y in range(len(table_layout[X])):
            if table_layout[X][Y] == now_obj:
                now_obj_location.append((X, Y))

    tip_location = ftc.robot_start_location if not flag_ignore_start else None

    # -- we will be identifying relative location w.r.t. origin (location of *action_now* target)
    #       for both *action_prev* and *action_next* target locations:

    if action_prev or tip_location:
        
        if not action_prev:
            # -- here, we will deal with the gripper's starting position 
            #       (based on a special tile), since the previous action is "None":
            prev_obj = tip_location
        else:
            # -- selecting the appropriate target object 
            #       (i.e., looking at the apposite table tile):
            action_prev_parts = action_prev[1:-1].split(' ')
            prev_obj = action_prev_parts[-1]
            while prev_obj in under_object_layout:
                # -- the target object (not surface) is given by the last argument;
                #       use the layout from perception to determine target location:
                prev_obj = under_object_layout[prev_obj]

        # NOTE: this would only be important for handling the coordinates for the long tiles,
        #       which span across the entirety of the y-axis of the table.
        for X in range(len(table_layout)):
            for Y in range(len(table_layout[X])):
                if table_layout[X][Y] == prev_obj:
                    prev_obj_location.append((X, Y))

        # -- coordinates will temporarily be in the form of (X, Y, euclidean distance from origin)!
        for PL in prev_obj_location:
            for NL in now_obj_location:
                if not coordinates['action_prev']:
                    distance = math.sqrt(
                        (NL[0] - PL[0]) ** 2 + (NL[1] - PL[1]) ** 2)
                    coordinates['action_prev'] = (
                        NL[0] - PL[0], NL[1] - PL[1], distance)
                else:
                    # -- measure distance for more representative coordinates:
                    distance = math.sqrt(
                        (NL[0] - PL[0]) ** 2 + (NL[1] - PL[1]) ** 2)
                    if distance < coordinates['action_prev'][2]:
                        coordinates['action_prev'] = (
                            NL[0] - PL[0], NL[1] - PL[1], distance)

        # print('prev: ' + prev_obj + ' - ' + str(prev_obj_location))

        if coordinates['action_prev']:
            # -- store the relative coordinates only if there was something calculated:
            coordinates['action_prev'] = (
                coordinates['action_prev'][0], coordinates['action_prev'][1])
            locations['action_prev'] = prev_obj        
    #endif

    if action_next:
        # -- selecting the appropriate target object:
        action_next_parts = action_next[1:-1].split(' ')
        next_obj = action_next_parts[-1]
        while next_obj in under_object_layout:
            # -- the target object (not surface) is given by the last argument;
            #       use the layout from perception to determine target location:
            next_obj = under_object_layout[next_obj]

        # NOTE: this would only be important for handling the coordinates for the long tiles,
        #       which span across the entirety of the y-axis of the table.
        for X in range(len(table_layout)):
            for Y in range(len(table_layout[X])):
                if table_layout[X][Y] == next_obj:
                    next_obj_location.append((X, Y))

        # -- coordinates will temporarily be in the form of (X, Y, euclidean distance from origin)!
        for PL in next_obj_location:
            for NL in now_obj_location:
                if not coordinates['action_next']:
                    distance = math.sqrt(
                        (NL[0] - PL[0]) ** 2 + (NL[1] - PL[1]) ** 2)
                    coordinates['action_next'] = (
                        NL[0] - PL[0], NL[1] - PL[1], distance)
                else:
                    # -- measure distance for more representative coordinates:
                    distance = math.sqrt(
                        (NL[0] - PL[0]) ** 2 + (NL[1] - PL[1]) ** 2)
                    if distance < coordinates['action_next'][2]:
                        coordinates['action_next'] = (
                            NL[0] - PL[0], NL[1] - PL[1], distance)

        # print('next: ' + next_obj + ' - ' + str(next_obj_location))

        if coordinates['action_next']:
            # -- store the relative coordinates only if there was something calculated:
            coordinates['action_next'] = (
                coordinates['action_next'][0], coordinates['action_next'][1])
            locations['action_next'] = next_obj
    #endif

    return coordinates, locations
#enddef


def _parseActionContexts(A):
    pred_now, pred_prev, pred_next = None, None, None

    # -- translating the 'action_prev' field from AC to the predicate format:
    action_arguments = ''
    try:
        for O in action_contexts[A].action_prev.args.tolist():
            action_arguments += (O + ' ') if isinstance(O, str) else ''
        if action_arguments:
            # -- generalizing different place actions to the same label:
            action_name = 'place' if 'place' in str(action_contexts[A].action_prev.name) else str(action_contexts[A].action_prev.name)
            pred_prev = '(' + action_name + ' ' + action_arguments.strip() + ')'
    except AttributeError:
        pred_prev = False

    # -- translating the 'action_now' field from AC to the predicate format:
    action_arguments = ''
    try:
        for O in action_contexts[A].action_now.args.tolist():
            action_arguments += (O + ' ') if isinstance(O, str) else ''
        if action_arguments:
            # -- generalizing different place actions to the same label:
            action_name = 'place' if 'place' in str(action_contexts[A].action_now.name) else str(action_contexts[A].action_now.name)
            pred_now = '(' + action_name + ' ' + action_arguments.strip() + ')'
    except AttributeError:
        pred_now = False

    # -- translating the 'action_next' field from AC to the predicate format:
    action_arguments = ''
    try:
        for O in action_contexts[A].action_next.args.tolist():
            action_arguments += (O + ' ') if isinstance(O, str) else ''
        if action_arguments:
            # -- generalizing different place actions to the same label:
            action_name = 'place' if 'place' in str(action_contexts[A].action_next.name) else str(action_contexts[A].action_next.name)
            pred_next = '(' + action_name + ' ' + action_arguments.strip() + ')'
    except AttributeError:
        pred_next = False

    return pred_now, pred_prev, pred_next
#enddef


def _findPlan(domain_file, problem_file, output_plan_file=None):
    # NOTE: define plan execution function that can be called for different parameters:
   
    global path_to_planners, planner_to_use, configs, algorithm, heuristic

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

    try:
        planner_output = subprocess.check_output(command)
    except subprocess.CalledProcessError:
        print('  -- [FOON-TAMP] : Error with planner execution! (planner used: ' + planner_to_use + ')')
        pass
    # endtry

    return planner_output
# enddef


def _checkPlannerOutput(output):
    # -- using planner-specific string checking to see if a plan was found:
    global planner_to_use
    if planner_to_use == 'fast-downward':
        return True if 'Solution found.' in str(output) else False
    elif planner_to_use == 'PDDL4J':
        # TODO: implement integration with other planners such as PDDL4J
        return False
    else:
        return False
#enddef


def main():
    global planner_to_use
    
    # NOTE: several important flags that will control the flow of planning and execution:
    global flag_perception, flag_skip_execution, flag_collision, flag_skip_demo
    global flag_action_contexts, action_contexts
 
    # -- execute function for preliminary operations (e.g., loading the scene):
    _preliminaries()

    global all_macro_POs, all_micro_POs

    ## -- parse through the newly created domain file and find all (macro) planning operators:
    all_macro_POs = {PO.name : PO for PO in _parseDomainPDDL(ftp.FOON_domain_file)}

    ## -- parse through all micro-POs defined in micro-domain and add them to this dictionary:
    all_micro_POs = {PO.name : PO for PO in _parseDomainPDDL(micro_domain_file)}

    print('\n -- [FOON-TAMP] : Initiating macro-level planning with ' + str(planner_to_use) + ' planner...')

    # -- checking to see if a macro level plan has been found (FD allows exporting to a file):
    outcome = _findPlan(
        domain_file=ftp.FOON_domain_file,   # NOTE: FOON_to_PDDL object will already contain the names of the 
        problem_file=ftp.FOON_problem_file,  #   corresponding macro-level domain and problem files
        output_plan_file=macro_plan_file
    )

    # TODO: write planning pipeline for other planners? 

    if _checkPlannerOutput(output=outcome):
        print("  -- [FOON-TAMP] : A macro-level plan for '" + str(FOON_subgraph_file) + "' was found!")

        # -- now that we have a plan, we are going to parse it for the steps:
        macro_file = open(macro_plan_file, 'r')

        # NOTE: counters for macro- and micro-level steps:
        macro_count, micro_count = 0, 0

        macro_plan_lines = list(macro_file)
        for L in macro_plan_lines:
            if L.startswith('('):
                print('\t\t\t' + str(macro_count) + ' : ' + L.strip())
            macro_count += 1
        print()

        macro_count = 0

        num_failures = 0

        # -- num_exact_AC_matches :- count of the number of action contexts 
        #           used that matched an exact context in our AC dataset:
        num_exact_AC_matches = 0

        # -- num_similar_AC_matches :- count of the number of action contexts 
        #           used that matched a similar context in our AC dataset:
        num_similar_AC_matches = 0

        # -- num_demoed_ACs :- count of the number of action contexts 
        #           that needed to be demonstrated in this trial:
        num_demoed_ACs = 0

        # -- this flag is used to keep track of whether the last executed PO required the hand being empty:
        flag = True

        # -- compile list of all steps from each FOON micro plan:
        complete_micro_plan = []

        total_cost = 0

        action_now, action_prev, action_next = None, None, None

        if flag_perception:
            print('-- [FOON-TAMP] : Reading INITIAL state of the environment (i.e., τ=0)...')
            _runPerception()

            global matlab_engine

            # -- get the initial state of the environment:
            state_history[micro_count] = {'on' : on_object_layout, 'under' : under_object_layout}
        
        print('\n--------------------------------------------------------' 
            + '--------------------------------------------------------\n')

        for L in range(len(macro_plan_lines)):

            if macro_plan_lines[L].startswith('('):
                # NOTE: this is where we have identified a macro plan's step; here, we check the contents of its PO definition for:
                #	1. preconditions - this will become a sub-problem file's initial states (as predicates)
                #	2. effects - this will become a sub-problem file's goal states (as predicates)

                macro_count += 1

                macro_PO_name = macro_plan_lines[L][1:-2].strip()
 
                print(" -- [FOON-TAMP] : Searching for micro-level plan for '" + macro_PO_name + "' macro-PO...")

                # -- try to find this step's matching planning operator definition:
                matching_PO_obj = all_macro_POs[macro_PO_name] if macro_PO_name in all_macro_POs else None

                # -- when we find the equivalent planning operator, then we proceed to treat it as its own problem:
                if matching_PO_obj:
                    # -- create sub-problem file (i.e., at the micro-level):
                    micro_problem_file = _defineMicroProblem(
                        macro_PO_name, 
                        preconditions=matching_PO_obj.getPreconditions(), 
                        effects=matching_PO_obj.getEffects(), 
                        is_hand_empty=flag
                    )

                    complete_micro_plan.append('; step ' + str(macro_count) + ' -- (' + macro_PO_name + '):')

                    need_to_replan = False

                    # -- try to find a sub-problem plan / solution:
                    outcome = _findPlan(
                        domain_file=micro_domain_file, 
                        problem_file=micro_problem_file, 
                        output_plan_file=str(micro_problems_dir + '/' + macro_PO_name + '_micro.plan')
                    )

                    print('\n\t' + 'step ' + str(macro_count) +' -- (' + macro_PO_name + ')')

                    if _checkPlannerOutput(outcome):

                        # -- saving history of micro-problem files:
                        micro_problem_history.append(micro_problem_file)

                        print('\t\t -- micro-level plan found as follows:')

                        # -- open the micro problem file, read each line referring to a micro PO, and save to list:
                        micro_file = open(
                            str(micro_problems_dir + '/' + macro_PO_name + '_micro.plan'), 'r')

                        # -- all except for the last line should be valid steps:
                        micro_file_lines = []
                        count = 0

                        for micro_line in micro_file:
                            if micro_line.startswith('('):
                                # -- parse the line and remove trailing newline character:
                                micro_step = micro_line.strip()
                                micro_file_lines.append(micro_step)

                                # -- print entire plan to the command line in format of X.Y, 
                                #       where X is the macro-step count and Y is the micro-step count:
                                count += 1
                                print('\t\t\t' + str(macro_count) + '.' + str(count) + ' : ' + micro_step)
                            else:
                                # -- this line will be in the format "; cost = X (unit cost)\n", 
                                #       where X is the total number of micro-steps in the file:
                                total_cost += int(micro_line.split(' ')[3])
                            #endif
                        #endfor

                        micro_file.close()

                        if flag_perception:
                            print('\n\t\t -- Executing micro plan...')

                            print('\n--------------------------------------------------------' 
                                + '--------------------------------------------------------')

                        for x in range(len(micro_file_lines)):
                            # -- set the current action as *action_now*:
                            action_now = micro_file_lines[x]

                            # -- set the following action as *action_next*
                            #       (however, if there is no following action, set it to null):
                            action_next = micro_file_lines[x+1].strip() if x+1 < len(micro_file_lines) else None

                            complete_micro_plan.append(action_now)

                            # -- keep track of the number of steps that have been executed:
                            micro_count += 1

                            # -- if a plan has been found and we have perception active,
                            #       then we should be executing actions at the same time:
                            if flag_perception:

                                print('-- [FOON-TAMP] : Attempting to execute action "' + action_now + '"...')

                                if not flag_skip_execution:
                                    print(' -- Checking if action preconditions are satisfied...')
    
                                    # -- running perception to check the present state of the environment:
                                    _runPerception()

                                    if verbose:
                                        print(in_object_layout)
                                        print(on_object_layout)
                                        print(under_object_layout)

                                    # -- now, we check if we satisfy the requirements of the action the robot 
                                    #       is about to execute (i.e., "action_now"):
                                    are_states_satisfied =_checkStateSatisfiability(next_action=action_now)

                                    if not are_states_satisfied:
                                        print(are_states_satisfied)

                                        # -- we need to trigger replanning at this point:

                                        # TODO: think about replanning!
                                        need_to_replan = True

                                this_AC_coord, this_AC_table_cells = _generalizeActionContext(table_layout, under_object_layout, action_now, action_prev, action_next)
                                while True:
                                    # -- if something is up with this AC due to some perception issue, run perception again:
                                    if (action_prev is not None and this_AC_coord['action_prev'] is not None) or (action_now is not None and this_AC_coord['action_now'] is not None):
                                        break

                                    _runPerception()
                                    this_AC_coord, this_AC_table_cells = _generalizeActionContext(table_layout, under_object_layout, action_now, action_prev, action_next)

                                    if verbose:
                                        print(in_object_layout)
                                        print(on_object_layout)
                                        print(under_object_layout)

                                # print('\naction context:')
                                print('\t\t<' + str(action_prev) + ',')
                                print('\t\t ' + str(action_now) + ',')
                                print('\t\t ' + str(action_next) + '>\n')
                                print('\t\t ' + str(this_AC_coord) + '\n')

                                # -- getting gripper directive from the motion's name:
                                if 'pick' in action_now:
                                    gripper_action = 1
                                elif 'place' in action_now or 'insert' in action_now:
                                    gripper_action = -1
                                else:
                                    # -- other motions such as mix, pour, sprinkle, etc. require the object to remain
                                    #       enclosed in the hand:
                                    gripper_action = 0
                                
                                action_now_parts = action_now[1:-1].split(' ')
                                if action_now in ['mix', 'stir']:
                                    # -- change the property of the container being stirred (for sake of simulation):
                                    ftc._changeObjectProperty(action_now_parts[-1], 'respondable', 0)
                                else:
                                    ftc._changeObjectProperty(action_now_parts[-1], 'respondable', 1)

                                trajectory_data = None

                                if flag_action_contexts:
                                    # -- first, try to load the action contexts file:

                                    exact_action_context = -1

                                    for A in range(len(action_contexts)):

                                        # NOTE: in each action context (AC) from the .MAT file created from Alejandro's pipeline,
                                        #   there will be DMPs associated to tuples <action_prev, action_now, action_next>
                                        pred_now, pred_prev, pred_next = _parseActionContexts(A)

                                        if pred_now == False or pred_prev == False or pred_next == False:
                                            continue

                                        if not action_next or not action_prev:
                                            # -- if there is a valid *action_next* that follows *action_now*,
                                            #       then pred_next will not be changed:
                                            pred_next = None

                                        if action_prev == pred_prev and action_now == pred_now and action_next == pred_next:
                                            if 'coord' in set(action_contexts[A].action_prev._fieldnames) & set(action_contexts[A].action_next._fieldnames):
                                                existing_AC_prev_coord = action_contexts[A].action_prev.coord if isinstance(action_contexts[A].action_prev.coord, str) else None
                                                existing_AC_next_coord = action_contexts[A].action_next.coord if isinstance(action_contexts[A].action_next.coord, str) else None
                                                if str(existing_AC_prev_coord) != str(this_AC_coord['action_prev']):
                                                    continue
                                                elif str(existing_AC_next_coord) != str(this_AC_coord['action_next']):
                                                    continue
                                                #endif
                                            #endif

                                            # -- we found the correct action context, so we select this one:
                                            print('   -- Found exact action context (# ' + str(A+1) + ')!')

                                            this_ac = {'action_now' : {'args' : None, 'name' : None}, 'dmps' : {'weights' : None, 'ini' : None, 'end' : None}}
                                            this_ac['action_now']['name'] = action_now_parts[0]
                                            this_ac['action_now']['args'] = [action_now_parts[X] for X in range(1, len(action_now_parts))]
                                            this_ac['dmps']['weights'] = matlab.double(action_contexts[A].dmps.weights.tolist())
                                            this_ac['dmps']['ini'] = matlab.double(action_contexts[A].dmps.ini.tolist())
                                            this_ac['dmps']['end'] = matlab.double(action_contexts[A].dmps.end.tolist())
                                            trajectory_data = matlab_engine.motion_planning(this_ac)
                                            
                                            exact_action_context = A
                                            
                                            num_exact_AC_matches += 1

                                            break
                                        #endif
                                    #endfor

                                    if trajectory_data is not None:
                                        # -- execute the trajectory represented as a DMP:
                                        if not flag_skip_execution:
                                            print('  -- Executing action "' + action_now + '" (with AC #' + str(exact_action_context+1) + ')!')
                                            ftc._replayAction_DMP(trajectory_data, gripper_action)

                                        # -- add used action context to the history:
                                        AC_history.append(exact_action_context+1)

                                    else:
                                        # -- this means that there was no exact action context found;
                                        #       we will find something that is similar to a previously seen AC based on tiles.

                                        print('   -- Exact AC not found! Finding alternative...')

                                        # NOTE: we will use the following ideas:
                                        # -- we will treat the location at *action_now* as coordinate (0,0).
                                        # -- depending on the type of action, we will infer the target object for its location.
                                        # -- we get the locations from the *table_layout* dictionary.

                                        this_action_name = action_now[1:-1].split(' ')[0]
                                        prev_action_name = action_prev[1:-1].split(' ')[0] if action_prev else None

                                        similar_action_context = -1

                                        # -- formatting new action context in format required:
                                        this_ac = {'action_now' : {}, 'action_prev': {}, 'action_next': {}, 'dmps' : {'weights' : [], 'ini' : [0] * 6, 'end' : [0] * 6}}
                                        action_now_parts = action_now[1:-1].split(' ')
                                        action_prev_parts = action_prev[1:-1].split(' ') if action_prev else ['', '']
                                        action_next_parts = action_next[1:-1].split(' ') if action_next else ['', '']

                                        this_ac['action_now']['name'] = action_now_parts[0]
                                        this_ac['action_now']['args'] = [action_now_parts[X] for X in range(1, len(action_now_parts))]
                                        this_ac['action_now']['coord'] = str(this_AC_coord['action_now'])

                                        this_ac['action_prev']['name'] = action_prev_parts[0]
                                        this_ac['action_prev']['args'] = [action_prev_parts[X] for X in range(1, len(action_prev_parts))]
                                        this_ac['action_prev']['coord'] = str(this_AC_coord['action_prev'])

                                        this_ac['action_next']['name'] = action_next_parts[0]
                                        this_ac['action_next']['args'] = [action_next_parts[X] for X in range(1, len(action_next_parts))]
                                        this_ac['action_next']['coord'] = str(this_AC_coord['action_next'])

                                        these_focus_objects = {}
                                        for pred in ['action_prev', 'action_now', 'action_next']:
                                            if not bool(this_ac[pred]['name']):
                                                these_focus_objects[pred] = None
                                            elif this_ac[pred]['name'] in ['pour', 'sprinkle']:
                                                these_focus_objects[pred] = this_ac[pred]['args'][1]
                                            else:
                                                these_focus_objects[pred] = this_ac[pred]['args'][0]

                                        if not action_prev:
                                            print('    -- Warning: Ignoring proceeding AC coordinates since this is first action!')

                                        for A in range(len(action_contexts)):
                                            # -- we will review all action contexts that we have learned, parse them for their respective predicates,
                                            #       and then we will generalize them based on a relative coordinate system:
                                            pred_now, pred_prev, pred_next = _parseActionContexts(A)

                                            if pred_now == False or pred_prev == False or pred_next == False:
                                                continue

                                            # -- set either the previous or next action predicate to None if there are some disjoints:
                                            #if not action_prev:
                                            #    pred_prev = None 
                                            # if not action_next:
                                            #     pred_next = None 

                                            existing_AC_name = action_contexts[A].action_now.name
                                            existing_AC_coord, existing_AC_table_cells = _generalizeActionContext(
                                                table_layout, under_object_layout, pred_now, pred_prev, pred_next)

                                            flag_same_coordinates = True
                                            if 'coord' in set(action_contexts[A].action_prev._fieldnames) & set(action_contexts[A].action_next._fieldnames):
                                                # -- instead of using the computed coordinates, we will instead read the coordinates loaded from the file:
                                                existing_AC_prev_coord = action_contexts[A].action_prev.coord if isinstance(action_contexts[A].action_prev.coord, str) else None
                                                existing_AC_next_coord = action_contexts[A].action_next.coord if isinstance(action_contexts[A].action_next.coord, str) else None

                                                existing_AC_coord['action_prev'] = existing_AC_prev_coord
                                                existing_AC_coord['action_next'] = existing_AC_next_coord
                                            #endif

                                            # -- checking if the preceding action coordinate matches:

                                            if not action_prev:
                                                # -- this means that we are at the very beginning of execution, 
                                                #       where there is NO previous action:

                                                # -- we will find a suitable action context based on two conditions:
                                                #       1. using the robot's (end-effector) starting location
                                                #       2. using the target table location (where the tables match exactly)
                                                if not flag_ignore_start and ftc.robot_start_location is not None and existing_AC_coord['action_prev']:
                                                    # -- in this case, we will rely upon the robot's starting location:
                                                    if str(existing_AC_coord['action_prev']) != str(this_AC_coord['action_prev']):
                                                        flag_same_coordinates = False

                                                elif flag_ignore_start and existing_AC_table_cells['action_now'] != this_AC_table_cells['action_now']:
                                                    # -- if there is no starting tile defined, then we will check for the exact table location
                                                    #       (assuming the numbering of tables is consistent throughout scenarios):
                                                    flag_same_coordinates = False

                                            if str(existing_AC_coord['action_prev']) != str(this_AC_coord['action_prev']):
                                                flag_same_coordinates = False
                                            
                                            # if flag_check_next and str(existing_AC_coord['action_next']) != str(this_AC_coord['action_next']):
                                            #     flag_same_coordinates = False

                                            # NOTE: also, it is important to check if the actions match as similar actions:
                                            flag_same_now_action = False
                                            if action_contexts[A].action_now.name in ['pour', 'sprinkle'] and this_action_name in ['pour', 'sprinkle']:
                                                flag_same_now_action = True
                                            elif action_contexts[A].action_now.name in ['stir', 'insert'] and this_action_name in ['stir', 'insert']:
                                                flag_same_now_action = True
                                            elif 'place' in action_contexts[A].action_now.name and 'place' in this_action_name:
                                                flag_same_now_action = True
                                            elif action_contexts[A].action_now.name == this_action_name:
                                                flag_same_now_action = True

                                            if action_prev:
                                                flag_same_prev_action = False
                                                if str(action_contexts[A].action_prev.name) in ['pour', 'sprinkle'] and prev_action_name in ['pour', 'sprinkle']:
                                                    flag_same_prev_action = True
                                                elif str(action_contexts[A].action_prev.name) in ['stir', 'insert'] and prev_action_name in ['stir', 'insert']:
                                                    flag_same_prev_action = True
                                                elif str(action_contexts[A].action_prev.name) and 'place' in prev_action_name:
                                                    flag_same_prev_action = True
                                                elif str(action_contexts[A].action_prev.name) == prev_action_name:
                                                    flag_same_prev_action = True
                                            else:
                                                # -- if there is no previous action, we will rely on the current location of the robot's end-effector:
                                                #       (this will be reflected by the 'flag_same_coordinates' variable)
                                                flag_same_prev_action = True

                                            if flag_same_now_action and flag_same_prev_action and flag_same_coordinates:
                                                those_focus_objects = {}
                                                for pred in ['action_prev', 'action_now', 'action_next']:
                                                    # -- because mat_structs work weirdly in Python, we have to do this:
                                                    if pred == 'action_prev':
                                                        temp_action_name = action_contexts[A].action_prev.name
                                                        temp_action_args = action_contexts[A].action_prev.args
                                                    elif pred == 'action_now':
                                                        temp_action_name = action_contexts[A].action_now.name
                                                        temp_action_args = action_contexts[A].action_now.args
                                                    else:
                                                        temp_action_name = action_contexts[A].action_next.name
                                                        temp_action_args = action_contexts[A].action_next.args
                                                    #endif

                                                    if not isinstance(temp_action_name, str):
                                                        those_focus_objects[pred] = None
                                                    elif temp_action_name in ['pour', 'sprinkle']:
                                                        those_focus_objects[pred] = temp_action_args[1]
                                                    else:
                                                        those_focus_objects[pred] = temp_action_args[0]
                                                    #endif
                                                #endfor

                                                # -- checking to see if the objects are all the same type:
                                                similar_shape = False

                                                num_matches = 0
                                                preds_to_check = ['action_prev', 'action_now']

                                                for pred in preds_to_check:
                                                    # NOTE: we need to check the preceding action, as the hand's orientation from a previous picking action
                                                    #           may be skewed and cause collisions if we do not check for similarity.
                                                    #  -- example: placing a bottle versus placing a cup
                                                    if these_focus_objects[pred] is None and those_focus_objects[pred] is None:
                                                        num_matches += 1
                                                        continue

                                                    for cat in ftc.object_categories:
                                                        if these_focus_objects[pred] in ftc.object_categories[cat] and those_focus_objects[pred] in ftc.object_categories[cat]:
                                                            num_matches += 1
                                                            break

                                                if num_matches == len(preds_to_check):
                                                    similar_shape = True

                                                if similar_shape:
                                                    print('\n   -- Found similar action context! Using the first one found (# ' + str(A+1) + ')..."')
                                                    print('\t\t<' + str(pred_prev) + ',')
                                                    print('\t\t ' + str(pred_now) + ',')
                                                    print('\t\t ' + str(pred_next) + '>\n')
                                                    print('\t\t ' + str(existing_AC_coord))
                                                    similar_action_context = A
                                                    break
                                        #endfor

                                        print()

                                        if similar_action_context > -1:
                                            # -- copy the DMP parameters (weights) for the most similar action context...
                                            this_ac['dmps']['weights'] = matlab.double(action_contexts[similar_action_context].dmps.weights.tolist())
                                            this_ac['dmps']['ini'] = matlab.double(action_contexts[similar_action_context].dmps.ini.tolist())
                                            this_ac['dmps']['end'] = matlab.double(action_contexts[similar_action_context].dmps.end.tolist())

                                            #   ... and run the motion planning method (from Alejandro's pipeline):
                                            trajectory_data = matlab_engine.motion_planning(this_ac)
 
                                            if not flag_skip_execution:
                                                print('  -- Executing action "' + action_now + '" (with AC #' + str(similar_action_context+1) + ')!')
                                                ftc._replayAction_DMP(trajectory_data, gripper_action)

                                            # -- add used action context to the history:
                                            AC_history.append(similar_action_context+1)

                                            num_similar_AC_matches += 1

                                        else:
                                            # -- save missing action context info to list:
                                            missing_AC_info = str('\t\t<' + str(action_prev) + ',') + \
                                                str('\t\t ' + str(action_now) + ',') + \
                                                str('\t\t ' + str(action_next) + '>\n') + \
                                                str('\t\t ' + str(this_AC_coord) + '\n')

                                            if this_AC_coord['action_prev'] is None and this_AC_coord['action_next'] is None:
                                                print(on_object_layout)
                                                print(under_object_layout)
                                                return {'success': False}

                                            AC_requiring_demo.append(missing_AC_info)

                                            if not flag_skip_execution:
                                                # -- if we are using the simulation and demonstration setup, 
                                                #       then we will ask the user to demonstrate the missing action:
                                                print('\n  -- No similar action context for "' + action_now + '" found!')

                                                # NOTE: in this case, a demonstration is required;
                                                #  -- we will use the recordAction method from the demonstration code to start recording
                                                #       from the present layout of the environment.
                                                
                                                if flag_skip_demo: return {'success': False}

                                                response = input('  -- Demonstration for action "' +
                                                    action_now + '" required!\n\tPress ENTER when ready! > ')                                            

                                                if response == 'end':
                                                    # -- this is to scrap the entire trial:
                                                    return {'success': False}

                                                try:
                                                    trajectory_data = ftc.loadmat(str(response), squeeze_me=True)
                                                except FileNotFoundError:
                                                    # -- if not, we can just record an entirely new trajectory:
                                                    trajectory_data, _ = ftc._recordAction(predicate=action_now)

                                                    # -- we will also save individual .MAT file containing the trajectory information:
                                                    from datetime import date
                                                    today = date.today().strftime("%d.%m.%Y")
                                                    demo_folder_name = 'recorded_data_' + today
                                                    if not os.path.exists(demo_folder_name):
                                                        os.makedirs(demo_folder_name)

                                                    MAT_file_name = demo_folder_name + '/' + ('0' if micro_count < 10 else '') + str(micro_count) + '.mat'

                                                    # -- randomly generating a new name for the demonstration:
                                                    while True:
                                                        letters = ''
                                                        for _ in range(random.randint(1, 4)):
                                                            letters += chr(random.randint(ord('a'), ord('z')))

                                                        MAT_file_name = demo_folder_name + '/' + ('0' if micro_count < 10 else '') + str(micro_count) + str(letters) + '.mat'
                                                        if not os.path.exists(MAT_file_name):
                                                            break

                                                    ftc.savemat(MAT_file_name, trajectory_data)
                                                    
                                                    # NOTE: we only have to deal with a dictionary object in this case:
                                                    trajectory_data = np.array(trajectory_data['Target']['trajectory']).transpose().tolist()

                                                    num_demoed_ACs += 1

                                                else:
                                                    # -- if we just want to replay a specific action as we have in a .MAT file:
                                                    ftc._replayAction(trajectory_data, gripper_action)
                                                    trajectory_data = np.array(trajectory_data['Target']['trajectory'].item()).transpose().tolist()  
                                                #endtry

                                                # -- once we have recorded the trajectory, we can then directly add the new action context
                                                #       to the database file (i.e., "actioncontexts.mat").

                                                # -- however, this requires generating DMP parameters from the trajectory.
                                                this_ac['dmps'] = matlab_engine.generate_motion_params(matlab.double(trajectory_data))
                                                return_code = matlab_engine.add_new_action_context(this_ac, -1, AC_location)

                                                if return_code == 1:
                                                    # -- after adding the new action context, we will reload the file:
                                                    action_contexts = ftc.loadmat(AC_location, squeeze_me=True, struct_as_record=False)
                                                    action_contexts = action_contexts['actioncontexts']

                                                    # -- add used action context to the history:
                                                    AC_history.append(len(action_contexts))

                                                #endif

                                                # NOTE: having added the new AC from a demonstration, we can just carry on with the next action.

                                                #endif

                                            #endif
                                    #endif

                                    if not flag_skip_execution:
                                        # -- perform secondary motion for special actions 
                                        #       (at this point, action_now == action_next):
                                        ftc._performAuxiliarySteps( 
                                            motion_name=this_ac['action_now']['name'], 
                                            args=this_ac['action_now']['args'],
                                            macro_PO=all_macro_POs[macro_PO_name]
                                        )

                                        ftc._adjustCamera()

                                else:
                                    if not flag_skip_execution:
                                        # -- load the demonstrated trajectory file instead:
                                        response = input('  -- Demonstration for action "' + action_now + '" required!\n\tPress ENTER when ready! > ')                                            
                                        try:
                                            trajectory_data = ftc.loadmat(str(response), squeeze_me=True)
                                        except FileNotFoundError:
                                            pass
                                        else:
                                            ftc._replayAction(trajectory_data, gripper_action)
                                    #endif
                                #endif

                                if not flag_skip_execution and ftc._checkGripperStatus(action_now, gripper_action) == False:
                                    # -- this means that something went wrong with the micro plan, 
                                    #       and so we will count this action as a failure:
                                    print('\n\t\t[ WARNING: step ' + str(macro_count) + ' -- (' + macro_PO_name + '):\taction failed! ]\n')
                                    num_failures += 1
                                #end
    
                                print('-- [FOON-TAMP] :  Finished executing action "' + action_now + '"!')

                                # -- do we need to run perception after each micro-action?
                                #       Answer: YES! This is done to ensure that actions were executed as planned.
                                _runPerception()

                                if not flag_skip_execution:
                                    print('\n -- Checking if action effects have been satisfied...')
    
                                    # -- check whether the executed action's effects have been satisfied/occurred:
                                    are_states_satisfied =_checkStateSatisfiability(prev_action=action_now)

                                    if not are_states_satisfied:
                                        print(are_states_satisfied)

                                # -- record the state of the environment:
                                state_history[micro_count] = {
                                    'in' : in_object_layout, 
                                    'on' : on_object_layout, 
                                    'under' : under_object_layout
                                }

                                print('\n--------------------------------------------------------' 
                                            + '--------------------------------------------------------\n')
            
                            #endif
        
                            # -- now we will keep track of the executed action as the previous step:
                            action_prev = action_now

                    else:
                        print('\t\t -- micro-level plan was NOT found!')
                        complete_micro_plan.append('; < missing solution! >')

                        # -- keep track of the number of macro-POs (functional units) that could not be executed:
                        num_failures += 1

                    # end

                    print()
                # endif
            else:
                pass
            # end

        # endfor

        print('-- [FOON-TAMP] : End of manipulation!')

        if complete_micro_plan:
            print(' -- [FOON-TAMP] : Saved micro-level plan as "' + str(micro_plan_file) + '"!')

            # -- parse through all the steps that were found by the planning phase
            #       and save them all to the micro-level plan file:
            micro_file = open(micro_plan_file, 'w')
            for step in complete_micro_plan:
                micro_file.write(step + '\n')

            # NOTE: for now, there are no costs for execution, so all steps have a cost of 1:
            micro_file.write('; cost = ' + str(total_cost) + ' (unit cost)' + '\n')

            micro_file.close()
        # endif

        if flag_perception:
            _runPerception()

            # -- record the state of the environment:
            state_history[micro_count] = {'on' : on_object_layout, 'under' : under_object_layout}

            # -- give a report about what happened in this trial:
            print('\n-- result of robotic manipulation:')

            if num_failures > 0:
                print(' -- [FOON-TAMP] : Trial was not fully completed! ' + str(num_failures) + ' functional units could not be executed.')
            else:
                print(' -- [FOON-TAMP] : Trial was successfully completed!')
            #endif

            print('  -- [FOON-TAMP] : Number of action contexts that had exact match in AC set:\t\t' + str(num_exact_AC_matches))
            print('  -- [FOON-TAMP] : Number of action contexts that had similar match in AC set:\t\t' + str(num_similar_AC_matches))
            print('  -- [FOON-TAMP] : Number of action contexts that were demonstrated in this trial:\t' + str(num_demoed_ACs))

            if flag_skip_execution:
                # -- if we skip the simulation, then this means we are performing the AC progression experiment:
                return {
                    'success': True if num_failures == 0 else False, 
                    'exact_matches': num_exact_AC_matches, 
                    'action_context_history': AC_history, 
                    'action_contexts_missing': AC_requiring_demo,
                    'micro_plan_length' : micro_count
                }

            else:
                if verbose:
                    for X in state_history:
                        print(X)
                        print(state_history[X])

                # -- otherwise, we return True/False depending on the success of this trial:
                return {   
                    'success': True if num_failures == 0 else False, 
                    'exact_matches': num_exact_AC_matches, 
                    'micro_plan_length': micro_count, 
                    'history': state_history
                }
            #endif
        #endif
    else:
        print("  -- [FOON-TAMP] : A macro-level plan for '" + str(FOON_subgraph_file) + "' was NOT found!")
        # TODO: write planning pipeline for other planners? 
    #endif

    print()
#enddef


if __name__ == '__main__':

    # -- check arguments from the command line (optional):
    try:
        opts, _ = getopt.getopt(sys.argv[1:],
                    'file:sce:plan:alg:heu:per:ac:ran:drop:ign:nst:con:sim:port:h',
                    [
                        'file=',    # -- name of the subgraph / universal FOON file used for translation and planning
                        'scenario=', # -- name of a JSON file containing properties and file name to use in CoppeliaSim / VREP
                        'planner=', # -- type of planner to use in the pipeline (default: "FD" -- Fast-Downward) 
                        'algorithm=', # -- type of searching algorithm to use
                        'heuristic=',   # -- type of heuristic to use for search for Fast-Downward (default: "lmcut()" -- landmark cut)
                        'use_perception',   # -- flag to enable perception step before each functional unit is executed (default: False -- OFF) 
                        'use_action_contexts',   # -- flag to enable use of action contexts for execution (default: False -- OFF)
                        'random',   # -- flag to enable randomized configuration of the scene
                        'dropout',  # -- flag to enable dropout of ingredients (automatic selection)
                        'ignore_items=',    # -- identifying objects to drop out (manual selection)
                        'no_start', # -- ignore robot's starting position
                        'contact', # -- disable collision check for contact between objects (for more robust state detection)
                        'no_sim', # -- assume micro-problem goals are always true
                        'port=',
                        'help'
                    ]
                )
        if opts:
            print(' -- [FOON-TAMP] : Provided the following arguments to this program:')
            for opt, arg in opts:

                if opt in ('-file', '--file'):
                    print("  -- File '" + str(arg) + "' will be converted to PDDL and used for TAMP!")
                    FOON_subgraph_file = str(arg)

                elif opt in ('-plan', '--planner'):
                    planner_to_use = str(arg)
                    print('  -- Selected the "' + planner_to_use + '" planner!')

                elif opt in ('-heu', '--heuristic'):
                    response = str(arg)
                    if 'FF' in response.upper():
                        # -- use FF heuristic:
                        heuristic = 1
                    else:
                        # -- use LM-Cut heuristic:
                        heuristic = 0

                elif opt in ('-alg', '--algorithm'):
                    response = str(arg).lower()
                    if response in configs:
                        algorithm = response
                    else:
                        # -- by default, use the A* search algorithm:
                        algorithm = 'astar'

                elif opt in ('-per', '--use_perception'):
                    flag_perception = True
                    try:
                        import FOON_to_CoppeliaSim as ftc
                    except ImportError:
                        print('  -- Disabling perception!')
                        flag_perception = False

                elif opt in ('-ran', '--random'):
                    flag_randomize_scene = True
                    print('  -- Scene configuration will be RANDOMIZED!')

                elif opt in ('-drop', '--dropout'):
                    flag_dropout = True
                    print('  -- Randomly selected ingredients will be DROPPED!')

                elif opt in ('-ign', '--ignore_items'):
                    # -- we can optionally pass a list of ingredients for the planner to ignore from the recipe:
                    ftp.ingredients_to_ignore = ast.literal_eval(arg)

                elif opt in ('-ac', '--use_action_contexts'):
                    print('  -- Loading action contexts (i.e., we will be using DMPs!)')
                    flag_action_contexts = True

                elif opt in ('-sce', '--scenario'):
                    json_scene_file = str(arg)

                elif opt in ('-nst', '--no_start'):
                    flag_ignore_start = True
                    print('  -- Assuming fixed starting location (i.e., "start_4").')

                elif opt in ('-con', '--contact'):
                    flag_collision = True
                    print('  -- Enabling collision check (for more rigourous state detection)!')

                elif opt in ('-sim', '--no_sim'):
                    flag_skip_execution = True

                elif opt in ('-port', '--port'):
                    try:
                        port_nums = ast.literal_eval(arg)
                    except Exception:
                        pass

                else:
                    print('  -- Invalid argument provided! Please review code for possible arguments.')
                    sys.exit()
                    
            if planner_to_use == 'fast-downward':
                # NOTE: Fast-Downward provides several kinds of heuristics; 
                #           please check their documentation for more details.
                if heuristic == 1:
                    print('  -- Using the FF heuristic with A* algorithm!')
                else:
                    print('  -- Using the LMCUT heuristic with A* algorithm!')
            print()
    except getopt.GetoptError:
        sys.exit()
    # endtry

    main()
#endif

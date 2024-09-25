def sysCall_init():
    sim = require('sim')
    simIK = require('simIK')
    simOMPL = require('simOMPL')
    math = require('math')

    sim.addLog(sim.getInt32Param(sim.intparam_verbosity), "[OMPLement] : Loading OMPL motion planning script...")


def sysCall_addOnScriptSuspend():
    pass


def sysCall_cleanup():
    sim.addLog(sim.getInt32Param(sim.intparam_verbosity), "[OMPLement] : Unloading OMPL motion planning script...")


def getConfig():
    config = [-1] * len(self.joint_handles)
    for J in range(len(self.joint_handles)):
        config[J] = sim.getJointPosition(self.joint_handles[J])

    return config


def setConfig(config):
    for J in range(len(self.joint_handles)):

        sim.setJointPosition(self.joint_handles[J], config[J])


def stateValidationCallback(state):
    savedState = simOMPL.readState(self.ompl_task)
    simOMPL.writeState(self.ompl_task, state)
    res, d, *_ = sim.checkDistance(self.robotCollection, sim.handle_all, self.maxDistance)
    is_valid = (res == 1) and (d[6] > self.minDistance)
    simOMPL.writeState(self.ompl_task, savedState)
    return is_valid


def configurationValidationCallback(config):
    # -- check if a configuration is valid, i.e. doesn't collide
    # -- save current config:
    tmp = getConfig()

    # -- apply new config:
    setConfig(config)

    # -- does new config collide?
    objs_in_collision = []
    is_collision, handles = sim.checkCollision(self.robotCollection,sim.handle_all)
    if is_collision == 1 and handles[1] not in objs_in_collision:
        objs_in_collision.append(sim.getObjectAlias(handles[1]))

    # -- restore original config:
    setConfig(tmp)

    if bool(objs_in_collision):
        sim.addLog(sim.getInt32Param(sim.intparam_verbosity), "collision found with:", objs_in_collision)

    return not bool(is_collision)


def path_planning(args):
    """
    This function requires a dictionary containing the following fields:
        1. "robot" :- the name of the robot's base in the scenario
        2. "goal" :- the object handle for a target (this should be some kind of dummy object -- refer to Python code for example)
        3. "algorithm" :- the name of the motion planning algorithm to use (by default, "RRTstar" will be used)
        4. "num_attempts" :- the number of times to run OMPL (default: 20)
        5. "max_compute" :- the maximum time (in seconds) allotted to computing a solution
        6. "max_simplify" :- the maximum time (in seconds) allotted to simplifying a solution
        7. "len_path" :- the number of states for path generation (default: leave it to OMPL)
    """

    # -- first check if the robot name and goal handle have been provided to the function:
    assert "robot" in args, "[OMPLement] : Robot name not defined!"
    robot_name = args["robot"]

    robot = sim.getObject(f'/{robot_name}')

    # NOTE: we will create a dummy object representing the target for planning!
    # -- extract the goal object given as input to this function:
    assert "goal" in args, "[OMPLement] : Goal not defined!"
    goal = args["goal"]

    pose = sim.getObjectPose(goal, robot)

    algorithm = simOMPL.Algorithm.RRTstar
    if "algorithm" in args: algorithm = args["algorithm"]

    max_compute = 5
    if "max_compute" in args: max_compute = args["max_compute"]

    max_simplify = -1
    if "max_simplify" in args: max_simplify = args["max_simplify"]

    # -- len_path :- number of states for the path (default: 0 -- we leave it to OMPL)
    len_path = 0
    if "len_path" in args: len_path = args["len_path"]

    # -- arm_prefix :- you can define the name format for joints (in the case where maybe there is a particular
    #   set of joints for which you want to do motion planning -- e.g., Spot robot has arm joints separate to legs)
    if "arm_prefix" in args:
        joint_prefix = f"/{args['arm_prefix']}_joint"
        tip = sim.getObject(f"/{args['arm_prefix']}_tip")
    else:
        joint_prefix = f"/{robot_name}_joint"
        tip = sim.getObject(f'/{robot_name}/tip')

    # -- num_ompl_attempts :- we have this functionality because simOMPL.compute() can reuse previously computed data
    #   Source: https://manual.coppeliarobotics.com/en/pathAndMotionPlanningModules.htm
    num_ompl_attempts = 5
    # -- check if the number of attempts for OMPL to solve a problem has been defined:
    if "num_attempts" in args: num_ompl_attempts = args["num_attempts"]

    assert robot != -1, "[OMPLement] : Robot base not defined!"
    assert tip != -1, "[OMPLement] : End-effector tip not defined!"

    ################################################################################################

    # NOTE: you need to know how many joints the robot you're using has;
    #   ideally, these joints should have some naming convention like in the loop below:
    self.joint_handles = []

    num_joints = 1
    while True:
        # -- using "noError" so default handle is -1 (if not found);
        #    read more here: https://manual.coppeliarobotics.com/en/regularApi/simGetObject.htm
        obj_handle = sim.getObject(f'{joint_prefix}{num_joints}', {"noError": True})

        if obj_handle == -1: break

        self.joint_handles.append(obj_handle)
        num_joints += 1

    # -- we will only use the first three joints (3) for projections:
    self.joint_projections = list([1] * 3) + list([0] * (len(self.joint_handles)-3))

    sim.addLog(sim.getInt32Param(sim.intparam_verbosity), f'[OMPLement] : Number of joints for robot "{robot_name}" - {len(self.joint_handles)}')

    # -- Prepare robot collection:
    self.robotCollection = sim.createCollection()
    sim.addItemToCollection(self.robotCollection, sim.handle_tree, robot, 0)

    # -- prepare an ik task (in order to be able to find configs that match specific end-effector poses):
    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    ikElement, simToIkObjectMapping, _ = simIK.addElementFromScene(ikEnv,ikGroup,robot,tip,goal,simIK.constraint_pose)
    simIK.syncFromSim(ikEnv, [ikGroup])

    # -- get a few handles from the IK world:
    ikJointHandles = []
    for J in range(len(self.joint_handles)):
        ikJointHandles.append(simToIkObjectMapping[self.joint_handles[J]])

    ikGoal=simToIkObjectMapping[goal]
    ikBase=simToIkObjectMapping[robot]
    ikTip=simToIkObjectMapping[tip]

    #path=simIK.generatePath(ikEnv,ikGroup,ikJointHandles,ikTip,500)
    simIK.setObjectPose(ikEnv,ikGoal,ikBase,pose)

    # -- check here for more info on how a valid configuration is found via IK: https://manual.coppeliarobotics.com/en/simIK.htm#simIK.findConfigs
    configs = simIK.findConfigs(
        ikEnv,ikGroup,ikJointHandles,
        {
            'maxDist': 0.05,
            'maxTime': 10,
            'findMultiple': False, # -- change to True to find multiple solutions
            'pMetric': [0.05,0.05,0.05,0.1],
            'cb': configurationValidationCallback
        })

    final_path = None

    if len(configs) > 0:
        # -- found a robot config that matches the desired pose!
        sim.addLog(sim.getInt32Param(sim.intparam_verbosity), "[OMPLement] : valid configuration found!")

        self.minDistance, self.maxDistance = float('1.0e-010'), 1.75

        # -- Now find a collision-free path (via path planning) that brings us from current config to the found config:
        self.ompl_task = simOMPL.createTask('task')
        simOMPL.setAlgorithm(self.ompl_task, algorithm)
        simOMPL.setStateSpaceForJoints(self.ompl_task, self.joint_handles, self.joint_projections)
        simOMPL.setCollisionPairs(self.ompl_task,[self.robotCollection, sim.handle_all])
        simOMPL.setStartState(self.ompl_task,getConfig())
        simOMPL.setGoalState(self.ompl_task,configs[0])
        simOMPL.setStateValidityCheckingResolution(self.ompl_task, 0.001)
        simOMPL.setStateValidationCallback(self.ompl_task, stateValidationCallback)
        simOMPL.setVerboseLevel(self.ompl_task, 1)
        simOMPL.setup(self.ompl_task)

        for _ in range(num_ompl_attempts):
            # -- read more about compute operation here: https://manual.coppeliarobotics.com/en/simOMPL.htm#compute
            result, path = simOMPL.compute(self.ompl_task,max_compute,max_simplify,len_path)

            # -- we will see if there was an exact solution found;
            #    that way we know if we might need to loop back around again to find the solution
            is_exact_solution = simOMPL.hasExactSolution(self.ompl_task)
            sim.addLog(sim.getInt32Param(sim.intparam_verbosity), f"[OMPLement] : Exact solution? - {is_exact_solution}")

            # -- if no exact solution was found... then maybe we will compute again?

            if result and is_exact_solution:
                # -- We found a collision-free path!
                sim.addLog(sim.getInt32Param(sim.intparam_verbosity), f"[OMPLement] : Length of path: {int(simOMPL.getPathStateCount(self.ompl_task,path))}")

                # NOTE: the path contains a Mx1 vector, which needs to be transformed to NxJ vector, where N = M/J.
                # -- the final path will be stored as a NxJ matrix, where N = number of points in trajectory and J = number of joints.
                final_path = []

                for x in range(0, len(path), len(self.joint_handles)):
                    final_path.append(path[x:x+len(self.joint_handles)])

                # NOTE: old way of parsing through the path -- it is very slow
                #for i in range(int(simOMPL.getPathStateCount(ompl_task,path))):
                #    conf=simOMPL.getPathState(ompl_task,path,i+1)
                #    final_path.append(conf)

                assert simOMPL.getPathStateCount(self.ompl_task,path) == len(final_path), "[OMPLement] : error in path rebuild?"

                break

        simOMPL.destroyTask(self.ompl_task)

    else:
        sim.addLog(sim.getInt32Param(sim.intparam_verbosity), "[OMPLement] : no configuration found!")

    return final_path
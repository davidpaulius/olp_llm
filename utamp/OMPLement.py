#python

'''luaExec
-- Here we have a special Lua section:
function sysCall_info()
    return {menu = 'OMPLement'}
end
'''

def sysCall_init():
    print("[OMPLement]: Initializing motion planning script with OMPL...")
    sim = require('sim')
    simIK = require('simIK')
    simOMPL = require('simOMPL')

def sysCall_addOnScriptSuspend():
    pass

def sysCall_cleanup():
    print("[OMPLement]: Cleaning up script...")


def getConfig(joint_handles):
    config = [-1] * len(joint_handles)
    for J in range(len(joint_handles)):
        config[J] = sim.getJointPosition(joint_handles[J])

    return config


def setConfig(joint_handles, config):
    for J in range(len(joint_handles)):
        sim.setJointPosition(joint_handles[J], config[J])


def configurationValidationCallback(robotCollection, config):
    # -- check if a configuration is valid, i.e. doesn't collide
    # -- save current config:
    tmp = getConfig()

    # -- apply new config:
    setConfig(config)

    # -- does new config collide?
    is_collision = sim.checkCollision(robotCollection,sim.handle_all)
    print('a', is_collision)
    # -- restore original config:
    setConfig(tmp)

    return not bool(is_collision)


def path_planning(target_object, gripper_action=-1, robot_name='Panda'):

    robot = sim.getObject(f'/{robot_name}')
    gripper = sim.getObject(f'/{robot_name}_target')
    tip = sim.getObject(f'/{robot_name}_tip')

    # NOTE: the Panda arm has 7 joints:
    joint_handles = [-1] * (7 if robot_name == 'Panda' else 6)
    joint_projections = [0] * len(joint_handles)
    for J in range(len(joint_handles)):
        joint_handles[J] = sim.getObject(f'/{robot_name}_joint{J+1}')
        joint_projections[J] += (1 if J < 3 else 0)


    goal_handle = sim.getObject(f'/{target_object}')

    # -- Prepare robot collection:
    robotCollection = sim.createCollection()
    sim.addItemToCollection(robotCollection, sim.handle_tree, robot, 0)

    # -- prepare an ik task (in order to be able to find configs that match specific end-effector poses):
    ikEnv = simIK.createEnvironment()
    ikGroup = simIK.createGroup(ikEnv)
    _, simToIkObjectMapping, _ = simIK.addElementFromScene(ikEnv,ikGroup,robot,tip,gripper,simIK.constraint_pose)
    # -- get a few handles from the IK world:
    ikJointHandles = []
    for J in range(len(joint_handles)):
        ikJointHandles.append(simToIkObjectMapping[joint_handles[J]])
    print(ikJointHandles)

    iktarget_object=simToIkObjectMapping[gripper]
    ikBase=simToIkObjectMapping[robot]

    # -- Find a collision-free config that matches a specific pose:
    pose = sim.getObjectPose(goal_handle, robot)
    simIK.setObjectPose(ikEnv,iktarget_object,ikBase,pose)

    config = simIK.findConfig(ikEnv,ikGroup,ikJointHandles,0.5,0.5,[1,1,1,0.1],configurationValidationCallback)
    print(config)
    if config:
        # -- found a collision-free config that matches the desired pose!
        # -- Now find a collision-free path (via path planning) that brings us from current config to the found config:
        ompl_task = simOMPL.createTask('task')
        simOMPL.setAlgorithm(ompl_task, simOMPL.Algorithm.RRTConnect)
        simOMPL.setStateSpaceForJoints(ompl_task, joint_handles, joint_projections)
        simOMPL.setCollisionPairs(ompl_task,[robotCollection, robotCollection])
        simOMPL.setStartState(ompl_task,getConfig())
        simOMPL.setGoalState(ompl_task,config)
        # -- now we could add more goal states with simOMPL.addGoalState, to increase the chance to find a collision-free path. But we won't do it for simplicity's sake
        simOMPL.setup(ompl_task)
        result,path = simOMPL.compute(ompl_task,4,-1,300)
        if result and path:
            # -- We found a collision-free path
            # -- Now move along the path in a very simple way:
            for i in range(simOMPL.getPathStateCount(ompl_task,path)):
                conf=simOMPL.getPathState(ompl_task,path,i)
                setConfig(conf)
                simOMPL.drawPath(ompl_task, path, 1, [255, 255, 255], )

        simOMPL.destroyTask(ompl_task)

    # -- use the gripper actions to decide whether to open or close gripper (-1 and 1 respectively, otherwise 0 -- no change):
    if gripper_action > -1:
        if gripper_action == 1:
            print('close')
            sim.setInt32Signal('close_gripper', 0)
        else:
            print('open')
            sim.setInt32Signal('close_gripper', 1)
    #endif



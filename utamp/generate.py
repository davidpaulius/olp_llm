import sys
import os
import math
from random import randint, choice

client = None

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print(' -- ERROR: Set up CoppeliaSim ZeroMQ client as described here: '\
          'https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm')
    sys.exit()

def randomize_blocks(path):
    client = RemoteAPIClient(host='localhost')

    sim = client.require('sim')
    sim.stopSimulation()
    sim.closeScene()

    try:
        sim.loadScene(os.path.abspath(path))
    except Exception:
        return None

    # -- randomize the scene with a certain number of blocks
    colours = {'red': 0, 'blue': 0, 'yellow': 0, 'green': 0}

    generated_objs = []

    for C in colours:
        # -- set colour of the new block:
        rgb = [0, 0, 0]
        if C == 'red':
            rgb = [255, 0, 0]
        elif C == 'green':
            rgb = [0, 255, 0]
        elif C == 'blue':
            rgb = [0, 0, 255]
        elif C == 'yellow':
            rgb = [255, 255, 0]

        # -- normalize between 0 and 1:
        rgb = [x / 255.0 for x in rgb]

        # -- programmatically create a new tile:
        new_tile = sim.createPrimitiveShape(
            sim.primitiveshape_cuboid,
            [0.08, 0.08, 0.01],
            0)

        # -- set the colour of the object:
        sim.setObjectColor(new_tile, 0, sim.colorcomponent_ambient_diffuse, rgb)
        sim.setObjectInt32Param(new_tile, sim.shapeintparam_edge_visibility, 1)

        # -- set the name of the newly created object:
        sim.setObjectAlias(new_tile, f'{C}_tile')

        # -- make the tile static but keep it respondable:
        sim.setObjectInt32Param(new_tile, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(new_tile, sim.shapeintparam_respondable, 1)

        sim.setObjectOrientation(new_tile, [0, 0, -math.pi], -1)

        generated_objs.append(sim.getObject(f'/{C}_tile'))


    for C in colours:
        # -- set colour of the new block:
        rgb = [0, 0, 0]
        if C == 'red':
            rgb = [255, 0, 0]
        elif C == 'green':
            rgb = [0, 255, 0]
        elif C == 'blue':
            rgb = [0, 0, 255]
        elif C == 'yellow':
            rgb = [255, 255, 0]

        # -- normalize between 0 and 1:
        rgb = [x / 255.0 for x in rgb]

        # -- randomly decide on number of blocks to place:
        colours[C] = randint(1, 5)

        for x in range(colours[C]):
        # -- programmatically create a new block:
            new_block = sim.createPrimitiveShape(
                sim.primitiveshape_cuboid,
                [0.05, 0.05, 0.05],
                0)

            # -- set the colour of the object:
            sim.setObjectColor(new_block, 0, sim.colorcomponent_ambient_diffuse, rgb)
            sim.setObjectInt32Param(new_block, sim.shapeintparam_edge_visibility, 1)

            # -- make the block dynamic and respondable:
            sim.setObjectInt32Param(new_block, sim.shapeintparam_static, 0)
            sim.setObjectInt32Param(new_block, sim.shapeintparam_respondable, 1)

            # -- set the name of the newly created object:
            sim.setObjectAlias(new_block, f'{C}_block_{x+1}')

            sim.setObjectOrientation(new_block, [0, 0, -(math.pi / randint(1,4))], -1)

            generated_objs.append(sim.getObject(f'/{C}_block_{x+1}'))

    # -- we will randomly distribute objects on the table based on dummies:
    dummy_objs = []

    while True:
        try:
            dummy = sim.getObject('/Dummy', {'index': len(dummy_objs)})
        except Exception:
            break

        dummy_objs.append(dummy)

    on_objects = {x : None for x in dummy_objs + generated_objs}

    for O in generated_objs:
        stack = 0
        if 'block' in sim.getObjectAlias(O):
            # -- we will flip a coin to determine whether the object will be stacked on another object or not:
            stack = randint(0, 1)

        while True:
            if 'block' in sim.getObjectAlias(O):
                # -- blocks can either be placed on other blocks or in empty spaces:
                selected = choice(dummy_objs + generated_objs)
            else:
                # -- tiles cannot be stacked on other objects:
                selected = choice(dummy_objs)

            if selected >= O:
                continue

            if not bool(stack):
                if bool(on_objects[selected]):
                    # -- if there is something in the spot, then we cannot stack it:
                    continue
            elif bool(stack):
                if bool(on_objects[selected]):
                    # -- if something is on the randomly selected object, then we cannot put something on it:
                    continue

            # NOTE: we must add an offset to the z-axis that accounts for the height of the object being stacked
            #       as well as the object below it:
            obj1_position = sim.getObjectPosition(selected, -1)
            _, obj1_dz = sim.getObjectFloatParameter(selected, sim.objfloatparam_objbbox_max_z)
            _, obj2_dz = sim.getObjectFloatParameter(O, sim.objfloatparam_objbbox_max_z)
            obj1_position[2] += obj1_dz + obj2_dz

            # -- place the current object on the randomly selected object/position:
            sim.setObjectPosition(O, obj1_position, -1)

            # -- now we mark the randomly selected object as being "occupied" (i.e., object is blocking it)
            on_objects[selected] = O

            break

    new_path = os.path.abspath(path.replace('prototype', 'randomized'))
    sim.saveScene(new_path)

    return new_path, colours

if __name__ == '__main__':
    randomize_blocks('C:/Users/david/OLP_LLM/utamp/scenes/panda_blocks_tiles_prototype.ttt')

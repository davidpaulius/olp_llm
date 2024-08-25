import sys
import os
import math

from random import randint, choice, sample, shuffle
from utamp.driver import Interfacer

client = None

textures_dir = '/textures/'

def randomize_blocks(
        path: str,
        attributes: dict = {
            'colours': {
                'red': [255, 0, 0],
                'green': [0, 255, 0],
                'blue': [0, 0, 255],
            },
            'alphabets': {
                x: os.path.join(textures_dir, f'{x.lower()}.png') for x in ['A', 'B', 'C',]
            }
        },
        block_type: int = 0, # NOTE: 0 -- coloured blocks, 1 -- alphabet blocks
        add_tiles: bool = False,
        all_shapes: bool = False,
        rotate: bool = False,
        num_empty_spots: int = 2,
        max_instances: int = None,
    ) -> tuple[str, dict]:

    # NOTE: if 'all_shapes' is set to True, we will use all the dimensions listed below:
    block_sizes = {'1x1': [0.05, 0.05, 0.05],
                   '3x1': [0.15, 0.05, 0.05], }

    tally = None
    # -- randomize the scene with a certain number of blocks
    if bool(block_type):
        tally = {'block': {L: 0 for L in attributes['alphabets']}}
    else:
        tally = {'block': {C: 0 for C in attributes['colours']}}

    all_objects = []

    sim_interfacer = Interfacer(scene_file_name=path)

    if bool(block_type):
        texture_objs = {}
        for L in attributes['alphabets']:
            handle, texture_id, _ = sim_interfacer.sim.createTexture(attributes['alphabets'][L], 13)
            texture_objs[L] = (handle, texture_id)

    # -- first, let's deal with generating blocks:
    for atr in tally['block']:

        # -- randomly decide on number of blocks to place:
        if not bool(block_type):
            tally['block'][atr] = randint(1, (max_instances if max_instances else 4))
        else:
            tally['block'][atr] = randint(1, (max_instances if max_instances else 3))

        block_shapes = list(block_sizes.keys())

        for x in range(tally['block'][atr]):
            # -- programmatically create a new block:
            block_size = block_sizes[choice(block_shapes) if all_shapes else '1x1']

            new_block = sim_interfacer.sim.createPrimitiveShape(
                sim_interfacer.sim.primitiveshape_cuboid,
                block_size,
                0,
            )

            # -- set the colour of the object:
            if not bool(block_type):
                # -- set colour of the new object and normalize between 0 and 1:
                rgb = [x / 255.0 for x in attributes['colours'][atr]]
                sim_interfacer.sim.setObjectColor(new_block, 0, sim_interfacer.sim.colorcomponent_ambient_diffuse, rgb)
            else:
                # -- set the texture for the alphabetic blocks:
                sim_interfacer.sim.setShapeTexture(new_block, texture_objs[atr][1], sim_interfacer.sim.texturemap_cube, 15, block_size[:-1], None, [0, 0, math.pi])

            # -- set the edge to being solid and visible:
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_edge_visibility, 1)

            # -- make the block dynamic and respondable:
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_static, 0)
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_respondable, 1)

            # -- set the name of the newly created object:
            sim_interfacer.sim.setObjectAlias(new_block, f'{atr}_block_{x+1}')

            sim_interfacer.sim.setObjectOrientation(new_block, [0, 0, -(math.pi / randint(1,4)) if rotate else 0], -1)

            all_objects.append(sim_interfacer.sim.getObject(f'/{atr}_block_{x+1}'))

    # -- we will randomly distribute objects on the table based on dummies:
    object_spots = []

    # -- shuffle the order of block placement:
    shuffle(all_objects)

    if bool(block_type):
        # -- delete texture objects if they are present:
        for L in texture_objs:
            sim_interfacer.sim.removeObjects([texture_objs[L][0]])

    if add_tiles:

        tally['tile'] = {C: 0 for C in attributes['colours']}

        for atr in tally['tile']:
            # -- set colour of the new object and normalize between 0 and 1:
            rgb = [x / 255.0 for x in attributes['colours'][atr]]

            # -- programmatically create a new tile:
            new_tile = sim_interfacer.sim.createPrimitiveShape(
                sim_interfacer.sim.primitiveshape_cuboid,
                [0.08, 0.08, 0.02], 0,
            )

            # -- set the colour of the object:
            sim_interfacer.sim.setObjectColor(new_tile, 0, sim_interfacer.sim.colorcomponent_ambient_diffuse, rgb)
            sim_interfacer.sim.setObjectInt32Param(new_tile, sim_interfacer.sim.shapeintparam_edge_visibility, 1)

            # -- set the name of the newly created object:
            sim_interfacer.sim.setObjectAlias(new_tile, f'{atr}_tile')

            # -- make the tile static but keep it respondable:
            sim_interfacer.sim.setObjectInt32Param(new_tile, sim_interfacer.sim.shapeintparam_static, 1)
            sim_interfacer.sim.setObjectInt32Param(new_tile, sim_interfacer.sim.shapeintparam_respondable, 1)

            sim_interfacer.sim.setObjectOrientation(new_tile, [0, 0, -math.pi], -1)

            all_objects.append(sim_interfacer.sim.getObject(f'/{atr}_tile'))

            tally['tile'][atr] = 1

        # -- make sure that the tiles are placed first, then the blocks:
        all_objects.reverse()

    while True:
        try:
            dummy = sim_interfacer.sim.getObject('/Dummy', {'index': len(object_spots)})
        except Exception:
            break

        object_spots.append(dummy)

    # -- randomly remove some table spots from consideration:
    for x in range(num_empty_spots):
        object_spots.pop(randint(0, len(object_spots) - 1))

    on_objects = {x : None for x in object_spots + all_objects}

    for O in all_objects:
        stack = 0
        if 'block' in sim_interfacer.sim.getObjectAlias(O):
            # -- we will flip a coin to determine whether the object will be stacked on another object or not:
            stack = randint(0, 1)

        while True:
            if 'block' in sim_interfacer.sim.getObjectAlias(O):
                # -- blocks can either be placed on other blocks or in empty spaces:
                selected = choice(object_spots + all_objects)
            else:
                # -- tiles cannot be stacked on other objects:
                selected = choice(object_spots)

            if selected not in object_spots and all_objects.index(selected) >= all_objects.index(O): continue

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
            obj1_position = sim_interfacer.sim.getObjectPosition(selected, -1)
            obj1_dz = sim_interfacer.sim.getObjectFloatParam(selected, sim_interfacer.sim.objfloatparam_objbbox_max_z)
            obj2_dz = sim_interfacer.sim.getObjectFloatParam(O, sim_interfacer.sim.objfloatparam_objbbox_max_z)
            obj1_position[2] += obj1_dz + obj2_dz

            # -- place the current object on the randomly selected object/position:
            sim_interfacer.sim.setObjectPosition(O, obj1_position, -1)

            # -- now we mark the randomly selected object as being "occupied" (i.e., object is blocking it)
            on_objects[selected] = O

            break

    new_path = os.path.abspath(path.replace('prototype', 'randomized'))
    sim_interfacer.sim.saveScene(new_path)

    return new_path, tally


def randomize_packing(
        path: str,
        colours: dict = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
        },
        num_empty_spots: int = 2,
    ) -> tuple[str, dict]:

    sim_interfacer = Interfacer(scene_file_name=path)

    # intuition: we will randomly generate 1~4 boxes on the robot's table.
    num_spots = 4
    num_boxes = randint(1, num_spots)

    box_colours = sample(list(colours.items()), num_boxes)

    taken_spots, box_handles = [], []

    # -- get the archetype box:
    proto_box = sim_interfacer.sim.getObject('/box')

    tally = {
        'box': {C: 0 for C in colours},
        'toy': {C: 0 for C in colours},
    }

    for colour, rgb in box_colours:
        tally['box'][colour] = 1

        # -- copy the prototype box and rename it to the randomly selected colour:
        new_box = sim_interfacer.sim.copyPasteObjects([proto_box])[0]

        sim_interfacer.sim.setObjectAlias(new_box, f'{colour}_box')

        # -- change the colour of the box (since it is a group object, we need to change each part's colour):
        for side in range(5):
            sim_interfacer.sim.setObjectColor(new_box, side, sim_interfacer.sim.colorcomponent_ambient_diffuse, [x / 255.0 for x in rgb])

        # -- select one of the box spots:
        while True:
            selected = randint(0, num_spots-1)
            if selected not in taken_spots:
                taken_spots.append(selected)
                dummy = sim_interfacer.sim.getObject('/box_spot', {'index': taken_spots[-1]})
                break

        # -- set the position at the right place
        sim_interfacer.sim.setObjectPosition(new_box, sim_interfacer.sim.getObjectPosition(dummy, -1), -1)

        # -- make the box static but keep it respondable:
        sim_interfacer.sim.setObjectInt32Param(new_box, sim_interfacer.sim.shapeintparam_static, 0)
        sim_interfacer.sim.setObjectInt32Param(new_box, sim_interfacer.sim.shapeintparam_respondable, 1)

        box_handles.append(new_box)

    # -- remove the generic box:
    sim_interfacer.sim.removeObjects([proto_box])

    # -- randomize the scene with a certain number of "toys" (blocks)
    all_objects = []

    for C in colours:
        # -- set colour of the new object and normalize between 0 and 1:
        rgb = [x / 255.0 for x in colours[C]]

        # -- randomly decide on number of blocks to place:
        tally['toy'][C] = randint(1, 5)

        block_sizes = {
            '1x1': [0.05, 0.05, 0.05],
            '1x2': [0.06, 0.06, 0.10],
        }

        block_shapes = {
            'cylinder': sim_interfacer.sim.primitiveshape_cylinder,
            'cube': sim_interfacer.sim.primitiveshape_cuboid,
        }

        for x in range(tally['toy'][C]):
            # -- programmatically create a new block:
            new_block = sim_interfacer.sim.createPrimitiveShape(
                block_shapes[choice(list(block_shapes.keys()))],
                block_sizes[choice(list(block_sizes.keys()))], 0,
            )

            # -- set the colour of the object and add a solid edge:
            sim_interfacer.sim.setObjectColor(new_block, 0, sim_interfacer.sim.colorcomponent_ambient_diffuse, rgb)
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_edge_visibility, 1)

            # -- make the block dynamic and respondable:
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_static, 0)
            sim_interfacer.sim.setObjectInt32Param(new_block, sim_interfacer.sim.shapeintparam_respondable, 1)

            # -- set the name of the newly created object:
            sim_interfacer.sim.setObjectAlias(new_block, f'{C}_toy_{x+1}')

            all_objects.append(sim_interfacer.sim.getObject(f'/{C}_toy_{x+1}'))

    object_spots = []

    while True:
        try:
            dummy = sim_interfacer.sim.getObject('/Dummy', {'index': len(object_spots)})
        except Exception:
            break

        object_spots.append(dummy)

    # -- randomly remove some table spots from consideration:
    for x in range(num_empty_spots):
        object_spots.pop(randint(0, len(object_spots) - 1))

    on_objects = {x : None for x in object_spots + all_objects}

    for O in all_objects:
        # -- we will flip a coin to determine whether the object will be stacked on another object or not:
        stack = randint(0, 1)

        while True:
            # -- blocks can either be placed on other blocks or in empty spaces:
            selected = choice(object_spots + all_objects)

            # -- we need to make sure that we consider placements in order:
            if selected not in object_spots and all_objects.index(selected) >= all_objects.index(O): continue

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
            obj1_position = sim_interfacer.sim.getObjectPosition(selected, -1)
            obj1_dz = sim_interfacer.sim.getObjectFloatParam(selected, sim_interfacer.sim.objfloatparam_objbbox_max_z)
            obj2_dz = sim_interfacer.sim.getObjectFloatParam(O, sim_interfacer.sim.objfloatparam_objbbox_max_z)
            obj1_position[2] += obj1_dz + obj2_dz

            # -- place the current object on the randomly selected object/position:
            sim_interfacer.sim.setObjectPosition(O, obj1_position, -1)

            # -- now we mark the randomly selected object as being "occupied" (i.e., object is blocking it)
            on_objects[selected] = O

            break

    new_path = os.path.abspath(path.replace('prototype', 'randomized'))
    sim_interfacer.sim.saveScene(new_path)

    return new_path, tally


def randomize_cocktail(
        path: str,
) -> tuple[str, dict]:

    prototypes = [
        'bottle',
        'cup',
        'knife',
        'spoon',
        'can',
    ]


if __name__ == '__main__':
    all_colours = [
        {'red': [255, 0, 0]},
        {'green': [0, 255, 0]},
        {'blue': [0, 0, 255]},
        {'yellow': [255, 255, 0]},
        {'pink': [255, 0, 255]},
        {'purple': [128, 0, 255]},
        {'orange': [255, 128, 0]},
        {'cyan': [0, 255, 255]},
        {'black': [64, 64, 64]},
        {'white': [255, 255, 255]},
    ]

    colours = {}
    for C in sample(all_colours, randint(4, len(all_colours))):
        colours.update(C)

    # randomize_blocks('C:/Users/david/OLP_LLM/utamp/scenes/panda_blocks_tiles_prototype.ttt')

    randomize_packing(
        path='C:/Users/david/OLP_LLM/utamp/scenes/panda_packing_prototype.ttt',
        colours=colours
    )
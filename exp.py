import copy

from olp_lib import *

from utamp.generate import *
from utamp.driver import *

import traceback

# NOTE: keep checking the pricing for details on best model to use: https://openai.com/api/pricing/

models = ["gpt-4-turbo", "chatgpt-4o-latest", "gpt-4o"]

# -- use the custom-made class for accessing OpenAI models:
openai_driver = OpenAIInterfacer(
    model_embed="text-embedding-3-small",
    model_text_gen="chatgpt-4o-latest",
)

# NOTE: all incontext examples will be stored within a JSON file:
# -- Q: can we randomly sample from the set of incontext examples?
# -- Q: should we also select an example "closest" to the provided task?

incontext_file = "all_fewshot_examples.json"
fewshot_examples = json.load(open(incontext_file))

experimental_results = []

coppelia_port_number = None
coppelia_robot = "Panda"
coppelia_gripper = "Panda_gripper"

path_planning_method = 1
terminate_upon_failure = True

setting = "blocks"

verbose = False

is_alphabetic = True

def initialize_scene():
    while True:
        all_colours = [
            {'red': {'rgb': [204, 0, 0], 'count': 0}},
            {'green': {'rgb': [0, 204, 0], 'count': 0}},
            {'blue': {'rgb': [0, 0, 204], 'count': 0}},
            {'yellow': {'rgb': [204, 204, 0], 'count': 0}},
            # {'pink': {'rgb': [204, 0, 204], 'count': 0}},
            {'purple': {'rgb': [102, 0, 204], 'count': 0}},
            # {'orange': {'rgb': [255, 128, 0], 'count': 0}},
            {'black': {'rgb': [64, 64, 64], 'count': 0}},
            {'white': {'rgb': [250, 250, 250], 'count': 0}},
        ]

        all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U']

        attributes = {}

        if task == 'spell':
            is_alphabetic = True

            min_attributes = 4      # -- number of letters to place on the table (3~7)
            max_instances = 3       # -- max number of blocks per letter
            num_vowels = 0          # -- enforcing a certain number of vowels for proper words

            attributes['alphabets'] = {
                'R': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/r.png'), 'count': 1},
                'O': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/o.png'), 'count': 2},
                'B': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/b.png'), 'count': 1},
                'T': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/t.png'), 'count': 1},
                'I': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/i.png'), 'count': 1},
                'C': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/c.png'), 'count': 1},
                'S': {'texture': os.path.join(os.getcwd(), f'./utamp/textures/s.png'), 'count': 1},
            }

            # # -- we want to guarantee at least a certain number of vowels:
            # attributes['alphabets'].update({
            #     x: {'texture': os.path.join(os.getcwd(), './utamp/textures/', f'{x.lower()}.png'), 'count': randint(1, max_instances)} for x in sample(['A', 'E', 'I', 'O'], 2)
            # })

            # -- let's now add the remaining letters sans vowels:
            attributes['alphabets'].update({
                x: {'texture': os.path.join(os.getcwd(), './utamp/textures/', f'{x.lower()}.png'), 'count': randint(1, max_instances)} for x in sample(list(set(all_letters) - set(['R', 'O', 'B', 'T', 'I', 'C'])), (min_attributes - num_vowels))
            })

        elif task == 'organize':
            # -- we will randomly decide if to organize lettered blocks or coloured blocks:
            # is_alphabetic = bool(randint(0, 1))

            min_attributes = 4      # -- number of type/attribute to handle
            max_instances = 3       # -- max number of blocks per type/attribute
            total_num_blocks = 10     # -- total number of blocks to place on the table across all attributes

            # if is_alphabetic:
            #     subset_alphabets = {
            #         x: {'texture': os.path.join(os.getcwd(), './utamp/textures/', f'{x.lower()}.png'), 'count': 0} for x in sample(all_letters, min_attributes)
            #     }
            #     attributes['alphabets'] = subset_alphabets

            # else:
            #     subset_colours = {}
            #     for C in sample(all_colours, min_attributes):
            #         subset_colours.update(C)

            #     attributes['colours'] = subset_colours

            # for atr in list(attributes.keys()):
            #     remainder = total_num_blocks
            #     for A in attributes[atr]:
            #         attributes[atr][A]['count'] = min(max_instances, max(0, remainder))
            #         remainder -= max_instances

            attributes['alphabets'] = {
                'M': {'texture': os.path.join(os.getcwd(), f"./utamp/textures/M.png"), 'count': 4},
                'K': {'texture': os.path.join(os.getcwd(), f"./utamp/textures/K.png"), 'count': 4},
                'D': {'texture': os.path.join(os.getcwd(), f"./utamp/textures/D.png"), 'count': 4},
            }

        else:
            min_attributes = 3 # -- the total number of attributes/types to handle

            subset_colours = {}
            for C in sample(all_colours, randint(min_attributes, len(all_colours))):
                subset_colours.update(C)

            attributes = {
                'colours': subset_colours,
                'alphabets': {
                    x: {'texture': os.path.join(os.getcwd(), './utamp/textures/', f'{x.lower()}.png'), 'count': 0} for x in sample(all_letters, randint(min_attributes, len(all_letters)))
                }
            }


def main(
        attr: dict,
        timestamp: str,
        trial: int,
        arg: str = None,
    ):

    global verbose, driver
    global setting, task
    global path_planning_method, terminate_upon_failure

    global coppelia_port_number, coppelia_gripper, coppelia_robot

    # NOTE: we will potentially have 3 task scenes:
    #   1. 'blocks' :- a scene focusing on block stacking
    #           (either based on colour or alphabet);
    #   2. 'packing' :- a scene focusing on packing away "toys" in boxes;
    #   3. 'cocktail' :- a scene focusing on the ability to make different
    #           cocktails based on available ingredients
    # setting = choice(['blocks', 'packing', 'cocktail'])

    fpath = f"./utamp/scenes/panda_{setting}_prototype.ttt"
    sim_interfacer = Interfacer(
        scene_file_name=fpath,
        robot_name=coppelia_robot,
        robot_gripper=coppelia_gripper,
        port_number=coppelia_port_number,
    )

    # -- we have different functions depending on the scene type:
    block_type = f"str(\"{'alphabets' if is_alphabetic else 'colours'}\")"
    fpath, tally = eval((
        f"randomize_{setting}("
        "sim_interfacer,"
        "fpath,"
        f"{attr},"
        f"block_type={block_type},"
        f"suffix=\"{timestamp}\","
        ")"
    ))

    print(f"{'*' * 25}\nSCENE HAS BEEN GENERATED\n{'*' * 25}")

    object_phrases = []
    for item in tally:
        for C in tally[item]:
            if bool(tally[item][C]):
                num_as_text = "one" if tally[item][C] == 1 else "two" if tally[item][C] == 2 else "three" if tally[item][C] == 3 else "four" if tally[item][C] == 4 else "five"
                object_phrases.append(
                        f"{num_as_text} ({tally[item][C]}) "
                        # + str(('' if not bool(is_alphabetic) else '\'') + f"{C}" + ('' if not bool(is_alphabetic) else '\''))
                        f"{C}{(' ' if not bool(is_alphabetic) else '-')}{item}{'s' if tally[item][C] > 1 else ''}"
                )

    sim_interfacer = Interfacer(
        scene_file_name=fpath,
        robot_name=coppelia_robot,
        robot_gripper=coppelia_gripper,
        port_number=coppelia_port_number,
    )

    print(f"\n{'*' * 4} AFTER COPPELIASIM RANDOMIZATION: {'*' * 4}")
    print("objects presently in scene:", sim_interfacer.objects_in_sim)
    print("\nobjects in sim for LLM:", object_phrases)

    user_task = {
        "objects": object_phrases,
        "query": None
    }

    use_cache = False

    # evaluated_methods = ['FOON', 'LLM-Planner', 'LLM+P', 'DELTA']
    evaluated_methods = ['FOON', ]
    # evaluated_methods = ['FOON', 'DELTA']
    # evaluated_methods = ['OLP-UTAMP', 'OLP', 'LLM-Planner', 'LLM-Planner-UTAMP', 'LLM+P']

    # %% [markdown]
    # ### Prompt User for Task
    user_task['query'] = None
    task_file_name = None

    if user_task['query'] is None:

        if task == "organize":
            user_task['query'] = choice([
                "Organize the table by stacking all similar blocks into piles.",
                "Make piles of matching blocks on the table."
            ])
            task_file_name = f"organize_table_trial{trial}.txt"

        elif task == "spelling":
            # -- maybe we can spell one word or two words:
            words = [arg]
            if len(words) == 1:
                user_task['query'] = choice([
                    f"Spell the word \"{words[0]}\" as a block tower.",
                    f"Make a tower of blocks spelling out the word \"{words[0]}\".",
                    # f"Spell the word \"{words[0]}\" in reverse.",
                    # f"Make a tower of blocks spelling out the word \"{words[0]}\" in reverse.",
                ])
                task_file_name = f"spell_the_word_{words[0]}_trial{trial}.txt"
            else:
                user_task['query'] = choice([
                    f"Spell the words \"{words[0]}\" and \"{words[1]}\" vertically.",
                    f"Make 2 towers that spell the words \"{words[1]}\" and \"{words[0]}\".",
                    f"Spell \"{words[0]}\" and \"{words[1]}\" in reverse.",
                    f"Make 2 towers of blocks spelling the words \"{words[0]}\" and \"{words[1]}\" but in reverse.",
                ])
                task_file_name = f"spell_the_words_{words[0]}_and_{words[1]}.txt"

            user_task['query'] += (
                " Spell the word vertically."
                " For example, to spell \"BOY\", you must stack each letter in reverse: first place \"O\" on top of \"Y\", then \"B\" on top of \"O\" so it reads \"BOY\" from top to bottom."
            )
        elif task == "towers":
            user_task['query'] = choice([
                    f"Stack exactly {arg} blocks.",
                    f"Make a tower with {arg} blocks.",
                    # f"Make a stack of blocks that is no higher than {arg-1} blocks tall.",
                ])
            task_file_name = f"stack_{arg}_blocks_trial{trial}.txt"

        else:
            while not user_task['query']:
                user_task['query'] = input(f"The robot has the following objects available to it on a table: {str(user_task['objects'])}.\nWhat would you like the robot to do?")

            task_file_name = f"{str.lower(re.compile('[^a-zA-Z]').sub('_', user_task['query'])[:-1])}.txt"

    print('User Task:', user_task['query'])

    # %% [markdown] Step 1 :- LLM -> Object-level Plan
    #
    # 1. Initialize libraries needed for FOON (``FOON_graph_analyser.py``) as well as OpenAI api.
    #
    # 2. Perform 2-stage prompting for recipe prototype.
    #     - In the first stage, we ask the LLM for a *high-level recipe* (list of instructions) and a *list of objects* needed for completing the recipe.
    #     - In the second stage, we ask the LLM for a breakdown of *state changes* that happen for each step of the recipe; specifically, we ask for the *preconditions* and *effects* of each action, which is similar to how a functional unit in FOON has *input* and *output object nodes*.

    # ### FOON-based OLP Generation

    if bool(set(['FOON', 'FOON-UTAMP']) & set(evaluated_methods)):

        print(f"\n\nUser Task: {user_task['query']}")
        print(f"Available objects: {user_task['objects']}")

        # -- load prototype/reference FOON graphs:
        loaded_foons = []
        for foon_file in os.listdir('./foon_prototypes/'):
            fga._resetFOON(); fga._constructFOON(os.path.join('./foon_prototypes/', foon_file))
            loaded_foons.append({
                "foon": fga.FOON_lvl3,
            })

        total_time = time.time()

        llm_output, interaction = generate_from_FOON(
            openai_obj=openai_driver,
            query=user_task['query'],
            FOON_samples=loaded_foons,
            scenario={
                "name": setting,
                "objects": user_task['objects'] if isinstance(user_task, dict) else None,
            },
            system_prompt_file='llm_prompts/foon_system_prompt.txt',
            human_feedback=False,
            verbose=False,
        )

        # -- save unparsed output to a JSON file:
        json.dump(llm_output, open('raw_FOON.json', 'w'), indent=4)

        print("\nTASK:",  user_task)
        print(f"\n{'*' * 25}\n    High level plan\n{'*' * 23}\n{llm_output['language_plan']}")
        print(f"\nAll Objects: {llm_output['all_objects']}")
        print(f"\n\n{'*' * 25}\n    DETAILED \n{'*' * 23}\n{json.dumps(llm_output['object_level_plan'], indent=4)}")

    # ### Regular OLP Generation

    if bool(set(['OLP', 'OLP-UTAMP']) & set(evaluated_methods)):
        # -- using the "legacy" OLP prompting method:
        print('User Task:', user_task['query'])
        print('Available objects:', user_task['objects'])

        llm_output = None

        if use_cache:
            while True:
                llm_output = repair_olp(
                    openai_obj=openai_driver,
                    query=user_task['query'],
                    available_objects=user_task['objects'],
                    verbose=False
                )

                if isinstance(llm_output, int):
                    if llm_output == 0:
                        # -- return 0 means that there are no items in cache:
                        llm_output = None
                        break
                    elif llm_output == -1:
                        # -- return -1 means that some error was encountered (e.g. LLM did not say "okay"):
                        pass

                else: break

        if not llm_output:
            # -- stick to the regular OLP creation pipeline powered by LLM goodness:
            while True:
                try:
                    llm_output, interaction = generate_olp(
                        openai_driver,
                        query=user_task["query"],
                        fewshot_examples=fewshot_examples,
                        scenario={
                            "name": setting,
                            "objects": user_task['objects'] if isinstance(user_task, dict) else None,
                        },
                        stage1_sys_prompt_file="llm_prompts/olp_stage1_prompt.txt",
                        stage2_sys_prompt_file="llm_prompts/olp_stage2_prompt.txt",
                        stage3_sys_prompt_file="llm_prompts/olp_stage3_prompt.txt",
                        stage4_sys_prompt_file="llm_prompts/olp_stage4_prompt.txt",
                        verbose=False
                    )
                except Exception as e: print('-- something went wrong:', type(e), e.args)
                else:
                    if llm_output: break

        # -- save unparsed output to a JSON file:
        json.dump(llm_output, open('raw_OLP.json', 'w'), indent=4)

        print("\nTASK:",  user_task)
        print(f"\n{'*' * 25}\n    High level plan\n{'*' * 23}\n{llm_output['language_plan']}\n")
        print(f"\All Objects: {llm_output['all_objects']}")
        print(f"\n\n{'*' * 25}\n    DETAILED \n{'*' * 23}\n{json.dumps(llm_output['object_level_plan'], indent=4)}")


    # %% [markdown] Step 2 :- Object-level Plan -> FOON

    # ### Creating new functional units
    # - This involves parsing a JSON structure produced by the LLM to create FOON functional units.
    if bool(set(['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP']) & set(evaluated_methods)):

        FOON_prototype = []

        for x in range(len(llm_output['object_level_plan']['plan'])):

            new_unit = olp_to_FOON(llm_output, index=x)

            # NOTE: in order to define a macro-problem, we need to properly identify all goal nodes;
            #       we will do this with the help of the LLM:
            if 'termination_steps' in llm_output and (x+1) in llm_output['termination_steps']:
                # -- set output objects as goal nodes for the functional units deemed as terminal steps:
                print(f'\nFunctional unit {x+1} has terminal goals!')
                for N in range(new_unit.getNumberOfOutputs()):
                    new_unit.getOutputNodes()[N].setAsGoal()

            # -- add the functional unit to the FOON prototype:

            if not new_unit.isEmpty():
                # -- we should only add a new functional unit if it is not empty, meaning it must have the following:
                #    1. >=1 input node and >= 1 output node
                #    2. a valid motion node
                FOON_prototype.append(new_unit)
                FOON_prototype[-1].print_functions[-1]()
            else:
                print('NOTE: the following functional unit has an error, so skipping it:')
                new_unit.print_functions[-1]()

            print('\n', '*' * 40)

            print()

    #  Save Generated FOON
    # 1. Use ```FOON_parser.py``` to clean and ensure FOON labels are correct
    # 2. Add the cleaned FOON to the embedded cache of object-level plans

    results_dir = f"results_{timestamp}/"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if bool(set(['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP']) & set(evaluated_methods)):

        temp_file_name = 'prototype.txt'

        verbose = False

        preprocess_dir = os.path.join(os.getcwd(), results_dir, 'preprocess/')
        postprocess_dir = os.path.join(os.getcwd(), results_dir, 'postprocess/')

        # -- save the prototype FOON graph as a text file, which we will then run with a parser to correct numbering:
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)

        if not os.path.exists(postprocess_dir):
            os.makedirs(postprocess_dir)

        with open(os.path.join(preprocess_dir, temp_file_name), 'w') as prototype:
            prototype.write("//\tFOON Prototype\n")
            prototype.write(f"//\t-- Task Prompt: {user_task['query']}\n")
            prototype.write(f"//\t-- Required Objects: {llm_output['all_objects']}\n")
            prototype.write("//\n")
            for x in range(len(FOON_prototype)):
                prototype.write(f"//Action {x+1}: {llm_output['object_level_plan']['plan'][x]['instruction']}\n")
                prototype.write(FOON_prototype[x].getFunctionalUnitText())
                if verbose:
                    print(f'{FOON_prototype[x].print_functions[2]()}//')

        # -- running parsing module to ensure that FOON labels and IDs are made consistent for further use:
        #		(it is important that each object and state type have a *UNIQUE* identifier)
        fpa.skip_JSON_conversion = True		# -- we don't need JSON versions of a FOON
        fpa.skip_index_check = True			# -- always create a new set of index files

        fpa.source_dir = preprocess_dir
        fpa.target_dir = postprocess_dir
        fpa._run_parser()

        with open(os.path.join(postprocess_dir, temp_file_name), 'r') as input_file:
            with open(os.path.join(postprocess_dir, task_file_name), 'w') as output_file:
                for line in input_file:
                    output_file.write(line)

        if verbose:
            print(f"\n-- [LLM-to-OLP] : File has been saved as \"./postprocess/{task_file_name}\"")
            print()
            print(os.path.join(os.getcwd(), './postprocess/', task_file_name))

        fga._resetFOON()
        fga._constructFOON(os.path.join(postprocess_dir, task_file_name))

        # embed_olp(
        #     openai_driver,
        #     llm_output,
        #     func_units=fga.FOON_functionalUnits[-1]
        # )

        # print("\n-- [LLM-to-OLP] : a new FOON has been added to embedding file!")

    # %% [markdown] Step 3 :- FOON -> ```FOON-TAMP```/```FOON_to_PDDL```
    # 1. Parse FOON file -- this step is important to ensure that all labels are unique and that the generated file follows the FOON syntax.
    #
    # 2. Run ``FOON_to_PDDL.py`` script to generate FOON macro-operators

    import FOON_TAMP as fta

    all_macro_POs = None

    robot_domain_file = './robot_skills.pddl'

    # ### Generate PDDL files using ```FOON_to_PDDL``` module

    if bool(set(['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP']) & set(evaluated_methods)):

        foon_dir = os.path.join(os.getcwd(), f'results_{timestamp}/', 'output_foon')
        if not os.path.exists(foon_dir):
            os.mkdir(foon_dir)

        FOON_subgraph_file = os.path.join(postprocess_dir, task_file_name)
        micro_problems_path = f'{foon_dir}/micro_problems-' + Path(FOON_subgraph_file).stem

        # -- definition of macro and micro plan file names:
        macro_plan_file = os.path.splitext(FOON_subgraph_file)[0] + '_macro.plan'
        micro_plan_file = os.path.splitext(FOON_subgraph_file)[0] + '_micro.plan'

        fta.micro_domain_file = robot_domain_file

        fta.flag_perception = False

        if os.path.exists(micro_problems_path):
            shutil.rmtree(micro_problems_path)

        # -- create a new folder for the generated problem files and their corresponding plans:
        os.makedirs(micro_problems_path)
        fta.micro_problems_dir = micro_problems_path

        # -- save unparsed output to a JSON file:
        json.dump(llm_output, open(f'{micro_problems_path}/{Path(task_file_name).stem}.json', 'w'), indent=4)

        ftp = fta.ftp

        # -- perform conversion of the FOON subgraph file to PDDL:
        ftp.FOON_subgraph_file = FOON_subgraph_file
        ftp._convert_to_PDDL('OCP')

        ## -- parse through the newly created domain file and find all (macro) planning operators:
        all_macro_POs = {PO.name : PO for PO in fta._parseDomainPDDL(ftp.FOON_domain_file)}

    # %% [markdown] OMPL Parameters
    OMPL_algorithm = 'RRTConnect'
    # OMPL_algorithm = 'LazyPRMstar'
    # OMPL_algorithm = 'BKPIECE1'
    OMPL_attempts = 5
    OMPL_max_compute = 40
    OMPL_max_simplify = 30
    OMPL_len_path = 0

    # %% [markdown] OLP (OMPL+skills)
    # NOTE: all_methods = ['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP', 'OLP-cache', 'OLP-cache-UTAMP', 'LLM-Planner', 'LLM-Planner-UTAMP', 'LLM-UTAMP', 'LLM+P', 'DELTA']


    if bool(set(['OLP', 'FOON']) & set(evaluated_methods)):

        method = 'OLP'

        setup_time = time.time() - total_time

        plan_time = 0

        sim_interfacer = Interfacer(
            scene_file_name=fpath,
            robot_name=coppelia_robot,
            robot_gripper=coppelia_gripper,
            port_number=coppelia_port_number,
        )
        sim_interfacer.start()

        # for i in ["opengl", "opengl3"]:
        for i in ["opengl"]:
            initial_state_img = os.path.join(os.getcwd(), f'results_{timestamp}/', f'{Path(task_file_name).stem}-{i}_trial{trial}.png')
            sim_interfacer.take_snapshot(file_name=initial_state_img, render_mode=i)

        # NOTE: counters for macro- and micro-level steps:
        macro_count, micro_count = 0, 0

        macro_plan = []

        total_success = 0

        all_micro_actions = []

        olp_interaction = list(interaction)

        print("OLP/FOON:")
        print('-- User Task:', user_task['query'])

        # -- before executing the plan, we will prompt the LLM for grounding all objects in the FOON to simulation objects:
        objects_in_FOON = []
        for N in range(len(ftp.fga.FOON_lvl3)):
            for O in ftp.fga.FOON_lvl3[N].getInputNodes() + ftp.fga.FOON_lvl3[N].getOutputNodes():
                objects_in_FOON.append(ftp._reviseObjectLabels(O.getObjectLabel()))
                for x in range(O.getNumberOfStates()):
                    related_obj = O.getRelatedObject(x)
                    if related_obj:
                        objects_in_FOON.append(ftp._reviseObjectLabels(related_obj))

        object_mapping, grounding_interaction = llm_grounding_sim_objects(
            openai_driver,
            objects_in_OLP=objects_in_FOON,
            objects_in_sim=sim_interfacer.objects_in_sim,
            state_as_text=sim_interfacer.perform_sensing(method=3, check_collision=False),
            task=user_task['query'],
        )

        olp_interaction.extend(grounding_interaction)

        print(" -- plan-to-simulation object grounding:", object_mapping)

        for N in range(len(ftp.fga.FOON_lvl3)):
            # NOTE: this is where we have identified a macro plan's step; here, we check the contents of its PO definition for:
            #	1. preconditions - this will become a sub-problem file's initial states (as predicates)
            #	2. effects - this will become a sub-problem file's goal states (as predicates)

            before_planning = time.time()

            macro_PO_name = f'{ftp._reviseObjectLabels(ftp.fga.FOON_lvl3[N].getMotion().getMotionLabel())}_{N}'

            print(" -- [FOON-TAMP] : Searching for micro-level plan for '" + macro_PO_name + "' macro-PO...")

            # print(f"\n{'*' * 30}")
            # ftp.fga.FOON_lvl3[N].print_functions[-1]()
            # print(f"{'*' * 30}\n")

            # -- try to find this step's matching planning operator definition:
            matching_PO_obj = all_macro_POs[macro_PO_name] if macro_PO_name in all_macro_POs else None

            # -- when we find the equivalent planning operator, then we proceed to treat it as its own problem:
            if not matching_PO_obj:
                continue

            # -- parse the goal predicates and replace the generic object names with those of the sim objects:
            print('-- performing object grounding...')
            original_predicates = {
                'preconditions': matching_PO_obj.getPreconditions(),
                'effects': matching_PO_obj.getEffects()
            }
            grounded_predicates = dict(original_predicates)

            for key in grounded_predicates:
                for x in range(len(grounded_predicates[key])):
                    grounded_pred_parts = []
                    for obj in grounded_predicates[key][x][1:-1].split(' '):
                        # -- some predicate args will be split with trailing parentheses (in the case of "not" predicates):
                        obj_no_parentheses = obj.replace('(', '').replace(')', '')
                        # -- we should only do label swapping if the whole argument exists in the grounding map:
                        grounded_pred_parts.append(obj if obj_no_parentheses not in object_mapping else object_mapping[obj_no_parentheses])

                    # -- overwrite the ungrounded predicate with the grounded in simulation version:
                    grounded_predicates[key][x] = f"({' '.join(grounded_pred_parts).strip()})"

            if verbose:
                print("before grounding:", json.dumps(original_predicates, indent=4))
                print("after grounding:", json.dumps(grounded_predicates, indent=4))

            assert bool(grounded_predicates['effects']), f"Error: empty list of goals! Check the generated micro-problem file '{micro_problem_file}'"

            # -- create sub-problem file (micro-level/task-level):
            micro_problem_file = create_problem_file(
                micro_fpath=micro_problems_path,
                action_name=macro_PO_name,
                preconditions=grounded_predicates['preconditions'],
                effects=grounded_predicates['effects'],
                state=sim_interfacer.perform_sensing(check_collision=False, verbose=False)
            )

            # -- create step-relevant domain file (micro-level/task-level):
            micro_domain_file = create_domain_file(
                micro_fpath=micro_problems_path,
                template_domain_fpath=robot_domain_file,
                objects_in_sim=sim_interfacer.objects_in_sim,
                # typing=llm_grounding_pddl_types(openai_driver, sim_interfacer.objects_in_sim),
            )

            micro_plan_file = None
            if not micro_plan_file:
                micro_plan_file = 'sas_plan'

            print('\n\t' + 'step ' + str(N+1) +' -- (' + macro_PO_name + ')')
            macro_plan.append('; step ' + str(N+1) + ' -- (' + macro_PO_name + '):')

            setup_time += time.time() - before_planning

            before_planning = time.time()

            # -- try to find a sub-problem plan / solution:
            result, fd_time = solve(
                find_plan(
                    domain_file=micro_domain_file,
                    problem_file=micro_problem_file,
                    verbose=verbose,
                ),
                verbose=verbose)

            # -- if FD returns something valid, then use FD's time:
            if fd_time:
                plan_time += fd_time
            else:
                plan_time += (time.time() - before_planning)

            successful_execution = True

            if result:
                macro_count += 1

                # -- open the micro problem file, read each line referring to a micro PO, and save to list:
                print('\t-- micro-level plan found as follows:')
                micro_plan = []

                if os.path.exists(os.path.join(os.getcwd(), micro_plan_file)):
                    with open(micro_plan_file, 'r') as micro_file:
                        for L in micro_file:
                            if L.startswith('('):
                                # -- parse the line and remove trailing newline character:
                                micro_step = L.strip()
                                micro_plan.append(micro_step)

                                # -- print entire plan to the command line in format of X.Y,
                                #       where X is the macro-step count and Y is the micro-step count:
                                print('\t\t' + str(N+1) + '.' + str(len(micro_plan)) + '\t:\t' + micro_step)

                micro_count += len(micro_plan)

                all_micro_actions.extend(micro_plan)

                print(f"\n{'*' * 10} ROBOT EXECUTION {'*' * 10}")

                for x in range(len(micro_plan)):
                    micro_step = micro_plan[x]
                    print(f"-- running step {N+1}.{x+1}: {micro_step} -- ", end="")

                    if 'pick' in micro_step:
                        target_object = micro_step[1:-1].split(' ')[1]
                    elif 'place' in micro_step:
                        target_object = micro_step[1:-1].split(' ')[2]

                    # NOTE: seems like Fast-Downward makes everything lowercase...
                    for obj in sim_interfacer.objects_in_sim:
                        if obj.lower() == target_object.lower():
                            target_object = obj

                    print(target_object, f"pick={bool('pick' in micro_step)}", "...", end="")

                    try:
                        result = sim_interfacer.execute(
                            target_object,
                            gripper_action=int(bool('pick' in micro_step)),
                            algorithm=OMPL_algorithm,
                            num_ompl_attempts=OMPL_attempts,
                            max_compute=OMPL_max_compute,
                            max_simplify=OMPL_max_simplify,
                            len_path=OMPL_len_path,
                            method=path_planning_method,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        sim_interfacer.return_home(method=path_planning_method)
                        successful_execution = False
                    else:
                        print(f" {'success' if result else 'failed'}!")
                        total_success += int(result)
                        if terminate_upon_failure and not result:
                            successful_execution = False
                            break

                if not successful_execution:
                    break

            else:
                print('\t-- no micro-level plan found!')
                successful_execution = False
            print()

        print(f"{'*' * 30}\n\n  RESULTS (method=\"{method}\"):")
        print(f"\t-- total number of macro-actions: {macro_count}")
        print(f"\t\t-- success: {macro_count/ len(ftp.fga.FOON_lvl3) * 100.0}% ({macro_count} / {len(ftp.fga.FOON_lvl3)}))")
        print(f"\t-- total number of micro-actions: {micro_count}")
        print(f"\t\t-- success: {(total_success / micro_count * 100.0 if micro_count != 0 else 0.0)}% ({total_success} / {micro_count})")

        with open(f'results_{timestamp}/plan_{method}_trial{trial}.txt', 'w') as plan_file:
            for x in range(len(all_micro_actions)):
                plan_file.write(f"{all_micro_actions[x]}\n")

        # -- count total number of tokens required by method:
        total_tokens = 0
        for msg in olp_interaction:
            total_tokens += openai_driver.num_tokens(msg['content'])
        print(f"\t-- total number of tokens: {total_tokens}")

        json.dump(olp_interaction, open(f'results_{timestamp}/interaction_{method}_trial{trial}.json', 'w'))

        time_taken = f'total time taken: {sim_interfacer.get_elapsed_time()}'
        print(f'\n{time_taken}')
        sim_interfacer.sim_print(time_taken)

        sim_interfacer.return_home(method=path_planning_method)
        sim_interfacer.pause()

        # -- take a screenshot of the final state of the world:
        # for i in ["opengl", "opengl3"]:
        for i in ["opengl"]:
            final_state_img = os.path.join(os.getcwd(), f'results_{timestamp}/', f'{Path(task_file_name).stem}-{method}_trial{trial}-{i}.png')
            sim_interfacer.take_snapshot(file_name=final_state_img, render_mode=i)

        experimental_results.append({
            'result_id': timestamp,
            'method': method,
            'num_blocks': len(sim_interfacer.objects_in_sim),
            'total_robot_actions': micro_count,
            'total_subgoals': len(ftp.fga.FOON_lvl3),
            'total_plan_setup_time': setup_time,
            'total_plan_solve_time': plan_time,
            'total_time': setup_time + plan_time,
            'total_tokens': total_tokens,
            'success': int(successful_execution),
        })

    # %% [markdown] LLM-Planner (OMPL+skills)

    # NOTE: all_methods = ['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP', 'OLP-cache', 'OLP-cache-UTAMP', 'LLM-Planner', 'LLM-Planner-UTAMP', 'LLM-UTAMP', 'LLM+P', 'DELTA']

    if bool(set(['LLM-Planner']) & set(evaluated_methods)):

        method = 'LLM-Planner'

        total_time = time.time()

        # -- use the LLM as a planner to acquire a task plan, where each step will be executed using programmed skills:
        sim_interfacer = Interfacer(
            scene_file_name=fpath,
            robot_name=coppelia_robot,
            robot_gripper=coppelia_gripper,
            port_number=coppelia_port_number,
        )
        sim_interfacer.start()

        # -- we are going to feed the LLM with the prompt of coming up with a plan using the pre-defined skills:
        llm_planner_sys_prompt = open('llm_prompts/llm_planner_system_prompt.txt', 'r').read()

        prompt_context = f"There is a scenario with the following objects: {sim_interfacer.objects_in_sim}. Please await further instructions."
        print("objects presently in scene:", sim_interfacer.objects_in_sim)

        interaction = [
            {"role": "system", "content": llm_planner_sys_prompt},
            {"role": "user", "content": prompt_context}
        ]

        _, response = openai_driver.prompt(interaction)
        interaction.extend([{"role": "assistant", "content": response}])

        prompt_goal = (
            f"Your task is as follows: {user_task['query']}."
            " Transform this instruction into a PDDL goal specification in terms of 'on' relations. Do not add any explanation."
        )

        interaction.extend([{"role": "user", "content": prompt_goal}])

        _, response = openai_driver.prompt(interaction)
        interaction.extend([{"role": "assistant", "content": response}])

        print(f"LLM-generated goal: {response}")

        llm_planner_user_prompt = (
            "Find a task plan in PDDL to achieve this goal given the initial state below."
            " Only specify the list of actions needed."
            " Use the actions defined above. Do not add any explanation.\n\n"
            f"Initial state:\n{sim_interfacer.perform_sensing(method=3).replace('air', 'nothing')}"
        )

        interaction.extend([{"role": "user", "content": llm_planner_user_prompt}])

        _, response = openai_driver.prompt(interaction)
        interaction.extend([{"role": "assistant", "content": response}])

        steps = response.split('\n')
        parsed_steps = []

        # -- first, we need to parse through the plan that the LLM gives us by splitting:
        total_steps = 0

        llm_planner_dir = os.path.join(os.getcwd(), f'results_{timestamp}/', 'output_llm-plan')
        if not os.path.exists(llm_planner_dir):
            os.mkdir(llm_planner_dir)

        # -- write the PDDL domain file generated by the LLM to a file:
        llm_planner_files = f"{llm_planner_dir}/problem-{Path(task_file_name).stem}"

        if not os.path.exists(llm_planner_files):
            os.mkdir(llm_planner_files)

        while True:
            total_steps += 1
            found = False

            for x in range(len(steps)):
                if steps[x].startswith(f"{total_steps}."):
                    parsed_steps.append(steps[x])
                    found = True

            if not found: break

        print("Complete list of actions:")
        print("\n".join(parsed_steps))

        total_time = time.time() - total_time
        plan_time = total_time

        print(f"\n{'*' * 25}\n")

        # -- now that we know the total number of steps, we can go ahead and start to execute each step:
        total_success = 0

        print('LLM-Planner')
        print('-- User Task:', user_task['query'])

        successful_execution = True

        if parsed_steps:

            # -- write the LLM plan's to a file:
            with open(f"{llm_planner_files}/task.plan", "w") as plan_file:
                for x in range(len(parsed_steps)):
                    micro_step = re.search("\(.+\)", parsed_steps[x])[0]
                    plan_file.write(f"{micro_step}\n")

            with open(f'results_{timestamp}/plan_{method}_trial{trial}.txt', 'w') as plan_file:
                for x in range(len(parsed_steps)):
                    micro_step = re.search("\(.+\)", parsed_steps[x])[0]
                    plan_file.write(f"{micro_step}\n")

            for x in range(len(parsed_steps)):
                # -- use regex to help us split the string into smaller components:
                micro_step = re.search("\(.+\)", parsed_steps[x])[0]

                print(f"-- running step {x+1}: {micro_step} -- ", end="")

                if 'pick' in micro_step:
                    target_object = micro_step[1:-1].split(' ')[1]
                elif 'place' in micro_step:
                    target_object = micro_step[1:-1].split(' ')[2]

                try:
                    # -- we will keep track of every successful execution:
                    result = sim_interfacer.execute(
                        target_object,
                        gripper_action=int('pick' in micro_step),
                        algorithm=OMPL_algorithm,
                        num_ompl_attempts=OMPL_attempts,
                        max_compute=OMPL_max_compute,
                        max_simplify=OMPL_max_simplify,
                        len_path=OMPL_len_path,
                        method=path_planning_method,
                    )
                except Exception as e:
                    traceback.print_exc()
                    sim_interfacer.return_home(method=path_planning_method)
                    successful_execution = False
                    break
                else:
                    print(f" {'success' if result else 'failed'}!")
                    total_success += int(result)
                    if terminate_upon_failure and not result:
                        successful_execution = False
                        break

            print('\n\n% Success rate:', float(total_success / len(parsed_steps) * 100.0), f"({total_success}/{len(parsed_steps)})" )

        else:
            successful_execution = False

        # -- count total number of tokens required by method:
        total_tokens = 0
        for msg in interaction:
            total_tokens += openai_driver.num_tokens(msg['content'])
        print(f"\t-- total number of tokens: {total_tokens}")

        json.dump(interaction, open(f'results_{timestamp}/interaction_{method}_trial{trial}.json', 'w'))

        time_taken = f'total time taken: {sim_interfacer.get_elapsed_time()}'
        print(f'\n{time_taken}')
        sim_interfacer.sim_print(time_taken)

        sim_interfacer.return_home(method=path_planning_method)
        sim_interfacer.pause()

        # -- take a screenshot of the final state of the world:
        # for i in ["opengl", "opengl3"]:
        for i in ["opengl"]:
            final_state_img = os.path.join(os.getcwd(), f'results_{timestamp}/', f'{Path(task_file_name).stem}-{method}_trial{trial}-{i}.png')
            sim_interfacer.take_snapshot(file_name=final_state_img, render_mode=i)

        experimental_results.append({
            'result_id': timestamp,
            'method': method,
            'num_blocks': len(sim_interfacer.objects_in_sim),
            'total_robot_actions': len(parsed_steps),
            'total_subgoals': '-',
            'total_plan_setup_time': plan_time,
            'total_plan_solve_time': plan_time,
            'total_time': plan_time,
            'total_tokens': total_tokens,
            'success': int(successful_execution),
        })


    # %% [markdown] ~LLM+P (generate problem file, use FD, run OMPL+skills)

    # NOTE: all_methods = ['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP', 'OLP-cache', 'OLP-cache-UTAMP', 'LLM-Planner', 'LLM-Planner-UTAMP', 'LLM-UTAMP', 'LLM+P', 'DELTA']

    # **NOTE**: This method is akin to [LLM+P (Liu et al. 2023)](https://github.com/Cranial-XIX/llm-pddl/tree/main).
    # We adopt a similar approach where we do the following steps:
    # 1. <u>Problem file generation</u> - given an example problem file, the current state of the environment, and the task, the LLM will generate a PDDL problem file that matches the task.
    # 2. <u>PDDL planning</u> - use a PDDL solver to find a task plan using a pre-defined domain file.
    # 3. <u>Robot execution</u> - if a plan was found in the previous step, run it with OMPL-based skills.

    if bool(set(['LLM+P']) & set(evaluated_methods)):

        method = 'LLM+P'

        setup_time = time.time()

        sim_interfacer = Interfacer(
            scene_file_name=fpath,
            robot_name=coppelia_robot,
            robot_gripper=coppelia_gripper,
            port_number=coppelia_port_number,
        )
        sim_interfacer.start()

        llm_plus_p_dir = os.path.join(os.getcwd(), f'results_{timestamp}/', 'output_llm+p')
        if not os.path.exists(llm_plus_p_dir):
            os.mkdir(llm_plus_p_dir)

        pddl_sys_prompt_file = "llm_prompts/llm+p_system_prompt.txt"

        pddl_system_prompt = open(pddl_sys_prompt_file, 'r').read().replace('<problem_file_example>', top_fewshot_examples(
            openai_driver, fewshot_examples, user_task['query'], method=['llm+p', setting],)[0]['pddl'])

        interaction = [{"role": "system", "content": pddl_system_prompt}]

        pddl_user_prompt = ("Now I have a new planning problem and its description is as follows:\n"
                            f"These objects are on the table: {sim_interfacer.objects_in_sim}."
                            f" The current state of the world is:\n{sim_interfacer.perform_sensing(method=3, check_collision=False)}.")

        interaction.extend([{"role": "user", "content": pddl_user_prompt}])

        pddl_user_prompt = (f"\nYour goal is to achieve this task: {user_task['query']}. "
                            "Provide me with the problem PDDL file that describes the new planning problem directly without further explanations.")

        interaction.extend([{"role": "user", "content": pddl_user_prompt}])

        _, response = openai_driver.prompt(chat_history=interaction, verbose=False)

        interaction.extend([{"role": "assistant", "content": response}])
        if verbose:
            print(json.dumps(interaction, indent=4))

        llm_plus_p_pddl_files = f"{llm_plus_p_dir}/problem-{Path(task_file_name).stem}"
        if os.path.exists(llm_plus_p_pddl_files):
            shutil.rmtree(llm_plus_p_pddl_files)
        os.mkdir(llm_plus_p_pddl_files)

        llm_plus_p_problem_file = f"{llm_plus_p_pddl_files}/problem.pddl"

        with open(llm_plus_p_problem_file, "w") as llm_plus_p_problem:
            llm_plus_p_problem.write(parse_llm_code(response, separator="\n"))

        # -- create step-relevant domain file (task-level):
        llm_plus_p_domain_file = create_domain_file(
            micro_fpath=llm_plus_p_pddl_files,
            template_domain_fpath=robot_domain_file,
            objects_in_sim=sim_interfacer.objects_in_sim,
            domain_name=setting,
            # typing=llm_grounding_pddl_types(openai_driver, sim_interfacer.objects_in_sim),
        )

        micro_plan = []

        setup_time = time.time() - setup_time

        before_planning = time.time()

        # -- try to find a sub-problem plan / solution:
        result, fd_time = solve(
            find_plan(
                domain_file=llm_plus_p_domain_file,
                problem_file=llm_plus_p_problem_file,
                verbose=verbose,
            ),
            verbose=verbose)

        # -- if FD returns something valid, then use FD's time:
        plan_time = (time.time() - before_planning)
        if fd_time:
            plan_time = fd_time

        successful_execution = True

        if result:
            print(f"\n{'*' * 25}\n")

            print('LLM+P plan:')
            print('-- User Task:', user_task['query'])

            print('\t-- micro-level plan found as follows:')

            # -- open the micro problem file, read each line referring to a micro PO, and save to list:
            if os.path.exists("sas_plan"):
                with open('sas_plan', 'r') as micro_file:
                    for L in micro_file:
                        if L.startswith('('):
                            # -- parse the line and remove trailing newline character:
                            micro_step = L.strip()
                            micro_plan.append(micro_step)

                            # -- print entire plan to the command line in format of X.Y,
                            #       where X is the macro-step count and Y is the micro-step count:
                            print('\t\t' + str(macro_count) + '.' + str(len(micro_plan)) + '\t:\t' + micro_step)

                os.remove("sas_plan")

                with open(f'results_{timestamp}/plan_{method}_trial{trial}.txt', 'w') as plan_file:
                    for x in range(len(micro_plan)):
                        plan_file.write(f"{micro_plan[x]}\n")


            print(f"\n{'*' * 25}\n")

            print('LLM+P execution:')

            total_success = 0

            for x in range(len(micro_plan)):
                micro_step = micro_plan[x]
                print(f"-- running step {macro_count}.{x+1}: {micro_step}... ", end="")

                if 'pick' in micro_step:
                    target_object = micro_step[1:-1].split(' ')[1]
                elif 'place' in micro_step:
                    target_object = micro_step[1:-1].split(' ')[2]

                # NOTE: seems like Fast-Downward makes everything lowercase...
                for obj in sim_interfacer.objects_in_sim:
                    if obj.lower() == target_object.lower():
                        target_object = obj

                print(target_object, f"pick={bool('pick' in micro_step)}", '...', end='')

                try:
                    result = sim_interfacer.execute(
                        target_object,
                        gripper_action=int('pick' in micro_step),
                        algorithm=OMPL_algorithm,
                        num_ompl_attempts=OMPL_attempts,
                        max_compute=OMPL_max_compute,
                        max_simplify=OMPL_max_simplify,
                        len_path=OMPL_len_path,
                        method=path_planning_method,
                    )
                except Exception as e:
                    traceback.print_exc()
                    sim_interfacer.return_home()
                    successful_execution = False
                    break
                else:
                    print(f" {'success' if result else 'failed'}!")
                    total_success += int(result)
                    if terminate_upon_failure and not result:
                        successful_execution = False
                        break

            print(f"{'*' * 30}\n\n  RESULTS (method=\"{method}\"):")
            print(f"\t-- total number of micro-actions: {len(micro_plan)}")
            if len(micro_plan):
                print(f"\t\t-- success: {total_success / len(micro_plan) * 100.0}%")
            else:
                print("\t\t-- success: 100.0%")
        else:
            print(f"{'*' * 30}\n\n  RESULTS (method=\"{method}\"):")
            print("\t-- no plan found!\n\t\tsuccess: 0%")
            successful_execution = False

        # -- count total number of tokens required by method:
        total_tokens = 0
        for msg in interaction:
            total_tokens += openai_driver.num_tokens(msg['content'])
        print(f"\t-- total number of tokens: {total_tokens}")

        json.dump(interaction, open(f'results_{timestamp}/interaction_{method}_trial{trial}.json', 'w'))

        experimental_results.append({
            'result_id': timestamp,
            'method': method,
            'num_blocks': len(sim_interfacer.objects_in_sim),
            'total_robot_actions': len(micro_plan),
            'total_subgoals': '-',
            'total_plan_setup_time': setup_time,
            'total_plan_solve_time': plan_time,
            'total_time': setup_time + plan_time,
            'total_tokens': total_tokens,
            'success': int(successful_execution),
        })

        time_taken = f'total time taken: {sim_interfacer.get_elapsed_time()}'
        print(f'\n{time_taken}')
        sim_interfacer.sim_print(time_taken)

        sim_interfacer.return_home(method=path_planning_method)
        sim_interfacer.pause()

        # -- take a screenshot of the final state of the world:
        # for i in ["opengl", "opengl3"]:
        for i in ["opengl"]:
            final_state_img = os.path.join(os.getcwd(), f'results_{timestamp}/', f'{Path(task_file_name).stem}-{method}_trial{trial}-{i}.png')
            sim_interfacer.take_snapshot(file_name=final_state_img, render_mode=i)

    # %% [markdown] ~DELTA (generate domain+problem files, run FD, run OMPL+skills)

    # **NOTE**: This baseline is akin to [DELTA (Liu et al. 2024)](https://arxiv.org/abs/2404.03275) but with slight modifications to work with a similar pipeline.
    # This method works as follows:
    # 1. <u>Domain file generation</u> - given an example domain file with pick and place planning operators, the LLM generates a new domain file.
    # 2. <u>Problem file generation</u> - given an example problem file, the LLM generates a comprehensive problem file (similar to LLM+P).
    # 3. <u>Subgoal problem generation</u> - given an example of subgoal decomposition, the LLM generates a set of subgoals, from which PDDL problem files are generated per subgoal.
    # 4. <u>PDDL planning</u> - use a PDDL solver to find a task plan using the generated domain file (step 1) and each subgoal problem.
    # 5. <u>Robot execution</u> - if a plan was found in the previous step, run it with OMPL-based skills.

    # NOTE: all_methods = ['FOON', 'FOON-UTAMP', 'OLP', 'OLP-UTAMP', 'OLP-cache', 'OLP-cache-UTAMP', 'LLM-Planner', 'LLM-Planner-UTAMP', 'LLM-UTAMP', 'LLM+P', 'DELTA']

    if bool(set(['DELTA']) & set(evaluated_methods)):

        method = 'DELTA'

        setup_time = time.time()

        sim_interfacer = Interfacer(
            scene_file_name=fpath,
            robot_name=coppelia_robot,
            robot_gripper=coppelia_gripper,
            port_number=coppelia_port_number,
        )
        sim_interfacer.start()

        delta_dir = os.path.join(os.getcwd(), f'results_{timestamp}/', 'output_delta')
        if not os.path.exists(delta_dir):
            os.mkdir(delta_dir)

        # -- make a folder for all PDDL files for this task:
        delta_pddl_files = f"{delta_dir}/problem-{Path(task_file_name).stem}"
        if os.path.exists(delta_pddl_files):
            shutil.rmtree(delta_pddl_files)
        os.mkdir(delta_pddl_files)

        best_example = top_fewshot_examples(
                openai_driver,
                fewshot_examples,
                user_task['query'],
                method=['delta', setting],)[0]

        domain_prompt = open("llm_prompts/delta_domain_prompt.txt", 'r').read().replace(
            "<domain_file_example>",
            best_example['domain_file_prompt']
            ).replace(
                "<objects_in_sim>",
                str(sim_interfacer.objects_in_sim)
            )

        interaction = [{"role": "user", "content": domain_prompt}]
        _, pddl_domain = openai_driver.prompt(interaction, verbose=False)
        interaction.extend([{"role": "assistant", "content": pddl_domain}])

        if verbose:
            print(pddl_domain)

        # -- write the PDDL domain file generated by the LLM to a file:
        with open(f"{delta_pddl_files}/domain.pddl", "w") as domain_file:
            domain_file.write(parse_llm_code(pddl_domain, separator="\n"))

        problem_prompt = open("llm_prompts/delta_problem_prompt.txt", 'r').read().replace(
            "<problem_file_example>",
            best_example['problem_file_prompt']
            ).replace(
                "<new_task>",
                f"{user_task['query']}"
            ).replace(
                "<current_state>",
                f"{sim_interfacer.perform_sensing(method=3, check_collision=False)}"
            )

        interaction.extend([{"role": "user", "content": problem_prompt}])
        _, pddl_problem = openai_driver.prompt(interaction, verbose=False)
        interaction.extend([{"role": "assistant", "content": pddl_problem}])

        if verbose:
            print(pddl_problem)

        # -- write the PDDL problem file generated by the LLM to a file:
        with open(f"{delta_pddl_files}/problem.pddl", "w") as problem_file:
            problem_file.write(parse_llm_code(pddl_problem, separator="\n"))

        # -- now we do subgoal generation:
        subtasks_prompt = open("llm_prompts/delta_subgoals_prompt.txt", 'r').read().replace(
            "<subgoals_example>",
            best_example['subgoals_prompt']
        )

        interaction.extend([{"role": "user", "content": subtasks_prompt}])
        _, pddl_subgoals = openai_driver.prompt(interaction, verbose=False)
        interaction.extend([{"role": "assistant", "content": pddl_subgoals}])

        parsed_subgoals = {}

        steps = [x.strip() for x in pddl_subgoals.split('\n')]

        # -- first, we need to parse through the plan that the LLM gives us by splitting:
        total_subgoals = 1
        total_actions = 0

        while True:
            found_subgoal = False

            for x in range(len(steps)):
                if steps[x].startswith(f"{total_subgoals}. "):
                    found_subgoal = True    # -- we found a valid subgoal action specified in natural language
                    found_pddl = False      # -- we now need to find all related PDDL subgoals

                    # -- we will compile all subgoals in this dictionary:
                    parsed_subgoals[total_subgoals] = {'description': steps[x].strip(), 'pddl': []}
                    count = 0

                    while True:
                        count += 1
                        found_pddl = False
                        for y in range(x, len(steps)):
                            if steps[y].startswith(f"{total_subgoals}.{count}."):
                                # -- use regex to extract the PDDL subgoal from text:
                                pddl_subgoal = re.search("\(.+\)", steps[y])
                                if pddl_subgoal:
                                    parsed_subgoals[total_subgoals]['pddl'].append(pddl_subgoal[0])
                                    found_pddl = True

                        if not found_pddl: break

            if not found_subgoal:
                total_subgoals -= 1
                break

            total_subgoals += 1


        total_success = 0

        all_actions = []

        setup_time, plan_time = time.time() - setup_time, 0

        successful_execution = True

        print('DELTA')
        print('-- User Task:', user_task['query'])

        if not parsed_subgoals or not total_subgoals:
            successful_execution = False
        else:
            for x in sorted(list(parsed_subgoals.keys())):

                more_setup = time.time()

                # -- create sub-problem file (micro-level/task-level):
                micro_problem_file = create_problem_file(
                    micro_fpath=delta_pddl_files,
                    preconditions=[],
                    effects=parsed_subgoals[x]['pddl'],
                    action_name=f"sub_problem-{x}",
                    domain_name=re.search("\(:domain.+\)", pddl_problem)[0][1:-1].split(' ')[1],
                    state=sim_interfacer.perform_sensing(check_collision=False, verbose=False),
                )

                setup_time += time.time() - more_setup

                before_planning = time.time()

                # -- try to find a sub-problem plan / solution:
                result, fd_time = solve(
                    find_plan(
                        domain_file=f"{delta_pddl_files}/domain.pddl",
                        problem_file=f"{delta_pddl_files}/sub_problem-{x}.pddl",
                        verbose=verbose,
                    ),
                    verbose=verbose)

                # -- if FD returns something valid, then use FD's time:
                if fd_time:
                    plan_time += fd_time
                else:
                    plan_time += time.time() - before_planning

                if result:
                    print(f"\n{'*' * 25}\n")

                    print(f"\t-- plan found for sub-goal problem {x}: \"{parsed_subgoals[x]['description']}\"!")

                    # -- open the micro problem file, read each line referring to a micro PO, and save to list:
                    micro_plan = []

                    micro_plan_file = 'sas_plan'
                    if os.path.exists(micro_plan_file):
                        with open(micro_plan_file, 'r') as micro_file:
                            for L in micro_file:
                                if L.startswith('('):
                                    # -- parse the line and remove trailing newline character:
                                    micro_step = L.strip()
                                    micro_plan.append(micro_step)

                                    # -- print entire plan to the command line in format of X.Y,
                                    #       where X is the macro-step count and Y is the micro-step count:
                                    print('\t\t' + str(x) + '.' + str(len(micro_plan)) + '\t:\t' + micro_step)

                    all_actions.extend(micro_plan)

                    print(f"\n{'*' * 25}\n")

                    print('DELTA execution:')

                    total_success += 1

                    for y in range(len(micro_plan)):
                        micro_step = micro_plan[y]
                        print(f"-- running step {x}.{y+1}: {micro_step}... ", end="")

                        if 'pick' in micro_step:
                            target_object = micro_step[1:-1].split(' ')[1]
                        elif 'place' in micro_step:
                            target_object = micro_step[1:-1].split(' ')[2]

                        # NOTE: seems like Fast-Downward makes everything lowercase...
                        for obj in sim_interfacer.objects_in_sim:
                            if obj.lower() == target_object.lower():
                                target_object = obj

                        print(target_object, f"pick={bool('pick' in micro_step)}", '...', end='')

                        try:
                            result = sim_interfacer.execute(
                                target_object,
                                gripper_action=int('pick' in micro_step),
                                algorithm=OMPL_algorithm,
                                num_ompl_attempts=OMPL_attempts,
                                max_compute=OMPL_max_compute,
                                max_simplify=OMPL_max_simplify,
                                len_path=OMPL_len_path,
                                method=path_planning_method,
                                )
                        except Exception as e:
                            traceback.print_exc()
                            sim_interfacer.return_home()
                            successful_execution = False
                            break
                        else:
                            print(f" {'success' if result else 'failed'}!")
                            total_actions += int(result)
                            if terminate_upon_failure and not result:
                                successful_execution = False
                                break

                    if not successful_execution:
                        break

                else:
                    print(f"\t-- no plan found for sub-goal problem {x}: \"{parsed_subgoals[x]['description']}\"!")
                    successful_execution = False

        print(f"{'*' * 30}\n\n  RESULTS (method=\"{method}\"):")
        if total_subgoals:
            print(f"\t-- total number of sub-tasks: {total_subgoals}")
            print(f"\t\t-- success: {total_success / total_subgoals * 100.0}% ({total_success} / {total_subgoals})")
            print(f"\t-- total number of micro-actions: {len(all_actions)}")
            if len(all_actions):
                print(f"\t\t-- success: {total_actions / len(all_actions) * 100.0}% ({total_actions} / {len(all_actions)})")
            else:
                print(f"\t\t-- success: 100.0% ({total_actions} / {len(all_actions)})")

        with open(f'results_{timestamp}/plan_{method}_trial{trial}.txt', 'w') as plan_file:
            for x in range(len(all_actions)):
                plan_file.write(f"{all_actions[x]}\n")

        # -- count total number of tokens required by method:
        total_tokens = 0
        for msg in interaction:
            total_tokens += openai_driver.num_tokens(msg['content'])
        print(f"\t-- total number of tokens: {total_tokens}")

        json.dump(interaction, open(f'results_{timestamp}/interaction_{method}_trial{trial}.json', 'w'))

        experimental_results.append({
            'result_id': timestamp,
            'method': method,
            'num_blocks': len(sim_interfacer.objects_in_sim),
            'total_robot_actions': len(all_actions),
            'total_subgoals': total_subgoals,
            'total_plan_setup_time': setup_time,
            'total_plan_solve_time': plan_time if plan_time > 0 else '-',
            'total_time': setup_time + plan_time,
            'total_tokens': total_tokens,
            'success': int(successful_execution)
        })

        time_taken = f'total time taken: {sim_interfacer.get_elapsed_time()}'
        print(f'\n{time_taken}')
        sim_interfacer.sim_print(time_taken)

        sim_interfacer.return_home(method=path_planning_method)
        sim_interfacer.pause()

        # -- take a screenshot of the final state of the world:
        # for i in ["opengl", "opengl3"]:
        for i in ["opengl"]:
            final_state_img = os.path.join(os.getcwd(), f'results_{timestamp}/', f'{Path(task_file_name).stem}-{method}_trial{trial}-{i}.png')
            sim_interfacer.take_snapshot(file_name=final_state_img, render_mode=i)

    sim_interfacer.stop()


    # %% [markdown]
    # ## Step 5 :- Write all results to a file
    # columns = ['result_id', 'method', 'total_plan_generation_time', 'total_robot_actions', 'total_subgoals', 'total_tokens', 'num_blocks']
    results_fpath = f'./results_{timestamp}/task={task}_results_trial{trial}.csv'
    df = pd.DataFrame(experimental_results)
    df.to_csv(results_fpath, index=False)

    os.remove(fpath)

    return results_fpath


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default="23000"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="organize",
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default="10"
    )

    parser.add_argument(
        "-v",
        type=bool,
    )

    args = parser.parse_args()
    if args.v:
        verbose = True

    # -- get args from command line:
    coppelia_port_number = args.port
    task = args.task
    num_trials = args.ntrials

    all_results = []
    count_exceptions = 0

    all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U']

    # NOTE: distribution of blocks for organizing task:
    distribution = [
        # [2, 2], [3, 2], [3, 3], [4, 3], [4, 4],
        # [2, 2, 2], [3, 2, 2], [3, 3, 2], [3, 3, 3], [4, 3, 3], [4, 4, 3],
        [4, 4, 4],
        # [4, 4, 4], [4, 4, 3], [4, 3, 3], [3, 3, 3], [3, 3, 2], [3, 2, 2], [2, 2, 2],
        # [2, 2, 2, 2], [3, 2, 2, 2], [3, 3, 2, 2], [3, 3, 3, 2], [3, 3, 3, 3],
        # [4, 3, 3, 3], [4, 4, 3, 3],
        # [4, 4, 4, 3], [4, 4, 4, 4],
    ]
    # distribution.reverse()

    # myconda; conda activate olp; python3 exp.py --task "organize" --port 23017

    # NOTE: words for spelling task:
    words = [
        # list("DR"),
        # list("FUN"),
        # list("PLAN"),
        # list("SKILL"),
        # list("MOTION"),
        list("DREAMER"),
        # list("ROBOTICS"),
    ]

    # NOTE: heights for "towers" task:
    heights = [x for x in range(7, 8)]

    last_exec = None

    for item in (words if task == "spelling" else heights if task == "towers" else distribution):
        last_exec = item

        # -- select a random subset of letters:
        attributes = {"alphabets": {}}
        if task == "spelling":
            for x in range(len(item)):
                if item[x] not in attributes['alphabets']:
                    attributes['alphabets'][item[x]] = {
                        'texture': os.path.join(os.getcwd(), f'./utamp/textures/{item[x].lower()}.png'),
                        'count': 1,
                    }
                else:
                    attributes['alphabets'][item[x]]['count'] += 1

        # -- create a folder to store the results from a certain run:
        timestamp = dt.today().strftime('%Y-%m-%d_%H-%M-%SS')
        if not os.path.exists(os.path.join(os.getcwd(), f'results_{timestamp}/')):
            os.mkdir(os.path.join(os.getcwd(), f'results_{timestamp}/'))

        end = False

        count = 0
        while count < num_trials:

            coppelia_process = subprocess.Popen(
                # ["python3", "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliasim.py", "-G", f"\"zmqRemoteApi.rpcPort={coppelia_port_number}\"" ],
                ["C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/coppeliaSim.exe", f"-GzmqRemoteApi.rpcPort={coppelia_port_number}"],
                # ["../CoppeliaSim/coppeliaSim", f"-GzmqRemoteApi.rpcPort={coppelia_port_number}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )

            attr = copy.deepcopy(attributes)
            if task == "spelling":
                # -- randomly add distractor objects:
                subset = sample(list(set(all_letters) - set(item)), randint(1, 3))
                # for x in range(len(subset)):
                #     attr['alphabets'].update({
                #         subset[x]: {
                #             "texture": os.path.join(os.getcwd(), f'./utamp/textures/{subset[x].lower()}.png'),
                #             "count": randint(1, 3)
                #         }
                #     })

            elif task == "towers":
                subset = sample(all_letters, item+1)
                for x in range(len(subset)):
                    attr['alphabets'].update({
                        subset[x]: {
                            "texture": os.path.join(os.getcwd(), f'./utamp/textures/{subset[x].lower()}.png'),
                            "count": 1
                        }
                    })

            elif task == "organize":
                subset = sample(all_letters, len(item))
                for x in range(len(subset)):
                    attr['alphabets'].update({
                        subset[x]: {
                            "texture": os.path.join(os.getcwd(), f'./utamp/textures/{subset[x].lower()}.png'),
                            "count": item[x]
                        }
                    })
            try:
                arg = None
                if task == "spelling":
                    arg = str("".join(item))
                elif task == "towers":
                    arg = int(item)

                results_fpath = main(attr, timestamp, count, arg)
            except Exception as e:
                traceback.print_exc()
                if isinstance(e, KeyboardInterrupt):
                    end = True
                    print("task:", task, ":- last exec:", last_exec)
                    break
                else:
                    continue

            # -- kill CoppeliaSim for this trial:
            coppelia_process.terminate()
            coppelia_process.wait()

            all_results.append(results_fpath)
            count += 1

        if end: break


    if all_results:
        experimental_results = {}
        for F in all_results:
            with open(F, 'r') as results_file:
                df = pd.read_csv(results_file)
                experimental_results.update(df.to_dict())

        if experimental_results:
            df = pd.DataFrame(experimental_results)
            df.to_csv('./all_experimental_results.csv', index=False)

        print('exception count:', count_exceptions)

    print(f"{'-' * 5} end of task '{task}' {'-' * 5}")
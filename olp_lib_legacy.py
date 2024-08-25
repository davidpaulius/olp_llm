#############################################    Imports   #####################################################
import openai
import json
import sys
import random
import re
import os
# import spacy

from pathlib import Path

##############################################################################################################

FOON_API_path = './foon_to_pddl/foon_api'
if FOON_API_path not in sys.path:
    sys.path.append(FOON_API_path)

try:
    import FOON_graph_analyser as fga
    import FOON_parser as fpa
except ImportError:
    sys.exit()


##############################################################################################################


def prompt_LLM(given_prompt, model_name='gpt-4', verbose=False):
    if verbose:
        print(f"Model: {model_name}\nComplete prompt:")
        print(json.dumps(given_prompt, indent=4))

    # -- create a completion request:
    completion = openai.chat.completions.create(
        model=model_name,
        messages=given_prompt,
        temperature=0.25,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    if verbose:
        print("*************************************************************************")

    response = completion.choices[0].message

    return (response.role, response.content)


def generate_OLP(query_task, incontext_file, llm_model,
                 stage1_sys_prompt_file="./llm_prompts/stage1_system_prompt.txt",
                 stage2_sys_prompt_file="./llm_prompts/stage2_system_prompt.txt",
                 verbose=True):
    # -- we will keep track of the entire interaction for context:
    message = []

    ######################################################################################
    # NOTE: Stage 1 prompting:
    ######################################################################################

    def get_stage1_objects(response):
        # NOTE: this function will parse the output generated from Stage 1 prompting to identify all objects needed for a task:

        # -- we will look for the line in the output indicat
        unique_objects = []
        for line in response.lower().split("\n"):
            if "unique_objects" in line or "unique objects" in line:
                try:
                    # -- we use eval() to remove any quotations that indicate some type of string:
                    unique_objects = eval(line.split(":")[1].strip())
                except Exception:
                    # -- if for whatever reason the LLM does not list objects enclosed in quotations,
                    #       then we just split like normal:
                    objects = line.split(":")[1].strip().split(",")
                    unique_objects = [x.strip().replace(".", "")
                                      for x in objects]

        return unique_objects
    # enddef

    if verbose:
        print("*************************************************************************\n"
              "Stage 1 Prompting\n"
              "*************************************************************************")

    # NOTE: here is the list of instructions given to the LLM for generating a high-level plan as part of stage 1:
    system_prompt = open(stage1_sys_prompt_file, 'r').read()

    # -- get a list of instructions that satisfies the given prompt:
    stage1_prompt = f"Generate a high-level plan for the following task prompt:\n{query_task}"

    message.extend([{"role": "system", "content": system_prompt},
                    {"role": "user", "content": stage1_prompt}])

    _, stage1_response = prompt_LLM(message, llm_model, verbose)
    if verbose:
        print("\n*************************************************************************")
        print(f"Stage 1 Response: \n{stage1_response}")
        print("*************************************************************************\n")

    # -- get the list of all objects used in the generated plan:
    stage1b_prompt = "List all objects created by or needed in the high-level plan."\
        " Follow the format: 'unique_objects:[\"object_1\", \"object_2\", \"object_3\", ...]'"

    message.extend([{"role": "assistant", "content": stage1_response},
                    {"role": "user", "content": stage1b_prompt}])

    _, stage1b_response = prompt_LLM(message, llm_model, verbose)
    unique_objects = get_stage1_objects(stage1b_response)
    if verbose:
        print(f"\nExtracted unique_objects:", unique_objects)

    ######################################################################################
    # NOTE: Stage 2 prompting:
    ######################################################################################

    if verbose:
        print("*************************************************************************\n",
              "Stage 2 Prompting\n",
              "*************************************************************************")

    stage2_user_msg = open(stage2_sys_prompt_file, 'r').read().replace("<obj_set>", str(unique_objects))

    if '.json' in str(incontext_file).lower():
        # NOTE: we will be selecting a random example from a JSON file containing examples:
        incontext_examples = json.load(open(incontext_file))
        selected_example = random.choice(incontext_examples)

        stage2_prompt = f"{stage2_user_msg}\n\nFormat your output as a JSON structure like in the following example:\n{json.dumps(selected_example, indent=4)}"
    else:
        # -- use older format of incontext examples:
        incontext_examples = generate_incontext_examples(
            incontext_file)  # generate incontext examples
        stage2_prompt = f"\n{stage2_user_msg}\n{incontext_examples}"

    message.extend([{"role": "assistant", "content": stage1_response},
                    {"role": "user", "content": stage2_prompt}])

    _, stage2_response = prompt_LLM(message, llm_model, verbose)

    if verbose:
        print("\n*************************************************************************")
        print(f"Stage 2 Response: \n{stage2_response}")
        print("*************************************************************************\n")

    if '.json' in str(incontext_file).lower():
        # -- use eval() function to parse through the Stage 2 prompt response obtained from LLM:
        object_level_plan = eval(stage2_response)
    else:
        # -- use regex to parse the output obtained from the LLM:
        object_level_plan = []
        for olp_unit in stage2_response.split("\n\n"):
            object_level_plan.append(parse_regex_unit(olp_unit))

    if verbose:
        print("*************************************************************************\n",
              "Stage 3 Prompting\n",
              "*************************************************************************")

    stage3_prompt = "Which step(s) describe the final state required to fulfill the task?"\
        " Think back to the high-level plan generated above."\
        " List the step numbers in an array, such as: [X, Y, Z], where X to Z are integers."
    message.extend([{"role": "user", "content": stage3_prompt}])

    _, stage3_response = prompt_LLM(message, llm_model, verbose)
    if verbose:
        print(stage3_response, '\n')

    stage3_terminalSteps = eval(re.findall(r'\[.+?\]', stage3_response)[0])

    return {
        'PlanSketch': f'{stage1_response}\n\n{stage1b_response}',
        'OLP': object_level_plan,
        'TerminalSteps': stage3_terminalSteps,
        'RelevantObjects': unique_objects,
    }


def generate_incontext_examples(incontext_file):
    ret = ''
    lines = []
    with open(incontext_file, "r") as f:
        lines.extend(f.readlines())
    for line in lines:
        ret += line
    return ret


def parse_regex_states(state_changes_str):
    state_changes = {}
    # Splitting at '},', which indicates the end of an entry, followed by a new object name
    entries = re.split(r'\},\s*(?=[\w\s]+:)', state_changes_str.strip())

    for entry in entries:
        # Extract object name
        object_name_match = re.match(r'([\w\s]+):{', entry)
        if object_name_match:
            object_name = object_name_match.group(1).strip()
            # Rest of the string contains the changes
            changes_str = entry[len(object_name) + 2:].rstrip('}')
            changes_dict = {}
            for change_detail in re.finditer(r'(\w+):\s*\[(.+?)\]', changes_str):
                change_type, change_values = change_detail.groups()
                changes_dict[change_type] = change_values.split(', ')
            state_changes[object_name] = changes_dict

    return state_changes


def parse_regex_unit(olp_unit):
    # Regular expression patterns
    # step_pattern = r'Step [0-9]+: (.+)\.'
    step_pattern = r'Step [0-9]+: (.+)'
    objects_pattern = r'Objects: \[(.+)\]'
    action_pattern = r'Action: (.+)'
    state_changes_pattern = r'StateChanges:{(.+?)}\s*}$'

    # Extracting data using regular expressions
    step_match = re.search(step_pattern, olp_unit)
    objects_match = re.search(objects_pattern, olp_unit)
    action_match = re.search(action_pattern, olp_unit)
    state_changes_match = re.search(state_changes_pattern, olp_unit, re.DOTALL)

    # Parsing extracted data
    step = step_match.group(1) if step_match else None
    objects = objects_match.group(1).split(', ') if objects_match else []
    action = action_match.group(1) if action_match else None
    state_changes = parse_regex_states(
        state_changes_match.group(1)) if state_changes_match else {}

    print(step)
    print(objects)
    print(action)
    print(state_changes)
    print(state_changes_match.group(1))
    input()

    # Constructing the dictionary
    instruction_dict = {
        'Step': eval(step),
        'Objects': [eval(x) for x in objects],
        'Action': action,
        'StateChanges': state_changes
    }

    return instruction_dict


def create_functionalUnit_ver1(plan_step, plan_objects):
    functionalUnit = fga.FOON.FunctionalUnit()

    used_objects = plan_step['Objects']
    action = plan_step['Action']
    step_info = plan_step['Step']
    print("Creating functional unit for:", step_info)
    print("Objects:", used_objects)
    print(plan_step['StateChanges'].keys())

    geometric_states = ['in', 'on', 'under', 'contains']

    for obj in used_objects:
        # create object node
        print("Current object is :", obj)

        try:
            # check object state changes in step
            object_state_changes = plan_step['StateChanges'][obj]
        except KeyError:
            print(f'Missing object: {obj}')
            continue
        else:
            print("\t", obj, "state changes:", object_state_changes)

        for state_type in ['Precondition', 'Effect']:
            # -- create a FOON object node for which we will then add attributes:
            new_object = fga.FOON.Object(objectLabel=obj)

            for state in object_state_changes[state_type]:
                # -- check if state effect involves another object in step
                related_obj = None

                parsed_state = state

                for O in plan_objects:
                    if O in state:
                        # -- check if there is no object left after removing the related object given by the LLM:
                        state_sans_obj = parsed_state.replace(O, "").strip()
                        # -- we remove the name of the related object from the state attribute string:
                        if state_sans_obj in geometric_states:
                            related_obj = O
                            parsed_state = parsed_state.replace(
                                related_obj, "").strip()

                if not related_obj:
                    # -- this means we have an object not listed initially by the LLM:
                    #       (usually references to containers or states not foreseen at the time of recipe generation)
                    for G in geometric_states:
                        if f'{G} ' in parsed_state:
                            related_obj = parsed_state.split(f'{G} ')[1]
                            parsed_state = parsed_state.replace(
                                related_obj, "").strip()

                if "contains" in state:
                    # -- this means we have a state expressing some kind of containment:
                    try:
                        new_object.addContainedObject(related_obj)
                    except Exception:
                        print(state)
                        print(related_obj)
                    related_obj = None

                # -- add each state effect to object node:
                if related_obj:
                    new_object.addNewState([None, parsed_state, related_obj])
                else:
                    new_object.addNewState([None, parsed_state, None])

                print("\t\t", state_type, ":", state,
                      "|| related objects:", related_obj)
            # -- add the node to the functional unit:
            functionalUnit.addObjectNode(objectNode=new_object, is_input=(
                True if state_type == 'Precondition' else False))

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action)
    functionalUnit.setMotionNode(newMotion)

    return functionalUnit


def create_functionalUnit_ver2(llm_output, index):
    # NOTE: version 2 -- strict JSON format

    # -- create a functional unit prototype:
    functionalUnit = fga.FOON.FunctionalUnit()

    olp = llm_output['OLP']['Instructions'][index]
    olp_objects = llm_output['OLP']['CompleteObjectSet']

    used_objects = olp['RelatedObjects']
    action = olp['Action']
    print(f"Creating functional unit for Step {olp['Step']}: {olp['Instruction']}")
    print("-> related objects:", used_objects, "\n")

    geometric_states = ['in', 'on', 'under', 'contains']

    for obj in used_objects:

        if obj in ['table', 'surface']:
            # -- we will remove any references to the table for object nodes:
            continue

        # -- create object node:
        print("Current object is :", obj)

        try:
            # check object state changes in step
            object_state_changes = olp['State'][obj]
        except KeyError:
            print(f'Missing object: {obj}')
            continue
        else:
            print("\t", obj, "state changes:", object_state_changes)

        for state_type in ['Precondition', 'Effect']:
            # -- create a FOON object node for which we will then add attributes:
            new_object = fga.FOON.Object(objectLabel=obj)

            for state in object_state_changes[state_type]:
                # -- check if state effect involves another object in step
                related_obj = None

                parsed_state = state

                for O in olp_objects:
                    if O in state:
                        # -- check if there is no object left after removing the related object given by the LLM:
                        state_sans_obj = parsed_state.replace(O, "").strip()
                        # -- we remove the name of the related object from the state attribute string:
                        if state_sans_obj in geometric_states:
                            related_obj = O
                            parsed_state = parsed_state.replace(
                                related_obj, "").strip()

                if not related_obj:
                    # -- this means we have an object not listed initially by the LLM:
                    #       (usually references to containers or states not foreseen at the time of recipe generation)
                    for G in geometric_states:
                        if f'{G} ' in parsed_state:
                            related_obj = parsed_state.split(f'{G} ')[1]
                            parsed_state = parsed_state.replace(
                                related_obj, "").strip()


                if "contains" in state:
                    # -- this means we have a state expressing some kind of containment:
                    try:
                        new_object.addContainedObject(related_obj)
                    except Exception:
                        print(state)
                        print(related_obj)
                    related_obj = None

                # -- add each state effect to object node:
                if related_obj:
                    new_object.addNewState([None, parsed_state, related_obj])
                else:
                    new_object.addNewState([None, parsed_state, None])

                print("\t\t", state_type, ":", state,
                      "|| related objects:", related_obj)
            # -- add the node to the functional unit:
            functionalUnit.addObjectNode(objectNode=new_object, is_input=(
                True if state_type == 'Precondition' else False))

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action)
    functionalUnit.setMotionNode(newMotion)

    return functionalUnit

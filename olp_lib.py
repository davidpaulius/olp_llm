#############################################    Imports   #####################################################
import sys
import os
import re
import pickle
import json

from openai import OpenAI
from random import randint, choice, shuffle

FOON_API_path = './foon_to_pddl/foon_api'
if FOON_API_path not in sys.path:
    sys.path.append(FOON_API_path)

import FOON_graph_analyser as fga
import FOON_parser as fpa


##############################################################################################################

class OpenAIInterfacer(object):
    def __init__(self, model_embed="text-embedding-3-small", model_text_gen="gpt-4-turbo"):
        self.model_text_gen = model_text_gen
        self.model_embed = model_embed
        self.client = OpenAI()

    def prompt(self, prompt,
               args={'temperature': 0.1, 'max_tokens': 2000, 'frequency_penalty': 0, 'presence_penalty': 0},
               verbose=False):

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_text_gen}\nComplete prompt:\n{json.dumps(prompt, indent=4)}")

        # -- create a completion request:
        completion = self.client.chat.completions.create(
            model=self.model_text_gen,
            messages=prompt,
            temperature=args['temperature'],
            max_tokens=args['max_tokens'],
            top_p=1,
            frequency_penalty=args['frequency_penalty'],
            presence_penalty=args['presence_penalty'],
        )

        if verbose:
            print("*" * 75)

        # -- extract the message component from the returned completion object:
        response = completion.choices[0].message

        return (response.role, response.content)

    def embed(self, text, verbose=False):

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_embed}\nText to embed: '{text}'")

        return self.client.embeddings.create(
            input=text, model=self.model_embed).data[0].embedding


def add_to_embeddings(openai_obj, text, func_unit, embedding_fpath='olp_embeddings.pkl', verbose=False):
    # NOTE: this function will be used to create a storage file with embeddings that will be used for retrieval

    embeds = []
    # -- check if there is an existing embedding file:
    if os.path.exists(embedding_fpath):
        while True:
            try:
                embeds.append(pickle.load(embedding_fpath))
            except EOFError:
                break

    embeds.append({
        'original_text': text,
        'embedding': openai_obj.embed(text,verbose=verbose),
        'functional_units': func_unit,
        })

    pickle.dump(open(embedding_fpath, 'wb'))


def generate_OLP(openai_obj, query, incontext_file,
                 stage1_sys_prompt_file="./llm_prompts/stage1_system_prompt.txt",
                 stage2_sys_prompt_file="./llm_prompts/stage2_system_prompt.txt",
                 stage3_sys_prompt_file="./llm_prompts/stage3_system_prompt.txt",
                 verbose=True):

    # -- we need to make sure that we are passing a valid object for interacting with the OpenAI platform:
    assert openai_obj != None, "WARNING: OpenAIInterfacer object has not been created!"

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
        print(f"{'*' * 75}\nStage 1 Prompting:\n{'*' * 75}")

    # NOTE: here is the list of instructions given to the LLM for generating a high-level plan as part of stage 1:
    system_prompt = open(stage1_sys_prompt_file, 'r').read()

    # -- get a list of instructions that satisfies the given prompt:
    stage1_prompt = f"Generate a high-level plan for the following task prompt:\n{query}"

    message.extend([{"role": "system", "content": system_prompt},
                    {"role": "user", "content": stage1_prompt}])

    _, stage1_response = openai_obj.prompt(message, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 1 Response:\n{stage1_response}\n{'*' * 75}")

    # -- get the list of all objects used in the generated plan:
    stage1b_prompt = "List all objects created by or needed in the high-level plan."\
        " Follow the format: 'unique_objects:[\"object_1\", \"object_2\", \"object_3\", ...]'"

    message.extend([{"role": "assistant", "content": stage1_response},
                    {"role": "user", "content": stage1b_prompt}])

    _, stage1b_response = openai_obj.prompt(message, verbose=verbose)
    unique_objects = get_stage1_objects(stage1b_response)
    if verbose:
        print(f"\nExtracted unique_objects:", unique_objects)

    ######################################################################################
    # NOTE: Stage 2 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 2 Prompting:\n{'*' * 75}")

    stage2_user_msg = open(stage2_sys_prompt_file, 'r').read().replace(
        "<obj_set>", str(unique_objects))

    # NOTE: we will be selecting a random example from a JSON file containing examples:
    incontext_examples = json.load(open(incontext_file))
    selected_example = choice(incontext_examples)

    stage2_prompt = f"{stage2_user_msg}\n\nFormat your output as a JSON structure like in the following example:\n{json.dumps(selected_example, indent=4)}"

    message.extend([{"role": "assistant", "content": stage1_response},
                    {"role": "user", "content": stage2_prompt}])

    _, stage2_response = openai_obj.prompt(message, verbose=verbose)

    if verbose:
        print(f"{'*' * 75}\nStage 2 Response:\n{stage2_response}\n{'*' * 75}")

    # -- use eval() function to parse through the Stage 2 prompt response obtained from LLM:
    object_level_plan = eval(stage2_response)

    ######################################################################################
    # NOTE: Stage 3 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 3 Prompting:\n{'*' * 75}")

    stage3_prompt = open(stage3_sys_prompt_file, 'r').read()
    message.extend([{"role": "user", "content": stage3_prompt}])

    _, stage3_response = openai_obj.prompt(message, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 3 Response:\n{stage3_response}\n{'*' * 75}")

    stage3_terminalSteps = eval(re.findall(r'\[.+?\]', stage3_response)[0])

    return {
        'PlanSketch': f'{stage1_response}\n\n{stage1b_response}',
        'OLP': object_level_plan,
        'TerminalSteps': stage3_terminalSteps,
        'RelevantObjects': unique_objects,
    }


def create_functionalUnit(llm_output, index):
    # NOTE: version 2 -- strict JSON format

    # -- create a functional unit prototype:
    functionalUnit = fga.FOON.FunctionalUnit()

    olp = llm_output['OLP']['Instructions'][index]
    olp_objects = llm_output['OLP']['CompleteObjectSet']

    used_objects = olp['RelatedObjects']
    action = olp['Action']
    print(f"Creating functional unit for Step {olp['Step']}: {olp['Instruction']}")
    print("\t-> related objects:", used_objects, "\n")

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

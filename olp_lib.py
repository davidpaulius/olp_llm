#############################################    Imports   #####################################################
import sys
import os
import re
import pickle
import json
import tiktoken
import numpy as np
import pandas as pd
from datetime import datetime as dt

FOON_API_path = './foon_to_pddl/foon_api'
if FOON_API_path not in sys.path:
    sys.path.append(FOON_API_path)

from foon_to_pddl.foon_api import FOON_graph_analyser as fga
from foon_to_pddl.foon_api import FOON_parser as fpa

# NOTE: import for Google's Gemini API:
# -- check more here: https://ai.google.dev/gemini-api/docs/quickstart?lang=python
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except ImportError:
    pass

# NOTE: imports for OpenAI's API:
from openai import OpenAI

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import Type

##############################################################################################################
# NOTE: LLM INTERFACER OBJECTS BELOW:
##############################################################################################################

class OpenAIInterfacer(object):
    def __init__(
            self,
            model_embed: str = "text-embedding-3-small",
            model_text_gen: str = "gpt-4-turbo"
        ) -> None:

        self.model_text_gen = model_text_gen
        self.model_embed = model_embed
        self.client = OpenAI()
        self.encoding = tiktoken.encoding_for_model(model_embed)


    def prompt(
            self,
            chat_history: str,
            args: dict = {},
            verbose: bool = False
        ) -> tuple:

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_text_gen}\nComplete prompt:\n{json.dumps(chat_history, indent=4)}")

        # -- make sure we add default arguments:
        if 'temperature' not in args: args['temperature'] = 0.1
        if 'max_tokens' not in args: args['max_tokens'] = 2500
        if 'frequency_penalty' not in args: args['frequency_penalty'] = 0
        if 'presence_penalty' not in args: args['presence_penalty'] = 0

        # -- create a completion request:
        completion = self.client.chat.completions.create(
            model=self.model_text_gen,
            messages=chat_history,
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

    def embed(
            self,
            text: str,
            verbose: bool = False
        ) -> list:

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_embed}\nText to embed: '{text}'")

        return self.client.embeddings.create(
            input=text, model=self.model_embed).data[0].embedding

    def num_tokens(
            self,
            text: str,
        ) -> int:

        return len(self.encoding.encode(text))


class GeminiInterfacer(object):
    def __init__(
            self,
            model_embed: str = "text-embedding-004",
            model_text_gen: str = "gemini-1.5-flash"
        ) -> None:

        self.model_text_gen = model_text_gen
        self.model_embed = model_embed


    def prompt(
            self,
            prompt: str,
            chat_history: list = [],
            args: dict = {'temperature': 0.1, 'max_tokens': 2000},
            verbose: bool = False,
        ) -> str:

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_text_gen}\nComplete prompt:\n{json.dumps(prompt, indent=4)}")

        # -- create a completion request:
        chat = genai.GenerativeModel(model_name=self.model_text_gen).start_chat(history=chat_history)
        response = chat.send_message(prompt)

        if verbose:
            print("*" * 75)

        # -- extract the message component from the returned completion object:

        return response.text

    def embed(
            self,
            text: str,
            verbose: bool = False
        ) -> list:

        if verbose:
            print("*" * 75)
            print(f"Model: {self.model_embed}\nText to embed: '{text}'")

        return genai.embed_content(model="models/text-embedding-004", content=text)

##############################################################################################################

def cos_similarity(
        openai_obj: Type[OpenAIInterfacer],
        str_1: str = None,
        str_2: str = None,
        vec_1: Type[np.array] = None,
        vec_2: Type[np.array] = None,
    ) -> float:

    # NOTE: this is a helper function that will make cosine similarity easier:

    vector_1, vector_2 = vec_1, vec_2

    if not vector_1:
        # -- embed the strings provided to the function:
        vector_1 = openai_obj.embed(text=str_1)
    if not vector_2:
        # -- embed the strings provided to the function:
        vector_2 = openai_obj.embed(text=str_2)

    return cosine_similarity(
        np.array(vector_1).reshape(1, -1),
        np.array(vector_2).reshape(1, -1))[0]


def chat_with_llm(
        openai_obj: Type[OpenAIInterfacer],
        chat_history: list,
        args: dict = {},
    ):

    while True:
        prompt = input("Type your new prompt for the LLM or press ENTER to end >> ")

        if not bool(prompt): break

        chat_history.extend([{"role": "user", "content": prompt}])

        _, response = openai_obj.prompt(chat_history, args)
        chat_history.extend([{"role": "assistant", "content": response}])

        print(f"\n{'*' * 50}\nuser: {prompt}\nGPT: {response}\n{'*' * 50}\n")

    return chat_history


def parse_llm_code(
        llm_output: str,
        separator: str = " ",
    ) -> str:
    if "```" not in llm_output:
        return llm_output

    valid_lines = []
    for line in llm_output.split('\n'):
        if "`" not in line:
            valid_lines.append(line)
    return separator.join(valid_lines)


def llm_grounding_sim_objects(
        openai_obj: Type[OpenAIInterfacer],
        objects_in_OLP: list[str],
        objects_in_sim: list[str],
        state_as_text: str = None,
        task: str = None,
        verbose: bool = False,
    ) -> dict:

    # -- prompt LLMs to perform object grounding
    #       (remove any objects that do not require grounding -- these are handled by the task planner system):
    objects_in_OLP = list(set(objects_in_OLP) - set(['hand', 'air', 'nothing', 'robot', 'table', 'work surface']))

    if verbose:
        print("\t-> objects in OLP:", objects_in_OLP)
        print("\t-> objects in scene:", objects_in_sim)
        print()

    real_objects, sim_objects = list(objects_in_OLP), list(objects_in_sim)

    object_mapping = dict()

    interaction = []

    assert bool(real_objects), "empty objects?"

    # -- we only prompt the LLM if there are objects we haven't gotten groundings for:

    if state_as_text and task:
        prompt_with_state = (
            f"Your task is to map object names to simulation objects for the task \"{task}\"). "
            f"Create a mapping that will reduce the number of actions needed by the robot. "
            "Example 1:"
            "\n- Action: \"Put first block on second block\""
            "\n- Object names: [\"first block\", \"second block\"]"
            "\n- Simulated objects: [\"block_1\", \"block_2\"]"
            "\n- Environment State:"
            "\n  - \"block_1\" is on \"block_2\""
            "\n  - \"block_2\" is under \"block_1\""
            "\n- Object mapping: \{\"first block\": \"block_1\", \"second block\": \"block_2\"\} (since \"block_1\" on \"block_2\" satisfies the action)"
            "\n\nExample 2:"
            "\n- Action: \"Put first block on second block\""
            "\n- Object names: [\"first block\", \"second block\"]"
            "\n- Simulated objects: [\"block_1\", \"block_2\"]"
            "\n- Environment State:"
            "\n  - \"block_2\" is on \"block_1\""
            "\n  - \"block_1\" is under \"block_2\""
            "\n- Object mapping: \{\"first block\": \"block_2\", \"second block\": \"block_1\"\} (since \"block_1\" on \"block_2\" satisfies the action)"
        )

        interaction.extend([{"role": "system", "content": prompt_with_state}])

    prompt_for_mapping = (
        "Map every object name to the best simulation object candidate from the provided list. "
        "Do not assign each object more than once. "
        "Format your answer as a Python dictionary without any explanation. "
        f"\nObject names: {real_objects}"
        f"\nSimulation objects: {sim_objects}"
        f"\nEnvironment State:\n{state_as_text}" if state_as_text else ""
    )

    interaction.extend([{"role": "user", "content": prompt_for_mapping}])

    while True:
        _, response = openai_obj.prompt(interaction)
        regex_matches = re.findall(r'\{(?<={)[^}]*\}', str(parse_llm_code(response)))

        if verbose:
            print(json.dumps(interaction, indent=4))

        if bool(regex_matches):
            # -- this means that the LLM has proposed some object groundings for us:

            # -- now we will consolidate all object groundings into an updated dictionary:
            object_mapping.update(eval(regex_matches.pop()))

            interaction.extend([{"role": "assistant", "content": response}])

            break

    return object_mapping, interaction


def llm_grounding_pddl_types(
        openai_obj: Type[OpenAIInterfacer],
        objects_in_sim: list[str],
        pddl_types: list = ['container'],
        verbose: bool = False,
    ) -> dict:

    # -- we have special object types for task-level planning;
    #       we will use the LLM to identify objects of special types.
    groundings = {}

    interaction = []
    for T in pddl_types:
        interaction = [{
            "role": "user",
            "content": (f"Which objects listed below can be classified as a \"{T}\" based on their names alone?"
                        " Format your answer as a Python list, and do not include any explanation."
                        f"\nSimulated objects: {objects_in_sim}")
        }]

        _, response = openai_obj.prompt(interaction)
        interaction.extend([{
            "role": "assistant",
            "content": response,
        }])

        groundings[T] = eval(parse_llm_code(response))

    if verbose:
        print(groundings)

    return groundings


def codify_FOON(
        FOON: list[fga.FOON.FunctionalUnit],
        verbose: bool = False,
    ):

    # NOTE: this function will "codify" functional units as a JSON structure, which will make it easier for a LLM to process.
    all_objects = []
    encoded_FOON = []

    for N in range(len(FOON)):
        # -- get a single functional unit from the FOON graph we've provided as a list:
        functional_unit = FOON[N]

        # -- first, let's populate this dictionary mapping object names to their preconditions/effects:
        object_states = {}
        for O in list(functional_unit.getInputNodes() + functional_unit.getOutputNodes()):
            obj_name = O.getObjectLabel()
            if obj_name not in object_states:
                object_states[obj_name] = {'preconditions': [], 'effects': []}

        all_objects.extend(list(object_states.keys()))

        # -- now we will go through all the input nodes to extract preconditions:
        for state_type in ['preconditions', 'effects']:
            # -- one loop to rule them all...
            nodes_to_iterate = functional_unit.getInputNodes()
            if state_type != 'preconditions':
                nodes_to_iterate = functional_unit.getOutputNodes()

            for O in nodes_to_iterate:
                for x in range(O.getNumberOfStates()):
                    if O.getStateLabel(x) != 'contains':
                        state = f"{O.getStateLabel(x)}{' 'if not bool(O.getRelatedObject(x)) else f' {O.getRelatedObject(x)}'}"
                        object_states[O.getObjectLabel()][state_type].append(state)
                    else:
                        for ingredient in O.getIngredients():
                            object_states[O.getObjectLabel()][state_type].append(f"contains {ingredient}")

        # -- now let's put it all together:
        encoded_FOON.append({
            'step': N+1,
            'action': functional_unit.getMotionNode().getMotionLabel(),
            'required_objects': list(object_states.keys()),
            'object_states': object_states,
            'instruction': functional_unit.toSentence(),
        })

        if verbose:
            print(json.dumps(encoded_FOON[-1], indent=4))

    return {
        "all_objects": all_objects,
        "plan": encoded_FOON,
    }


def llm_summarize_FOON(
        openai_obj: Type[OpenAIInterfacer],
        FOON: list[fga.FOON.FunctionalUnit],
        verbose: bool = False,
    ) -> list[str]:

    interaction = [{
        "role": "user",
        "content": (
            "Below is a plan given as a JSON: a plan consists of a list of actions to complete a task. "
            "Summarize each action into a concise set of instructions. "
            f"Use only one sentence per action.\n\n{codify_FOON(FOON, verbose)}"
        )
    }]

    _, task_steps = openai_obj.prompt(interaction)
    interaction.extend([{
        "role": "assistant",
        "content": task_steps,
    }])

    interaction.extend([{
        "role": "user",
        "content": "Write a concise sentence describing the plan's objective or task."
    }])

    _, task_description = openai_obj.prompt(interaction)
    interaction.extend([{
        "role": "assistant",
        "content": task_description,
    }])

    if verbose:
        print(interaction)

    return task_description, list(filter(None, task_steps.split('\n')))


def generate_from_FOON(
        openai_obj: Type[OpenAIInterfacer],
        query: str,
        FOON_samples: list,
        scenario: dict = None,
        system_prompt_file: str = "./llm_prompts/foon_system_prompt.txt",
        human_feedback: bool = False,
        use_embedding: bool = True,
        top_k: int = 3,
        verbose: bool = True,
    ) -> tuple[dict, list]:

    # -- we need to make sure that we are passing a valid object for interacting with the OpenAI platform:
    assert openai_obj != None, "WARNING: OpenAIInterfacer object has not been created!"

    ######################################################################################
    # NOTE: Priming stage:
    ######################################################################################

    # -- first, we are gonna give LLM context about the FOON generation task as a system prompt:
    system_prompt = open(system_prompt_file, 'r').read()

    # -- we will keep track of the entire interaction for context:
    interaction = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Your task will be to create a step-by-step plan for the following prompt: {query}."
                + (f" The following objects are available in the scene: {scenario['objects']}. " if scenario['objects'] else " ")
                + "Say 'Okay!' if you understand the task."
        }
    ]

    _, prelims_response = openai_obj.prompt(interaction, verbose=verbose)
    if "okay" not in prelims_response.lower():
        return None

    interaction.extend([
        {"role": "assistant", "content": prelims_response},
    ])

    ######################################################################################
    # NOTE: Stage 1 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 1 Prompting:\n{'*' * 75}")

    encoded_FOONs = []
    for x in range(len(FOON_samples)):
        # -- use the LLM to summarize the FOON samples we have in our repository:
        task, steps = llm_summarize_FOON(openai_obj, FOON_samples[x]['foon'])

        encoded_FOONs.append({
            'task_description': task,
            'language_plan': "\n".join(steps),
            'json': codify_FOON(FOON_samples[x]['foon']),
        })

    top_k_foons = None

    if not use_embedding:
        # -- first, we are gonna give LLM context about the FOON generation task as a system prompt:
        stage1a_prompt = (
            "Below are a list of prototype task descriptions."
            f" Pick no more than {top_k} tasks that are most similar to the new task prompt."
            " Give your answer as a Python list without explanation like \"[<num>, <num>, <num>]\", where <num> refers to the tasks below."
            "\n\nPrototype tasks:"
        )

        for x in range(len(encoded_FOONs)):
            stage1a_prompt = f"{stage1a_prompt}\n- Task Description #{x+1}:\n{encoded_FOONs[x]['task_description']}"

        if verbose:
            print(stage1a_prompt)

        interaction.extend([{"role": "user", "content": stage1a_prompt}, ])

        _, stage1a_response = openai_obj.prompt(interaction, verbose=verbose)
        if verbose:
            print(f"{'*' * 75}\nStage 1 Response:\n{stage1a_response}\n{'*' * 75}")

        interaction.extend([{"role": "assistant", "content": stage1a_response}, ])

        top_k_foons = eval(stage1a_response)

    else:
        # -- we will use text embedding to decide upon the top 3 most similar task descriptions:
        task_relevance_scores = []

        query_vec = openai_obj.embed(query, verbose=verbose)
        for x in range(len(encoded_FOONs)):
            # -- find text similarity using cosine similarity:
            score = cos_similarity(
                openai_obj=openai_obj,
                vec_1=query_vec,
                vec_2=openai_obj.embed(encoded_FOONs[x]['task_description'], verbose=verbose))

            task_relevance_scores.append((x+1, score))

        # -- sort them in descending order:
        task_relevance_scores.sort(key=lambda x: x[1], reverse=True)

        top_k_foons = [x[0] for x in task_relevance_scores[:min(len(task_relevance_scores), top_k)]]

    # -- now that we've identified similar task descriptions, we will now ask the LLM to implicitly pick the FOON that closely resembles this current task:
    stage1b_prompt = (
        "Out of the following prototype task plans, select the one closest to the new task prompt."
        " Give your answer as \"Prototype #<num>\", where <num> refers to a task number."
    )

    for x in top_k_foons:
        stage1b_prompt = f"{stage1b_prompt}\n\nPrototype Task #{x}:\n{encoded_FOONs[x-1]['language_plan']}"
        if verbose:
            print(f"Prototype Task #{x}:\n{encoded_FOONs[x-1]['language_plan']}\n")

    interaction.extend([{"role": "user", "content": stage1b_prompt}, ])

    _, stage1b_response = openai_obj.prompt(interaction, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 1 Response:\n{stage1b_response}\n{'*' * 75}")

    interaction.extend([{"role": "assistant", "content": stage1b_response}, ])

    selected_example = int(stage1b_response.split("#")[-1]) - 1

    if verbose:
        print(selected_example)

    ######################################################################################
    # NOTE: Stage 2 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 2 Prompting:\n{'*' * 75}")

    # -- get a list of instructions that satisfies the given prompt:
    stage2a_prompt = (
        f"Generate a concise plan using the prototype as inspiration for the task: {query}{'.' if '.' not in query else ''}"
        " Follow all guidelines. "
        " Give evidence to support your plan logic."
    )

    interaction.extend([{"role": "user", "content": stage2a_prompt}])

    _, stage2a_response = openai_obj.prompt(interaction, args={'temperature': 0.2}, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 2a Response:\n{stage2a_response}\n{'*' * 75}")

    interaction.extend([
        {"role": "assistant", "content": stage2a_response}
    ])

    if human_feedback:
        input(f"Give your opinion on the plan sketch given by the LLM: {stage2a_response}")
        interaction = chat_with_llm(openai_obj, interaction, {"temperature": 0.2})

    stage2b_prompt = (
        "Make a Python list of used objects in the following format: [\"object_1\", \"object_2\", ...]'. "
        "If there are several instances of an object type, list them individually (e.g., ['first apple', 'second apple'] if two apples are used). "
        "Do not add any explanation."
    )

    interaction.extend([
        {"role": "user", "content": stage2b_prompt}
    ])

    _, stage2b_response = openai_obj.prompt(interaction, verbose=verbose,)
    if verbose:
        print(f"{'*' * 75}\nStage 2b Response:\n{stage2b_response}\n{'*' * 75}")

    required_objects = eval(parse_llm_code(stage2b_response))

    stage2c_prompt = (
        "Format your generated plan as a JSON dictionary. "
        "List as many states as possible when describing each object's preconditions and effects. "
        "Each required object should match a key in \"object_states\": Be consistent with object names across actions. "
        f"Use this JSON prototype as reference:\n\n{codify_FOON(FOON_samples[selected_example]['foon'])}"
    )

    interaction.extend([
        {"role": "assistant", "content": stage2b_response},
        {"role": "user", "content": stage2c_prompt}
    ])

    # -- check if there is some sort of break between responses given by the LLM:
    plan_as_json = None

    entire_response = []
    while True:
        _, stage2c_response = openai_obj.prompt(
            interaction,
            verbose=verbose)

        entire_response.append(stage2c_response)

        try:
            # -- use eval() function to parse response obtained from  as a dictionary:
            plan_as_json = eval(parse_llm_code("".join(entire_response)))
        except SyntaxError as err:
            print(f"--warning: EOL (overflow): {err.msg}")
            interaction.extend([{"role": "assistant", "content": stage2c_response}])
        else: break

    assert bool(plan_as_json), "Something went wrong here?"

    if verbose:
        print(plan_as_json)

    interaction.extend([{"role": "assistant", "content": "".join(entire_response)}])

    language_plan = []
    for action in plan_as_json['plan']:
        language_plan.append(f"{action['step']}. {action['instruction']}")

    # -- returning parsed content as well as chat history for further interaction (and testing):
    llm_output = {
        "task_prompt": query,
        "all_objects": required_objects,
        "language_plan": language_plan,
        "object_level_plan": plan_as_json,
    }

    return llm_output, interaction


###################################################################################################################
# NOTE: COMPREHENSIVE OBJECT-LEVEL PLANNING METHODS:
##############################################################################################################

def top_fewshot_examples(
        openai_obj: Type[OpenAIInterfacer],
        fewshot_examples: dict,
        query: str,
        method: list = ['olp', 'blocks'], # NOTE: ['olp'], ['llm+p', 'blocks'|'packing'|'llm+p-cocktail']
        verbose: bool = False,
    ) -> dict:

    if 'olp' in method:
        # -- we only find the closest examples for stage 2 prompting for OLP:
        select_examples = fewshot_examples[f'{method[0]}_examples']['stage2']
    elif 'llm+p' in method or 'delta' in method:
        # -- we will provide a few examples per task type:
        select_examples = fewshot_examples[f'{method[0]}_examples'][method[1]]
    else: return None

    sample_queries = [I['task_prompt'] for I in select_examples]

    # -- use text embedding and cosine similarity to score and find the most similar examples:
    scores = [cos_similarity(openai_obj, query, I) for I in sample_queries]

    # NOTE: 'score_mapping' is a dictionary that can be used to see the score mapped to its query text:
    score_mapping = {sample_queries[I]: scores[I] for I in range(len(sample_queries))}

    if verbose:
        print(scores)
        print(score_mapping)

    # -- sort examples in descending order of scoring and add them to the prompt:
    sorted_examples = [x for _, x in sorted(zip(scores, select_examples), reverse=True)]

    return sorted_examples


def embed_olp(
        openai_obj: Type[OpenAIInterfacer],
        llm_output: dict,
        func_units: Type[fga.FOON.FunctionalUnit],
        embedding_fpath: str = 'olp_embeddings.pkl',
        verbose: bool = False
    ) -> None:

    # NOTE: this function will be used to create a storage file with embeddings that will be used for retrieval

    embeds = []
    # -- check if there is an existing embedding file:
    if os.path.exists(embedding_fpath):
        with open(embedding_fpath, 'rb') as ef:
            while True:
                try:
                    embeds.extend(pickle.load(ef))
                except EOFError:
                    break

    text_to_embed = (
        f"Task Prompt: {llm_output['task_prompt']}"
        f"\nTask Plan:\n{llm_output['language_plan']}"
        f"\nRequired Objects:{llm_output['all_objects']}"
    )

    embeds.append({
        'llm_output': llm_output,
        'functional_units': func_units,
        'embedding': openai_obj.embed(text_to_embed, verbose=verbose),
    })

    if verbose: print(embeds)

    pickle.dump(embeds, open(embedding_fpath, 'wb'))


def find_similar_olp(
        openai_obj: Type[OpenAIInterfacer],
        query: str,
        embedding_fpath: str = 'olp_embeddings.pkl',
        verbose: bool = False,
        top_k: int = 3,
    ) -> dict:

    # NOTE: this function will iterate through all of the

    embeds = []
    # -- check if there is an existing embedding file:
    if os.path.exists(embedding_fpath):
        with open(embedding_fpath, 'rb') as ef:
            while True:
                try:
                    embeds.extend(pickle.load(ef))
                except EOFError:
                    break

    else: return None

    if not bool(embeds):
        return None

    task_scores = []
    for E in range(len(embeds)):
        score = cos_similarity(
            openai_obj=openai_obj,
            vec_1=embeds[E]['embedding'],
            str_2=query)

        task_scores.append((embeds[E], score))

    task_scores.sort(key=lambda x: x[1])

    return task_scores[:min(len(task_scores), top_k)]


def generate_olp(
        openai_obj: Type[OpenAIInterfacer],
        query: str,
        stage1_sys_prompt_file: str,
        stage2_sys_prompt_file: str,
        stage3_sys_prompt_file: str,
        stage4_sys_prompt_file: str,
        scenario: dict = None,
        fewshot_examples: dict = None,
        num_fewshot: int = 1,
        verbose: bool = True,
    ) -> tuple[dict, list]:

    # -- we need to make sure that we are passing a valid object for interacting with the OpenAI platform:
    assert openai_obj != None, "WARNING: OpenAIInterfacer object has not been created!"

    def get_stage1_objects(response: str) -> str:
        # NOTE: this function will parse the output generated from Stage 1 prompting to identify all objects needed for a task:

        # -- we will look for the line in the output indicat
        all_objects = []
        for line in response.lower().split("\n"):
            if "all_objects" in line:
                try:
                    # -- we use eval() to remove any quotations that indicate some type of string:
                    all_objects = eval(line.split(":")[1].strip())
                except Exception:
                    # -- if for whatever reason the LLM does not list objects enclosed in quotations,
                    #       then we just split like normal:
                    objects = line.split(":")[1].strip().split(",")
                    all_objects = [x.strip().replace(".", "")
                                      for x in objects]

        return all_objects
    # enddef


    ######################################################################################
    # NOTE: Priming stage:
    ######################################################################################

    # NOTE: here is the list of instructions given to the LLM for generating a high-level plan as part of stage 1:
    system_prompt = open(stage1_sys_prompt_file, 'r').read()
    system_prompt = system_prompt.replace("<insert_examples>", fewshot_examples["olp_examples"]["stage1"][scenario['name']])

    # -- we will keep track of the entire interaction for context:
    interaction = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Your task will be to create a step-by-step plan for the following prompt: {query}."
                + (f" The following objects are available in the scene: {scenario['objects']}. " if scenario['objects'] else " ")
                + "Say 'Okay!' if you understand the task."
        }
    ]

    _, prelims_response = openai_obj.prompt(interaction, verbose=verbose)
    if "okay" not in prelims_response.lower():
        return None

    interaction.extend([
        {"role": "assistant", "content": prelims_response},
    ])

    ######################################################################################
    # NOTE: Stage 1 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 1 Prompting:\n{'*' * 75}")

    # -- get a list of instructions that satisfies the given prompt:
    stage1a_prompt = f"Generate a plan for the task. Think step by step and follow all instructions."
    interaction.extend([{"role": "user", "content": stage1a_prompt}, ])

    _, stage1a_response = openai_obj.prompt(interaction, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 1 Response:\n{stage1a_response}\n{'*' * 75}")

    if not scenario['objects']:
        # -- get the list of all objects used in the generated plan:
        stage1b_prompt = "List all objects created by or needed in the high-level plan."\
            " Follow the format: 'all_objects': [\"object_1\", \"object_2\", \"object_3\", ...]'"

    else:
        # -- ask the LLM to see if the list of objects is complete:
        # stage1b_prompt = "Are there any objects missing from the above list?"\
        #     " Output a corrected list with the following format: 'all_objects: [\"object_1\", \"object_2\", \"object_3\", ...]'"
        stage1b_prompt = "Make a list of atomic objects used for this task in the following format: 'all_objects: [\"object_1\", \"object_2\", \"object_3\", ...]'. "\
            "If there are several instances of an object type, list them individually (e.g., 'first apple', 'second apple' if two apples are used)."

    interaction.extend([
        {"role": "assistant", "content": stage1a_response},
        {"role": "user", "content": stage1b_prompt}
    ])

    _, stage1b_response = openai_obj.prompt(interaction, verbose=verbose)
    all_objects = get_stage1_objects(stage1b_response)
    if verbose:
        print(f"\nExtracted all_objects:", all_objects)

    ######################################################################################
    # NOTE: Stage 2 prompting:
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 2 Prompting:\n{'*' * 75}")

    stage2_user_msg = open(stage2_sys_prompt_file, 'r').read().replace(
        "<obj_set>", str(all_objects))

    stage2_prompt = f"{stage2_user_msg}\n\nFormat your output as a JSON dictionary."

    # NOTE: we will be selecting a random example from a JSON file containing examples:
    # -- sort examples in descending order of scoring and add them to the prompt:
    sorted_examples = top_fewshot_examples(
        openai_obj,
        fewshot_examples,
        query,
        verbose=verbose
    )

    for x in range(num_fewshot): # NOTE: change the argument for more few-shot examples
        stage2_prompt = f"{stage2_prompt}\n\nExample #{x+1}:\n{json.dumps(sorted_examples[x], indent=4)}"

    if verbose:
        print(stage2_prompt)

    interaction.extend([
        {"role": "assistant", "content": stage1b_response},
        {"role": "user", "content": stage2_prompt}
    ])

    # -- this is to make sure we continue prompting in the case where the response from the LLM was cut off:
    entire_response = []
    while True:
        _, stage2_response = openai_obj.prompt(
            interaction,
            verbose=verbose)
        interaction.extend([{"role": "assistant", "content": stage2_response}])
        entire_response.append(stage2_response)

        try:
            # -- use eval() function to parse through the Stage 2 prompt response obtained from LLM:
            object_level_plan = eval(parse_llm_code("".join(entire_response)))
        except SyntaxError as err:
            print(f"--warning: EOL (overflow): {err.msg}")
        else: break

    assert bool(object_level_plan), "Something went wrong here?"

    if verbose:
        print(f"{'*' * 75}\nStage 2 Response:\n{stage2_response}\n{'*' * 75}")
        print(f" -- total number of tokens: {openai_obj.num_tokens(stage2_response)}")

    ######################################################################################
    # NOTE: Stage 3 prompting:
    #   -- this involves asking the LLM about the key step(s) that
    #       will represent the final state for the entire object-level plan
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 3 Prompting:\n{'*' * 75}")

    stage3_prompt = open(stage3_sys_prompt_file, 'r').read()
    interaction.extend([{"role": "user", "content": stage3_prompt}])

    _, stage3_response = openai_obj.prompt(interaction, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 3 Response:\n{stage3_response}\n{'*' * 75}")
        print(f" -- total number of tokens: {openai_obj.num_tokens(stage3_response)}")

    stage3_terminalSteps = eval(re.findall(r'\[.+?\]', stage3_response)[0])

    ######################################################################################
    # NOTE: Stage 4 prompting:
    #   -- this involves getting a state summary dictionary
    #       for archiving the generated OLP and making it easy to retrieve from cache
    ######################################################################################

    if verbose:
        print(f"{'*' * 75}\nStage 4 Prompting:\n{'*' * 75}")

    stage4_prompt = open(stage4_sys_prompt_file, 'r').read()

    interaction.extend([
        {"role": "assistant", "content": stage3_response},
        {"role": "user", "content": stage4_prompt},
    ])

    _, stage4_response = openai_obj.prompt(interaction, verbose=verbose)
    if verbose:
        print(f"{'*' * 75}\nStage 4 Response:\n{stage4_response}\n{'*' * 75}")

    interaction.extend([{"role": "assistant", "content": stage4_response}, ])

    language_plan = []
    for action in object_level_plan:
        language_plan.append(f"{action['step']}. {action['instruction']}")

    # -- returning parsed content as well as chat history for further interaction (and testing):
    llm_output = {
        'task_prompt': query,
        'all_objects': all_objects,
        'language_plan': language_plan,
        'plan': object_level_plan,
        'final_state': eval(parse_llm_code(stage4_response)),
        'termination_steps': stage3_terminalSteps,
    }

    return llm_output, interaction


def repair_olp(
        openai_obj: Type[OpenAIInterfacer],
        query: str,
        available_objects: list,
        embedding_fpath: str = 'olp_embeddings.pkl',
        threshold: float = 0.8,
        skip_feedback: bool = True,
        verbose: bool = True,
    ) -> dict:
    """
    This function reviews all embedded object-level plans (if any exist) and modifies the closest one
    to achieve the user-specified task while also accounting for the objects in the scene.

    :param openai_obj: OpenAIInterfacer object for interacting with GPT
    :param query: a string containing the task prompt given by a human user
    :param available_objects: a list of strings referring to objects from the scene
    :param embedding_fpath: a string referring to the path of the embedding file
    :param threshold: a float value referring to the percentage similarity needed (between 0 and 1.0)
    :param verbose: a boolean value to set verbose comments

    :return -1: insufficient object set (missing objects prevent plan completion)
    :return 0: no cached plans are available for modification
    :return dict: a plan has been returned (either modified or original)
    """

    # -- find the most similar object-level plan that's been stored in embedding file:
    task_scores = find_similar_olp(openai_obj, query, embedding_fpath, verbose)

    if not task_scores:
        print(f"{' ' * 3}-- warning: no available FOONs for reference\n")
        return 0

    if task_scores[-1][1] < threshold:
        print(f"{' ' * 3}-- warning: similarity is pretty low :- {task_scores[-1][1]}")

    cached_olp = task_scores[-1][0]

    interaction = [
        {
            "role": "user",
            "content": (
                f"You will modify an existing plan so that it completes the following prompt: {query}."
                f" These objects are available in the scene: {available_objects}. "
                "Reply with 'Okay!' if you understand the task."
            )
        }
    ]

    _, prelims_response = openai_obj.prompt(interaction)
    if "okay" not in prelims_response.lower():
        return -1

    interaction.extend([
        {"role": "assistant", "content": prelims_response},
    ])

    if verbose:
        print('user:', interaction[-1]['content'])
        print(f'GPT: {prelims_response}')


    # interaction.extend([{
    #     "role": "user",
    #     "content": (
    #         "Which one of the following sample tasks sound similar to the current task?\n"
    #         str("\n".join([f"{task_scores[x][0]['object_level_plan']['task_prompt'] for x in task_scores}"]))
    #     )
    # }])

    if verbose:
        print(f"\nclosest OLP:\n{cached_olp['llm_output']['language_plan']}\n")

    # -- trying a chain-of-thought (CoT) reasoning step, where we have the LLM reason
    #       about whether the plan can be sufficiently fixed to satisfy the task prompt.
    interaction.extend([{
        "role": "user",
        "content": (
            f"Here is a prototype plan sketch that may be related to the task:\n{cached_olp['llm_output']['language_plan'].replace('.', '')}."
            "\n\nDoes this plan achieve the above task?"
            " Think step by step, and be concise with your reasoning."
            " Do not write a modified plan yet."
        )
    }])

    _, response = openai_obj.prompt(interaction, args={'temperature': 0.0})
    interaction.extend([{"role": "assistant", "content": response}])

    if verbose:
        print(f'\n{response}')

    interaction.extend([{
        "role": "user",
        "content": (
            "Does the above task specify any key objects required to complete it? Say \"yes\" or \"no\"."
        )
    }])

    _, response = openai_obj.prompt(interaction, args={'temperature': 0.0})
    interaction.extend([{"role": "assistant", "content": response}])

    if verbose:
        print(f'\n{response}')

    interaction.extend([{
        "role": "user",
        "content": (
            "Which of the following can you conclude about the plan? Provide your answer with no explanation."
            "\n1) Plan is complete: no changes are needed."
            "\n2) Plan is incomplete: actions must be added/removed or objects from scene must be added."
            "\n3) Plan is not solvable: key objects are missing from the scene."
        )
    }])

    print(f"\n{cached_olp['llm_output']['language_plan']}")
    print(f"\nsystem: {interaction[-1]['content']}")

    _, response = openai_obj.prompt(interaction)
    interaction.extend([{"role": "assistant", "content": response}])

    print(f"\nGPT: {response}")
    print(f"\n{'*' * 30}")

    if "2)" in response:
        # -- this means that there needs to be some kind of modification to the plan:
        interaction.extend([{
            "role": "user",
            "content": (
                "Show me your revised plan sketch. Make changes by missing objects or steps."
                " When thinking of object states, only use the geometric relations \"in\", \"on\", and \"under\". Do not add any explanation."
            )
        }])
        _, response = openai_obj.prompt(interaction)

        interaction.extend([{"role": "assistant", "content": response}])
        print(f'\n{response}')

        # -- Possible idea: use human verification to ensure the plan correction is good?
        if skip_feedback:
            input(("To the user: you will now interact with the LLM."
                " Check below for the proposed plan sketch for your given task"
                " If there are any issues with the plan sketch, tell it to the LLM."))
            interaction = chat_with_llm(openai_obj, interaction)

        # -- once the plan sketch looks good, we can then ask the LLM to correct the OLP sketch:
        olp_adapt_prompt = (
            "Okay, now that the high-level plan looks good, let's modify the more detailed plan."
            f" Make changes to the JSON below to match the plan. Do not add any explanation.\n\nJSON:\n{cached_olp['llm_output']['object_level_plan']}"
        )

        interaction.extend([{"role": "user", "content": olp_adapt_prompt}])

        entire_response = []
        while True:
            # -- this is to make sure we continue prompting in the case where the response from the LLM was cut off:
            _, response = openai_obj.prompt(interaction)
            interaction.extend([{"role": "assistant", "content": response}])
            entire_response.append(response)

            try:
                revised_olp = eval(parse_llm_code("".join(entire_response)))
            except SyntaxError as err:
                print(f"--warning: EOL (overflow): {err.msg}")
            else: break

        new_plan_sketch = '\n'.join([f"{str(x['step'])}. {x['instruction']}" for x in revised_olp['plan']])
        new_reqs = []
        for x in revised_olp['plan']:
            new_reqs.extend(x['required_objects'])

        if verbose:
            print(f'\n{new_plan_sketch}')

        return {
            'object_level_plan': revised_olp,
            'language_plan': f"High-level plan:\n{new_plan_sketch}\nRequired",
            'all_objects': f"{list(set(new_reqs))}"
        }

    elif "3)" in response:
        interaction.extend([{
            "role": "user",
            "content": "List the objects missing from the scene."
        }])
        _, response = openai_obj.prompt(interaction)

        interaction.extend([{"role": "assistant", "content": response}])
        print(f'\n{response}')

        return 0

    return {
        'object_level_plan': cached_olp['llm_output']['object_level_plan'],
        'language_plan': cached_olp['llm_output']['language_plan'],
        'all_objects': cached_olp['all_objects']
    }


def olp_to_FOON(
        llm_output: dict,
        index: int
    ) -> Type[fga.FOON.FunctionalUnit]:

    # NOTE: version 2 -- strict JSON format

    # -- create a functional unit prototype:
    functionalUnit = fga.FOON.FunctionalUnit()

    olp = llm_output['object_level_plan']['plan'][index]
    olp_objects = llm_output['all_objects']

    used_objects, action = olp['required_objects'], olp['action']

    print(f"Creating functional unit for Step {olp['step']}: {olp['instruction']}")
    print("\t-> related objects:", used_objects, "\n")

    geometric_states = ['in', 'on', 'under', 'contains', 'left of', 'right of', 'above', 'below']

    temp_objects = []

    for obj in used_objects:

        if obj in ['table', 'surface']:
            # -- we will remove any references to the table for object nodes:
            continue

        # -- create object node:
        print("Current object is :", obj)

        try:
            # check object state changes in step
            object_state_changes = olp['object_states'][obj]
        except KeyError:
            print(f'Missing object: {obj}')
            continue
        else:
            print("\t", obj, "state changes:", object_state_changes)

        for state_type in ['preconditions', 'effects']:
            # -- create a FOON object node for which we will then add attributes:
            new_object = fga.FOON.Object(
                objectID=used_objects.index(obj),
                objectLabel=obj,
            )

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
            functionalUnit.addObjectNode(
                objectNode=new_object,
                is_input=(True if 'precondition' in state_type.lower() else False),
            )

            temp_objects.append(new_object)

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action)
    functionalUnit.setMotionNode(newMotion)

    return functionalUnit

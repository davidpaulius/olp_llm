#############################################    Imports   #####################################################
import openai
import json
import sys
import spacy
import re


# NOTE: we need to import some files from the other FOON directory:
path_to_FOON_code = './foon_api/'
if path_to_FOON_code not in sys.path:
    sys.path.append(path_to_FOON_code)
    try:
        import FOON_graph_analyser as fga
    except ImportError:
        print(" -- ERROR: Missing 'FOON_graph_analyser.py' file! Make sure you have downloaded the FOON API scripts!")
        sys.exit()
##############################################################################################################
import FOON_graph_analyser as fga

def create_functionalUnit(object_list, action_verb):
    # -- create a blank functional unit (no object nodes nor motion node):
    functionalUnit = fga.FOON.FunctionalUnit()

    for obj in object_list:
        # NOTE: each key is an object name (noun); we will assume that the ID is neglible
        #   (since we can perform parsing):

        # -- we will be looking for the keys referring to preconditions and effects 
        #       ('input' and 'output' respectively)
        for x in ['Precondition', 'Effect']:
            if x in object_list[obj]:
                # -- add an input node if we find that there is an "ini" entry:
                new_object = fga.FOON.Object(objectLabel=obj) 
                for state in object_list[obj][x]:
                    if 'contains' in state['name']:
                        new_object.addNewState([None, state['name'], None])
                        new_object.addIngredient(state['related_obj'])
                    else:
                        new_object.addNewState([None, state['name'], state['related_obj']])
                # -- add the node to the functional unit:
                functionalUnit.addObjectNode(objectNode=new_object, is_input=(True if x == 'input' else False))

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action_verb)
    functionalUnit.setMotionNode(newMotion)

    return functionalUnit

def create_olp_functionalUnit(plan_step):
    functionalUnit = fga.FOON.FunctionalUnit()
   
    step_objects = plan_step['Objects']
    action = plan_step['Action']
    step_info = plan_step['Step']
    print("Creating functional unit for:", step_info)
    print("Objects:", step_objects)

    for object in step_objects:
        #create object node 
        print("Current object is :", object)
        new_object = fga.FOON.Object(objectLabel=object)

        #check object state changes in step
        object_state_changes = plan_step['StateChanges'][object]
        print("\t", object, "state changes:", object_state_changes)

        for state_type in ['Precondition', 'Effect']:
            current_state_effects = object_state_changes[state_type]
            
            for state_effect in current_state_effects:
                related_obj = [obj for obj in step_objects if obj in state_effect] #check if state effect involves another object in step
                if "contains" in state_effect:
                    new_object.addNewState([None, state_effect , None]) #add each state effect to object node
                    if related_obj!=[]: 
                        new_object.addIngredient(related_obj[0])  #maybe we should rename to addRelatedObject? instead of addIngredient to be more general outside of cooking domain
                else:
                    if related_obj!=[]: 
                        new_object.addNewState([None, state_effect, related_obj[0]])
                print("\t\t", state_type, ":", current_state_effects, "|| related objects:", related_obj)
            # -- add the node to the functional unit:
            functionalUnit.addObjectNode(objectNode=new_object, is_input=(True if state_type == 'Precondition' else False))

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action)
    functionalUnit.setMotionNode(newMotion)

    return functionalUnit

def generate_response_from_llm(given_prompt, model_name, verbose):
    if verbose: print(f"Model: {model_name}\nComplete prompt:")
    if verbose: print(json.dumps(given_prompt,indent=4))
    # create a completion
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=given_prompt,
        temperature=0.3,
        max_tokens = 2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if verbose: print("*************************************************************************")
    response = completion.choices[0].message
    return(response.role,response.content)


def get_objects_frm_llmresponse(response):
    unique_objects = []
    for line in response.lower().split("\n"):
        if "unique_objects" in line or "unique objects" in line:
            objects = line.split(":")[1].strip().split(",") 
            unique_objects = [x.strip().replace(".","") for x in objects]
    return unique_objects


def generate_incontext_examples(incontextfile):
    ret =''
    lines=[]
    with open(incontextfile,"r") as f:
        lines.extend(f.readlines())
    for line in lines:
        ret += line
    return ret


def generate_olp(query_task,incontextfile,llm_model="gpt-3.5-turbo", verbose=True):
    message = []
    # Stage 1 prompt
    if verbose: print(f"*************************************************************************\nStage1 Prompting\n*************************************************************************")
    system_prompt = "You are an LLM that understands how to generate concise high level plans for arbitrary tasks involving objects. Only focus on what happens to objects.  Stay consistent with object names and use one verb per step. Assume you have all objects necessary for the plan and do not need to obtain anything."
    user_prompt1 = "After the complete high level plan, list all the unique objects ignoring state changes to those objects. Follow the format unique_objects:object_1, obect_two, ..."
    query1 = query_task+ f"\n{user_prompt1}"
    message.extend([{"role":"system", "content":system_prompt},
                    {"role":"user", "content":query1}])

    stage1_response_role,stage1_response_content = generate_response_from_llm(message,llm_model,verbose)
    if verbose: print(f"Stage1 Response: \n{stage1_response_content}")
    unique_objects = get_objects_frm_llmresponse(stage1_response_content)
    if verbose: print(f"\nExtracted unique_objects:",unique_objects)
    
    # Stage 2 prompt
    if verbose: print(f"*************************************************************************\nStage2 Prompting\n*************************************************************************")
    user_prompt2 = f"List all states for each object in each step of the generated plan strictly referring to the object names from this set:{unique_objects}. Do not merge states, keep states atomic, mention them separately.\nFollow this example:"
    incontext_examples = generate_incontext_examples(incontextfile) #generate incontext examples 
    # print("In context examples:",incontext_examples)
    query2 = f"\n{user_prompt2}\n{incontext_examples}"
    message.extend([{"role":stage1_response_role,"content":stage1_response_content},
                     {"role":"user", "content":query2}])
    stage2_response_role,stage2_response_content = generate_response_from_llm(message,llm_model,verbose)

    if verbose: print(f"Stage2 Response: \n{stage2_response_content}")

    #extract object list and action to create FOON
    object_level_plan = []
    for olp_unit in stage2_response_content.split("\n\n"):
        object_level_plan.append(parse_olp_unit(olp_unit))
    
    return stage1_response_content, object_level_plan, unique_objects




def parse_state_changes(state_changes_str):
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

def parse_olp_unit(olp_unit):
    # Regular expression patterns
    step_pattern = r'Step_[0-9]+: (.+)\.'
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
    state_changes = parse_state_changes(state_changes_match.group(1)) if state_changes_match else {}

    # Constructing the dictionary
    instruction_dict = {
        'Step': step,
        'Objects': objects,
        'Action': action,
        'StateChanges': state_changes
    }

    return instruction_dict

# def parse_plan_steps(plan_steps):
#     plan_steps = plan_steps.split("\n")
#     print(

def build_foon():
    pass
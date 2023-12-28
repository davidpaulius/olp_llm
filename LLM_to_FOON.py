import os
import sys
import json
import spacy
import openai
import random
import datetime
import pickle 
import tqdm


###############################################################################################################################
#
# NOTE: Intuition of OLP-LLM pipeline:
#   1. first, create a plan schema as a FOON using LLM prompting (let's call it Plan Alpha -- P_α)
#   2. in parallel, we will use perception (object detector) to get the names of objects in the scene
#   3.
###############################################################################################################################

# NOTE: using S-BERT for sentence embedding and similarity
# -- read more here: https://www.sbert.net/
from sentence_transformers import SentenceTransformer, util

# NOTE: we will be using spaCy for doing POS tagging and other stuff:
try:
    import spacy
except ImportError:
    print(' ERROR: Missing spaCy library! Check here: https://spacy.io/')
    sys.exit()
else:
    spacy_model = spacy.load('en_core_web_sm')
    print(' -- [spaCY] : Loaded NLP pipeline...')
#endtry

# NOTE: we need to import some files from the other FOON directory:
path_to_FOON_code = './foon_api/'
if path_to_FOON_code not in sys.path:
    sys.path.append(path_to_FOON_code)

    # NOTE: this is a module used for opening and parsing FOON graph files:
    try:
        import FOON_graph_analyser as fga
    except ImportError:
        print(" -- ERROR: Missing 'FOON_graph_analyser.py' file! Make sure you have downloaded the FOON API scripts!")
        sys.exit()

###############################################################################################################################

# NOTE: flags for controlling functionality of the LLM-to-FOON pipeline:
'''
    flag_use_API: if True, use OpenAI's API for accessing language models
    flag_use_spacy: if True, use spaCy NLP pipeline for tasks such as POS-tagging and lemmatization
    flag_preamble: if True, provide a prompt to the LLM indicating verbs to use when generating language instructions
    flag_LLM_unit_to_sent: if True, use the LLM to construct a sentence to describe a functional unit
    flag_fewshot: if True, use the few-shot learning approach for generating a FOON
'''
flag_use_API = True
flag_use_spacy = True
flag_preamble = True
flag_LLM_unit_to_sent = True
flag_fewshot = False

# fewshot_method :- either use 'step' (1) or 'task' (2)
fewshot_method = 1

verbose = True

###############################################################################################################################

# NOTE: list of all available models: https://platform.openai.com/docs/models/model-endpoint-compatibility
# -- using chat variations (GPT-3.5 is probably best to use since it is cheaper to use):
openai_chat_models = ['gpt-3.5-turbo', 'gpt-4']
openai_model = openai_chat_models[0]

if flag_use_API:
    # NOTE: info on ChatGPT and OpenAI's API: https://platform.openai.com/docs/api-reference/chat
    openai.api_key = os.getenv('OPENAI_API_KEY')

# NOTE: set parameters for GPT prompting:
params = {
    'temperature': 0.3,     # -- the lower the temperature, the more deterministic the outcome
    'max_tokens': 256,      # -- keeping it relatively low to mitigate costs
    'presence_penalty': 0.0   # -- it's fine for GPT to repeat certain words since we are inquiring about a recipe
}

# NOTE: create a system prompt that will prime GPT for answering queries for object-level planning:
# -- guidelines :- list of strings that will make up the system prompt
guidelines = [
    'You are proposing recipes for execution.',
    # 'Make each instruction as concise as possible!', 
    'Use simple words for objects (ingredients, containers, and utensils) needed for each step of the recipe.',
    'Do not include quantities and assume that all ingredients are washed and ready for use.',
    'Use one verb per instruction.',
    # 'Make the final step say "Enjoy!".',
]

OCP_relations= ['in', 'on', 'under', 'left', 'right', 'center']

if flag_preamble:
    # -- we can inform the LLM of the following actions that we expect it to give us:
    guidelines.append(
        'You must use one of the following verbs: '\
            '"place", "slice", "cut", "chop", "pour", "sprinkle", "spread", "mix", "stir", "insert", and "cook".'
    )

system_prompt = {
    'role': 'system',
    'content' : '\n'.join(guidelines)
}

# TODO: confirm that these are a good set of tasks for our demonstrations
task_list = [
    'How can I wash dishes?',
    'How can I stack two blocks?',
    'How can I make an axe in Minecraft?'
    'How can I make a Black Russian cocktail?',
    'How can I slice a tomato?',
    'How can I make coffee?',
    # 'How can I set a dining table?',
]

# -- we will be storing the entire history of GPT interactions in this variable:
chat_log = [system_prompt]

FOON_subgraph_file = None
FOON_prototype = []

use_FOON_basis = False

use_context = True

class FewShotPrompter:
    # NOTE: this class definition is for the few-shot learning approach to generating FOONs from language models.

    # NOTE: this will work as follows:
    # -- Use a language prompt to generate a recipe/instruction set from a LLM.
    # -- We can then provide in-context examples to the LLM in two styles:
    #       1. For each step, we will find the closest functional unit example to it and include it in the prompt.
    #       2. For a given recipe, we can find the closest subgraph or FOON for that task.
    # -- Q: how do we find the closest recipe to a given meal? Could we just prompt the LLM directly?

    def __init__(self, method='step', sent_model=None, incontext_examples_dir=None, FOON_file=None):
        '''
        An object for performing few-shot learning and prompting for FOON generation from a language model.
        
        Constructor Parameters:
            few_shot_method (str): A variable indicating the type of method to use ('step' for method #1 of step-by-step example, 'task' for method #2 of whole task example)
            sentence_model (sentence_transformers.SentenceTransformer): A SentenceTransformer object that will be used for sentence embedding
            incontext_examples_dir (str): A string containing the path to a set of in-context examples of FOON subgraphs
            FOON_file (str): A string containing the path and name to a universal FOON or subgraph 
        '''
        self.few_shot_method = method
        self.sentence_model = self.load_SentenceTransformer(sent_model)
        self.incontext_examples = self.build_map(incontext_examples_dir)
        
        # -- get the FOON graph loaded:
        self.foon_object = fga
        if self.few_shot_method == 'step':
            if FOON_file:
                self.load_universalFOON(FOON_file)
            else:
                print(' -- ERROR: Missing FOON graph for few-shot examples!')
        #endif
        self.unit_to_sentences = {}
    #enddef

    def build_map(self, examples_loc):
        if not examples_loc:
            return None

        incontext_examples = {}
        for subgraph in os.listdir(examples_loc):
            # -- get the basename for a given subgraph file and map to its absolute path:
            subgraph_name = os.path.basename(subgraph)
            incontext_examples[subgraph_name] = os.path.abspath(subgraph)

        return incontext_examples
    #enddef

    def load_universalFOON(self, _file):
        # -- first, we need to ensure that a universal FOON graph is loaded:
        if not self.foon_object._isFOONLoaded():
            self.foon_object._constructFOON(_file)
    #enddef

    def load_SentenceTransformer(self, model):
        if not model:
            # -- by default, use S-BERT: https://www.sbert.net/docs/usage/semantic_textual_similarity.html
            return SentenceTransformer('sentence-transformers/sentence-t5-base')

        return SentenceTransformer(model)
    #enddef

    def findExample_subgraph(self, sent):
        # TODO: how should we find a similar unit?
        # -- one idea may be to take each subgraph and convert them into functional unit sentence instructions, 
        #       then we use the LLM to decide if the recipe is close enough or not (questionable performance?)
        # -- another would be to assume that we have a sentence description for each subgraph 
        #       (a probably reasonable assumption, given that recipes or other tasks may have titles/descriptions)

        # NOTE: we can assume that we have some title or description for each in-context example:

        def get_FOON_descriptor(subgraph_file):
            # -- read contents of the subgraph file so we can parse its metadata:
            subgraph_contents = open(subgraph_file, 'r').readlines()
        
            task_description = None

            for line in subgraph_contents:
                if line.startswith("# Task"):
                    task_description = line.split('\t')[1].strip('\n')

            return task_description

        if not self.incontext_examples:
            return
        
        # -- we will record the best in-context example from the set of subgraphs we have:
        best_match = {
            'similarity_score': None,
            'subgraph': None
        }

        # -- we will iterate through all of the examples and find the closest one 
        #       based on the new task and example task's embeddings:
        for task in self.incontext_examples:
            example_task_description = get_FOON_descriptor(self.incontext_examples[task])

            # -- embed the instruction/step provided by the LLM and the sentence description of this functional unit:
            embedding_1 = self.sentence_model.encode(sent, convert_to_tensor=True)
            embedding_2 = self.sentence_model.encode(example_task_description, convert_to_tensor=True)
        
            # -- compute cosine-similarity between both sentences:
            similarity_score = util.cos_sim(embedding_1, embedding_2).item()

            # -- take note of the subgraph with the highest similarity score:
            if not best_match['similarity_score'] or similarity_score > best_match['similarity_score']:
                best_match['similarity_score'] = similarity_score

        return best_match
    #enddef

    def findExample_unit(self, sent, use_LLM_for_generation=False, top_k=1, hierarchy_level=3):
        # NOTE: the intuition here is as follows: given a natural language instruction, 
        #   find a functional unit in FOON that closely aligns with the intentions of the command.

        # -- we will record the closest functional unit as a dictionary...
        best_match = {
            'functional_unit': None,
            'similarity_score': None,
            'sentence': None
        }

        # ... and we will also record the top-k closest units:
        candidate_units = []

        list_functionalUnits = self.foon_object.FOON_functionalUnits[hierarchy_level-1]

        # -- next, we embed each functional unit's equivalent sentence using the sentence model;
        #       however, we need to first generate a sentence for each functional unit.
    
        for unit in list_functionalUnits:

            unit_to_sentence = None

            # NOTE: we have two approaches for sentence generation:
            #   1. LLM-generated sentences :- use the LLM to propose a possible sentence description,
            #           given a list of objects and the action verb from a functional unit.
            #   2. regular sentence generation :- use a functional unit's native toSentence() function,
            #           which generates a sentence description using a rule-based approach.

            if use_LLM_for_generation:

                if unit not in self.unit_to_sentences:
                    # -- iterate through all objects in the functional unit's input and output node lists and save their names/types:
                    referred_objects = []
                    for node in unit.inputNodes + unit.outputNodes:
                        referred_objects.append(node.getName())

                    # -- get the name of the verb describing this functional unit, i.e., the motion node's label:
                    action_verb = unit.getMotion().getMotionType()

                    # -- format the prompt to send to the LLM:
                    prompt_for_sent = {
                        'role' : 'user',
                        'content' : 'Can you form an instructional sentence with the following words?\n' \
                            'Nouns: {0}\nVerb: {1}'.format(str(set(referred_objects)), action_verb)
                    }

                    # -- send the prompt to the LLM without context (save on number of tokens for this task):
                    response = prompt_LLM(prompt=prompt_for_sent)

                    # -- check the output returned by the LLM:
                    if response:
                        unit_to_sentence = response

                        #  -- record the generated description for each functional unit to save on number of requests:
                        self.unit_to_sentences[unit] = unit_to_sentence

                else:
                    # -- use the cached LLM-generated description of the functional unit:
                    unit_to_sentence = self.unit_to_sentences[unit]

            else:
                # -- use the rule-based approach to translate each functional unit into a sentence (see "FOON_classes.py"):
                unit_to_sentence = unit.toSentence()

            #endif

            if not unit_to_sentence:
                continue
            
            # NOTE: sentence embedding is based on content from here: https://www.sbert.net/docs/usage/semantic_textual_similarity.html

            # -- embed the instruction/step provided by the LLM and the sentence description of this functional unit:
            embedding_1 = self.sentence_model.encode(sent, convert_to_tensor=True)
            embedding_2 = self.sentence_model.encode(unit_to_sentence, convert_to_tensor=True)

            # -- compute cosine-similarity between both sentences:
            similarity_score = util.cos_sim(embedding_1, embedding_2).item()

            if len(candidate_units) < top_k:
                # -- add the functional unit to the list of top-3 candidate units:
                candidate_units.append(
                    {
                        'similarity_score': similarity_score,
                        'functional_unit' : unit,
                        'sentence' : unit_to_sentence
                    }
                )
            else:
                # -- we will kick out the unit with the lowest score:
                lowest_index = None
                for can in candidate_units:
                    if not lowest_index or can['similarity_score'] < candidate_units[lowest_index]['similarity_score']:
                        lowest_index = candidate_units.index(can)

                if similarity_score < candidate_units[lowest_index]['similarity_score']:
                    candidate_units[lowest_index] = {
                        'similarity_score': similarity_score,
                        'functional_unit' : unit,
                        'sentence' : unit_to_sentence
                    }

            # -- we will also keep track of the best scoring unit (i.e., with the most similarity)
            if not best_match['similarity_score'] or similarity_score > best_match['similarity_score']:
                best_match['similarity_score'] = similarity_score
                best_match['functional_unit'] = unit
                best_match['sentence'] = unit_to_sentence

        #endfor

        if top_k > 1:
            return candidate_units

        # -- return the best matching unit by default:
        return best_match
    #enddef

    def generate(self, recipe):
        # -- this approach assumes that we have some form of FOON that we will use as reference;
        #       this will be very similar to what was done in Sakib et al. 2022.

        if self.few_shot_method == 'step':
            # -- we will go step-by-step to find in-context functional unit examples for each instruction in the recipe:

            print('  -- Few-shot generation method #1: step-by-step...')

            for step in recipe:
                # -- get a single instruction from the recipe/instruction set; remove the numbering:
                instruction = recipe[step]['instruction'].split(str(step) + '. ')[1]

                # -- search for the nearest functional unit:
                best_match = self.findExample_unit(instruction, use_LLM_for_generation=flag_LLM_unit_to_sent)

                # -- with the best match, we will construct a special prompt through which the LLM generates a similar one:
                few_shot_example = str()

                # TODO: format the functional unit in an HTML-like way?

                # -- adding preconditions section:
                few_shot_example += '<preconditions>\n'
                for node in best_match['functional_unit'].inputNodes:
                    few_shot_example += '\t<object name="{0}">\n'.format(node.getName())
                    for x in range(node.getNumberOfStates()):
                        # -- check if there is a state relative to this current object:
                        if node.getRelatedObject(x):
                            few_shot_example += '\t\t<state description="{0} {1}">'.format(node.getStateLabel(x), node.getRelatedObject(x))
                        elif 'contains' in node.getStateLabel(x):
                            few_shot_example += '\t\t<state description="{0} {1}">'.format(node.getStateLabel(x), node.getIngredients())
                        else:
                            few_shot_example += '\t\t<state description="{0}">'.format(node.getStateLabel(x))
                        few_shot_example += '\n'
                    few_shot_example += '\t</object>\n'
                few_shot_example += '</preconditions>\n'

                # -- adding action section:
                few_shot_example += '<action_name = "{0}">\n'.format(best_match['functional_unit'].getMotion().getName())

                # -- adding effects section:
                few_shot_example += '<effects>\n'
                for node in best_match['functional_unit'].outputNodes:
                    few_shot_example += '\t<object name="{0}">\n'.format(node.getName())
                    for x in range(node.getNumberOfStates()):
                        # -- check if there is a state relative to this current object:
                        if node.getRelatedObject(x):
                            few_shot_example += '\t\t<state description="{0} {1}">'.format(node.getStateLabel(x), node.getRelatedObject(x))
                        elif 'contains' in node.getStateLabel(x):
                            few_shot_example += '\t\t<state description="{0} {1}">'.format(node.getStateLabel(x), node.getIngredients())
                        else:
                            few_shot_example += '\t\t<state description="{0}">'.format(node.getStateLabel(x))
                        few_shot_example += '\n'
                    few_shot_example += '\t</object>\n'
                few_shot_example += '</effects>\n'

                # -- now that we have the few-shot example, 
                #       we will provide this in a prompt for a similar functional unit:
                few_shot_prompt = {
                    'role': 'user',
                    'content': 'Write code for the sentence "{0}" similar to the example below:\n' \
                        '{1}'.format(instruction, few_shot_example)
                }

                response = prompt_LLM(prompt=few_shot_prompt)
                input(response)

                if response:
                    print(response)
                else:
                    return None

        elif self.few_shot_method == 'task':
            print('  -- Few-shot generation method #2: whole-task... NOT IMPLEMENTED!')
            pass
    #enddef
#endclass

class RecursivePrompter:
    # NOTE: this class definition is for a ruled-based approach to generating FOONs with the help of language models.

    # NOTE: this will work as follows:
    # -- Use a language prompt to generate a recipe/instruction set from a LLM.
    # -- Given each step of the generated recipe, we can do the following:
    #       1. use an LLM-heavy approach ("full"): ask LLM to infer object-state labels and parse its output
    #       2. use an LLM-medium-heavy approach ("assisted"): ask the LLM to infer object-state labels as #1,
    #           but we will use NLP parsers (spaCy) to help in prompt designing
    #       3. use an LLM-light approach ("lite"): use NLP approaches to lessen the number of prompt requests to LLM
    #           -- Q: can this approach use some sort of embedding of allowable state labels?
    #           -- R: perhaps we can require that a list of state labels are available for use

    def __init__(self, method='full', sent_model=None, state_file=None):
        # NOTE: see notes above for the meanings of different approaches/methods to take for prompting:
        self.method = 'lite' if method != 'full' and method != 'assisted' else 'assisted' if method !='full' else 'full'
        if self.method == 'lite':
            # -- when using 'lite' mode, we will be using a list of pre-defined states as reference:
            self.state_list = self.load_stateList(state_file)
        self.sentence_model = self.load_SentenceTransformer(sent_model)
    #enddef

    def load_stateList(self, state_file):
        state_list = []
        if '.txt' in state_file.lower():
            # -- this means we loaded a classical FOON state list file as a text file:
            with open(state_file, 'r') as f:
                for line in f:
                    state_list.append(line.split('\t')[1])
        #endif
        return state_list
    #enddef

    def load_SentenceTransformer(self, model):
        if not model:
            # -- by default, use S-BERT: https://www.sbert.net/docs/usage/semantic_textual_similarity.html
            return SentenceTransformer('sentence-transformers/sentence-t5-base')

        return SentenceTransformer(model)
    #enddef

    def generate(self, recipe):
        # -- now that we know how many instructions there are,
        #		we can go ahead and do some prompting per instruction.

        # NOTE: remember that FOON functional units have the following parts:
        #	1. object nodes:
        #		a) object name -- noun (name/alias) of object as referred to in natural language, 
        # 		b) object state(s) -- adjectives or phrases that describe the object's current state
        #	2. motion nodes:
        #		a) motion name -- verb that describes the manipulation or effect that is incurred by objects
        
        objs_per_step = {}
        for step in recipe:
            # -- if we are using 'assisted' mode, we should try to use some NLP to determine language hints:
            if self.method == 'assisted':
                # -- we will provide the LLM with some hints about the objects by including 
                #       what we think could be object candidates (i.e., words given as nouns):

                # -- we should perform some pre-processing to identify keywords (nouns, verbs) from language:
                nlp_hints = {
                    'objects': [],          # -- in short: all object names identified (based on nouns)
                    'action_verb': None,    # -- in short: this refers to the verb that was found in the sentence
                    'relations': [],        # -- in short: any relevant adjectives, adverbs, prepositions, etc.
                    'relevant_words': [],   # -- in short: ALL nouns, adjectives, adverbs, and verbs that are relevant to this sentence:
                }

                tokens = spacy_model(recipe[step]['instruction'])    
                
                global verbose
    
                for T in range(len(tokens)):
                    if verbose:
                        # NOTE: spacy.explain(...) gives more context about the POS tag that was identified by the model
                        print(tokens[T], tokens[T].pos_, spacy.explain(tokens[T].pos_))

                    # -- we will be looking for specific parts:
                    if tokens[T].pos_ == 'VERB':
                        # -- the verb will be used to name the planning operator:
                        if not nlp_hints['action_verb']:
                            nlp_hints['action_verb'] = tokens[T].lemma_
                        else:
                            nlp_hints['action_verb'] += '-and-' + tokens[T].lemma_
                    
                    if tokens[T].pos_ == 'NOUN':    
                        nlp_hints['objects'].append(tokens[T].lemma_)

                    if tokens[T].lemma_ in OCP_relations or tokens[T].pos_ == 'ADP':  
                        if len(list(tokens[T].subtree)) > 1:
                            # -- example: the phrase "pick up the basketball", subtree for "up" would just be ['up'],
                            #       but "put the basketball on the ground", subtree for "on" would be ['on', 'the', 'ground']
                            nlp_hints['relations'].append(tokens[T].text)

                    if tokens[T].pos_ in ['NOUN', 'VERB', 'ADP']:
                        nlp_hints['relevant_words'].append(T)

                #endfor

                if verbose:
                    print('verb:', nlp_hints['action_verb'])
                    print('objects:', nlp_hints['objects'])
                    print('OCRs:', nlp_hints['relations'])
                    print(nlp_hints['relevant_words'])
                    input()    
            #endif

            # -- formulate a prompt for each step that was previously obtained:
            if self.method == 'full':
                # -- we will *explicitly* ask the LLM to tell us what the objects are without hints:
                prompt_for_objects = {
                    'role': 'user',
                    'content': 'What objects are needed for step {0}?' \
                        ' List each object type on a separate line.'.format(step)
                }
            elif self.method == 'assisted':
                prompt_for_objects = {
                    'role': 'user',
                    'content': 'Are any of the following objects used in step {0}? Use a bulleted list.\n' \
                        'Possible objects: {1}'.format(step, str(nlp_hints['objects']))
                }
            #endif

            # NOTE: we will store each object in a dictionary, where an object key maps to a list of its relevant state attributes:
            necessary_objs = {}

            response = prompt_LLM(prompt=prompt_for_objects)
            if response:
                # -- we will iterate through the response given by GPT and add to a list:
                necessary_objs = post_process(intent='objects', content=response)

            print('step', step, '--', 'objects required:', necessary_objs)

            # NOTE: object states in FOON are typically of the following types:
            #   1) physical states of matter :- sliced, chopped, mashed, ground, liquid
            #   2) geometrical states :- use object-centered relations (e.g., in, on, under)
            #   3) containment :- objects that contain other objects (e.g., a bowl contains ingredients)

            if self.method != 'lite':
                # -- we will prompt the LLM directly for usable object states:

                # -- intuition: should we ask the LLM directly for states of each category above?
                #       e.g., we can ask the LLM for all physical, geometrical or containment states by providing a predefined set.

                if openai_model == 'gpt-4':
                    primer = {
                        'role': 'user',
                        'content': 'You will use the following attributes to describe object state: ' \
                            '1) geometrical states ("in", "on", "under", "contains") and its related object (e.g., "in bowl", "on cutting board"); ' \
                            '2) physical states ("whole", "raw", "cooked", "chopped", "sliced", "unmixed", "mixed", "clean", "empty"). ' \
                            '3) status ("used", "unused"). ' \
                            'Say "okay" if you understand.'
                    }
                else:
                    primer = {
                        'role': 'user',
                        'content': 'You will use as many of these states to describe objects: ' \
                            '1) geometrical states ("in", "on", "under") and its related object (e.g., "in bowl", "on cutting board"); ' \
                            '2) physical states ("whole", "raw", "cooked", "chopped", "unmixed", "mixed", "clean", "no state"); ' \
                            '3) containment state (e.g., "contains tomato", "contains tomato, lettuce and cucumber", "empty")'
                            'Say "okay" if you understand.'
                    }
                #endif

                response = prompt_LLM(prompt=primer)

                # -- this should never happen, but this is for testing purposes:
                if 'okay' not in '\n'.join(response).lower():
                    print(response)
                    sys.exit()


                reminder =  {
                        'role': 'user',
                        'content': 'What happens in step {0}?'.format(step)
                    }                
                response = prompt_LLM(prompt=reminder)
                input(response)
 
                # -- we should formulate prompts for each object:
                for obj in necessary_objs:
                    prompt_for_state = {
                        'role': 'user',
                        'content': 'List all states for the "{0}" object before and after step {1} is performed.\n' \
                            ''.format(obj, step)
                    }

                    fewshot_examples = [
                        [
                            'Follow this example: "Pour water from a cup into a bowl."\n' \
                            'For the "water" object:\nBefore step:\n- Water (in cup)\nAfter step:\n- Water (in bowl)\n' \
                            'For the "bowl" object:\nBefore step:\n- Bowl (empty; on table)\nAfter step:\n- Bowl (contains water; on table)\n' \
                            'For the "cup" object:\nBefore step:\n- Cup (contains water; on table)\nAfter step:\n- Cup (empty; on table)\n' \

                        ],
                        [
                            'Follow this example: "Place apple on plate."\n' \
                            'For the "apple" object:\nBefore step:\n- Apple (whole; on table)\nAfter step:\n- Apple (whole; on plate)\n' \
                            'For the "plate" object:\nBefore step:\n- Plate (empty; on table)\nAfter step:\n- Plate (contains apple; on table)\n' \

                        ]
                    ]

                    # -- add an additional example to help the LLM out:
                    prompt_for_state['content'] += ''.join(fewshot_examples[random.randint(0, len(fewshot_examples) - 1)])

                    response = prompt_LLM(prompt=prompt_for_state)

                    print(response)

                    # -- change all text to lowercase for ease:
                    necessary_objs[obj]['all_states'] = [x.lower() for x in response]
                #endfor

                print(necessary_objs)
                input()

                # -- now that we have gotten some clues from the LLM about possible object states, 
                #       let's see if we can create some preliminary FOONs:
                for obj in necessary_objs:
                    # -- this variable will be used to switch between precondition/effect lists:
                    is_input = True

                    # -- make sure to initialize the precondition/effect lists ahead of time:
                    for x in ['input', 'output']:
                        if x not in necessary_objs[obj]:
                            necessary_objs[obj][x] = []

                    for state in necessary_objs[obj]['all_states']:
                        if 'after' in state or 'effects' in state:
                            is_input = False

                        elif state.startswith('- '):
                            # -- this means we are on a state string line:
                            node_type = 'input' if is_input else 'output'                                                       

                            if self.method == 'assisted':
                                # -- use spacy to preprocess the output from LLM:
                                tokens = spacy_model(state.split('- ')[1])    
                                if verbose:
                                    for T in range(len(tokens)):
                                        # NOTE: spacy.explain(...) gives more context about the POS tag that was identified by the model
                                        print(tokens[T], tokens[T].pos_, spacy.explain(tokens[T].pos_))

                                if 'contains' in state:
                                    # -- we need to identify the thing that now contains a certain object:
                                    relevant_contents = []

                                    for T in range(len(tokens)):                                
                                        if tokens[T].pos_ == 'NOUN':    
                                            relevant_contents.append(tokens[T].lemma_)

                                    related_object = ' '.join(relevant_contents)

                                    necessary_objs[obj][node_type].append({'name': 'contains', 'related_obj': related_object})

                                else:
                                    # -- check if we are considering a state from the OCP perspective:
                                    relevant_OCP = None
                                    for rel in OCP_relations:
                                        if rel in state:
                                            relevant_OCP = rel
                                            break

                                    if relevant_OCP and not state.startswith('not'):
                                        # -- we have something that is OCP related (in, on, under, etc.), which has a related object
                                        #       with which the description is defined:
                                        relevant_contents = []

                                        # -- we need to identify the related object:
                                        for T in range(len(tokens)):                                
                                            if tokens[T].pos_ == 'NOUN':    
                                                relevant_contents.append(tokens[T].lemma_)

                                        related_object = ' '.join(relevant_contents)
                                        necessary_objs[obj][node_type].append({'name': relevant_OCP, 'related_obj': related_object})

                                    else:
                                        necessary_objs[obj][node_type].append({'name': state.split('- ')[1], 'related_obj': None})

                            else:
                                # -- parse through the terms in parentheses, which will also be comma-separated:
                                tokenized_states = state.split('(')[1].split(')')[0].split(',')

                                for attr in tokenized_states:
                                    # -- check if there is some mention of containment or use of object-centered predicates:
                                    keywords = ['contains', 'in', 'on', 'under']
                                    matched_keyword = False

                                    for key in keywords:
                                        token = str(key + ' ')
                                        if token in attr:
                                            # -- we need to make sure we split the string for the right related object name:
                                            ingredient = attr.split(token)[1]
                                            necessary_objs[obj][node_type].append({'name': key, 'related_obj': ingredient})

                                            matched_keyword = True

                                    if not matched_keyword:
                                        # -- if there are no special attributes, then we will just add this state as is:
                                        necessary_objs[obj][node_type].append({'name': attr.strip(), 'related_obj': None})
                                #endfor
                            #endif

                for obj in necessary_objs:
                    print(obj, necessary_objs[obj])
                input()

                # -- keep track of all objects deemed necessary for completing each step:
                objs_per_step[step] = necessary_objs

                action_type = None
                if self.method != 'assisted':
                    prompt_for_verb = {
                        'role': 'user',
                        'content': 'What action (verb) is happening in step {0}? Be concise!'.format(step)
                    }

                    response = prompt_LLM(prompt=prompt_for_verb)
                    input(response)

                    action_type = response[0].lower()
                else:
                    action_type = nlp_hints['action_verb']
                #endif

                # -- now we will create functional units that follow the FOON format:
                new_unit = create_functionalUnit(object_list=necessary_objs, action_verb=action_type)        
                new_unit.print_functions[2](version=1)
                input()

                # -- add the functional unit to the FOON prototype:
                FOON_prototype.append(new_unit)

            # else:

            #     for index in range(len(relevant_words)):
            #         if tokens[relevant_words[index]].pos_ == 'ADP':
            #             # -- this means that the previous word has some relation to the next:
            #             necessary_objs[tokens[relevant_words[index-1]].lemma_]['end'] = {
            #                 'state': tokens[relevant_words[index]].text,
            #                 'related_obj': tokens[relevant_words[index+1]].lemma_
            #             }

            #             # -- taking care of complementary state relations:
            #             if tokens[relevant_words[index]].text == 'in':
            #                 necessary_objs[tokens[relevant_words[index+1]].lemma_]['end'] = {
            #                     'state': 'contains',
            #                     'related_obj': tokens[relevant_words[index-1]].lemma_
            #                 }
            #             elif tokens[relevant_words[index]].text == 'on':
            #                 necessary_objs[tokens[relevant_words[index+1]].lemma_]['end'] = {
            #                     'state': 'under',
            #                     'related_obj': tokens[relevant_words[index-1]].lemma_
            #                 }


        return
    #enddef
#endclass

def get_args():
    from getopt import getopt, GetoptError

    global verbose, flag_use_API, flag_preamble, flag_fewshot, fewshot_method, FOON_subgraph_file

    try:
        opts, _ = getopt(
            sys.argv[1:],
            'vhf:opg:', 
            ['verbose', 'help', 'fewshot=', 'offline', 'predefine', 'graph=']
        )
    except GetoptError:
        pass
    else:
        for opt, arg in opts:
            if opt in ('-f', '--fewshot'):
                # -- by default, this flag will be set to false:
                flag_fewshot = True
                fewshot_method = int(arg)

            if opt in ('-g', '--graph'):
                FOON_subgraph_file = str(arg)

            if opt in ('-o', '--offline'):
                # -- connect to OpenAI's services:
                flag_use_API = False

            if opt in ('-p', '--predefine'):
                flag_preamble = True

            if opt in ('-v', '--verbose'):
                verbose = True
    #endtry
#enddef


def post_process(intent, content):
    # NOTE: the purpose of this function is to perform post-processing on the results obtained from the LLM.

    def _parse_objects():
        object_map = {}
        for L in content:
            # -- search for all lines that begin with a bullet:
            if L.startswith('- '):
                object_map[str(L.split('- ')[1]).lower()] = {}
        
        return object_map

    def _parse_states():
        return

    # NOTE: depending on the intent (i.e., 'objects', 'states', 'none'), 
    #   we want to perform some kind of post-processing on the response from the LLM:
    if not intent:
        return None;

    if intent == 'objects':
        return _parse_objects()
    
    if intent == 'states':
        return _parse_states()
#enddef

def save_chat_log():
    # -- save the entire interaction to a text file:
    if verbose:
        print(chat_log)

    today = datetime.datetime.now().strftime("%d-%m-%Y_%H%M")
    log_file = open('llm_to_FOON_log-{}.txt'.format(today), 'w')
    for entry in chat_log:
        log_file.write(entry['role'] + ': ' + entry['content'] + '\n')

    log_file.close()
#enddef


def fake_prompt_LLM(prompt):
    print(' -- [LOG] Faking GPT interaction...')

    # -- load some samples of prompts 
    gpt_sample_task_prompts = json.load(open('d:/Documents/Brown/Research/OLP-LLM/gpt_samples.json', 'r'))
    for entry in gpt_sample_task_prompts['prototypes']:
        if entry['task_prompt'] == prompt:
            for sample in entry['examples']:
                # -- checking if we find an offline sample that fits a specific set of conditions:
                #       1) completions were generated by the requested OpenAI model and with the same temperature
                #       2) whether the fixed action set was included in the system prompt or not
                if sample['fixed_action_set'] == flag_preamble and sample['model'] == openai_model and sample['temperature'] == params['temperature']:
                    return sample['instructions']
        
    return None
#enddef


def select_recipe(method='wikihow'):
    path_to_wikihow = 'D:/Documents/USF/Research/Wiki-How/articles/'

    wikihow_articles = [
        'HowtoCleanWaterBottles2.txt',
        'HowtoSetaTable2.txt',
        'HowtoMakeandDrinkGreenTea.txt'
    ]

    if method == 'wikihow':
        return wikihow_articles[random.randint(0, len(wikihow_articles)-1)]

    if method == 'recipe1m':

        path_to_recipe1m = 'D:/Documents/USF/Research/Recipe1M+/'
        recipe1m_files = ['recipe1m_train.pkl', 'recipe1m_test.pkl', 'recipe1m_val.pkl']

        recipe1m_recipes = []

        for F in recipe1m_files:
            with open(str(path_to_recipe1m + F), 'rb') as _file:
                recipe1m_pkl = pickle.load(_file)

            for R in tqdm.tqdm(recipe1m_pkl, desc='Parsing Recipe1M+ recipes...'):
                parsed_recipe = {'instructions': []}
                for I in R['instructions']:
                    recipe1m_sentence = I.strip(' ,;').replace('“', '"').replace("”", '"').replace("`", "'")
                    parsed_recipe['instructions'].append(recipe1m_sentence)

                # -- once the recipe has been parsed, we will add it to the complete list of recipes:
                recipe1m_recipes.append(parsed_recipe)
            #endfor

        return recipe1m_recipes[random.randint(0, len(recipe1m_recipes)-1)]

#enddef

def prompt_LLM(prompt):

    global use_context

    # -- first, append the prompt to the entire log (give entire context of interaction):
    chat_log.append(prompt)

    if flag_use_API:
        print('\n -- [LOG] Accessing GPT (using', openai_model, ')...')

        while True:
            if use_context:
                # -- using the defined parameters, create a prompt request using OpenAI's API, 
                #       providing all the necessary context:
                response = openai.ChatCompletion.create(
                    model=openai_model,
                    messages=chat_log,
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    presence_penalty=params['presence_penalty']
                )

            else:
                # -- don't pass the entire chat context to the LLM:
                response = openai.ChatCompletion.create(
                    model=openai_model,
                    messages=[prompt],
                    temperature=params['temperature'],
                    max_tokens=params['max_tokens'],
                    presence_penalty=params['presence_penalty']
                )

            # -- ensure that there was no error in response:
            if response.choices[0]['finish_reason'] in ['stop', 'length']:
                break
            
            
        #endwhile
    else:
        # -- fake the interaction by accessing content that's in our collection of GPT samples:
        response = fake_prompt_LLM(prompt=prompt['content'])

    # -- add the response from GPT to the entire context;
    #		this is necessary so that GPT understands what you're asking about"
    assistant_prompt = {
        'role': 'assistant',
        'content': response.choices[0].message.content if flag_use_API else '\n'.join(response)
    }

    chat_log.append(assistant_prompt)

    # print(response)

    # -- split the string corresponding to the response given by GPT:
    return response.choices[0].message.content.split('\n') if flag_use_API else response
#enddef


def create_functionalUnit(object_list, action_verb):
    # -- we will use the code from FGA to create and parse functional units:

    # -- create a blank functional unit (no object nodes nor motion node):
    functionalUnit = fga.FOON.FunctionalUnit()

    for obj in object_list:
        # NOTE: each key is an object name (noun); we will assume that the ID is neglible
        #   (since we can perform parsing):

        # -- we will be looking for the keys referring to preconditions and effects 
        #       ('input' and 'output' respectively)
        for x in ['input', 'output']:
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
            #endif
        #endfor

    #endfor

    # -- make a new motion node and add it to the functional unit:
    newMotion = fga.FOON.Motion(motionID=None, motionLabel=action_verb)
    functionalUnit.setMotionNode(newMotion)

    # -- now that we have the functional unit prototype, we can return this:

    return functionalUnit
#enddef

def main():
    # -- select a random task from the list of available tasks:
    selected_task = task_list[random.randint(1, len(task_list)) - 1]
    selected_task = task_list[-1]

    # -- first, we will start by asking GPT for a base recipe:
    user_prompt = {
        'role': 'user',
        'content': selected_task
    }
    print(selected_task)

    # -- we are expecting a numbered set of instructions that addresses the given prompt:
    response = prompt_LLM(prompt=user_prompt)

    # -- we will manually iterate through each line of the retrieved content to count the number of instructions:
    num_steps = 0
    recipe = {}
    for step in response:
        if step.startswith( str(str(num_steps+1) + '.') ):
            recipe[num_steps+1] = {'instruction': step}
            num_steps += 1
    
            if verbose:
                print(step)

    print()


    ##############################################################################################################

    prompter = None
    if flag_fewshot:
        print(' -- [FOON-LLM] : Starting few-shot prompting pipeline...')
        prompter = FewShotPrompter(
            method=('step' if fewshot_method == 1 else 'task'), 
            FOON_file=FOON_subgraph_file, 
            incontext_examples_dir='D:/Documents/Brown/Research/OLP-LLM/Few-shot Examples/',
        )
    else:
        print('[FOON-LLM] -- Starting rule-based prompting pipeline...')
        prompter = RecursivePrompter(
            method='full',
        )
    #endfor

    prompter.generate(recipe)

    if FOON_prototype:
        for unit in FOON_prototype:
            unit.print_functions[2](version=1)
            print('//')
        input()
    else:
        print(' -- No FOON was generated!')
        return


    while True:
        # -- testing further prompts that will ask the LLM for plan repairs and modifications:
        question = input()
        new_prompt = {
            'role': 'user',
            'content': question
        }
        response = prompt_LLM(new_prompt)

        print(response)

#enddef


if __name__ == '__main__':
    get_args()
    main()
    # save_chat_log()
#end

# def method_recursivePrompts(recipe):
#     # -- now that we know how many instructions there are,
#     #		we can go ahead and do some prompting per instruction.

#     # NOTE: remember that FOON functional units have the following parts:
#     #	1. object nodes:
#     #		a) object name -- noun (name/alias) of object as referred to in natural language, 
#     # 		b) object state(s) -- adjectives or phrases that describe the object's current state
#     #	2. motion nodes:
#     #		a) motion name -- verb that describes the manipulation or effect that is incurred by objects
    
#     global flag_use_API, flag_use_spacy; flag_use_API = True

#     print(recipe)

#     objs_per_step = {}
#     for step in recipe:
#         # -- formulate a prompt for each step that was previously obtained:
#         if not flag_use_spacy:
#             # -- we will *explicitly* ask the LLM to tell us what the objects are without hints:
#             prompt_for_objects = {
#                 'role': 'user',
#                 'content': 'What objects are needed for step {0}?' \
#                     ' List each object type on a separate line.'.format(step)
#             }
#         else:
#             # -- we will provide the LLM with some hints about the objects by including 
#             #       what we think could be object candidates (i.e., words given as nouns):

#             # -- we should perform some pre-processing to identify keywords (nouns, verbs) from the instructions
#             objects_in_action = []
#             action_verb = None
#             relations = []
#             relevant_words = []

#             tokens = spacy_model(recipe[step]['instruction'])    
#             if verbose:
#                 for T in range(len(tokens)):
#                     # NOTE: spacy.explain(...) gives more context about the POS tag that was identified by the model
#                     print(tokens[T], tokens[T].pos_, spacy.explain(tokens[T].pos_))

#             for T in range(len(tokens)):
#                 # -- we will be looking for specific parts:
#                 if tokens[T].pos_ == 'VERB':
#                     # -- the verb will be used to name the planning operator:
#                     if not action_verb:
#                         action_verb = tokens[T].lemma_
#                     else:
#                         action_verb += '-and-' + tokens[T].lemma_

#                     relevant_words.append(T)
                
#                 elif tokens[T].pos_ == 'NOUN':    
#                     objects_in_action.append(tokens[T].lemma_)
#                     relevant_words.append(T)

#                 if tokens[T].lemma_ in OCP_relations or tokens[T].pos_ == 'ADP':  
#                     if len(list(tokens[T].subtree)) > 1:
#                         # -- example: the phrase "pick up the basketball", subtree for "up" would just be ['up'],
#                         #       but "put the basketball on the ground", subtree for "on" would be ['on', 'the', 'ground']
#                         relations.append(tokens[T].text)
#                         relevant_words.append(T)

#             if verbose:
#                 print('verb:', action_verb)
#                 print('objects:', objects_in_action)
#                 print('OCRs:', relations)
#                 print(relevant_words)
#                 input()    

#             prompt_for_objects = {
#                 'role': 'user',
#                 'content': 'Are any of the following words objects used in step {0}? Possible objects: {1}' \
#                     ' Use a bulleted list.\n'.format(step, str(objects_in_action))
#             }
#         #endif

#         response = prompt_LLM(prompt=prompt_for_objects)

#         necessary_objs = {}
#         if response:
#             # -- we will iterate through the response given by GPT and add to a list:
#             for L in response:
#                 if L.startswith('- '):
#                     necessary_objs[L.split('- ')[1]] = {}

#         print('step', step, '--', 'objects required:', necessary_objs)

#         # NOTE: object states in FOON are typically of the following types:
#         #   1) physical states of matter :- sliced, chopped, mashed, ground, liquid
#         #   2) geometrical states :- use object-centered relations (e.g., in, on, under)
#         #   3) containment :- objects that contain other objects (e.g., a bowl contains ingredients)

#         # flag_use_spacy = False

#         if flag_use_spacy:
#             # -- we will prompt the LLM directly for usable object states:

#             # -- intuition: should we ask the LLM directly for states of each category above?
#             #       e.g., we can ask the LLM for all physical, geometrical or containment states by providing a predefined set.

#             if step == 1:
#                 # -- idea: try to give a primer statement to prepare the LLM for proper parsing;
#                 #       give the primer to the LLM:

#                 if openai_model == 'gpt-4':
#                     primer = {
#                         'role': 'user',
#                         'content': 'Use the following attributes to describe object state: ' \
#                             '1) geometrical states ("in", "on", "under", "contains") and its related object (e.g., "in bowl"); ' \
#                             '2) physical states ("whole", "raw", "cooked", "chopped", "sliced", "unmixed", "mixed", "clean", "empty"). ' \
#                             '3) status ("used", "unused"). ' \
#                             'Say "okay" if you understand!'
#                     }
#                 else:
#                     primer = {
#                         'role': 'user',
#                         'content': 'I want you to use as many of these states to describe objects: ' \
#                             '1) geometrical states ("in", "on", "under") and its related object (e.g., "in bowl"); ' \
#                             '2) physical states ("whole", "raw", "cooked", "chopped", "unmixed", "mixed", "clean", "empty", "no state"); ' \
#                             'Say "okay" if you understand!'
#                     }


#                 response = prompt_LLM(prompt=primer)

#                 if 'okay' not in '\n'.join(response).lower():
#                     print(response)
#                     sys.exit()

#             # -- we should formulate prompts for each object:
#             for obj in necessary_objs:

#                 prompt_for_state = {
#                     'role': 'user',
#                     'content': 'Create a bulleted list of states for "{0}" before and after step {1} is executed.'.format(obj, step)
#                 }

#                 response = prompt_LLM(prompt=prompt_for_state)

#                 # -- change all text to lowercase for ease:
#                 necessary_objs[obj]['all_states'] = [x.lower() for x in response]

#             print(necessary_objs)
#             input()

#             # -- now that we have gotten some clues from the LLM about possible object states, 
#             #       let's see if we can create some preliminary FOONs:
#             for obj in necessary_objs:
#                 # -- this variable will be used to switch between precondition/effect lists:
#                 is_input = True

#                 # -- make sure to initialize the precondition/effect lists ahead of time:
#                 for x in ['ini', 'end']:
#                     if x not in necessary_objs[obj]:
#                         necessary_objs[obj][x] = []

#                 for state in necessary_objs[obj]['all_states']:
#                     if state.startswith('after'):
#                         is_input = False

#                     elif state.startswith('- '):
#                         # -- this means we are on a state string line:
#                         key = 'ini' if is_input else 'end'                                                       

#                         # -- use spacy to preprocess the output from LLM:
#                         tokens = spacy_model(state.split('- ')[1])    
#                         if verbose:
#                             for T in range(len(tokens)):
#                                 # NOTE: spacy.explain(...) gives more context about the POS tag that was identified by the model
#                                 print(tokens[T], tokens[T].pos_, spacy.explain(tokens[T].pos_))

#                         if 'contains' in state:
#                             # -- we need to identify the thing that now contains a certain object:
#                             relevant_contents = []

#                             for T in range(len(tokens)):                                
#                                 if tokens[T].pos_ == 'NOUN':    
#                                     relevant_contents.append(tokens[T].lemma_)

#                             related_object = ' '.join(relevant_contents)

#                             necessary_objs[obj][key].append({'name': 'contains', 'related_obj': related_object})

#                         else:
#                             # -- check if we are considering a state from the OCP perspective:
#                             relevant_OCP = None
#                             for rel in OCP_relations:
#                                 if rel in state:
#                                     relevant_OCP = rel
#                                     break

#                             if relevant_OCP and not state.startswith('not'):
#                                 # -- we have something that is OCP related (in, on, under, etc.), which has a related object
#                                 #       with which the description is defined:
#                                 relevant_contents = []

#                                 # -- we need to identify the related object:
#                                 for T in range(len(tokens)):                                
#                                     if tokens[T].pos_ == 'NOUN':    
#                                         relevant_contents.append(tokens[T].lemma_)

#                                 related_object = ' '.join(relevant_contents)
#                                 necessary_objs[obj][key].append({'name': relevant_OCP, 'related_obj': related_object})

#                             else:
#                                 necessary_objs[obj][key].append({'name': state.split('- ')[1], 'related_obj': None})

#             for obj in necessary_objs:
#                 print(obj, necessary_objs[obj])
#             input()

#             # -- keep track of all objects deemed necessary for completing each step:
#             objs_per_step[step] = necessary_objs

#             if not action_verb:
#                 prompt_for_verb = {
#                     'role': 'user',
#                     'content': 'What action (verb) is happening in step {0}? Be concise!'.format(step)
#                 }

#                 response = prompt_LLM(prompt=prompt_for_verb)
#                 input(response)


#             # -- now we will create functional units that follow the FOON format:
#             new_unit = create_functionalUnit(object_list=necessary_objs, action_verb=action_verb)        
#             new_unit.print_functions[2](version=1)
#             input()

#             # -- add the functional unit to the FOON prototype:
#             FOON_prototype.append(new_unit)

#         else:

#             for index in range(len(relevant_words)):
#                 if tokens[relevant_words[index]].pos_ == 'ADP':
#                     # -- this means that the previous word has some relation to the next:
#                     necessary_objs[tokens[relevant_words[index-1]].lemma_]['end'] = {
#                         'state': tokens[relevant_words[index]].text,
#                         'related_obj': tokens[relevant_words[index+1]].lemma_
#                     }

#                     # -- taking care of complementary state relations:
#                     if tokens[relevant_words[index]].text == 'in':
#                         necessary_objs[tokens[relevant_words[index+1]].lemma_]['end'] = {
#                             'state': 'contains',
#                             'related_obj': tokens[relevant_words[index-1]].lemma_
#                         }
#                     elif tokens[relevant_words[index]].text == 'on':
#                         necessary_objs[tokens[relevant_words[index+1]].lemma_]['end'] = {
#                             'state': 'under',
#                             'related_obj': tokens[relevant_words[index-1]].lemma_
#                         }


#     return
# #enddef

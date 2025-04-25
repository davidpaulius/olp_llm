
import os
import subprocess
import argparse
from pathlib import Path

# NOTE: make sure that fast-downward submodule is pulled and built:
path_to_planners = {
    'fast-downward': './downward/fast-downward.py'
}

planner_to_use = 'fast-downward'

# NOTE: algorithm is a key in the configs dictionary, while
#   heuristic is an index for the list of configs per algorithm:
configs = {
    'astar': ['astar(lmcut())', 'astar(ff())'],
    'eager': ['eager(lmcut())', 'eager(ff())'],
    'lazy': ['lazy(lmcut())', 'lazy(ff())']
}
algorithm = 'astar'
heuristic = 0


class PlanningOperator(object):

    def __init__(self, PO_name=None):
        self.name = PO_name
        self.preconditions = []
        self.effects = []
        self.parameters = {}
        self.args = []

    def addParameter(self, variable, type):
        self.parameters[variable] = type
        self.args.append(variable)

    def addPrecondition(self, pred):
        self.preconditions.append(pred)

    def addEffect(self, pred):
        self.effects.append(pred)

    def getPreconditions(self):
        return self.preconditions

    def getEffects(self):
        return self.effects

    def getParameters(self):
        return self.parameters, self.args

    def getActionName(self):
        return self.name

    def printPlanningOperator(self):
        if self.name:
            print('(:action ' + self.name)

            # -- print parameters for the planning operator:
            if bool(len(self.args)):
                print('\t:parameters (')
                for P in self.args:
                    print('\t\t' + str(P) + ' - ' + str(self.parameters[P]))
                print('\t)')
            else:
                print('\t:parameters ( )')

            # -- print preconditions..
            print('\t:precondition (and ')
            for pred in self.preconditions:
                if isinstance(pred, list):
                    # -- this is an *or* statement:
                    print('\t\t(or')
                    for subpred in pred:
                        print('\t\t\t' + subpred)
                    print('\t\t)')
                else:
                    print('\t\t' + pred)
            print('\t)')

            # -- print effects..
            print('\t:effect (and')
            for pred in self.effects:
                print('\t\t' + pred)
            print('\t)')


def create_domain_file(
        objects_in_sim: list,
        micro_fpath: str,
        template_domain_fpath: str,
        domain_name: str = 'olp-2024',
        typing: list = None,
    ) -> str:

    # -- we are going to inject new objects into the
    micro_domain_fpath = os.path.join(micro_fpath, template_domain_fpath)

    with open(micro_domain_fpath, 'w') as pddl_file:

        # -- the template file contains all of the default skills for our robot:
        with open(template_domain_fpath) as f:
            domain_file_lines = f.readlines()
            for L in domain_file_lines:
                if '(:constants' in L:
                    pddl_file.write(f'{L}')
                    for O in objects_in_sim:
                        # -- check if an object in the scene has been classified as special PDDL type:
                        object_type = 'object'
                        if typing:
                            for T in typing:
                                if O in typing[T]:
                                    pddl_file.write(f'\t\t{O} - {T}\n')

                        else:
                            pddl_file.write(f'\t\t{O} - {object_type}\n')

                elif '(domain' in L:
                    pddl_file.write(f'(define (domain {domain_name})')

                else:
                    pddl_file.write(f'{L}')

    return micro_domain_fpath


def create_problem_file(
        preconditions: list,
        effects: list,
        micro_fpath: str,
        action_name: str,
        state: dict,
        domain_name: str = 'olp-2024',
    ) -> str:

    on_object_layout = state['on']
    in_object_layout = state['in']
    under_object_layout = state['under']

    micro_problem_fpath = os.path.join(micro_fpath, f"{action_name}.pddl",)

    with open(micro_problem_fpath, 'w') as pddl_file:

        pddl_file.write(f'(define (problem {Path(micro_fpath).stem})\n')
        pddl_file.write(f'(:domain {domain_name})\n')
        pddl_file.write('(:init' + '\n')

        # -- dictionary that will ground object names from conceptual (domain-independent) space to
        # 	simulated (domain-specific) space:
        grounded_objects = {}

        if 'hand' in in_object_layout:
            if in_object_layout['hand'] == 'air':
                # -- hand should be empty (i.e., contains 'air')
                pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
            else:
                # -- hand is not empty for whatever reason:
                pddl_file.write('\t; hand/end-effector contains some type of object:\n')

            pddl_file.write('\t' + '(in hand ' + str(in_object_layout['hand']) +')' + '\n')
            pddl_file.write('\t' + '(under hand ' + str(under_object_layout['hand']) +')' + '\n')
        else:
            # -- hand will be empty by default:
            pddl_file.write('\t; hand/end-effector must be empty (i.e. contains "air"):\n')
            pddl_file.write('\t' + '(in hand air)' + '\n')

        if 'olp' in domain_name:
            pddl_file.write('\t' + '(not (is-mobile))' + '\n')

        # -- identifying all objects that are being referenced as preconditions:
        all_objects, on_objects = set(), set()
        for pred in preconditions:
            predicate_parts = pred[1:-1].split(' ')
            # -- we are just reviewing all of the objects referenced by predicates...
            all_objects.add(predicate_parts[1])
            if len(predicate_parts) > 2:
                all_objects.add(predicate_parts[2])

            # ... but we also need to note all that have already been referenced by the "on" predicate:
            if predicate_parts[0] == 'on' and predicate_parts[1] != 'cutting_board':
                on_objects.add(predicate_parts[1])
        # endfor

        # -- we will write the current object layout as obtained from the "perception" step.
        written_predicates = []

        for obj in list(on_object_layout.keys()):
            # -- writing the "on" relation predicate:
            on_pred = str('(on ' + obj + ' ' + on_object_layout[obj] + ')')
            written_predicates.append(on_pred)

            if obj in under_object_layout:
                # -- writing the "under" relation predicate:
                under_pred = str('(under ' + obj + ' ' + under_object_layout[obj] + ')')

                # -- write the predicate to the micro-problem file:
                written_predicates.append(under_pred)

        for obj in list(under_object_layout.keys()):
            # if under_object_layout[obj] not in on_object_layout and under_object_layout[obj] not in ['air']:
            if under_object_layout[obj] not in ['air']:
                on_pred = str('(on ' + under_object_layout[obj] + ' ' + obj + ')')
                written_predicates.append(on_pred)

        if written_predicates:
            # -- remove any duplicates:
            written_predicates = sorted(list(set(written_predicates)))

            pddl_file.write('\n\t; current state of environment (object layout from perception):\n')
            for pred in written_predicates:
                pddl_file.write('\t' + pred + '\n')

        written_predicates = []

        # NOTE: some objects in the simulation space have names that will combine two words together:
        #	e.g., we could have:	1. (in bottle vodka) -- bottle_vodka
        #				2. (in cup lemon_juice) -- cup_lemon_juice
        # -- because of this, we need to review those predicates that remain for "grounding"
        # 	to simulation instances.

        for pred in preconditions:
            # -- if we use the perception-based method to acquire initial state,
            #	then we will need to change required FOON states to current state:
            pred_parts = pred[1:-1].split(' ')

            # -- when using the perception-based method, we will acquire "on" or "under" conditions for the tables;
            #	therefore, we could ignore these states in particular.

            if len(pred_parts) < 3:
                written_predicates.append(pred)

            elif 'table' in [pred_parts[1], pred_parts[2]]:
                continue

            else:
                # -- in this case, we should only be considering predicates
                #       of relation "in":
                if pred_parts[0] != 'in':
                    continue

                # -- we would have to consider grounding the objects to their simulated counter-parts.
                obj1, obj2 = pred_parts[1], pred_parts[2]
                if str(pred_parts[1] + '_' + pred_parts[2]) in on_object_layout:
                    obj1 = pred_parts[1] + '_' + pred_parts[2]
                if str(pred_parts[2] + '_' + pred_parts[1]) in on_object_layout:
                    obj2 = pred_parts[2] + '_' + pred_parts[1]

                # -- we will preserve a mapping between domain-independent and domain-specific objects:
                grounded_objects[pred_parts[1]] = obj1
                grounded_objects[pred_parts[2]] = obj2

                grounded_pred = str('(' + pred_parts[0] + ' ' + obj1 + ' ' + obj2 + ')')
                if obj1 != 'air' and grounded_pred not in written_predicates:
                    written_predicates.append(grounded_pred)

                    # -- we will also put the reverse fact that obj1 is under obj2 if it is in obj1
                    reversed_pred = str('(' + 'under' + ' ' + obj2 + ' ' + obj1 + ')')
                    written_predicates.append(reversed_pred)

        if written_predicates:
            # -- remove any duplicates:
            written_predicates = list(set(sorted(written_predicates)))

            pddl_file.write('\n\t; precondition predicates obtained directly from macro PO:\n')
            for pred in written_predicates:
                pddl_file.write('\t' + pred + '\n')

        pddl_file.write(')\n\n')

        pddl_file.write('(:goal (and\n')

        # -- hand must be empty after executing action to free it for next macro-PO:
        if "(in hand air)" not in effects:
            pddl_file.write('\t; hand/end-effector must be also be empty after execution (i.e. contains "air"):\n')
            pddl_file.write('\t(in hand air)\n\n')

        pddl_file.write('\t; effect predicates obtained directly from macro PO:\n')

        # -- the effects (post-conditions) for a macro PO will form the goals:

        grounded_goals = {'reg': [], 'neg': []}

        for pred in effects:
            predicate_parts = pred[1:-1].split(' ')

            # -- we need to treat negation predicates special so we print them correctly:
            is_negation = False
            if predicate_parts[0] == 'not':
                is_negation = True
                predicate_parts = list(filter(None, predicate_parts[1:]))

                for x in range(len(predicate_parts)):
                    # -- remove any trailing parentheses characters:
                    predicate_parts[x] = predicate_parts[x].replace('(', '')
                    predicate_parts[x] = predicate_parts[x].replace(')', '')

            if 'mix' in Path(micro_fpath).stem and (len(set(predicate_parts[1:]) - set(all_objects)) > 0 or is_negation):
                # -- if we are mixing ingredients, we need to omit all negations of individual ingredients;
                #	this is because, on the micro level, only the container is viewed as being "mixed".

                # -- here, we omit the following cases of predicates when mixing:
                # 	1. individual ingredients making up will be negated after the process of mixing
                #	2. when an object is mixed, we form a new object that was not referenced as an input or precondition
                #		(e.g. when mixing salad ingredients, we create a new object "salad")
                pass

            else:
                # -- first, check if there is any grounding that needs to be done to simulation objects:
                obj1 = grounded_objects[predicate_parts[1]] if str(
                    predicate_parts[1]) in grounded_objects else str(predicate_parts[1])

                obj2 = ''
                if len(predicate_parts) > 2:
                    obj2 = grounded_objects[predicate_parts[2]] if str(
                        predicate_parts[2]) in grounded_objects else str(predicate_parts[2])

                # # -- if any of the referenced objects are "table", then we need to ground the table to the proper grid-space:
                # if 'table' in [obj1, obj2]:
                #     # -- this means that we have an "on" or "under" predicate if something is on the table:
                #     is_table_obj1 = bool(obj1 == 'table')

                #     found_mapping = False
                #     for item in on_object_layout:
                #         if on_object_layout[item] == (obj2 if is_table_obj1 else obj1):
                #             if is_table_obj1:
                #                 obj1 = item
                #             else:
                #                 obj2 = item
                #             found_mapping = True
                #             break

                #     if not found_mapping:
                #         # -- if we could not find a grounding for the table in the scene, skip this predicate:
                #         continue

                if is_negation:
                    grounded_goals['neg'].append(f"\t(not ({predicate_parts[0]} {obj1} {obj2 if obj2 else ''}))\n")
                else:
                    grounded_goals['reg'].append(f"\t({predicate_parts[0]} {obj1} {obj2 if obj2 else ''})\n")
                    pddl_file.write(grounded_goals['reg'][-1])

        # print(f'\n\n{grounded_goals}\n\n')

        # -- checking to see if there are any contradictions:
        # for pred1 in grounded_goals['neg']:
        #     is_contradiction = False
        #     for pred2 in grounded_goals['reg']:
        #         is_contradiction = str(pred2).strip() in pred1
        #         if is_contradiction: break

        #     if not is_contradiction:
        #         pddl_file.write(pred1)


        pddl_file.write('))\n')
        pddl_file.write('\n)')

    return micro_problem_fpath


def find_plan(
        domain_file: str,
        problem_file: str,
        plan_file: str = None,
        verbose: bool = False,
    ) -> str:

    # NOTE: define plan execution function that can be called for different parameters:

    # -- this will store the output that would normally be seen in the terminal
    planner_output = None
    command = None

    # -- based on the planner we use, the required command changes:
    if planner_to_use == 'fast-downward':
        # NOTE: more information on aliases for Fast-Downward can be found here: https://www.fast-downward.org/IpcPlanners
        # -- you can use a different algorithm for more optimal or satisficing results:
        method = configs[algorithm][heuristic]

        command = ['python3', str(path_to_planners[planner_to_use])]
        if plan_file:
            command.extend(['--plan-file', plan_file])
        command.extend([domain_file, problem_file, '--search', method])

    elif planner_to_use == 'PDDL4J':
        command = ['java', '-jar', str(path_to_planners[planner_to_use]),  '-o', domain_file, '-f', problem_file]

    if verbose:
        print(f'run this command: {str(" ".join(command))}')

    try:
        planner_output = subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        print(f"  -- [FOON-TAMP] : Planner error (planner:  {planner_to_use}, error code: {e.returncode})!\n\t-- Actual message: {e.output}")

    return planner_output


def solve(
        output: str,
        verbose: bool = False,
    ) -> tuple[bool, str]:

    # -- using planner-specific string checking to see if a plan was found:
    global planner_to_use

    if verbose:
        print(str(output))

    if planner_to_use == 'fast-downward':
        time_taken = None
        if "Total time: " in str(output):
            time_taken = str(output).split("Total time: ")[1].split("s")[0]
        return bool('Solution found.' in str(output)), (float(time_taken) if time_taken else None)
    elif planner_to_use == 'PDDL4J':
        # TODO: implement integration with other planners such as PDDL4J
        return False

    return False


def parse_domain_file(file_name):
    # NOTE: this function is used to parse a domain file with any type of planning operator.

    # -- this function adds them all to a list, and then return them to the main function.
    domain_file = list(open(file_name, 'r'))

    # -- a list of planning operators will be returned:
    planning_operators = []

    # -- variable name for newly creating PO instance:
    new_planning_operator = None

    for x in range(len(domain_file)):

        line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()

        if line.startswith(';'):
            # -- this is a comment; just ignore:
            pass

        elif line.startswith('(:action'):
            # -- this is the beginning of a planning operator (action) from PDDL file:
            macro_PO_name = line.split(' ')[1]

            # -- create new temp planning operator instance and add to list:
            new_planning_operator = PlanningOperator(macro_PO_name)
            planning_operators.append(new_planning_operator)

        elif line.startswith(':parameters') and not line.endswith(')'):
            # -- this is the beginning of the definition of parameters for this PO:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- splitting each line of a parameter based on variable name and type:
                parameter_parts = [X for X in line.split(' ') if X != '-']

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and len(parameter_parts) == 2:
                    # -- save the parameter to the new planning operator object:
                    new_planning_operator.addParameter(variable=parameter_parts[0], type=parameter_parts[1])

                x += 1

        elif line.startswith(':precondition'):
            # -- this is where we start reading the preconditions of the planning operator:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- remember that a predicate will be of the form: <relation> <obj1> <obj2>
                new_predicate = line

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and line.startswith('('):
                    # -- we need to check for potential *OR* operators in the preconditions:
                    if line.startswith('(or'):
                        # -- we will treat an *or* structure as a single precondition list with its arguments:
                        or_predicates = []

                        x += 1
                        while True:
                            # -- properly tokenize and clean up the line:
                            line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                            if line.startswith(')'):
                                break
                            x += 1

                            or_predicates.append(line)

                        # -- add the list of or-predicates to the list of preconditions:
                        new_planning_operator.addPrecondition(or_predicates)
                    else:
                        # -- just add this predicate as normal:
                        new_planning_operator.addPrecondition(new_predicate)

                # -- move onto the next line of the file:
                x += 1

        elif line.startswith(':effect'):
            # -- this is where we start reading the effects (post-conditions) of the planning operator:
            x += 1

            while True:
                line = [L for L in domain_file[x].split('\t') if L != ''][0].strip()
                if line.startswith(')'):
                    break

                # -- remember that a predicate will be of the form: <relation> <obj1> <obj2>
                new_predicate = line

                # -- skip any comments or new-lines on the domain file POs:
                if not line.startswith(';') and line.startswith('('):
                    new_planning_operator.addEffect(new_predicate)

                # -- move onto the next line of the file:
                x += 1

        else:
            pass

    #endfor

    return planning_operators
#enddef


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="This specifies the PDDL problem file.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="This specifies the PDDL domain file.",
    )

    args = parser.parse_args()

    if args.problem and args.domain:
        find_plan(
            problem_file=args.problem,
            domain_file=args.domain,
        )

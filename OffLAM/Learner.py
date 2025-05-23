import copy
import itertools
import os
import re
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
from OffLAM.Action import Action
from OffLAM.ActionModel import ActionModel
from OffLAM.Observation import Observation
from OffLAM.Operator import Operator
from OffLAM.Trace import Trace
from Util import metrics
from Util.PddlParser import PddlParser
from Util.util import powerset

template = "{Iter:^6}|{Real_precs:^10}|{Learn_precs:^11}|{Real_pos:^8}|{Learn_pos:^9}" \
           "|{Real_neg:^8}|{Learn_neg:^9}|{Ins_pre:^7}|{Del_pre:^7}|{Ins_pos:^7}|{Del_pos:^7}" \
           "|{Ins_neg:^7}|{Del_neg:^7}|{Precs_recall:^12}|{Pos_recall:^10}|{Neg_recall:^10}" \
           "|{Precs_precision:^15}|{Pos_precision:^13}|{Neg_precision:^13}" \
           "|{Tot_recall:^10}|{Tot_precision:^13}"


class Learner:

    def __init__(self, input_domain_path: str):

        # PDDL parser, used to update pddl problem file
        self.parser = PddlParser()

        # Initialize action model with input one
        # if os.path.exists('PDDL/domain_input.pddl'):
        if os.path.exists(input_domain_path):
            print(f'[Info] Reading input action model from file {input_domain_path}')
            self.action_model = ActionModel(input_file=input_domain_path)
            os.makedirs('PDDL', exist_ok=True)
            self.action_model.write('PDDL/domain_learned.pddl')
        else:
            self.action_model = ActionModel(input_file='PDDL/domain_learned.pddl')

        self.eval = pd.DataFrame()
        self.iter = 0
        self.trace = None
        self.traces = None

        self.traces_updates = dict()
        self.operator_updates = {op.operator_name: True for op in self.action_model.operators}


    def learn_offline(self, input_trace, greedy=False):
        """
        Solve the problem instance
        :param input_trace: the input plan trace
        :return: None
        """

        self.trace = input_trace

        # all_facts = self.compute_facts_superset()
        if self.trace.name not in self.traces_updates.keys():
            self.traces_updates[self.trace.name] = [True for _ in range(len(self.trace.observations))]

        print(f'[Info] Processing trace {self.trace.name}')
        self.learn_action_model(self.action_model, self.trace, greedy=greedy)

    def post_process_obs(self, action_model, obs):
        post_processed_obs = copy.deepcopy(obs)
        all_fict_preds = [p.split('(')[0] for p in action_model.predicates if p[1:].startswith('_')]

        all_original_preds = list({"_".join([el for el in p.split('_') if len(el) > 1]) for p in all_fict_preds})

        obs_positive_literals = set.union(*post_processed_obs.positive_literals.values())
        obs_negative_literals = set.union(*post_processed_obs.negative_literals.values())

        obs_positive_split = {original_pred: [p[:-1].split('(') for p in obs_positive_literals
                                              if p.split('(')[0].endswith(f'_{original_pred}')]
                              for original_pred in all_original_preds}

        obs_negative_split = defaultdict(lambda: defaultdict(set))
        [obs_negative_split[original_pred][p[4:-1].split('(')[1]].add(p[4:-1].split('(')[0])
         for original_pred in all_original_preds
         for p in obs_negative_literals
         if p.split('(')[0].endswith(f'_{original_pred}') ]

        for original_pred in all_original_preds:
            fict_preds = {p for p in all_fict_preds if p.endswith(f"_{original_pred}")}

            pos_objs_comb = {p[1] for p in obs_positive_split[original_pred]}

            for obj_comb in pos_objs_comb:
                obs_positive_split_obj = [p for p in obs_positive_split[original_pred] if p[1].endswith(obj_comb)]
                if fict_preds.issubset({p[0] for p in obs_positive_split_obj}):
                    original_pos = {f"{original_pred}({p[1]})" for p in obs_positive_split_obj
                                    if p[0] in fict_preds}

                    original_pos = list(original_pos)[0]
                    post_processed_obs.positive_literals[original_pos.split('(')[0]].add(original_pos)  # TODO check why only the first original pos is added

            neg_objs_comb = obs_negative_split[original_pred].keys()

            for obj_comb in neg_objs_comb:

                if fict_preds.issubset(obs_negative_split[original_pred][obj_comb]):
                    original_neg = f"not_{original_pred}({obj_comb})"

                    post_processed_obs.negative_literals[original_neg.split('(')[0]].add(original_neg)

            for fict_pred in fict_preds:
                del post_processed_obs.positive_literals[fict_pred]
                del post_processed_obs.negative_literals[fict_pred]

        return post_processed_obs

    def post_process_action_model(self, action_model):

        post_processed_action_model = ActionModel()
        post_processed_action_model.input_file = action_model.input_file
        # post_processed_action_model.types_hierarchy = post_processed_action_model.read_object_types_hierarchy("PDDL/domain_input.pddl")
        post_processed_action_model.types_hierarchy = post_processed_action_model.read_object_types_hierarchy(post_processed_action_model.input_file)
        post_processed_action_model.predicates = action_model.predicates
        post_processed_action_model.operators = [Operator(op.operator_name, op.parameters,
                                                          eff_neg_cert={p for p in op.eff_neg_cert},
                                                          eff_pos_cert={p for p in op.eff_pos_cert},
                                                          eff_neg_uncert={p for p in op.eff_neg_uncert},
                                                          eff_pos_uncert={p for p in op.eff_pos_uncert},
                                                          precs_cert={p for p in op.precs_cert},
                                                          precs_uncert={p for p in op.precs_uncert})
                                                 for op in action_model.operators]

        all_fict_preds = defaultdict(set)
        [all_fict_preds["_".join([el for el in p.split('(')[0].split('_') if len(el) > 1])].add(p.split('(')[0])
         for p in post_processed_action_model.predicates if p[1] == '_']

        for original_pred, fict_preds in all_fict_preds.items():

            for op in post_processed_action_model.operators:

                # Replace certain preconditions
                precs_cert_fict = {p for p in op.precs_cert if p.split('(')[0] in fict_preds}
                if len(precs_cert_fict) > 0:
                    original_precs = {p[p.find(original_pred):] for p in precs_cert_fict}
                    op.precs_cert |= original_precs

                op.precs_cert -= precs_cert_fict

                # Replace uncertain preconditions
                precs_uncert_fict = {p for p in op.precs_uncert if p.split('(')[0] in fict_preds}

                if len(precs_uncert_fict) > 0:
                    original_precs = {p[p.find(original_pred):] for p in precs_uncert_fict}
                    op.precs_uncert |= original_precs

                op.precs_uncert -= precs_uncert_fict

                # Replace certain positive effects
                op_params_comb = {p.split('(')[1][:-1] for p in op.eff_pos_cert
                                  if p.split('(')[0].endswith(f'_{original_pred}')}
                for param_comb in op_params_comb:

                    if fict_preds.issubset({p.split('(')[0] for p in op.eff_pos_cert if p.endswith(f"{param_comb})")}):
                        original_effs_pos = {p[p.find(original_pred):] for p in op.eff_pos_cert if p.split('(')[0] in fict_preds
                                             and p.endswith(f"{param_comb})")}
                        op.eff_pos_cert |= original_effs_pos

                op.eff_pos_cert = {p for p in op.eff_pos_cert if p.split('(')[0] not in fict_preds}

                # Replace uncertain positive effects
                if len([p for p in op.eff_pos_uncert if p.split('(')[0] in fict_preds]) > 0:

                    original_effs_pos = {p[p.find(original_pred):] for p in op.eff_pos_uncert if p.split('(')[0] in fict_preds}
                    op.eff_pos_uncert |= original_effs_pos

                op.eff_pos_uncert = {p for p in op.eff_pos_uncert if p.split('(')[0] not in fict_preds}

                # Add uncertain positive effects whether some of their fictitious predicates (but not all) are in
                # the set of certain positive effects
                original_op = [o for o in action_model.operators if o.operator_name == op.operator_name][0]

                if len([p for p in original_op.eff_pos_cert if p.split('(')[0] in fict_preds]) > 0:
                    original_effs_pos = {p[p.find(original_pred):] for p in original_op.eff_pos_cert if p.split('(')[0] in fict_preds}

                    op.eff_pos_uncert |= {p for p in original_effs_pos if p not in op.eff_pos_cert}

                # Replace certain negative effects
                op_params_comb = {p.split('(')[1][:-1] for p in op.eff_neg_cert
                                  if p.split('(')[0].endswith(f'_{original_pred}')}
                for param_comb in op_params_comb:
                    if fict_preds.issubset({p.split('(')[0] for p in op.eff_neg_cert if p.endswith(f"{param_comb})")}):
                        original_effs_neg = {p[p.find(original_pred):] for p in op.eff_neg_cert if p.split('(')[0] in fict_preds
                                             and p.endswith(f"{param_comb})")}
                        op.eff_neg_cert |= original_effs_neg

                op.eff_neg_cert = {p for p in op.eff_neg_cert if p.split('(')[0] not in fict_preds}

                # Replace uncertain negative effects
                if len([p for p in op.eff_neg_uncert if p.split('(')[0] in fict_preds]) > 0:
                    original_effs_neg = {p[p.find(original_pred):] for p in op.eff_neg_uncert if p.split('(')[0] in fict_preds}
                    op.eff_neg_uncert |= original_effs_neg

                # Add uncertain negative effects whether some of their fictitious predicates (but not all) are in
                # the set of certain negative effects
                original_op = [o for o in action_model.operators if o.operator_name == op.operator_name][0]

                if len([p for p in original_op.eff_neg_cert if p.split('(')[0] in fict_preds]) > 0:
                    original_effs_neg = {p[p.find(original_pred):] for p in original_op.eff_neg_cert if p.split('(')[0] in fict_preds}
                    op.eff_neg_uncert |= {p for p in original_effs_neg if p not in op.eff_neg_cert}

                op.eff_neg_uncert = {p for p in op.eff_neg_uncert if p.split('(')[0] not in fict_preds}

        post_processed_action_model.predicates = post_processed_action_model.read_predicates(post_processed_action_model.input_file)

        return post_processed_action_model

    # More efficient
    def get_relevant_objects_in_trace(self, trace):

        observed_objects = defaultdict(set)

        # Get objects in trace actions
        for action in trace.actions:
            if type(action) == Action:
                action_operator = [o for o in self.action_model.operators if o.operator_name == action.operator_name][0]
                [observed_objects[o].add(t) for o, t in zip(action.parameters, list(action_operator.parameters.values()))]

        # Get relevant literals, i.e. literals that changed their truth value in the trace
        all_literals = set()
        for obs in trace.observations:

            obs_positive_literals = set.union(*obs.positive_literals.values())
            obs_negative_literals = set.union(*obs.negative_literals.values())

            all_literals |= {l for l in obs_positive_literals}
            all_literals |= {l[4:] for l in obs_negative_literals}

        # Get relevant literals object names and their types
        for l in all_literals:
            l_predicate = l.split('(')[0]
            l_objects = [o for o in l.split('(')[1].strip()[:-1].split(',') if o != '']
            p = [p for p in self.action_model.predicates if p.startswith(f"{l_predicate}(")][0]
            p_types = [t for t in p.split('(')[1].strip()[:-1].split(',') if t != '']

            [observed_objects[o].add(t) for o, t in zip(l_objects, p_types)]

        for obj_name, obj_type in observed_objects.items():
            if len(obj_type) > 1:
                supertypes = list({t for t in obj_type if t in self.action_model.types_hierarchy.keys()})
                subtypes = {t for t in obj_type if t not in supertypes}

                # Check if the subtype cannot be observed
                if len(subtypes) == 0:
                    for i in range(len(supertypes)):
                        if not np.any([t in self.action_model.types_hierarchy[supertypes[i]] for t in supertypes
                                       if t != supertypes[i]]):
                            subtypes = [supertypes[i]]
                            break
                assert len(subtypes) == 1, f'Detected multiple subtypes {subtypes} for object {obj_name}'
                observed_objects[obj_name] = subtypes

        assert np.all([len(v) == 1 for v in observed_objects.values()]), 'Check object type hierarchy, predicate ' \
                                                                         'object types, and actions object types'

        observed_objects = {k: list(v)[0] for k, v in observed_objects.items()}

        return observed_objects

    def get_domain_name(self, action_model):
        with open(action_model.input_file, 'r') as f:
            data = [el.strip() for el in f.read().split("\n")]
            domain_name = re.findall(r"domain.+?\)","".join(data))[0].strip()[:-1].split()[-1].strip()
        return domain_name

    def parse_trace(self, input_trace):

        with open(input_trace, 'r') as f:
            data = [el.strip() for el in f.read().split("\n") if el.strip() != '']

            data = [r for r in data if r.strip().startswith('(:state') or r.startswith('(:action')]

            states = []
            actions = []
            adding_state = True
            adding_action = False
            for r in data:
                if adding_state:
                    if r.startswith('(:state'):
                        states.append(r.replace('(:state', '').strip()[:-1].strip())
                        adding_action = True
                        adding_state = False
                    elif r.startswith('(:action'):
                        states.append('')
                        actions.append(r.replace('(:action', '').strip()[:-1].strip())
                    else:
                        print(f'Error when parsing input trace {input_trace}')
                        exit()

                elif adding_action:
                    if r.startswith('(:action'):
                        actions.append(r.replace('(:action', '').strip()[:-1].strip())
                        adding_action = False
                        adding_state = True
                    elif r.startswith('(:state'):
                        actions.append(None)
                        states.append(r.replace('(:state', '').strip()[:-1].strip())
                    else:
                        print(f'Error when parsing input trace {input_trace}')
                        exit()

            trace_observations = []
            trace_actions = []

            for s in states:
                neg_literals = [e.strip()[1:-1].replace('not', '', 1).strip() for e in re.findall(r"\(not[^)]*\)\)", s)
                                  if not len(e.replace('(and', '').replace(')', '').strip()) == 0]
                pos_literals = [e.strip() for e in re.findall(r"\([^()]*\)", s)
                                  if e not in neg_literals and not len(e.replace('(and', '').replace(')', '').strip()) == 0]
                pos_literals = [f"{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                                for l in pos_literals]
                neg_literals = [f"not_{l.strip()[1:-1].split()[0]}({f','.join([o for o in l.strip()[1:-1].split()[1:] if o != ''])})"
                                for l in neg_literals]

                trace_observations.append(Observation(pos_literals + neg_literals))

            for a in actions:

                if a is None:
                    trace_actions.append(None)
                else:
                    a_name = a.strip()[1:-1].split()[0]
                    # a_objs = [o for o in a.strip()[1:-1].split()[1:] if o != '']
                    a = a.strip()[1:-1]
                    # operator_name = a.split()[0]

                    operator = next((o for o in self.action_model.operators if o.operator_name == a_name), None)

                    if len(a.split()) > 1:
                        objects = a.split()[1:]
                        params_bind = {f'?param_{i + 1}': obj for i, obj in enumerate(objects)}
                    else:
                        objects = []
                    action_precs_cert = {self.ground_lifted_atom(params_bind, p) for p in operator.precs_cert}
                    action_precs_uncert = {self.ground_lifted_atom(params_bind, p) for p in operator.precs_uncert}
                    action_eff_pos_cert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_pos_cert}
                    action_eff_pos_uncert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_pos_uncert}
                    action_eff_neg_cert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_neg_cert}
                    action_eff_neg_uncert = {self.ground_lifted_atom(params_bind, p) for p in operator.eff_neg_uncert}
                    action = Action(a_name, objects, action_precs_cert, action_eff_pos_cert, action_eff_neg_cert,
                                    action_precs_uncert, action_eff_pos_uncert, action_eff_neg_uncert)

                    ground_model_actions = [str(a) for a in self.action_model.ground_actions[action.operator_name]]
                    if str(action) not in ground_model_actions:
                        trace_actions.append(action)
                        self.action_model.ground_actions[action.operator_name].append(action)
                        self.action_model.ground_action_labels.add(str(action))
                    else:
                        action_idx = ground_model_actions.index(str(action))
                        trace_actions.append(self.action_model.ground_actions[action.operator_name][action_idx])

        return Trace(input_trace, trace_observations, trace_actions)

    def instantiate_actions_model(self, ground_action_labels):

        ground_action_names = [a.split()[0][1:] for a in ground_action_labels]
        ground_action_params = [a[:-1].split()[1:] for a in ground_action_labels]
        ground_action_params = [{f'?param_{i + 1}': obj for i, obj in enumerate(params)} for params in ground_action_params]
        ground_action_operators = [next((o for o in self.action_model.operators if o.operator_name == a_name), None)
                                   for a_name in ground_action_names]

        ground_actions = []

        for action_name, action_params, action_operator in zip(ground_action_names, ground_action_params, ground_action_operators):

            # Ground certain preconditions
            precs_cert = {self.ground_lifted_atom(action_params, p) for p in action_operator.precs_cert}

            # Ground uncertain preconditions
            precs_uncert = {self.ground_lifted_atom(action_params, p) for p in action_operator.precs_uncert}

            # Ground certain positive effects
            eff_pos_cert = {self.ground_lifted_atom(action_params, p) for p in action_operator.eff_pos_cert}

            # Ground uncertain positive effects
            eff_pos_uncert = {self.ground_lifted_atom(action_params, p) for p in action_operator.eff_pos_uncert}

            # Ground certain negative effects
            eff_neg_cert = {self.ground_lifted_atom(action_params, p) for p in action_operator.eff_neg_cert}

            # Ground uncertain positive effects
            eff_neg_uncert = {self.ground_lifted_atom(action_params, p) for p in action_operator.eff_neg_uncert}

            new_action = Action(action_name, list(action_params.values()), precs_cert, eff_pos_cert,
                                eff_neg_cert, precs_uncert, eff_pos_uncert, eff_neg_uncert)
            ground_actions.append(new_action)
            self.action_model.ground_actions[action_name].append(new_action)
            self.action_model.ground_action_labels.add(str(new_action))

        return ground_actions

    def compute_possible_actions(self, trace_objects, action_model, eff_pos, eff_neg, next_positive, next_negative):

        action_model_dummy = ActionModel()
        action_model_dummy.types_hierarchy = action_model_dummy.read_object_types_hierarchy("PDDL/domain_input.pddl")
        action_model_dummy.predicates = []

        become_pos_names = [p.split('(')[0] for p in eff_pos]
        become_neg_names = [p.split('(')[0] for p in eff_neg]

        predicates = set()
        pred_names = set()
        for p in set(become_pos_names + become_neg_names):
            pred = [pred for pred in action_model.predicates if pred.startswith(f"{p}(")][0]
            predicates |= {f"make_true_{pred}", f"make_false_{pred}"}
            pred_names.add(pred.split('(')[0])
        for p in action_model.predicates:
            predicates |= {f"can_be_true_{p}", f"can_be_false_{p}"}

        dummy_operators = []
        for op in action_model.operators:

            eff_neg_names = {p.split('(')[0] for p in op.eff_neg_cert}
            eff_pos_names = {p.split('(')[0] for p in op.eff_pos_cert}

            impossible_op = False

            # Check if a literal l becoming positive belongs to the uncertain + certain positive effects of the operator
            # otherwise prevent the operator from being grounded
            for p in become_pos_names:
                p_eff_pos = [f"make_true_{e}" for e in op.eff_pos_uncert | op.eff_pos_cert if e.split('(')[0] == p]
                if len(p_eff_pos) == 0:
                    impossible_op = True

            # Check if a literal l becoming negative belongs to the uncertain + certain negative effects of the operator
            # otherwise prevent the operator from being grounded
            if not impossible_op:
                for p in become_neg_names:
                    p_eff_neg = [f"make_false_{e}" for e in op.eff_neg_uncert | op.eff_neg_cert if e.split('(')[0] == p]
                    if len(p_eff_neg) == 0:
                        impossible_op = True

            if not impossible_op:

                # Check if a literal becoming positive belongs to the uncertain positive effects of the operator
                all_eff_pos = []
                for p in become_pos_names:
                    p_eff_pos = [f"make_true_{e}" for e in op.eff_pos_uncert if e.split('(')[0] == p]

                    if len(p_eff_pos) > 0 and p not in eff_pos_names:
                        all_eff_pos.append([list(el) for el in powerset(p_eff_pos) if len(el) > 0])

                # Check if a literal becoming negative belongs to the uncertain negative effects of the operator
                all_eff_neg = []
                for p in become_neg_names:
                    p_eff_neg = [f"make_false_{e}" for e in op.eff_neg_uncert if e.split('(')[0] == p]

                    if len(p_eff_neg) > 0 and p not in eff_neg_names:
                        all_eff_neg.append([list(el) for el in powerset(p_eff_neg) if len(el) > 0])

                for p in op.eff_pos_cert:
                    if p.split('(')[0] in become_pos_names:
                        all_eff_pos.append([[f'make_true_{p}']])
                    all_eff_pos.append([[f'not(can_be_false_{p})']])
                for p in op.eff_neg_cert:
                    if p.split('(')[0] in become_neg_names:
                        all_eff_neg.append([[f'make_false_{p}']])
                    all_eff_neg.append([[f'not(can_be_true_{p})']])

                dummy_precs_combinations = [list([it for sublist in p for it in sublist]) for p in itertools.product(*(all_eff_pos + all_eff_neg))]

                filtered_dummy_precs_combinations = []
                for k, dummy_precs in enumerate(dummy_precs_combinations):
                    make_true_pred = {p[10:] for p in dummy_precs if p.startswith('make_true')}
                    make_false_pred = {p[11:] for p in dummy_precs if p.startswith('make_false')}

                    cannot_be_true_pred = {p.replace('not(can_be_true', '') for p in dummy_precs if p.startswith('not(can_be_true')}
                    cannot_be_false_pred = {p.replace('not(can_be_false', '') for p in dummy_precs if p.startswith('not(can_be_false')}

                    if not make_true_pred.intersection(make_false_pred) \
                            and not cannot_be_true_pred.intersection(cannot_be_false_pred) \
                            and not make_true_pred.intersection(cannot_be_true_pred) \
                            and not make_false_pred.intersection(cannot_be_false_pred):
                        filtered_dummy_precs_combinations.append(dummy_precs)

                removed = set()
                for i in range(len(filtered_dummy_precs_combinations) - 1):
                    if i not in removed:
                        for j in range(i + 1, len(filtered_dummy_precs_combinations)):
                            if j not in removed:
                                if set(filtered_dummy_precs_combinations[i]).issubset(set(filtered_dummy_precs_combinations[j])):
                                    removed.add(j)
                                elif set(filtered_dummy_precs_combinations[j]).issubset(set(filtered_dummy_precs_combinations[i])):
                                    removed.add(i)

                filtered_dummy_precs_combinations = [filtered_dummy_precs_combinations[i]
                                                     for i in range(len(filtered_dummy_precs_combinations))
                                                     if i not in removed]

                for k, dummy_precs in enumerate(filtered_dummy_precs_combinations):
                    dummy_operators.append(Operator(f"{op.operator_name}_{k}", op.parameters, precs_cert=set(dummy_precs),
                                                    eff_pos_cert=set(), eff_neg_cert=set(), eff_neg_uncert=set(),
                                                    eff_pos_uncert=set()))

        # Check dummy operators that allow all possible groundings, and avoid computing such groundings
        all_ground_op = set()
        partial_ground_op = set()
        for dummy_op in dummy_operators:
            if len(dummy_op.precs_cert) == 0:
                all_ground_op.add(dummy_op)
            else:
                partial_ground_op.add(dummy_op)

        all_actions = set()

        all_ground_op = {'_'.join(op.operator_name.split('_')[:-1]) for op in all_ground_op}

        if len(partial_ground_op) == 1 and len(all_ground_op) == 0:

            action_model_dummy.operators = partial_ground_op

            action_model_dummy.predicates += list(predicates)
            action_model_dummy.predicates.append('dummy()')

            # To get correct domain name
            action_model_dummy.input_file = 'PDDL/domain_input.pddl'

            action_model_dummy.write('PDDL/domain_tmp.pddl', precs_uncertain=False, eff_neg_uncertain=False, eff_pos_uncertain=False)

            # Write dummy pddl state
            domain_name = self.get_domain_name(action_model_dummy)
            dummy_literals = [f"make_true_{l}" for l in eff_pos]
            dummy_literals += [f"make_false_{l}" for l in eff_neg]
            dummy_literals += [f"can_be_true_{l}" for l in next_positive]
            dummy_literals += [f"can_be_false_{l[4:]}" for l in next_negative]

            self.parser.write_pddl_state(trace_objects, Observation(dummy_literals), domain_name, 'PDDL/facts_tmp.pddl')

            bash_command = ["./Util/Grounding/Instantiate", "PDDL/domain_tmp.pddl", "PDDL/facts_tmp.pddl"]

            process = subprocess.run(bash_command, capture_output=True)
            output = str(process.stdout).split('\\n')

            all_actions = re.findall(r"\([^()]*\)", re.findall(r"so far.*literals", "".join(output))[0])
            all_actions = [a[1:-1].split() for a in all_actions]
            all_actions = {f"({a[0].rsplit('_', 1)[0]} {' '.join(a[1:])})" for a in all_actions}

        return all_actions, all_ground_op

    def ground_lifted_atom(self, action_params_bind, lifted_atom):
        lifted_atom_split = lifted_atom.split('(')
        lifted_atom_params = [p for p in lifted_atom_split[1][:-1].split(',') if p != '']
        return f"{lifted_atom_split[0]}({','.join([action_params_bind[p] for p in lifted_atom_params])})"

    def lift_ground_atoms(self, ground_action, ground_atoms):
        lifted_precs = []
        action_params = ground_action.parameters
        action_params_bind = {p:[f'?param_{i + 1}' for i in range(len(action_params)) if action_params[i] == p]
                                 for p in action_params}

        for prec in ground_atoms:
            prec_objects = [o for o in prec.split('(')[1][:-1].split(',') if o.strip() != '']

            try:
                params_bind_combinations = [list(p) for p in itertools.product(*[action_params_bind[obj]
                                                                                 for obj in prec_objects])]
                for tup in params_bind_combinations:
                    lifted_prec = f"{prec.split('(')[0]}({','.join(tup)})"
                    lifted_precs.append(lifted_prec)
            except:
                print(f'Warning: cannot lift ground atom {prec} for action {ground_action}')

        return lifted_precs

    def remove_predicate(self, pred_name):
        self.action_model.remove_predicate(pred_name)
        [t.remove_predicates([pred_name]) for t in self.traces]

    def learn_action_model(self, action_model, trace, greedy=False):
        learning_in_progress = True
        prev_cert_size = {op.operator_name: len(op.eff_pos_cert) + len(op.eff_neg_cert) + len(op.precs_cert)
                         for op in action_model.operators}
        prev_uncert_size = {op.operator_name: len(op.eff_pos_uncert) + len(op.eff_neg_uncert) + len(op.precs_uncert)
                         for op in action_model.operators}
        prev_trace_actions = len([a for a in trace.actions if type(a) == Action])

        prev_trace_literals_pos = [sum([len(list(pos_literals)) for pos_name, pos_literals in o.positive_literals.items()])
                                   for o in trace.observations]
        prev_trace_literals_neg = [sum([len(list(neg_literals)) for neg_name, neg_literals in o.negative_literals.items()])
                                   for o in trace.observations]
        prev_trace_literals = [prev_trace_literals_pos[i] + prev_trace_literals_neg[i]
                               for i in range(len(prev_trace_literals_pos))]

        while learning_in_progress:
            learning_in_progress = False

            self.learn_from_trace(action_model, trace, greedy=greedy)

            self.fill_trace(action_model, trace)

            # Check if the trace or the action model have been updated
            cert_size = {op.operator_name: len(op.eff_pos_cert) + len(op.eff_neg_cert) + len(op.precs_cert)
                             for op in action_model.operators}
            uncert_size = {op.operator_name: len(op.eff_pos_uncert) + len(op.eff_neg_uncert) + len(op.precs_uncert)
                             for op in action_model.operators}
            trace_actions = len([a for a in trace.actions if type(a) == Action])

            trace_literals_pos = [sum([len(list(pos_literals)) for pos_name, pos_literals in o.positive_literals.items()])
                                  for o in trace.observations]
            trace_literals_neg = [sum([len(list(neg_literals)) for neg_name, neg_literals in o.negative_literals.items()])
                                  for o in trace.observations]
            trace_literals = [trace_literals_pos[i] + trace_literals_neg[i] for i in range(len(trace_literals_pos))]

            for op in self.action_model.operators:
                if cert_size[op.operator_name] != prev_cert_size[op.operator_name]:
                    self.operator_updates[op.operator_name] = True
                    learning_in_progress = True
                else:
                    self.operator_updates[op.operator_name] = False

            for op in self.action_model.operators:
                if not self.operator_updates[op.operator_name] and uncert_size[op.operator_name] != prev_uncert_size[op.operator_name]:
                    self.operator_updates[op.operator_name] = True
                    learning_in_progress = True

            for i in range(len(trace_literals)):
                if trace_literals[i] != prev_trace_literals[i]:
                    self.traces_updates[trace.name][i] = True
                    learning_in_progress = True
                else:
                    self.traces_updates[trace.name][i] = False

            if trace_actions != prev_trace_actions:
                learning_in_progress = True

            prev_cert_size = cert_size
            prev_uncert_size = uncert_size
            prev_trace_literals = trace_literals
            prev_trace_actions = trace_actions

    def learn_from_trace(self, action_model, trace, greedy=False):

        for i in range(len(trace.observations) - 1):

            prev_observation = trace.observations[i]
            prev_action = trace.actions[i]
            next_observation = trace.observations[i + 1]

            # if prev_action is not None:
            if type(prev_action) == Action:
                self.remove_uncertain_preconditions(action_model, prev_observation, prev_action)
                self.learn_certain_effects(action_model, prev_observation, prev_action, next_observation)
                self.remove_uncertain_eff_pos(action_model, prev_action, next_observation)
                self.remove_uncertain_eff_neg(action_model, prev_action, next_observation)
            else:
                if not greedy:
                    self.learn_from_possible_actions(trace, action_model, prev_observation, next_observation, i)

        return action_model

    def learn_from_possible_actions(self, trace, action_model, prev_obs, next_obs, action_idx):

        if trace.actions[action_idx] is None \
                or (self.traces_updates[trace.name][action_idx] or self.traces_updates[trace.name][action_idx + 1]) \
                or bool(np.any(list(self.operator_updates.values()))):
            possible_actions, all_ground_op = self.get_possible_actions(trace, prev_obs, next_obs, action_model, action_idx)
        else:
            possible_actions = trace.actions[action_idx]
            all_ground_op = trace.all_ground_op[action_idx]

        one_possible_op = len({str(a)[1:-1].split()[0] for a in possible_actions}) == 1 and len(all_ground_op) == 0

        if one_possible_op:
            print('[Debug] One possible operator found')

            ######################################################################################################
            ############################## COMPUTE COMMON EFFECTS AND PRECONDITIONS ##############################
            ######################################################################################################
            operator = next((o for o in action_model.operators
                             if o.operator_name == possible_actions[0].operator_name), None)

            ground_actions = action_model.ground_actions[operator.operator_name]

            # Get positive effects by comparing previous and next state observations

            prev_positive_literals = set.union(*prev_obs.positive_literals.values())
            prev_negative_literals = set.union(*prev_obs.negative_literals.values())
            next_positive_literals = set.union(*next_obs.positive_literals.values())
            next_negative_literals = set.union(*next_obs.negative_literals.values())

            effects_pos = [l[4:] for l in prev_negative_literals
                           if l[4:] in next_positive_literals]
            effects_neg = [l for l in prev_positive_literals if f"not_{l}" in next_negative_literals]


            ######################################################################################################
            ################################## COMPUTE COMMON POSITIVE EFFECTS ###################################
            ######################################################################################################
            common_lifted_effects_pos = self.get_common_eff_pos(possible_actions, effects_pos, action_model)


            ######################################################################################################
            #################################### ADD COMMON POSITIVE EFFECTS #####################################
            ######################################################################################################
            for effect_pos_param in common_lifted_effects_pos:

                operator.add_eff_pos_cert(effect_pos_param)

                # Add grounded certain positive effect on all ground actions compute by instantiating the operator
                [a.add_eff_pos_cert(self.ground_lifted_atom(a.params_bind, effect_pos_param))
                 for a in ground_actions]


            ######################################################################################################
            ################################## COMPUTE COMMON NEGATIVE EFFECTS ###################################
            ######################################################################################################
            common_lifted_effects_neg = self.get_common_eff_neg(possible_actions, effects_neg, action_model)


            ######################################################################################################
            #################################### ADD COMMON NEGATIVE EFFECTS #####################################
            ######################################################################################################
            for effect_neg_param in common_lifted_effects_neg:

                operator.add_eff_neg_cert(effect_neg_param)

                # Add grounded certain negative effect on all ground actions computed by instantiating the operator
                [a.add_eff_neg_cert(self.ground_lifted_atom(a.params_bind, effect_neg_param))
                 for a in ground_actions]


            ######################################################################################################
            ############################## REMOVE COMMON UNCERTAIN POSITIVE EFFECTS ##############################
            ######################################################################################################

            impossible_eff_pos = [l[4:] for l in next_negative_literals]
            common_impossible_eff_pos = self.get_common_impossible_eff_pos(possible_actions, impossible_eff_pos, action_model)

            # If the size of 'lifted_effs_pos' is greater than one, then there is an ambiguous objects binding,
            # but the ambiguity is not problematic, as in the following example
            # e.g.
            # uncertain positive effect p(param1, param2)
            # hypotetic positive effect p(param1, param3)
            # If the action a(c1,c2,c2) is executed AND p(c1,c2) is false in the destination state, then
            # both p(param1, param2) and p(param1, param3) are not positive effects.
            # Suppose that p(param1, param3) is a positive effect, then we have the contradiction that p(c1,c2) is
            # true in the destination state (even if there is a negative effect p(param1, param2) since, in case of
            # inconsistent effects, we give priority to the positive effect). Similarly for p(param1, param2).
            for lifted_eff_pos in common_impossible_eff_pos:

                # assert lifted_eff_pos not in operator.eff_pos_cert

                if lifted_eff_pos in operator.eff_pos_uncert:

                    # Remove lifted precondition from operator uncertain preconditions
                    operator.remove_eff_pos_uncert(lifted_eff_pos)

                    # Remove grounded positive effects from all ground actions computed by instantiating the operator
                    [a.remove_eff_pos_uncert(self.ground_lifted_atom(a.params_bind, lifted_eff_pos))
                     for a in ground_actions]


            ######################################################################################################
            ############################## REMOVE COMMON UNCERTAIN NEGATIVE EFFECTS ##############################
            ######################################################################################################

            impossible_eff_neg = [l for l in next_positive_literals]
            common_impossible_eff_neg = self.get_common_impossible_eff_neg(possible_actions, impossible_eff_neg, action_model)

            # If the size of 'lifted_effs_pos' is greater than one, then there is an ambiguous objects binding,
            # but the ambiguity is not problematic as in the following example
            # e.g.
            # uncertain positive effect p(param1, param2)
            # hypotetic positive effect p(param1, param3)
            # If the action a(c1,c2,c2) is executed AND p(c1,c2) is false in the destination state, then
            # both p(param1, param2) and p(param1, param3) are not positive effects.
            # Suppose that p(param1, param3) is a positive effect, then we have the contradiction that p(c1,c2) is
            # true in the destination state (even if there is a negative effect p(param1, param2) since, in case of
            # inconsistent effects, we give priority to the positive effect). Similarly for p(param1, param2).
            for lifted_eff_neg in common_impossible_eff_neg:

                # assert lifted_eff_pos not in operator.eff_pos_cert

                if lifted_eff_neg in operator.eff_neg_uncert:

                    # Remove lifted precondition from operator uncertain preconditions
                    operator.remove_eff_neg_uncert(lifted_eff_neg)

                    # Remove grounded positive effects from all ground actions computed by instantiating the operator
                    [a.remove_eff_pos_uncert(self.ground_lifted_atom(a.params_bind, lifted_eff_neg))
                     for a in ground_actions]


            ######################################################################################################
            ############################### REMOVE COMMON UNCERTAIN PRECONDITIONS ################################
            ######################################################################################################
            prev_negative_literals = {l[4:] for l in prev_negative_literals}
            fake_preconditions = [l for l in prev_negative_literals
                                  if set([o for o in l.split('(')[1][:-1].split(',') if o != ''])
                                      .issubset(set(possible_actions[0].parameters))]

            lifted_fake_preconditions = set(self.lift_ground_atoms(possible_actions[0], fake_preconditions))
            lifted_precs = {p for p in lifted_fake_preconditions if p in operator.precs_uncert}

            if len(lifted_precs) > 0:
                for j in range(1, len(possible_actions)):
                    action = possible_actions[j]
                    fake_preconditions = [l for l in prev_negative_literals
                                          if set([o for o in l.split('(')[1][:-1].split(',') if o != ''])
                                              .issubset(set(action.parameters))]
                    lifted_fake_preconditions = set(self.lift_ground_atoms(action, fake_preconditions))
                    lifted_precs = lifted_precs.intersection(lifted_fake_preconditions)

                    if len(lifted_precs) == 0:
                        break

            # Remove the intersection of uncertain preconditions that are false when all the possible actions
            # could have been executed
            for lifted_prec in lifted_precs:

                # Remove lifted precondition from operator uncertain preconditions
                operator.remove_prec_uncert(lifted_prec)

                # Remove grounded preconditions from all ground actions computed by instantiating the operator
                [a.remove_prec_uncert(self.ground_lifted_atom(a.params_bind, lifted_prec))
                 for a in ground_actions]


    def get_possible_action_names(self, trace_objects, prev_obs, next_obs, action_model):

        prev_obs = self.post_process_obs(action_model, prev_obs)
        next_obs = self.post_process_obs(action_model, next_obs)

        action_model = self.post_process_action_model(action_model)

        prev_positive_literals = set.union(*prev_obs.positive_literals.values())
        prev_negative_literals = set.union(*prev_obs.negative_literals.values())
        next_positive_literals = set.union(*next_obs.positive_literals.values())
        next_negative_literals = set.union(*next_obs.negative_literals.values())

        eff_pos = [l[4:] for l in prev_negative_literals if l[4:] in next_positive_literals]
        eff_neg = [l for l in prev_positive_literals if f"not_{l}" in next_negative_literals]

        ground_action_names, all_ground_op_names = self.compute_possible_actions(trace_objects, action_model, eff_pos,
                                                                                 eff_neg, next_positive_literals,
                                                                                 next_negative_literals)
        for obj_name, obj_type in trace_objects.items():
            if obj_type in self.action_model.types_hierarchy.keys():
                print(f'Warning: cannot observe the subtype of object {obj_name}. Supertype observed: {obj_type}')

                for possible_subtype in [t for t in self.action_model.types_hierarchy[obj_type] if t != obj_type]:
                    relevant_operators = [o.operator_name for o in self.action_model.operators
                                          if possible_subtype in o.parameters.values()]
                    trace_objects[obj_name] = possible_subtype

                    if len(relevant_operators) > 0:
                        # new_ground_actions, new_all_ground_op = self.compute_possible_actions(trace_objects, action_model, prev_obs, next_obs)
                        new_ground_actions, new_all_ground_op = self.compute_possible_actions(trace_objects, action_model, eff_pos,
                                                                                              eff_neg, next_positive_literals,
                                                                                              next_negative_literals)
                        ground_action_names |= new_ground_actions
                        all_ground_op_names |= new_all_ground_op

        return ground_action_names, all_ground_op_names


    def get_possible_actions(self, trace, prev_obs, next_obs, action_model, action_idx):

        trace_objects = trace.objects
        possible_action_names, all_ground_op_names = self.get_possible_action_names(trace_objects, prev_obs, next_obs, action_model)

        print(f'[Info] Grounding {len(possible_action_names)} possible actions.')
        existing_ground_actions = []
        if len(possible_action_names) > 0:
            existing_ground_actions = [a for op in action_model.operators
                                       for a in action_model.ground_actions[op.operator_name]
                                       if str(a) in possible_action_names]

        missing_ground_actions_names = possible_action_names - action_model.ground_action_labels
        existing_ground_action_names = possible_action_names - missing_ground_actions_names

        print(f'[Debug] Missing {len(missing_ground_actions_names)} actions.')
        print(f'[Debug] Existing {len(existing_ground_action_names)} actions.')
        missing_ground_actions = self.instantiate_actions_model(missing_ground_actions_names)
        possible_actions = existing_ground_actions + missing_ground_actions
        trace.actions[action_idx] = possible_actions

        all_ground_ops = {o for o in action_model.operators
                          if str(o).split('(')[0] in all_ground_op_names}
        trace.all_ground_op[action_idx] = all_ground_ops

        return trace.actions[action_idx], trace.all_ground_op[action_idx]


    def fill_trace(self, action_model, trace):

        for i in range(len(trace.observations) - 1):

            prev_observation = trace.observations[i]
            prev_action = trace.actions[i]
            next_observation = trace.observations[i + 1]

            if type(prev_action) == Action:
                self.complete_trace_with_effects_certain(prev_action, next_observation)
                self.complete_trace_with_precs_certain(action_model, prev_action, prev_observation)
                self.complete_forward_inertia(prev_observation, prev_action, next_observation)
                self.complete_backward_inertia(prev_observation, prev_action, next_observation)

            elif prev_action is not None and len(prev_action) > 0:

                possible_actions = trace.actions[i]
                all_ground_op = trace.all_ground_op[i]

                if (len(possible_actions) + len(all_ground_op)) == 1:
                    if len(possible_actions) > 0:
                        trace.actions[i] = possible_actions[0]  # Use already existing action reference

                #####################################
                ########## FORWARD INERTIA ##########
                #####################################

                prev_positive_literals = set.union(*prev_observation.positive_literals.values())
                prev_negative_literals = set.union(*prev_observation.negative_literals.values())
                next_positive_literals = set.union(*next_observation.positive_literals.values())
                next_negative_literals = set.union(*next_observation.negative_literals.values())
                forward_inertia_positives = prev_positive_literals - next_positive_literals
                forward_inertia_negatives = {l[4:] for l in prev_negative_literals} - {l[4:] for l in next_negative_literals}

                possible_eff_neg = {l for action in possible_actions for l in action.eff_neg_uncert | action.eff_neg_cert}
                possible_eff_pos = {l for action in possible_actions for l in action.eff_pos_uncert | action.eff_pos_cert}

                forward_inertia_positives = forward_inertia_positives - possible_eff_neg
                forward_inertia_negatives = forward_inertia_negatives - possible_eff_pos

                # Remove from forward inertia all predicates in the operator effects that admit all possible groundings
                possible_eff_neg_lift = {l.split('(')[0] for op in all_ground_op for l in op.eff_neg_uncert | op.eff_neg_cert}
                possible_eff_pos_lift = {l.split('(')[0] for op in all_ground_op for l in op.eff_pos_uncert | op.eff_pos_cert}

                forward_inertia_positives = {l for l in forward_inertia_positives if l.split('(')[0] not in possible_eff_neg_lift}
                forward_inertia_negatives = {l for l in forward_inertia_negatives if l.split('(')[0] not in possible_eff_pos_lift}

                # Augment next observation with forward inertia positive literals
                [next_observation.add_positive(l) for l in forward_inertia_positives]
                # Augment next observation with forward inertia negative literals
                [next_observation.add_negative(f"not_{l}") for l in forward_inertia_negatives]

                #####################################
                ########## BACKWARD INERTIA #########
                #####################################

                backward_inertia_positives = next_positive_literals - prev_positive_literals
                backward_inertia_negatives = {l[4:] for l in next_negative_literals} - {l[4:] for l in prev_negative_literals}

                backward_inertia_positives = backward_inertia_positives - possible_eff_pos
                backward_inertia_negatives = backward_inertia_negatives - possible_eff_neg

                # Remove from backward inertia all predicates in the operator effects that admit all possible groundings
                backward_inertia_positives = {l for l in backward_inertia_positives if l.split('(')[0] not in possible_eff_pos_lift}
                backward_inertia_negatives = {l for l in backward_inertia_negatives if l.split('(')[0] not in possible_eff_neg_lift}

                # Augment previous observation with backward inertia literals
                [prev_observation.add_positive(l) for l in backward_inertia_positives]
                [prev_observation.add_negative(f"not_{l}") for l in backward_inertia_negatives]

        return trace

    def complete_trace_with_precs_certain(self, action_model, prev_action, prev_observation):

        operator = next((o for o in action_model.operators if o.operator_name == prev_action.operator_name), None)

        if len(operator.precs_cert) > 0:
            for prec_cert in operator.precs_cert:
                prec_params = [e for e in prec_cert.split('(')[1][:-1].split(',') if e.strip() != '']
                action_prec = prec_cert

                for param in prec_params:
                    obj_idx = int(param.replace('?param_', '')) - 1
                    action_obj = prev_action.parameters[obj_idx]
                    action_prec = action_prec.replace(f'{param},', f'{action_obj},')
                    action_prec = action_prec.replace(f'{param})', f'{action_obj})')

                if action_prec not in prev_observation:
                    prev_observation.add_positive(action_prec)

    def get_common_eff_pos(self, possible_actions, effects_pos, action_model):

        common_lifted_effects_pos = set()

        operator = next((o for o in action_model.operators if o.operator_name == possible_actions[0].operator_name), None)

        for j in range(len(possible_actions)):

            action_lifted_effects_pos = set()

            for effect_pos in effects_pos:

                lifted_effect_pos = self.lift_ground_atoms(possible_actions[j], [effect_pos])

                # Check if the positive effect is valid, i.e., if it involves only possible action objects
                if len(lifted_effect_pos) == 0:
                    effects_pos.remove(effect_pos)

                # Ensure that the lifted positive effects are not generated by an already known certain positive effect
                if not set(operator.eff_pos_cert).intersection(lifted_effect_pos):

                    # Do not consider (possibly ambiguous) lifted positive effects that have already been removed from
                    # the list of uncertain positive effects.
                    # e.g. if the action a(c1,c2,c2) is executed, and p(c1,c2) became true in the destination state,
                    # then there are two ambiguous positive effects p(param1, param2) and p(param1, param3). However,
                    # if previously p(param1, param3) has already been removed from the list of certain and uncertain
                    # positive effects, then there is no more ambiguity, and p(param1, param2) can be added to the list
                    # of certain positive effects.
                    lifted_effect_pos = [e for e in lifted_effect_pos if e in operator.eff_pos_uncert]

                    if len(lifted_effect_pos) == 1:
                        action_lifted_effects_pos.add(lifted_effect_pos[0])

            if j == 0:
                common_lifted_effects_pos = action_lifted_effects_pos
            else:
                common_lifted_effects_pos = common_lifted_effects_pos.intersection(action_lifted_effects_pos)

        return common_lifted_effects_pos

    def get_common_eff_neg(self, possible_actions, effects_neg, action_model):

        common_lifted_effects_neg = set()

        operator = next((o for o in action_model.operators if o.operator_name == possible_actions[0].operator_name), None)

        for j in range(len(possible_actions)):

            action_lifted_effects_neg = set()

            for effect_neg in effects_neg:

                lifted_effect_neg = self.lift_ground_atoms(possible_actions[j], [effect_neg])

                # Check if the negative effect is valid, i.e., if it involves only action objects
                if len(lifted_effect_neg) == 0:
                    effects_neg.remove(effect_neg)

                # Ensure that the lifted positive effects are not generated by an already known certain positive effect
                if not set(operator.eff_neg_cert).intersection(lifted_effect_neg):

                    # Do not consider (possibly ambiguous) lifted positive effects that have already been removed from
                    # the list of uncertain positive effects.
                    # e.g. if the action a(c1,c2,c2) is executed, and p(c1,c2) became true in the destination state,
                    # then there are two ambiguous positive effects p(param1, param2) and p(param1, param3). However,
                    # if previously p(param1, param3) has already been removed from the list of certain and uncertain
                    # positive effects, then there is no more ambiguity, and p(param1, param2) can be added to the list
                    # of certain positive effects.
                    lifted_effect_neg = [e for e in lifted_effect_neg if e in operator.eff_neg_uncert]

                    if len(lifted_effect_neg) == 1:
                        action_lifted_effects_neg.add(lifted_effect_neg[0])

            if j == 0:
                common_lifted_effects_neg = action_lifted_effects_neg
            else:
                common_lifted_effects_neg = common_lifted_effects_neg.intersection(action_lifted_effects_neg)

        return common_lifted_effects_neg

    def get_common_impossible_eff_pos(self, actions, impossible_eff_pos, action_model):

        common_impossible_eff_pos = set()

        operator = next((o for o in action_model.operators if o.operator_name == actions[0].operator_name), None)

        impossible_eff_pos_objs = [set([o for o in l.split('(')[1][:-1].split(',') if o != ''])
                                   for l in impossible_eff_pos]

        for j in range(len(actions)):
            action = actions[j]
            action_params = set(action.parameters)
            action_impossible_eff_pos = [l for l, l_objs in zip(impossible_eff_pos, impossible_eff_pos_objs)
                                         if l_objs.issubset(action_params)]

            if j == 0:
                common_impossible_eff_pos = set([e for e in self.lift_ground_atoms(action, action_impossible_eff_pos)
                                                    if e in operator.eff_pos_uncert])
            else:
                common_impossible_eff_pos = set([e for e in self.lift_ground_atoms(action, action_impossible_eff_pos)
                                                    if e in operator.eff_pos_uncert and e in common_impossible_eff_pos])

            if len(common_impossible_eff_pos) == 0:
                break

        return common_impossible_eff_pos

    def get_common_impossible_eff_neg(self, actions, impossible_eff_neg, action_model):

        common_impossible_eff_neg = set()

        operator = next((o for o in action_model.operators if o.operator_name == actions[0].operator_name), None)

        impossible_eff_neg_objs = [set([o for o in l.split('(')[1][:-1].split(',') if o != ''])
                                   for l in impossible_eff_neg]
        for j in range(len(actions)):
            action = actions[j]
            action_params = set(action.parameters)
            action_impossible_eff_neg = [l for l, l_objs in zip(impossible_eff_neg, impossible_eff_neg_objs)
                                         if l_objs.issubset(action_params)
                                         and l not in action.eff_pos_cert | action.eff_pos_uncert]

            if j == 0:
                common_impossible_eff_neg = set([e for e in self.lift_ground_atoms(action, action_impossible_eff_neg)
                                                    if e in operator.eff_neg_uncert])
            else:
                common_impossible_eff_neg = set([e for e in self.lift_ground_atoms(action, action_impossible_eff_neg)
                                                    if e in operator.eff_neg_uncert and e in common_impossible_eff_neg])

            if len(common_impossible_eff_neg) == 0:
                break

        return common_impossible_eff_neg


    def remove_uncertain_preconditions(self, action_model, prev_observation, next_action):

        fake_preconditions = [l for l in next_action.precs_uncert if f"not_{l}" in prev_observation.negative_literals[l.split('(')[0]]]

        operator = next((o for o in action_model.operators if o.operator_name == next_action.operator_name), None)

        for prec in fake_preconditions:

            lifted_precs = self.lift_ground_atoms(next_action, [prec])

            for lifted_prec in lifted_precs:

                if lifted_prec in operator.precs_uncert:

                    # Remove lifted precondition from operator uncertain preconditions
                    operator.remove_prec_uncert(lifted_prec)

                    # Remove grounded preconditions from all ground actions computed by instantiating the operator
                    ground_actions = action_model.ground_actions[operator.operator_name]
                    for a in ground_actions:
                        ground_prec = self.ground_lifted_atom(a.params_bind, lifted_prec)
                        a.remove_prec_uncert(ground_prec)


    def remove_uncertain_eff_pos(self, action_model, previous_action, next_observation):

        impossible_eff_pos = [l for l in previous_action.eff_pos_uncert if f"not_{l}" in next_observation.negative_literals[l.split('(')[0]]]

        operator = next((o for o in action_model.operators if o.operator_name == previous_action.operator_name), None)

        for eff_pos in impossible_eff_pos:

            lifted_effs_pos = self.lift_ground_atoms(previous_action, [eff_pos])

            # If the size of 'lifted_effs_pos' is greater than one, then there is an ambiguous objects binding,
            # but the ambiguity is not problematic as in the following example
            # e.g.
            # uncertain positive effect p(param1, param2)
            # hypotetic positive effect p(param1, param3)
            # If the action a(c1,c2,c2) is executed AND p(c1,c2) is false in the destination state, then
            # both p(param1, param2) and p(param1, param3) are not positive effects.
            # Suppose that p(param1, param3) is a positive effect, then we have the contradiction that p(c1,c2) is
            # true in the destination state (even if there is a negative effect p(param1, param2) since, in case of
            # inconsistent effects, we give priority to the positive effect). Similarly for p(param1, param2).
            for lifted_eff_pos in lifted_effs_pos:

                if lifted_eff_pos in operator.eff_pos_uncert:

                    # Remove lifted precondition from operator uncertain preconditions
                    operator.remove_eff_pos_uncert(lifted_eff_pos)

                    # Remove grounded positive effects from all ground actions computed by instantiating the operator
                    ground_actions = action_model.ground_actions[operator.operator_name]
                    for a in ground_actions:
                        ground_eff_pos = self.ground_lifted_atom(a.params_bind, lifted_eff_pos)

                        # Check if the lifted effect is ambiguous, i.e., if the ground_eff_pos corresponds to multiple
                        # lifted effects in the set of possible positive effect of the operator.
                        # e.g. if you remove the uncertain positive effect p(c1,c2) with an action a(c1,c2), then you
                        # are removing p(x1,x2), but there may be another ground action p(c1,c1) and in this case you
                        # would remove p(x1,x2), p(x1,x1), p(x2,x2), p(x2,x1), which may be unsafe if inconsistent
                        # effects can be generated (as in some IPC benchmarks, e.g. rovers).
                        if len(set(self.lift_ground_atoms(a, [ground_eff_pos])) & operator.eff_pos_uncert) == 0:
                            a.remove_eff_pos_uncert(ground_eff_pos)


    def remove_uncertain_eff_neg(self, action_model, previous_action, next_observation):

        impossible_eff_neg = [l for l in previous_action.eff_neg_uncert if l in next_observation.positive_literals[l.split('(')[0]]]

        operator = next((o for o in action_model.operators if o.operator_name == previous_action.operator_name), None)

        for eff_neg in impossible_eff_neg:

            lifted_effs_neg = self.lift_ground_atoms(previous_action, [eff_neg])

            # If the size of 'lifted_effs_neg' is greater than one, then there is an ambiguous objects binding,
            # but the ambiguity can be resolved if the effect is not a (certain and uncertain) positive one.
            # e.g.
            # uncertain negative effect = p(param1, param2)
            # (either certain or uncertain) positive effect = p(param1, param3)
            # If the action a(c1,c2,c2) is executed, then p(c1,c2) is true in the destination state, but the
            # uncertain negative effect p(param1, param2) cannot be removed, indeed it could be a real one.
            # However, if p(c1,c2) is false in the destination state AND p(c1,c2) is not a (certain and uncertain)
            # positive effect, then the uncertain negative effect p(c1,c2) can be safely removed, even if the size
            # of 'lifted_effs_neg' is greater than one, i.e. if there is an ambiguous binding of action objects.
            if len(lifted_effs_neg) == 1 or eff_neg not in previous_action.eff_pos_cert | previous_action.eff_pos_uncert:
                for lifted_eff_neg in lifted_effs_neg:

                    if lifted_eff_neg in operator.eff_neg_uncert:

                        # Remove lifted precondition from operator uncertain preconditions
                        operator.remove_eff_neg_uncert(lifted_eff_neg)

                        # Remove grounded negative effects from all ground actions derived by instantiating the operator
                        ground_actions = action_model.ground_actions[operator.operator_name]
                        for a in ground_actions:
                            ground_eff_neg = self.ground_lifted_atom(a.params_bind, lifted_eff_neg)

                            if len(set(self.lift_ground_atoms(a, [ground_eff_neg])) & operator.eff_neg_uncert) == 0:
                                a.remove_eff_neg_uncert(ground_eff_neg)

    def complete_forward_inertia(self, prev_observation, prev_action, next_observation):

        # Add forward inertia literals, i.e. literals either not involved in next successfully executed action objects
        # or not involved in next successfully executed action effects
        prev_action_pos = prev_action.eff_pos_uncert | prev_action.eff_pos_cert
        prev_action_neg = prev_action.eff_neg_uncert | prev_action.eff_neg_cert

        prev_obs_pos = set.union(*prev_observation.positive_literals.values())
        prev_obs_neg = set.union(*prev_observation.negative_literals.values())

        next_obs_pos = set.union(*next_observation.positive_literals.values())
        next_obs_neg = set.union(*next_observation.negative_literals.values())

        # Augment next observation with forward inertia positive literals
        [next_observation.add_positive(l) for l in prev_obs_pos - next_obs_pos - prev_action_neg]
        # Augment next observation with forward inertia negative literals
        [next_observation.add_negative(l) for l in prev_obs_neg - next_obs_neg - {f"not_{e}" for e in prev_action_pos}]


    def complete_backward_inertia(self, prev_observation, prev_action, next_observation):

        # Add backward inertia literals, i.e. literals not involved in previous successfully executed action objects
        # or effects
        prev_action_pos = prev_action.eff_pos_uncert | prev_action.eff_pos_cert
        prev_action_neg = prev_action.eff_neg_uncert | prev_action.eff_neg_cert

        prev_obs_pos = set.union(*prev_observation.positive_literals.values())
        prev_obs_neg = set.union(*prev_observation.negative_literals.values())

        next_obs_pos = set.union(*next_observation.positive_literals.values())
        next_obs_neg = set.union(*next_observation.negative_literals.values())

        # Augment previous observation with backward inertia literals
        [prev_observation.add_positive(l) for l in next_obs_pos - prev_obs_pos - prev_action_pos]
        [prev_observation.add_negative(l) for l in next_obs_neg - prev_obs_neg - {f"not_{e}" for e in prev_action_neg}]

    def learn_certain_effects(self, action_model, prev_observation, next_action, next_observation):

        # Get positive effects by comparing previous and next state observations
        effects_pos = [l for l in next_action.eff_pos_uncert
                       if l in next_observation.positive_literals[l.split('(')[0]]
                       and f"not_{l}" in prev_observation.negative_literals[l.split('(')[0]]]

        effects_neg = [l for l in next_action.eff_neg_uncert
                       if f"not_{l}" in next_observation.negative_literals[l.split('(')[0]]
                       and l in prev_observation.positive_literals[l.split('(')[0]]]

        for effect_pos in effects_pos:

            lifted_effect_pos = self.lift_ground_atoms(next_action, [effect_pos])
            operator = next((o for o in action_model.operators if o.operator_name == next_action.operator_name), None)

            # Ensure that the lifted positive effects are not generated by an already known certain positive effect
            if not operator.eff_pos_cert.intersection(lifted_effect_pos):

                # Do not consider (possibly ambiguous) lifted positive effects that have already been removed from
                # the list of uncertain positive effects.
                # e.g. if the action a(c1,c2,c2) is executed, and p(c1,c2) became true in the destination state,
                # then there are two ambiguous positive effects p(param1, param2) and p(param1, param3). However,
                # if previously p(param1, param3) has already been removed from the list of certain and uncertain
                # positive effects, then there is no more ambiguity, and p(param1, param2) can be added to the list
                # of certain positive effects.
                lifted_effect_pos = [e for e in lifted_effect_pos if e in operator.eff_pos_uncert]

                if len(lifted_effect_pos) == 1:
                    effect_pos_param = lifted_effect_pos[0]
                    operator.add_eff_pos_cert(effect_pos_param)

                    # Add grounded certain positive effect on all ground actions compute by instantiating the operator
                    ground_actions = action_model.ground_actions[operator.operator_name]
                    for a in ground_actions:
                        ground_eff_pos = self.ground_lifted_atom(a.params_bind, effect_pos_param)
                        a.add_eff_pos_cert(ground_eff_pos)

        for effect_neg in effects_neg:

            lifted_effect_neg = self.lift_ground_atoms(next_action, [effect_neg])
            operator = next((o for o in action_model.operators if o.operator_name == next_action.operator_name), None)

            # Ensure that the lifted negative effects are not generated by an already known certain negative effect
            if not operator.eff_neg_cert.intersection(lifted_effect_neg):

                # Do not consider (possibly ambiguous) lifted negative effects that have already been removed from
                # the list of uncertain negative effects.
                # e.g. if the action a(c1,c2,c2) is executed, and p(c1,c2) became false in the destination state,
                # then there are two ambiguous negative effects p(param1, param2) and p(param1, param3). However,
                # if previously p(param1, param3) has already been removed from the list of certain and uncertain
                # negative effects, then there is no more ambiguity, and p(param1, param2) can be added to the list
                # of certain negative effects.
                lifted_effect_neg = [e for e in lifted_effect_neg if e in operator.eff_neg_uncert]

                if len(lifted_effect_neg) == 1:
                    effect_neg_param = lifted_effect_neg[0]
                    operator.add_eff_neg_cert(effect_neg_param)

                    # Add grounded certain negative effect on all ground actions computed by instantiating the operator
                    ground_actions = action_model.ground_actions[operator.operator_name]
                    for a in ground_actions:
                        ground_eff_neg = self.ground_lifted_atom(a.params_bind, effect_neg_param)
                        a.add_eff_neg_cert(ground_eff_neg)

    def complete_trace_with_effects_certain(self, prev_action, next_observation):

        for ground_eff_pos in prev_action.eff_pos_cert:

            if ground_eff_pos not in next_observation:

                # Check inconsistent effects due to ambiguous parameters binding, e.g. in the hanoi domain,
                # action move(disk1,peg1,peg1) generates the inconsistent literals not_on(disk1, peg1)
                # and on(disk1, peg1), since peg1 is both the second and third parameter. If such an action
                # can be executed, then we solve the inconsistency by keeping only the positive effect.
                if f'not_{ground_eff_pos}' in next_observation:
                    next_observation.remove_negative(f'not_{ground_eff_pos}')

                next_observation.add_positive(ground_eff_pos)

        for ground_eff_neg in prev_action.eff_neg_cert:
            if f"not_{ground_eff_neg}" not in next_observation:
                # Check inconsistent effects due to ambiguous parameters binding, e.g. in the hanoi domain,
                # action move(disk1,peg1,peg1) generates the inconsistent literals not_on(disk1, peg1)
                # and on(disk1, peg1), since peg1 is both the second and third parameter. If such an action
                # can be executed, then we solve the inconsistency by keeping only the positive effect.
                if ground_eff_neg not in next_observation:
                    next_observation.add_negative(f"not_{ground_eff_neg}")

    def eval_log(self, evaluated_model=None):
        """
        Evaluate metrics and print them in log file
        :return: None
        """

        assert evaluated_model is not None

        real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
        real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
        ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
        eff_pos_precision, eff_neg_precision, overall_recall, overall_precision = metrics.action_model_statistics(evaluated_model)


        evaluate = {'Iter': self.iter,
                    'Real_precs': int(real_precs_size),
                    'Learn_precs': int(learned_precs_size),
                    'Real_pos': int(real_eff_pos_size),
                    'Learn_pos': int(learned_eff_pos_size),
                    'Real_neg': int(real_eff_neg_size),
                    'Learn_neg': int(learned_eff_neg_size),
                    'Ins_pre': int(ins_pre),
                    'Del_pre': int(del_pre),
                    'Ins_pos': int(ins_eff_pos),
                    'Del_pos': int(del_eff_pos),
                    'Ins_neg': int(ins_eff_neg),
                    'Del_neg': int(del_eff_neg),
                    'Precs_recall': "{0:.2f}".format(precs_recall),
                    'Pos_recall': "{0:.2f}".format(eff_pos_recall),
                    'Neg_recall': "{0:.2f}".format(eff_neg_recall),
                    'Precs_precision': "{0:.2f}".format(precs_precision),
                    'Pos_precision': "{0:.2f}".format(eff_pos_precision),
                    'Neg_precision': "{0:.2f}".format(eff_neg_precision),
                    'Tot_recall': "{0:.2f}".format(overall_recall),
                    'Tot_precision': "{0:.2f}".format(overall_precision)
                    }

        self.eval = pd.concat([self.eval, pd.DataFrame([evaluate])], ignore_index=True)

        print("\n")
        print(template.format(  # header
            Iter="Iter", Real_precs="Real_precs", Learn_precs="Learn_precs",
            Real_pos="Real_pos", Learn_pos="Learn_pos", Real_neg="Real_neg", Learn_neg="Learn_neg",
            Ins_pre="Ins_pre", Del_pre="Del_pre", Ins_pos="Ins_pos", Del_pos="Del_pos",
            Ins_neg="Ins_neg", Del_neg="Del_neg", Precs_recall="Precs_recall",
            Pos_recall="Pos_recall", Neg_recall="Neg_recall", Precs_precision="Precs_precision",
            Pos_precision="Pos_precision", Neg_precision="Neg_precision",
            Tot_recall="Tot_recall", Tot_precision="Tot_precision"
        ))
        print(template.format(**evaluate))
        print("\n")

import copy
import itertools
import os
import re
import warnings
from collections import defaultdict

from OffLAM.Operator import Operator


class ActionModel:

    def __init__(self, input_file=None):
        self.input_file = input_file
        self.types_hierarchy = None
        self.operators = None
        self.predicates = None
        self.ground_actions = defaultdict(list)
        self.ground_action_labels = set()
        self.constants = None

        if input_file is not None:
            self.clean_pddl_domain_file(input_file)
            self.read(f'{input_file}_clean')
            self.fill_empty_uncertain()
            os.remove(f'{input_file}_clean')

    def __str__(self):
        return "\n\n".join(self.operators)

    def read(self, f_name):
        self.types_hierarchy = self.read_object_types_hierarchy(f_name)
        self.constants = self.read_constants(f_name)
        self.operators = self.read_operators(f_name)
        self.predicates = self.read_predicates(f_name)

    def empty(self):
        for o in self.operators:
            o.precs_cert = []
            o.precs_uncert = []
            o.eff_pos_cert = []
            o.eff_pos_uncert = []
            o.eff_neg_cert = []
            o.eff_neg_uncert = []

        if self.ground_actions is not None:
            for op_name, op_actions in self.ground_actions.items():
                for a in op_actions:
                    a.precs_cert = []
                    a.precs_uncert = []
                    a.eff_pos_cert = []
                    a.eff_pos_uncert = []
                    a.eff_neg_cert = []
                    a.eff_neg_uncert = []

    def fill_empty_uncertain(self):

        for operator in self.operators:
            if operator.precs_uncert is None:
                print(f'[Info] Initializing uncertain preconditions of {operator.operator_name} to the maximal superset of preconditions.')

                preconditions_superset = self.get_op_relevant_predicates(operator)

                # Format preconditions syntax
                precs_uncert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in preconditions_superset}
                operator.precs_uncert = precs_uncert

            if operator.eff_pos_uncert is None:
                print(f'[Info] Initializing uncertain positive effects of {operator.operator_name} to the maximal superset of possible effects.')

                eff_pos_superset = self.get_op_relevant_predicates(operator)

                # Format positive effects syntax
                eff_pos_superset = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                    if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_pos_superset}
                operator.eff_pos_uncert = eff_pos_superset

            if operator.eff_neg_uncert is None:
                print(f'[Info] Initializing uncertain negative effects of {operator.operator_name} to the maximal superset of possible effects.')

                eff_neg_superset = self.get_op_relevant_predicates(operator)

                # Format negative effects syntax
                eff_neg_superset = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                    if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_neg_superset}
                operator.eff_neg_uncert = eff_neg_superset

    def read_constants(self, f_name):
        with open(f_name, 'r') as f:
            data = f.read().split("\n")

            objects_row = [el.replace(")","").strip()
                           for el in re.findall(r":constants.*\(:predicates","++".join(data))[0].replace(":constants","").replace("(:predicates", "").split("++")
                           if el.strip() != ""]

            objects = defaultdict(list)
            obj_of_same_type = []

            for row in objects_row:
                row = row.replace("(", "").replace(")", "")
                if row.find("- ") != -1:
                    # [objects['objects'].append(el) for el in row.strip().split("- ")[0].split()]
                    # [objects['objects'].append(el) for el in row.strip().split("- ")[1].split()]
                    objects[row.strip().split("- ")[1].strip()].extend([el.strip()
                                                                        for el in row.strip().split("- ")[0].strip().split()]
                                                                       + obj_of_same_type
                                                                       + [row.strip().split("- ")[1].strip()])
                    obj_of_same_type = []
                else:
                    # [objects['objects'].append(el) for el in row.split()]
                    [obj_of_same_type.append(el) for el in row.split()]

            for object_key, object_values in objects.items():
                if object_key != 'objects':

                    for val in object_values:

                        for key in objects.keys():
                            if val == key:
                                objects[object_key] = [el for el in objects[object_key] + objects[val]
                                                       if el != object_key]

            for key in objects.keys():
                objects[key] = list(set(objects[key]))

        return objects

    def read_object_types_hierarchy(self, f_name):
        with open(f_name, 'r') as f:
            data = f.read().split("\n")

            objects_row = [el.replace(")","").strip()
                           for el in re.findall(r":types.*\(:predicates","++".join(data))[0].replace(":types","").replace("(:predicates", "").split("++")
                           if el.strip() != ""]
            if "(:constants" in '\n'.join(data):
                objects_row = [el.replace(")", "").strip()
                               for el in re.findall(r":types.*\(:constants", "++".join(data))[0]
                               .replace(":types", "").replace("(:constants", "").split("++")
                               if el.strip() != ""]

            objects = defaultdict(list)
            obj_of_same_type = []

            for row in objects_row:
                row = row.replace("(", "").replace(")", "")
                if row.find("- ") != -1:
                    [objects['objects'].append(el) for el in row.strip().split("- ")[0].split()]
                    [objects['objects'].append(el) for el in row.strip().split("- ")[1].split()]
                    objects[row.strip().split("- ")[1].strip()].extend([el.strip()
                                                                        for el in row.strip().split("- ")[0].strip().split()]
                                                                       + obj_of_same_type
                                                                       + [row.strip().split("- ")[1].strip()])
                    obj_of_same_type = []
                else:
                    [objects['objects'].append(el) for el in row.split()]
                    [obj_of_same_type.append(el) for el in row.split()]

            for object_key, object_values in objects.items():
                if object_key != 'objects':

                    for val in object_values:

                        for key in objects.keys():
                            if val == key:
                                objects[object_key] = [el for el in objects[object_key] + objects[val]]

            for key in objects.keys():
                objects[key] = list(set(objects[key]))

        return objects

    def read_operators(self, f_name):

        operators = dict()

        with open(f_name, "r") as f:
            data = [el.strip().lower() for el in f.read().split("\n")]
            all_action_schema = " ".join(data)[" ".join(data).index(":action"):]

            # Read certain operator preconditions and effects
            operators_cert = [o.strip().lower() for o in re.findall(r"action(.*?) :parameters", all_action_schema)
                              if not o.strip().lower().endswith('-uncert')]
            for operator_name in operators_cert:

                # Read operator parameters
                action_schema = re.findall(r":action {}(.*?)(?:action|$)".format(operator_name), all_action_schema)[0]
                op_params_row = re.findall(r":parameters(.*?):precondition", action_schema)[0].strip()[1:-1]
                params = [p.strip() for p in op_params_row.split() if p.strip() != '-']
                op_params = dict()

                params_of_type = []
                for el in params:
                    if '?' in el:
                        params_of_type.append(el)
                    else:
                        for p in params_of_type:
                            op_params[p] = el
                            params_of_type = []

                # Read operator certain preconditions
                op_precs_row = re.findall(r":precondition(.*?):effect", action_schema)[0].strip()[1:-1]
                precs_cert = {p.strip() for p in re.findall(r"\([^()]*\)", op_precs_row)
                            if not len(p.replace('(and', '').replace(')', '').strip()) == 0}

                # Read operator certain effects
                op_effects_row = re.findall(r":effect(.*?)(?:action|$)", action_schema)[0]
                eff_neg_cert = {e.strip()[1:-1].replace('not', '', 1).strip() for e in re.findall(r"\(not[^)]*\)\)", op_effects_row)
                                  if not len(e.replace('(and', '').replace(')', '').strip()) == 0}
                eff_pos_cert = {e.strip() for e in re.findall(r"\([^()]*\)", op_effects_row)
                                  if e not in eff_neg_cert and not len(e.replace('(and', '').replace(')', '').replace('(', '').strip()) == 0}

                # Format preconditions and effects syntax
                precs_cert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                         if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in precs_cert}
                eff_neg_cert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                  if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_neg_cert}
                eff_pos_cert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                  if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_pos_cert}

                operators[operator_name] = Operator(operator_name, op_params, precs_cert=precs_cert,
                                                    eff_pos_cert=eff_pos_cert, eff_neg_cert=eff_neg_cert)

            # Read uncertain operator preconditions and effects
            operators_uncert = [o.strip().lower() for o in re.findall(r"action(.*?) :parameters", all_action_schema)
                                if o.strip().lower().endswith('-uncert')]
            for operator_name in operators_uncert:

                # Read operator parameters
                action_schema = re.findall(r":action {}(.*?)(?:action|$)".format(operator_name), all_action_schema)[0]
                op_params_row = re.findall(r":parameters(.*?):precondition", action_schema)[0].strip()[1:-1]
                params = [p.strip() for p in op_params_row.split() if p.strip() != '-']
                op_params = dict()

                params_of_type = []
                for el in params:
                    if '?' in el:
                        params_of_type.append(el)
                    else:
                        for p in params_of_type:
                            op_params[p] = el
                            params_of_type = []

                # Read operator uncertain preconditions
                op_precs_row = re.findall(r":precondition(.*?):effect", action_schema)[0].strip()[1:-1]
                precs_uncert = {p.strip() for p in re.findall(r"\([^()]*\)", op_precs_row)
                            if not len(p.replace('(and', '').replace(')', '').strip()) == 0}

                # Read operator uncertain effects
                op_effects_row = re.findall(r":effect(.*?)(?:action|$)", action_schema)[0]
                eff_neg_uncert = {e.strip()[1:-1].replace('not', '', 1).strip() for e in re.findall(r"\(not[^)]*\)\)", op_effects_row)
                                  if not len(e.replace('(and', '').replace(')', '').strip()) == 0}
                eff_pos_uncert = {e.strip() for e in re.findall(r"\([^()]*\)", op_effects_row)
                                  if e not in eff_neg_cert and not len(e.replace('(and', '').replace(')', '').strip()) == 0}
                filtered_eff_pos_uncert = copy.deepcopy(eff_pos_uncert)
                for e in eff_neg_uncert:
                    if e in eff_pos_uncert:
                        filtered_eff_pos_uncert.remove(e)  # Remove only one instance
                eff_pos_uncert = filtered_eff_pos_uncert

                # Format preconditions and effects syntax
                precs_uncert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                         if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in precs_uncert}
                eff_neg_uncert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                  if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_neg_uncert}
                eff_pos_uncert = {f"{p[1:-1].split()[0]}({','.join(p[1:-1].split()[1:])})"
                                  if len(p[1:-1].split()) > 1 else f"{p[1:-1].split()[0]}()" for p in eff_pos_uncert}

                operators[operator_name.replace('-uncert', '')].precs_uncert = precs_uncert
                operators[operator_name.replace('-uncert', '')].eff_neg_uncert = eff_neg_uncert
                operators[operator_name.replace('-uncert', '')].eff_pos_uncert = eff_pos_uncert

        return list(operators.values())

    def read_predicates(self, f_name):

        with open(f_name, "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            predicates_row = re.findall(r":predicates(.*?):action", " ".join(data))[0]
            predicates_row = [p.strip() for p in re.findall(r"\([^()]*\)", predicates_row)]

            predicates = []
            for p in predicates_row:
                p_name = p[1:-1].split()[0].strip().lower()

                if len(p[1:-1].split()) > 1:
                    p_objs_num = 0
                    p_objs_types = []
                    for s in p[1:-1].split()[1:]:
                        if '?' in s:
                            p_objs_num += 1
                        elif s.strip() != '-':
                            p_objs_types.extend([s]*p_objs_num)
                            p_objs_num = 0
                else:
                    p_objs_types = []

                predicates.append(f"{p_name}({','.join(p_objs_types)})")

        return predicates

    def get_op_relevant_predicates(self, operator):

        op_params = " ".join([f"{k} - {v}" for k, v in operator.parameters.items()])

        obj_type_hierarchy = self.types_hierarchy

        # Get op param types
        single_obj_count = 0
        op_param_types = []
        op_param_supertypes = []
        for el in [el for el in op_params.strip().split() if el.strip() != "-"]:
            if el.startswith("?"):
                single_obj_count += 1
            else:
                [op_param_types.append([el]) if el not in obj_type_hierarchy.keys() else
                 op_param_types.append(obj_type_hierarchy[el])
                 for _ in range(single_obj_count)]

                [op_param_supertypes.append(el) for _ in range(single_obj_count)]
                single_obj_count = 0

        # Get all predicates
        with open(self.input_file, "r") as f:
            data = [el.strip() for el in f.read().split("\n")]
            preds = re.findall(r":predicates.+?:action","".join(data))[0]

        all_predicates = sorted(re.findall(r"\([^()]*\)", preds))

        relevant_predicates = []

        for predicate in all_predicates:

            pred_name = predicate.strip()[1:-1].split()[0]

            # Get predicate parameter types
            single_obj_count = 0
            pred_param_types = []
            pred_param_supertypes = []
            for el in [el for el in predicate.strip()[1:-1].strip().split()[1:] if el.strip() != "-"]:
                if el.startswith("?"):
                    single_obj_count += 1
                else:
                    [pred_param_types.append([el]) if el not in obj_type_hierarchy.keys()
                     else pred_param_types.append(obj_type_hierarchy[el])
                     for _ in range(single_obj_count)]

                    [pred_param_supertypes.append(el) for _ in range(single_obj_count)]

                    single_obj_count = 0

            # Check if predicate object types are contained into operator object types
            if all([any([el in [item for sublist in op_param_types for item in sublist]]
                        for el in pred_param_types[i]) for i in range(len(pred_param_types))]):

                all_pred_type_indices = []
                for pred_type in pred_param_types:
                    pred_type_indices = ["?param_{}".format(i+1)
                                         for i, op_pred_type in enumerate(op_param_types)
                                         if len([el for el in pred_type if el in op_pred_type]) > 0]
                                         # if op_pred_type == pred_type]
                    all_pred_type_indices.append(pred_type_indices)

                param_combinations = [list(p) for p in itertools.product(*all_pred_type_indices)]

                # Remove inconsistent combinations according to predicate input types
                param_comb_inconsistent = []
                for comb in param_combinations:

                    comb_param_types = []
                    for param in comb:
                        comb_param_types.append(op_param_supertypes[int(param.split("_")[1]) - 1])

                    for k, op_param_type in enumerate(comb_param_types):

                        if not ((pred_param_supertypes[k] in obj_type_hierarchy.keys() \
                                 and op_param_type in obj_type_hierarchy[pred_param_supertypes[k]]) \
                                or op_param_type == pred_param_supertypes[k]):

                            param_comb_inconsistent.append(comb)

                            break

                # Remove inconsistent combinations
                [param_combinations.remove(comb) for comb in param_comb_inconsistent]

                if len(all_pred_type_indices) > 0:
                    relevant_predicates.extend(["({} {})".format(pred_name, " ".join(pred_comb))
                                                for pred_comb in param_combinations])
                else:
                    relevant_predicates.extend(["({})".format(pred_name)])

        return sorted(relevant_predicates)


    def write(self, f_name, precs_certain=True, eff_pos_certain=True, eff_neg_certain=True,
              precs_uncertain=True, eff_pos_uncertain=True, eff_neg_uncertain=True, add_uncertain_operators=False):
        """
        Write the pddl action model into the file 'f_name'
        :param f_name: name of the pddl action model
        :param precs_certain: When set to True, the pddl operators contain the certain preconditions
        :param eff_pos_certain: When set to True, the pddl operators contain the certain positive effects
        :param eff_neg_certain: When set to True, the pddl operators contain the certain negative effects
        :param precs_uncertain: When set to True, the pddl operators contain the uncertain preconditions
        :param eff_pos_uncertain: When set to True, the pddl operators contain the uncertain positive effects
        :param eff_neg_uncertain: When set to True, the pddl operators contain the uncertain negative effects
        :param add_uncertain_operators: When set to True, the uncertain preconditions and effects are stored in
        additional fictitious pddl operators. If 'add_uncertain_operators' is set to True, then at least one of
        'precs_uncertain', 'eff_pos_uncertain' and 'eff_neg_uncertain' should be set to True
        :return: None
        """

        if add_uncertain_operators:
            if (precs_uncertain or eff_pos_uncertain or eff_neg_uncertain) is False:
                warnings.warn(f"writing pddl action model into file {f_name} with additional fictitious "
                              f"pddl operators for uncertain preconditions and effects. However 'precs_uncertain' and "
                              f"'eff_pos_uncertain' and 'eff_neg_uncertain' are all set to False. Therefore I will not "
                              f"write additional fictitious pddl operators.")
            self.write_with_uncertain_operators(f_name, precs_certain, eff_pos_certain, eff_neg_certain,
                                                precs_uncertain, eff_pos_uncertain, eff_neg_uncertain)
        else:
            self.write_without_uncertain_operators(f_name, precs_certain, eff_pos_certain, eff_neg_certain,
                                                   precs_uncertain, eff_pos_uncertain, eff_neg_uncertain)


    def write_with_uncertain_operators(self, f_name, precs_certain=True, eff_pos_certain=True, eff_neg_certain=True,
                                       precs_uncertain=True, eff_pos_uncertain=True, eff_neg_uncertain=True):

        with open(self.input_file, 'r') as f:
            data = [el.strip() for el in f.read().split("\n")]
            domain_name = re.findall(r"domain.+?\)","".join(data))[0].strip()[:-1].split()[-1].strip()

        with open(f_name, 'w') as f:

            # Write domain name and requirements
            f.write(f"(define (domain {domain_name})"
                    f"\n(:requirements)")

            # Write types
            f.write("\n(:types")

            # Remove redundant supertypes, i.e. supertypes in their subtypes
            types_hierarchy = copy.deepcopy(self.types_hierarchy)
            types_hierarchy = {k: [t for t in v if t != k] for k, v in types_hierarchy.items()}

            # Remove redundant subtypes, e.g. if a = super(b) and b = super(c),
            # remove a from c explicit subtypes since 'a' is implicitly subtype of 'c'
            for supertype, subtypes in types_hierarchy.items():
                for k, v in types_hierarchy.items():
                    if k != supertype and k != 'objects' and k in subtypes:
                        types_hierarchy[supertype] = [v for v in types_hierarchy[supertype] if v not in types_hierarchy[k]]

            for supertype, subtypes in types_hierarchy.items():
                subtypes = [t for t in subtypes if t != supertype]
                if supertype != 'objects':
                    f.write("\n\t{} - {}".format('\n\t'.join(subtypes), supertype))
                elif len(self.types_hierarchy.keys()) == 1 and supertype == 'objects':
                    f.write("\n\t{}".format('\n\t'.join(subtypes)))
            f.write("\n)")

            # Write constants
            if len(self.constants) > 0:
                f.write("\n(:constants")
                [f.write(f"\n{' '.join(v)} - {k}") for k, v in self.constants.items()]
                f.write("\n)\n")

            # Write predicates
            f.write("\n(:predicates")
            for p in self.predicates:
                p_name = p.split('(')[0]
                p_types = [t for t in p.split('(')[1][:-1].split(',') if t.strip() != '']
                if len(p_types) > 0:
                    f.write(f"\n\t({p_name} {' '.join([f'?param_{i + 1} - {p_type}' for i, p_type in enumerate(p_types)])})")
                else:
                    f.write(f"\n\t({p_name})")
            f.write("\n)\n\n")

            # Write operators with certain preconditions and effects
            if precs_certain or eff_pos_certain or eff_neg_certain:
                for operator in self.operators:
                    f.write(f"\n(:action {operator.operator_name}")
                    f.write("\n:parameters ({})".format(" ".join(['{} - {}'.format(param, obj_type) for param, obj_type in operator.parameters.items()])))

                    # Format preconditions and effects syntax
                    precs_cert = []
                    eff_pos_cert = []
                    eff_neg_cert = []
                    if precs_certain:
                        precs_cert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                      if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0].strip())
                                      for p in operator.precs_cert]
                    if eff_pos_certain:
                        eff_pos_cert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                        if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0])
                                        for p in operator.eff_pos_cert]
                    if eff_neg_certain:
                        eff_neg_cert = ["(not ({} {}))".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                        if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "(not ({}))".format(p.split('(')[0])
                                        for p in operator.eff_neg_cert]

                    f.write("\n:precondition\t(and {}\n)".format("\n".join(precs_cert)))

                    f.write("\n:effect\t(and {}\n)".format("\n".join(eff_pos_cert + eff_neg_cert)))
                    f.write('\n)\n\n')


            # Write operators with uncertain preconditions and effects
            if precs_uncertain or eff_pos_uncertain or eff_neg_uncertain:
                for operator in self.operators:
                    f.write(f"\n(:action {operator.operator_name}-uncert")
                    f.write("\n:parameters ({})".format(" ".join(['{} - {}'.format(param, obj_type) for param, obj_type in operator.parameters.items()])))

                    precs_uncert = []
                    eff_pos_uncert = []
                    eff_neg_uncert = []
                    # Format preconditions and effects syntax
                    if precs_uncertain:
                        precs_uncert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                        if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0].strip())
                                        for p in operator.precs_uncert]
                    if eff_pos_uncertain:
                        eff_pos_uncert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                          if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0])
                                          for p in operator.eff_pos_uncert]
                    if eff_neg_uncertain:
                        eff_neg_uncert = ["(not ({} {}))".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                          if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "(not ({}))".format(p.split('(')[0])
                                          for p in operator.eff_neg_uncert]

                    f.write("\n:precondition\t(and {}\n)".format("\n".join(precs_uncert)))

                    f.write("\n:effect\t(and {}\n)".format("\n".join(eff_pos_uncert + eff_neg_uncert)))
                    f.write('\n)\n\n')

            f.write('\n\n)')


    def write_without_uncertain_operators(self, f_name, precs_certain=True, eff_pos_certain=True, eff_neg_certain=True,
                                          precs_uncertain=True, eff_pos_uncertain=True, eff_neg_uncertain=True):

        with open(self.input_file, 'r') as f:
            data = [el.strip() for el in f.read().split("\n")]
            domain_name = re.findall(r"domain.+?\)","".join(data))[0].strip()[:-1].split()[-1].strip()

        with open(f_name, 'w') as f:

            # Write domain name and requirements
            f.write(f"(define (domain {domain_name})"
                    f"\n(:requirements)")

            # Write types
            f.write("\n(:types")

            # Remove redundant supertypes, i.e. supertypes in their subtypes
            types_hierarchy = copy.deepcopy(self.types_hierarchy)
            types_hierarchy = {k: [t for t in v if t != k] for k, v in types_hierarchy.items()}

            # Remove redundant subtypes, e.g. if a = super(b) and b = super(c), remove a from c explicit subtypes
            # since 'a' is implicitly subtype of 'c'
            for supertype, subtypes in types_hierarchy.items():
                for k, v in types_hierarchy.items():
                    if k != supertype and k != 'objects' and k in subtypes:
                        types_hierarchy[supertype] = [v for v in types_hierarchy[supertype] if v not in types_hierarchy[k]]


            for supertype, subtypes in types_hierarchy.items():
                subtypes = [t for t in subtypes if t != supertype]
                if supertype != 'objects':
                    f.write("\n\t{} - {}".format('\n\t'.join(subtypes), supertype))
                elif len(self.types_hierarchy.keys()) == 1 and supertype == 'objects':
                    f.write("\n\t{}".format('\n\t'.join(subtypes)))
            f.write("\n)")

            # Write constants
            if len(self.constants) > 0:
                f.write("\n(:constants")
                [f.write(f"\n{' '.join(v)} - {k}") for k, v in self.constants.items()]
                f.write("\n)\n")

            # Write predicates
            f.write("\n(:predicates")
            for p in self.predicates:
                p_name = p.split('(')[0]
                p_types = [t for t in p.split('(')[1][:-1].split(',') if t.strip() != '']
                if len(p_types) > 0:
                    f.write(f"\n\t({p_name} {' '.join([f'?param_{i + 1} - {p_type}' for i, p_type in enumerate(p_types)])})")
                else:
                    f.write(f"\n\t({p_name})")
            f.write("\n)\n\n")

            # Write operators with certain preconditions and effects
            for operator in self.operators:
                f.write(f"\n(:action {operator.operator_name}")
                f.write("\n:parameters ({})".format(" ".join(['{} - {}'.format(param, obj_type) for param, obj_type in operator.parameters.items()])))

                # Format preconditions and effects syntax
                precs_cert = []
                eff_pos_cert = []
                eff_neg_cert = []
                precs_uncert = []
                eff_pos_uncert = []
                eff_neg_uncert = []

                if precs_certain:

                    precs_cert = []
                    for p in operator.precs_cert:
                        if p.startswith('not('):
                            p = p[4:-1]
                            if len([o for o in "(".join(p.split('(')[1:])[:-1].split(',') if o != '']) > 0:
                                precs_cert.append("(not ({} {}))".format(p.split('(')[0], " ".join("(".join(p.split('(')[1:])[:-1].split(','))))
                            else:
                                precs_cert.append(f"(not ({p.split('(')[0].strip()}))")
                        elif len([o for o in "(".join(p.split('(')[1:])[:-1].split(',') if o != '']) > 0:
                            precs_cert.append("({} {})".format(p.split('(')[0], " ".join("(".join(p.split('(')[1:])[:-1].split(','))))
                        else:
                            precs_cert.append(f"({p.split('(')[0].strip()})")

                if eff_pos_certain:
                    eff_pos_cert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                    if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0])
                                    for p in operator.eff_pos_cert]

                if eff_neg_certain:
                    eff_neg_cert = ["(not ({} {}))".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                    if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "(not ({}))".format(p.split('(')[0])
                                    for p in operator.eff_neg_cert]

                if precs_uncertain:
                    precs_uncert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                    if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0].strip())
                                    for p in operator.precs_uncert]

                if eff_pos_uncertain:
                    eff_pos_uncert = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                      if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(p.split('(')[0])
                                      for p in operator.eff_pos_uncert]

                if eff_neg_uncertain:
                    eff_neg_uncert = ["(not ({} {}))".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                                      if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0 else "(not ({}))".format(p.split('(')[0])
                                      for p in operator.eff_neg_uncert]

                f.write("\n:precondition\t(and {}\n)".format("\n".join(precs_cert + precs_uncert)))

                f.write("\n:effect\t(and {}\n)".format("\n".join(eff_pos_cert + eff_pos_uncert + eff_neg_cert + eff_neg_uncert)))
                f.write('\n)\n\n')

            f.write('\n\n)')


    def clean_pddl_domain_file(self, f_name):

        with open(f_name, "r") as f:
            data = [el.lower() for el in f.read().split("\n") if not el.strip().startswith(";")]

        # Remove domain functions
        for i in range(len(data)):

            if data[i].find(":action-costs") != -1:
                data[i] = data[i].replace(":action-costs", "")

            if data[i].find(":functions") != -1:

                for j in range(i, len(data)):

                    if data[j].find(":action") != -1:
                        break
                    else:
                        data[j] = ""

        with open(f"{f_name}_clean", "w") as f:
            [f.write(el.lower() + "\n") for el in data]

        # Rename operator parameters in each action schema
        with open(f"{f_name}_clean", "w") as f:

            all_action_schema = []
            action_indices = []

            for i in range(len(data)):
                row = data[i]

                if row.find(":action ") != -1:
                    action_indices.append(i)

            for i in range(len(action_indices)):

                action_index = action_indices[i]

                if action_index != action_indices[-1]:

                    action_schema = "".join(data[action_index:action_indices[i + 1]])

                else:

                    action_schema = "".join(data[action_index:])

                action_schema = re.sub(' +|\t', ' ', action_schema).replace(":", "\n:").replace("\n:", ":", 1)
                params = [el for i, el in
                          enumerate(re.findall(r"\(.*\)", action_schema.split("\n")[1])[0][1:-1].split())
                          if el.startswith("?")]

                for k, param in enumerate(params):
                    action_schema = action_schema.replace("({} ".format(param), "(?param_{} ".format(k + 1))
                    action_schema = action_schema.replace(" {} ".format(param), " ?param_{} ".format(k + 1))
                    action_schema = action_schema.replace(" {})".format(param), " ?param_{})".format(k + 1))

                all_action_schema.append(action_schema)

            for i in range(len(data)):
                if data[i].find(":action ") != -1:
                    break
                f.write("\n" + data[i])

            [f.write("\n\n{}".format(action_schema)) for action_schema in all_action_schema]
            f.write("\n)")


    def remove_predicate(self, pred):

        # Replace predicate in predicate list
        self.predicates = [p for p in self.predicates if p.split('(')[0] != pred]

        # Replace predicate in operator action model
        for op in self.operators:

            # Replace predicate in operator action model
            op.precs_uncert = {p for p in op.precs_uncert if p.split('(')[0] != pred}
            op.precs_cert = {p for p in op.precs_cert if p.split('(')[0] != pred}
            op.eff_pos_uncert = {p for p in op.eff_pos_uncert if p.split('(')[0] != pred}
            op.eff_pos_cert = {p for p in op.eff_pos_cert if p.split('(')[0] != pred}
            op.eff_neg_uncert = {p for p in op.eff_neg_uncert if p.split('(')[0] != pred}
            op.eff_neg_cert = {p for p in op.eff_neg_cert if p.split('(')[0] != pred}

            # Replace predicate in ground action models
            ground_actions = self.ground_actions[op.operator_name]
            for a in ground_actions:
                a.precs_uncert = {p for p in a.precs_uncert if p.split('(')[0] != pred}
                a.precs_cert = {p for p in a.precs_cert if p.split('(')[0] != pred}
                a.eff_pos_uncert = {p for p in a.eff_pos_uncert if p.split('(')[0] != pred}
                a.eff_pos_cert = {p for p in a.eff_pos_cert if p.split('(')[0] != pred}
                a.eff_neg_uncert = {p for p in a.eff_neg_uncert if p.split('(')[0] != pred}
                a.eff_neg_cert = {p for p in a.eff_neg_cert if p.split('(')[0] != pred}

from collections import defaultdict
import numpy as np
from OLAM.Action import Action


class Trace:

    def __init__(self, name, observations, actions):
        self.name = name
        self.observations = observations
        self.actions = actions
        self.all_ground_op = [[] for _ in actions]
        self.objects = None

    def __str__(self):
        trace = '(observation'

        for obs, act in zip(self.observations[:-1], self.actions):
            trace += f"\n\n(:state {str(obs)})"
            if act is not None:
                trace += f"\n\n(:action {str(act)})"

        trace += f"\n\n(:state {str(self.observations[-1])})"
        trace += '\n\n)'
        return trace

    def add_observation(self, obs):
        self.observations.append(obs)

    def add_action(self, action):
        self.actions.append(action)

    def rename_pred(self, p_old_names, new_preds):
        for obs in self.observations:
            for p_old, new_ps in zip(p_old_names, new_preds):
                for new_p in new_ps:
                    obs.positive_literals[new_p] |= {f"{new_p}({p.split('(')[1]}" for p in obs.positive_literals[p_old]}
                    obs.negative_literals[new_p] |= {f"not_{new_p}({p.split('(')[1]}" for p in obs.negative_literals[p_old]}
                obs.positive_literals.pop(p_old)
                obs.negative_literals.pop(p_old)


    def remove_predicates(self, predicates):

        for i in range(len(self.observations)):
            obs = self.observations[i]
            # Remove predicate from positive literals
            for p in predicates:
                obs.positive_literals.pop(p.split('(')[0])
                obs.negative_literals.pop(p.split('(')[0])


    def write(self, file):

        with open(file, 'w') as f:
            f.write('(observation')

            for i in range(len(self.observations)):
                observed_pos = set.union(*self.observations[i].positive_literals.values())
                observed_neg = set.union(*self.observations[i].negative_literals.values())

                literals_pos = ["({} {})".format(l.split('(')[0], " ".join(l.split('(')[1][:-1].split(',')))
                                if len([o for o in l.split('(')[1][:-1].split(',') if o != '']) > 0 else "({})".format(l.split('(')[0].strip())
                                for l in observed_pos]
                literals_neg = ["(not ({} {}))".format(l.split('(')[0], " ".join(l.split('(')[1][:-1].split(',')))
                                if len([o for o in l.split('(')[1][:-1].split(',') if o != '']) > 0 else "(not ({}))".format(l.split('(')[0].strip())
                                for l in observed_neg]

                f.write(f"\n\n(:state {' '.join(literals_pos + literals_neg)})")

                if i != len(self.actions) and type(self.actions[i]) == Action:
                    f.write(f"\n\n(:action {str(self.actions[i])})")

            f.write(')')


    def set_objects(self, action_model):

        observed_objects = defaultdict(set)

        # Get objects in trace actions
        for action in self.actions:
            if type(action) == Action:
                action_operator = [o for o in action_model.operators if o.operator_name == action.operator_name][0]
                [observed_objects[o].add(t) for o, t in zip(action.parameters, list(action_operator.parameters.values()))]

        # Get relevant literals, i.e. literals that changed their truth value in the trace
        all_literals = set()
        for obs in self.observations:

            obs_positive_literals = set.union(*obs.positive_literals.values())
            obs_negative_literals = set.union(*obs.negative_literals.values())

            all_literals |= {l for l in obs_positive_literals}
            all_literals |= {l[4:] for l in obs_negative_literals}

        # Get relevant literals object names and their types
        for l in all_literals:
            l_predicate = l.split('(')[0]
            l_objects = [o for o in l.split('(')[1].strip()[:-1].split(',') if o != '']
            p = [p for p in action_model.predicates if p.startswith(f"{l_predicate}(")][0]
            p_types = [t for t in p.split('(')[1].strip()[:-1].split(',') if t != '']

            [observed_objects[o].add(t) for o, t in zip(l_objects, p_types)]

        for obj_name, obj_type in observed_objects.items():
            if len(obj_type) > 1:
                supertypes = list({t for t in obj_type if t in action_model.types_hierarchy.keys()})
                subtypes = {t for t in obj_type if t not in supertypes}

                # Check if the subtype cannot be observed
                if len(subtypes) == 0:
                    for i in range(len(supertypes)):
                        if not np.any([t in action_model.types_hierarchy[supertypes[i]] for t in supertypes
                                       if t != supertypes[i]]):
                            subtypes = [supertypes[i]]
                            break

                assert len(subtypes) == 1, f'Detected multiple subtypes {subtypes} for object {obj_name}'

                observed_objects[obj_name] = subtypes

        assert np.all([len(v) == 1 for v in observed_objects.values()]), 'Check object type hierarchy, predicate ' \
                                                                         'object types, and actions object types'

        observed_objects = {k: list(v)[0] for k, v in observed_objects.items()}

        self.objects = observed_objects

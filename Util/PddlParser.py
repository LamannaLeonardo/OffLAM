
class PddlParser:

    def __init__(self):
        pass

    def write_pddl_state(self, objs, obs, domain_name, facts_file, goal=None):
        obs_positive_literals = set.union(*obs.positive_literals.values())
        positive_literals = [f"({l.split('(')[0]} {' '.join(l[:-1].split('(')[1].split(','))})"
                             if l[:-1].split('(')[1].split(',') != [''] else f"({l.split('(')[0]})"
                             for l in obs_positive_literals]

        with open(facts_file, "w") as f:

            # Write problem header
            f.write(f'\n(define (problem prob-{domain_name})')
            f.write(f'\n(:domain {domain_name})')

            # Write objects
            f.write(f"\n(:objects ")
            [f.write(f"\n{obj_name} - {obj_type}") for obj_name, obj_type in objs.items()]
            f.write(f"\n)")

            # Write pddl state
            f.write('\n(:init')
            [f.write(f"\n{l}") for l in positive_literals]
            f.write('\n)')

            # Write empty goal
            if goal is None:
                f.write('\n(:goal (and ))')
            else:
                f.write(f'\n(:goal {goal})')

            # Write end of problem definition
            f.write('\n)')

from collections import defaultdict


class Observation:

    def __init__(self, literals):

        positive_literals = {l for l in literals if not l.startswith('not_')}
        negative_literals = {l for l in literals if l.startswith('not_')}
        positive_literals_names = {l.split('(')[0] for l in literals if not l.startswith('not_')}
        negative_literals_names = {l[4:].split('(')[0] for l in literals if l.startswith('not_')}

        self.positive_literals = defaultdict(set)
        self.negative_literals = defaultdict(set)
        self.positive_literals['dummy'] = set()
        self.negative_literals['dummy'] = set()

        for n in positive_literals_names:
            self.positive_literals[n] = {p for p in positive_literals if p.startswith(f"{n}(")}
        for n in negative_literals_names:
            self.negative_literals[n] = {p for p in negative_literals if p[4:].startswith(f"{n}(")}

    def __str__(self):
        positive_literals = []
        [positive_literals.extend(pos) for k, pos in self.positive_literals.items()]

        negative_literals = []
        [negative_literals.extend(neg) for k, neg in self.negative_literals.items()]

        pos_literals = ["({} {})".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                        if len([o for o in p.split('(')[1][:-1].split(',')
                                if o != '']) > 0 else "({})".format(p.split('(')[0].strip())
                        for p in self.positive_literals]
        # neg_literals = [p.replace('not_', '') for p in self.negative_literals]
        neg_literals = [p[4:] for p in self.negative_literals]
        neg_literals = ["(not ({} {}))".format(p.split('(')[0], " ".join(p.split('(')[1][:-1].split(',')))
                        if len([o for o in p.split('(')[1][:-1].split(',') if o != '']) > 0
                        else "({})".format(p.split('(')[0].strip())
                        for p in neg_literals]

        return " ".join(sorted(pos_literals) + sorted(neg_literals))

    def add_positive(self, positive_literal):
        self.positive_literals[positive_literal.split('(')[0]].add(positive_literal)

    def add_negative(self, negative_literal):
        self.negative_literals[negative_literal[4:].split('(')[0]].add(negative_literal)

    def remove_negative(self, negative_literal):
        # assert not negative_literal.truth_value
        self.negative_literals[negative_literal[4:].split('(')[0]].remove(negative_literal)

    def __contains__(self, literal):
        if literal.startswith('not_'):
            return literal in self.negative_literals[literal[4:].split('(')[0]]
        else:
            return literal in self.positive_literals[literal.split('(')[0]]

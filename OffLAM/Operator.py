import Configuration


class Operator:

    def __init__(self, operator_name, parameters,
                 precs_cert=None, eff_pos_cert=None, eff_neg_cert=None,
                 precs_uncert=None, eff_pos_uncert=None, eff_neg_uncert=None):

        self.operator_name = operator_name
        self.parameters = parameters
        self.precs_cert = precs_cert
        self.eff_pos_cert = eff_pos_cert
        self.eff_neg_cert = eff_neg_cert
        self.precs_uncert = precs_uncert
        self.eff_pos_uncert = eff_pos_uncert
        self.eff_neg_uncert = eff_neg_uncert

    def __str__(self):
        return f"{self.operator_name}({','.join(self.parameters)})"

    def add_prec_cert(self, precondition):
        if precondition not in self.precs_cert:
            print(f'[Info] Operator {self.operator_name}, adding certain precondition {precondition}')
            self.precs_cert.add(precondition)
        self.remove_prec_uncert(precondition)

    def add_prec_uncert(self, precondition):
        if precondition not in self.precs_uncert:
            print(f'[Info] Operator {self.operator_name}, adding uncertain precondition {precondition}')
            self.precs_uncert.add(precondition)
        self.remove_prec_cert(precondition)

    def remove_prec_uncert(self, precondition):
        if precondition in self.precs_uncert:
            print(f'[Info] Operator {self.operator_name}, removing uncertain precondition {precondition}')
            self.precs_uncert.remove(precondition)

    def remove_prec_cert(self, precondition):
        if precondition in self.precs_cert:
            print(f'[Info] Operator {self.operator_name}, removing certain precondition {precondition}')
            self.precs_cert.remove(precondition)

    def add_eff_pos_cert(self, effect):
        if effect not in self.eff_pos_cert:
            print(f'[Info] Operator {self.operator_name}, adding certain positive effect {effect}')
            self.eff_pos_cert.add(effect)
        self.remove_eff_neg_uncert(effect)
        self.remove_eff_pos_uncert(effect)

    def remove_eff_pos_uncert(self, effect):
        if effect in self.eff_pos_uncert:
            print(f'[Info] Operator {self.operator_name}, removing uncertain positive effect {effect}')
            self.eff_pos_uncert.remove(effect)

    def add_eff_neg_cert(self, effect):
        if effect not in self.eff_neg_cert:
            print(f'[Info] Operator {self.operator_name}, adding certain negative effect {effect}')
            self.eff_neg_cert.add(effect)
        self.remove_eff_neg_uncert(effect)
        self.remove_eff_pos_uncert(effect)

    def add_eff_neg_uncert(self, effect):
        if effect not in self.eff_neg_uncert:
            print(f'[Info] Operator {self.operator_name}, adding uncertain negative effect {effect}')
            self.eff_neg_uncert.add(effect)

    def add_eff_pos_uncert(self, effect):
        if effect not in self.eff_pos_uncert:
            print(f'[Info] Operator {self.operator_name}, adding uncertain positive effect {effect}')
            self.eff_pos_uncert.append(effect)

    def remove_eff_neg_uncert(self, effect):
        if effect in self.eff_neg_uncert:
            print(f'[Info] Operator {self.operator_name}, removing uncertain negative effect {effect}')
            self.eff_neg_uncert.remove(effect)

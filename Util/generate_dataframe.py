import os
import pandas as pd


def save_dataframe(num_traces, EXP, run=0):

    df = pd.DataFrame(columns=('Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision', 'Pos recall',
                               'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall', 'Overall precision'))

    RESULTS_DIR = f'Analysis/Results/{num_traces} traces/OffLAM/{EXP}/run{run}'

    for domain in [dir for dir in os.listdir(RESULTS_DIR) if '.xls' not in dir and 'Results' not in dir and '.png' not in dir]:
        for observability in sorted({float(el.split('_')[-1]) for el in os.listdir(os.path.join(RESULTS_DIR, domain))}):

            if observability == 1.0:
                observability = int(observability)

            metrics_file = os.path.join(RESULTS_DIR, domain, f'{observability}', 'log')

            with open(metrics_file, 'r') as f:
                data = [el.replace("|","") for el in f.read().split('\n') if el.strip() != ""]

                all_metrics = [str.replace("\t\t", " ").replace("|", "").split()
                               for str in data
                               if str.strip() != ""
                               and all([el.isnumeric() for el in str.replace("\t\t", " ").replace(".", "").replace("|", "").split()])]

                precs_recall = all_metrics[-1][13]
                pos_recall = all_metrics[-1][14]
                neg_recall = all_metrics[-1][15]
                precs_precision = all_metrics[-1][16]
                pos_precision = all_metrics[-1][17]
                neg_precision = all_metrics[-1][18]
                overall_recall = all_metrics[-1][19]
                overall_precision = all_metrics[-1][20]

                len_traces = int([l for l in data if 'processed traces' in l][0].split()[-1])
                cpu_time = round(float([l for l in data if 'cpu' in l.lower()][0].split()[-1]), 2)

                evaluate = {
                    'Domain': domain,
                    'Traces': int(len_traces),
                    'Observability': float(observability),
                    'CPU time': float(cpu_time),
                    'Precs recall': precs_recall,
                    'Precs precision': precs_precision,
                    'Pos recall': pos_recall,
                    'Pos precision': pos_precision,
                    'Neg recall': neg_recall,
                    'Neg precision': neg_precision,
                    'Overall recall': overall_recall,
                    'Overall precision': overall_precision
                }

                evaluate = pd.DataFrame([evaluate])
                df = pd.concat([df, evaluate.astype(df.dtypes)], ignore_index=True)

    writer = pd.ExcelWriter(os.path.join(RESULTS_DIR, f"run{run}_offlam_results.xlsx"))
    df.to_excel(writer, index=False, float_format="%0.2f")
    writer.close()


if __name__ == "__main__":

    num_traces = 10

    PARTIAL_STATE_EXP = 'partial_states'
    PARTIAL_ACTIONS_EXP = 'partial_actions'
    PARTIAL_STATEACTIONS_EXP = 'partial_states_actions'
    EXP = PARTIAL_STATE_EXP
    # EXP = PARTIAL_ACTIONS_EXP
    # EXP = PARTIAL_STATEACTIONS_EXP

    save_dataframe(num_traces, EXP)


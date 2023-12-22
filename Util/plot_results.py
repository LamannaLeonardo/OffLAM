import os
from collections import defaultdict
from statistics import stdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

from statistics import stdev
plt.style.use('ggplot')

def plot_overall_prec_recall(traces, EXP, run=0):

    # traces = 10

    PARTIAL_STATE_EXP = 'partial_states'
    PARTIAL_ACTIONS_EXP = 'partial_actions'
    PARTIAL_STATEACTIONS_EXP = 'partial_states_actions'

    OFFLAM_RESULTS_DIR = f'Analysis/Results/{traces} traces/OffLAM/{EXP}'

    all_precision_offlam = defaultdict(list)
    all_recall_offlam = defaultdict(list)

    # for run in range(1, len(os.listdir(OFFLAM_RESULTS_DIR)) + 1):

    df_path_polam = os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f'run{run}_offlam_results.xlsx')

    df_polam = pd.read_excel(df_path_polam)

    df_polam = df_polam[['Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision',
    'Pos recall', 'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall', 'Overall precision']]


    # Write merged dataframe of OffLAM and FAMA comparison
    # writer = pd.ExcelWriter(os.path.join(RESULTS_DIR, "results.xls"))
    # new_df.to_excel(writer, index=True, float_format="%0.2f")
    # writer.close()

    avg_precision_offlam = dict()
    avg_precision_fama = dict()

    avg_recall_offlam = dict()
    avg_recall_fama = dict()

    for observability in sorted(set(df_polam['Observability'].values)):

        observability_df = df_polam.loc[df_polam['Observability'] == observability]

        assert len(observability_df['Overall precision'].values) == len(set(df_polam['Domain'].values))

        avg_precision_offlam[observability] = observability_df['Overall precision'].sum() / len(observability_df['Overall precision'].values)
        # avg_precision_fama[observability] = observability_df['Overall precision'].sum() / len(observability_df['Overall precision'].values)

        avg_recall_offlam[observability] = observability_df['Overall recall'].sum() / len(observability_df['Overall recall'].values)
        # avg_recall_fama[observability] = observability_df['Overall recall'].sum() / len(observability_df['Overall recall'].values)

    for observability, avg_precision in avg_precision_offlam.items():
        all_precision_offlam[observability].append(avg_precision)

    for observability, avg_recall in avg_recall_offlam.items():
        all_recall_offlam[observability].append(avg_recall)


    mean_precision_offlam = {obs: np.mean(obs_avg_precision_offlam) for obs, obs_avg_precision_offlam in all_precision_offlam.items()}
    # stdev_precision_offlam = {obs: stdev(obs_avg_precision_offlam) for obs, obs_avg_precision_offlam in all_precision_offlam.items()}
    plt.plot(list(mean_precision_offlam.keys()), list(mean_precision_offlam.values()), label='OffLAM')

    # plt.fill_between(list(mean_precision_offlam.keys()),
    #                  np.array(list(mean_precision_offlam.values())) - np.array(list(stdev_precision_offlam.values())),
    #                  np.array(list(mean_precision_offlam.values())) + np.array(list(stdev_precision_offlam.values())),
    #                  color='b', alpha=0.2)



    if EXP == PARTIAL_ACTIONS_EXP:
        plt.xlabel('Actions observability')
    elif EXP == PARTIAL_STATE_EXP:
        plt.xlabel('States observability')
    elif EXP == PARTIAL_STATEACTIONS_EXP:
        plt.xlabel('States and actions observability')
    else:
        plt.xlabel('Observability')


    plt.ylabel('Overall precision')
    plt.legend()
    plt.title(f"Average precision over {len(set(df_polam['Domain'].values))} domains")
    # plt.savefig(os.path.join(f'Analysis/Results/{traces} traces/', f"{EXP}_precision_offlam.png"))
    plt.savefig(os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f"{EXP}_precision_offlam.png"))
    plt.close()






    mean_recall_offlam = {obs: np.mean(obs_avg_recall_offlam) for obs, obs_avg_recall_offlam in all_recall_offlam.items()}
    # stdev_recall_offlam = {obs: stdev(obs_avg_recall_offlam) for obs, obs_avg_recall_offlam in all_recall_offlam.items()}
    plt.plot(list(mean_recall_offlam.keys()), list(mean_recall_offlam.values()), label='OffLAM')

    # plt.fill_between(list(mean_recall_offlam.keys()),
    #                  np.array(list(mean_recall_offlam.values())) - np.array(list(stdev_recall_offlam.values())),
    #                  np.array(list(mean_recall_offlam.values())) + np.array(list(stdev_recall_offlam.values())),
    #                  color='b', alpha=0.2)

    if EXP == PARTIAL_ACTIONS_EXP:
        plt.xlabel('Actions observability')
    elif EXP == PARTIAL_STATE_EXP:
        plt.xlabel('States observability')
    elif EXP == PARTIAL_STATEACTIONS_EXP:
        plt.xlabel('States and actions observability')
    else:
        plt.xlabel('Observability')

    plt.ylabel('Overall recall')
    plt.legend()
    plt.title(f"Average recall over {len(set(df_polam['Domain'].values))} domains")
    # plt.savefig(os.path.join(f'Analysis/Results/{traces} traces/', f"{EXP}_recall_offlam.png"))
    plt.savefig(os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f"{EXP}_recall_offlam.png"))
    plt.close()



def plot_average_recall(traces, run=0):

    PARTIAL_STATE_EXP = 'partial_states'
    PARTIAL_ACTIONS_EXP = 'partial_actions'
    PARTIAL_STATEACTIONS_EXP = 'partial_states_actions'
    lines = ["-", "--", "-.", ":"]
    colors = ['royalblue', 'seagreen', 'tomato']
    linecycler = cycle(lines)
    colorcycler = cycle(colors)

    for EXP in [PARTIAL_STATE_EXP, PARTIAL_ACTIONS_EXP, PARTIAL_STATEACTIONS_EXP]:
        OFFLAM_RESULTS_DIR = f'../Analysis/Results/{traces} traces/OffLAM/{EXP}'\

        df_path_polam = os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f'run{run}_offlam_results.xlsx')
        df_polam = pd.read_excel(df_path_polam)
        df_polam = df_polam[['Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision',
        'Pos recall', 'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall', 'Overall precision']]

        mean_recall_offlam = dict()
        stdev_recall_offlam = dict()

        for observability in sorted(set(df_polam['Observability'].values)):
            observability_df = df_polam.loc[df_polam['Observability'] == observability]
            mean_recall_offlam[observability] = np.mean(observability_df['Overall recall'])
            stdev_recall_offlam[observability] = stdev(observability_df['Overall recall'])

        color = next(colorcycler)
        plt.plot(list(mean_recall_offlam.keys()), list(mean_recall_offlam.values()), label="partial " + EXP[8:].replace('_', ' and '),
                 linestyle=next(linecycler), c=color)

        plt.fill_between(list(mean_recall_offlam.keys()),
                         np.array(list(mean_recall_offlam.values())) - np.array(list(stdev_recall_offlam.values())),
                         np.array(list(mean_recall_offlam.values())) + np.array(list(stdev_recall_offlam.values())),
                         color=color, alpha=0.2)

    plt.xlabel('Observability degree', fontsize="14")

    plt.ylabel('Average recall', fontsize="14")
    plt.legend(loc='lower right', fontsize="12")
    plt.ylim(0.25, 1)
    plt.xticks([i*0.1 for i in range(1, 11)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.margins(0.)
    plt.tight_layout()
    # plt.title(f"Average recall over {len(set(df_polam['Domain'].values))} domains")
    plt.title(f"")
    plt.savefig(os.path.join(f'../Analysis/Results/{traces} traces/OffLAM/run{run}_recall_offlam.png'))
    plt.close()



def plot_average_prec(traces, run=0):

    PARTIAL_STATE_EXP = 'partial_states'
    PARTIAL_ACTIONS_EXP = 'partial_actions'
    PARTIAL_STATEACTIONS_EXP = 'partial_states_actions'
    lines = ["-", "--", "-.", ":"]
    colors = ['royalblue', 'seagreen', 'tomato']
    linecycler = cycle(lines)
    colorcycler = cycle(colors)

    for EXP in [PARTIAL_STATE_EXP, PARTIAL_ACTIONS_EXP, PARTIAL_STATEACTIONS_EXP]:
        OFFLAM_RESULTS_DIR = f'../Analysis/Results/{traces} traces/OffLAM/{EXP}'\

        df_path_polam = os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f'run{run}_offlam_results.xlsx')
        df_polam = pd.read_excel(df_path_polam)
        df_polam = df_polam[['Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision',
        'Pos recall', 'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall', 'Overall precision']]

        mean_prec_offlam = dict()
        stdev_prec_offlam = dict()

        for observability in sorted(set(df_polam['Observability'].values)):
            observability_df = df_polam.loc[df_polam['Observability'] == observability]
            mean_prec_offlam[observability] = np.mean(observability_df['Overall precision'])
            stdev_prec_offlam[observability] = stdev(observability_df['Overall precision'])

        color = next(colorcycler)

        plt.plot(list(mean_prec_offlam.keys()), list(mean_prec_offlam.values()), label="partial " + EXP[8:].replace('_', ' and '),
                 linestyle=next(linecycler), c=color)

        plt.fill_between(list(mean_prec_offlam.keys()),
                         np.array(list(mean_prec_offlam.values())) - np.array(list(stdev_prec_offlam.values())),
                         np.array(list(mean_prec_offlam.values())) + np.array(list(stdev_prec_offlam.values())),
                         color=color, alpha=0.2)

    plt.xlabel('Observability degree', fontsize="14")

    plt.ylabel('Average precision', fontsize="14")
    plt.legend(loc='lower right', fontsize="12")
    plt.ylim(0.25, 1)
    plt.xticks([i*0.1 for i in range(11)], fontsize=14)
    plt.yticks(fontsize=14)
    plt.margins(0.)
    plt.tight_layout()
    # plt.title(f"Average precision over {len(set(df_polam['Domain'].values))} domains")
    plt.title(f"")
    plt.savefig(os.path.join(f'../Analysis/Results/{traces} traces/OffLAM/run{run}_precision_offlam.png'))
    plt.close()



def domain_results_table(traces, run=0):


    columns = [('', '', 'Domain'),
               ("Observability degree", '$0.1$', '$P$'), ("Observability degree", '$0.1$', '$R$'),
               ("Observability degree", '$0.2$', '$P$'), ("Observability degree", '$0.2$', '$R$'),
               ("Observability degree", '$0.3$', '$P$'), ("Observability degree", '$0.3$', '$R$'),
               ("Observability degree", '$0.4$', '$P$'), ("Observability degree", '$0.4$', '$R$'),
               ("Observability degree", '$0.5$', '$P$'), ("Observability degree", '$0.5$', '$R$'),
               ("Observability degree", '$0.6$', '$P$'), ("Observability degree", '$0.6$', '$R$'),
               ("Observability degree", '$0.7$', '$P$'), ("Observability degree", '$0.7$', '$R$'),
               ("Observability degree", '$0.8$', '$P$'), ("Observability degree", '$0.8$', '$R$'),
               ("Observability degree", '$0.9$', '$P$'), ("Observability degree", '$0.9$', '$R$'),
               ("Observability degree", '$1.0$', '$P$'), ("Observability degree", '$1.0$', '$R$')]
    df = pd.DataFrame(columns=list(range(len(columns))))
    df.columns = pd.MultiIndex.from_tuples(columns)

    # for EXP in [PARTIAL_STATE_EXP, PARTIAL_ACTIONS_EXP, PARTIAL_STATEACTIONS_EXP]:
    # EXP = PARTIAL_STATE_EXP
    EXP = PARTIAL_ACTIONS_EXP
    # EXP = PARTIAL_STATEACTIONS_EXP
    OFFLAM_RESULTS_DIR = f'../Analysis/Results/{traces} traces/OffLAM/{EXP}'\

    df_path_polam = os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f'run{run}_offlam_results.xlsx')
    df_polam = pd.read_excel(df_path_polam)
    df_polam = df_polam[['Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision',
                         'Pos recall', 'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall',
                         'Overall precision']]


    for domain in sorted(set(df_polam['Domain'].values)):
        # for observability in sorted(set(df_polam['Observability'].values)):
        #     observability_df = df_polam.loc[df_polam['Observability'] == observability]
        domain_df = df_polam.loc[df_polam['Domain'] == domain]
        eval = {
            ('', '', 'Domain'): domain,
            ("Observability degree", '$0.1$', '$P$'): domain_df[domain_df['Observability'] == .1]['Overall precision'].values[0],
            ("Observability degree", '$0.1$', '$R$'): domain_df[domain_df['Observability'] == .1]['Overall recall'].values[0],
            ("Observability degree", '$0.2$', '$P$'): domain_df[domain_df['Observability'] == .2]['Overall precision'].values[0],
            ("Observability degree", '$0.2$', '$R$'): domain_df[domain_df['Observability'] == .2]['Overall recall'].values[0],
            ("Observability degree", '$0.3$', '$P$'): domain_df[domain_df['Observability'] == .3]['Overall precision'].values[0],
            ("Observability degree", '$0.3$', '$R$'): domain_df[domain_df['Observability'] == .3]['Overall recall'].values[0],
            ("Observability degree", '$0.4$', '$P$'): domain_df[domain_df['Observability'] == .4]['Overall precision'].values[0],
            ("Observability degree", '$0.4$', '$R$'): domain_df[domain_df['Observability'] == .4]['Overall recall'].values[0],
            ("Observability degree", '$0.5$', '$P$'): domain_df[domain_df['Observability'] == .5]['Overall precision'].values[0],
            ("Observability degree", '$0.5$', '$R$'): domain_df[domain_df['Observability'] == .5]['Overall recall'].values[0],
            ("Observability degree", '$0.6$', '$P$'): domain_df[domain_df['Observability'] == .6]['Overall precision'].values[0],
            ("Observability degree", '$0.6$', '$R$'): domain_df[domain_df['Observability'] == .6]['Overall recall'].values[0],
            ("Observability degree", '$0.7$', '$P$'): domain_df[domain_df['Observability'] == .7]['Overall precision'].values[0],
            ("Observability degree", '$0.7$', '$R$'): domain_df[domain_df['Observability'] == .7]['Overall recall'].values[0],
            ("Observability degree", '$0.8$', '$P$'): domain_df[domain_df['Observability'] == .8]['Overall precision'].values[0],
            ("Observability degree", '$0.8$', '$R$'): domain_df[domain_df['Observability'] == .8]['Overall recall'].values[0],
            ("Observability degree", '$0.9$', '$P$'): domain_df[domain_df['Observability'] == .9]['Overall precision'].values[0],
            ("Observability degree", '$0.9$', '$R$'): domain_df[domain_df['Observability'] == .9]['Overall recall'].values[0],
            ("Observability degree", '$1.0$', '$P$'): domain_df[domain_df['Observability'] == 1.0]['Overall precision'].values[0],
            ("Observability degree", '$1.0$', '$R$'): domain_df[domain_df['Observability'] == 1.0]['Overall recall'].values[0]
        }
        df = pd.concat([df, pd.DataFrame([eval])], ignore_index=True)

    with open(f"{EXP}_offlam.tex", "w") as f:
        f.write(df.to_latex(index=False,
                            label="tab:partial-states-offlam",
                            caption="Per domain results achieved by \\alg{} when learning from $10$ traces with partially "
                                    "observable actions "
                                    "and an observation degree ranging from $0$ to $1$. The precision and recall "
                                    "measures are averaged over $10$ runs",
                            escape=False,
                            float_format="{:0.2f}".format,
                            column_format='c|cc|cc|cc|cc|cc|cc|cc|cc|cc|cc|'))



def domain_results_table_vs_fama(traces=2, exp='partial_states'):


    columns = [('', '', 'Domain'),
               ("Observability degree", '$0.1$', '$P$'), ("Observability degree", '$0.1$', '$R$'),
               ("Observability degree", '$0.2$', '$P$'), ("Observability degree", '$0.2$', '$R$'),
               ("Observability degree", '$0.3$', '$P$'), ("Observability degree", '$0.3$', '$R$'),
               ("Observability degree", '$0.4$', '$P$'), ("Observability degree", '$0.4$', '$R$'),
               ("Observability degree", '$0.5$', '$P$'), ("Observability degree", '$0.5$', '$R$'),
               ("Observability degree", '$0.6$', '$P$'), ("Observability degree", '$0.6$', '$R$'),
               ("Observability degree", '$0.7$', '$P$'), ("Observability degree", '$0.7$', '$R$'),
               ("Observability degree", '$0.8$', '$P$'), ("Observability degree", '$0.8$', '$R$'),
               ("Observability degree", '$0.9$', '$P$'), ("Observability degree", '$0.9$', '$R$'),
               ("Observability degree", '$1.0$', '$P$'), ("Observability degree", '$1.0$', '$R$')]
    df = pd.DataFrame(columns=list(range(len(columns))))
    df.columns = pd.MultiIndex.from_tuples(columns)

    OFFLAM_RESULTS_DIR = f'../Analysis/Results/{traces} traces/OffLAM/{exp}'
    
    precision = {obs: dict() for obs in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}
    recall = {obs: dict() for obs in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]}
    all_domains = [d for d in sorted(set(os.listdir(f"{OFFLAM_RESULTS_DIR}/run0"))) if '.' not in d]
    
    for domain in all_domains:
        for obs in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            precision[obs][domain] = []
            recall[obs][domain] = []
    
    for run in range(10):

        df_path_polam = os.path.join(OFFLAM_RESULTS_DIR, f'run{run}', f'run{run}_offlam_results.xlsx')
        df_polam = pd.read_excel(df_path_polam)
        df_polam = df_polam[['Domain', 'Traces', 'Observability', 'Precs recall', 'Precs precision',
                             'Pos recall', 'Pos precision', 'Neg recall', 'Neg precision', 'Overall recall',
                             'Overall precision']]
        
        for domain in all_domains:
            domain_df = df_polam.loc[df_polam['Domain'] == domain]
            for obs in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
                precision[obs][domain].append(domain_df[domain_df['Observability'] == obs]['Overall precision'].values[0])
                recall[obs][domain].append(domain_df[domain_df['Observability'] == obs]['Overall recall'].values[0])
        
    for domain in all_domains:

        for obs in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            precision[obs][domain] = np.array(precision[obs][domain])
            recall[obs][domain] = np.array(recall[obs][domain])

        # eval = {
        #     ('', '', 'Domain'): domain,
        #     ("Observability degree", '$0.1$', '$P$'): f"${round(precision[.1][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.1][domain]), 2)}}}$",
        #     ("Observability degree", '$0.1$', '$R$'): f"${round(recall[.1][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.1][domain]), 2)}}}$",
        #     ("Observability degree", '$0.2$', '$P$'): f"${round(precision[.2][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.2][domain]), 2)}}}$",
        #     ("Observability degree", '$0.2$', '$R$'): f"${round(recall[.2][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.2][domain]), 2)}}}$",
        #     ("Observability degree", '$0.3$', '$P$'): f"${round(precision[.3][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.3][domain]), 2)}}}$",
        #     ("Observability degree", '$0.3$', '$R$'): f"${round(recall[.3][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.3][domain]), 2)}}}$",
        #     ("Observability degree", '$0.4$', '$P$'): f"${round(precision[.4][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.4][domain]), 2)}}}$",
        #     ("Observability degree", '$0.4$', '$R$'): f"${round(recall[.4][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.4][domain]), 2)}}}$",
        #     ("Observability degree", '$0.5$', '$P$'): f"${round(precision[.5][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.5][domain]), 2)}}}$",
        #     ("Observability degree", '$0.5$', '$R$'): f"${round(recall[.5][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.5][domain]), 2)}}}$",
        #     ("Observability degree", '$0.6$', '$P$'): f"${round(precision[.6][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.6][domain]), 2)}}}$",
        #     ("Observability degree", '$0.6$', '$R$'): f"${round(recall[.6][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.6][domain]), 2)}}}$",
        #     ("Observability degree", '$0.7$', '$P$'): f"${round(precision[.7][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.7][domain]), 2)}}}$",
        #     ("Observability degree", '$0.7$', '$R$'): f"${round(recall[.7][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.7][domain]), 2)}}}$",
        #     ("Observability degree", '$0.8$', '$P$'): f"${round(precision[.8][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.8][domain]), 2)}}}$",
        #     ("Observability degree", '$0.8$', '$R$'): f"${round(recall[.8][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.8][domain]), 2)}}}$",
        #     ("Observability degree", '$0.9$', '$P$'): f"${round(precision[.9][domain].mean(), 2)}_{{\\pm{round(stdev(precision[.9][domain]), 2)}}}$",
        #     ("Observability degree", '$0.9$', '$R$'): f"${round(recall[.9][domain].mean(), 2)}_{{\\pm{round(stdev(recall[.9][domain]), 2)}}}$",
        #     ("Observability degree", '$1.0$', '$P$'): f"${round(precision[1.0][domain].mean(), 2)}_{{\\pm{round(stdev(precision[1.0][domain]), 2)}}}$",
        #     ("Observability degree", '$1.0$', '$R$'): f"${round(recall[1.0][domain].mean(), 2)}_{{\\pm{round(stdev(recall[1.0][domain]), 2)}}}$",
        # }
        
        eval = {
            ('', '', 'Domain'): domain,
            ("Observability degree", '$0.1$', '$P$'): f"${round(precision[.1][domain].mean(), 2)}$",
            ("Observability degree", '$0.1$', '$R$'): f"${round(recall[.1][domain].mean(), 2)}$",
            ("Observability degree", '$0.2$', '$P$'): f"${round(precision[.2][domain].mean(), 2)}$",
            ("Observability degree", '$0.2$', '$R$'): f"${round(recall[.2][domain].mean(), 2)}$",
            ("Observability degree", '$0.3$', '$P$'): f"${round(precision[.3][domain].mean(), 2)}$",
            ("Observability degree", '$0.3$', '$R$'): f"${round(recall[.3][domain].mean(), 2)}$",
            ("Observability degree", '$0.4$', '$P$'): f"${round(precision[.4][domain].mean(), 2)}$",
            ("Observability degree", '$0.4$', '$R$'): f"${round(recall[.4][domain].mean(), 2)}$",
            ("Observability degree", '$0.5$', '$P$'): f"${round(precision[.5][domain].mean(), 2)}$",
            ("Observability degree", '$0.5$', '$R$'): f"${round(recall[.5][domain].mean(), 2)}$",
            ("Observability degree", '$0.6$', '$P$'): f"${round(precision[.6][domain].mean(), 2)}$",
            ("Observability degree", '$0.6$', '$R$'): f"${round(recall[.6][domain].mean(), 2)}$",
            ("Observability degree", '$0.7$', '$P$'): f"${round(precision[.7][domain].mean(), 2)}$",
            ("Observability degree", '$0.7$', '$R$'): f"${round(recall[.7][domain].mean(), 2)}$",
            ("Observability degree", '$0.8$', '$P$'): f"${round(precision[.8][domain].mean(), 2)}$",
            ("Observability degree", '$0.8$', '$R$'): f"${round(recall[.8][domain].mean(), 2)}$",
            ("Observability degree", '$0.9$', '$P$'): f"${round(precision[.9][domain].mean(), 2)}$",
            ("Observability degree", '$0.9$', '$R$'): f"${round(recall[.9][domain].mean(), 2)}$",
            ("Observability degree", '$1.0$', '$P$'): f"${round(precision[1.0][domain].mean(), 2)}$",
            ("Observability degree", '$1.0$', '$R$'): f"${round(recall[1.0][domain].mean(), 2)}$",
        }

        df = pd.concat([df, pd.DataFrame([eval])], ignore_index=True)

    with open(f"{exp}_{traces}_offlam.tex", "w") as f:
        f.write(df.to_latex(index=False,
                            label="tab:partial-states-vsfama",
                            caption=f"Per domain results achieved by \\alg when learning from ${traces}$ traces with partially "
                                    "observable actions "
                                    "and an observation degree ranging from $0$ to $1$.",
                            escape=False,
                            float_format="{:0.2f}".format,
                            column_format='c|cc|cc|cc|cc|cc|cc|cc|cc|cc|cc|'))


if __name__ == "__main__":
    plt.style.use('ggplot')

    traces = 10

    PARTIAL_STATE_EXP = 'partial_states'
    PARTIAL_ACTIONS_EXP = 'partial_actions'
    PARTIAL_STATEACTIONS_EXP = 'partial_states_actions'

    # EXP = PARTIAL_STATE_EXP
    # EXP = PARTIAL_ACTIONS_EXP
    # EXP = PARTIAL_STATEACTIONS_EXP
    # plot_overall_prec_recall(traces, EXP)

    # plot_average_recall(traces)
    # plot_average_prec(traces)

    domain_results_table(traces)



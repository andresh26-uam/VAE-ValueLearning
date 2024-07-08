import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

from src.values_and_costs import FULL_NAME_VALUES
combinations={((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'sec_eff_11_'),((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'sus_eff_11_'),((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'all_5_'),}
    #combinations={((1.0,0.0,0.0), (0.0,1.0,0.0)): (11, 'cor_sus_sec_11_'), ((1.0,0.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sus_eff_11_'),((0.0,1.0,0.0), (0.0,0.0,1.0)): (11, 'cor_sec_eff_11_'),((1.0,0.0,0.0),(0.0,1.0,0.0), (0.0,0.0,1.0)): (5, 'cor_all_5_'),}
PUT_OD_STD = False


# Define colors
colors = {
    'sus': ('darkgreen', 'green'),
    'eff': ('red', 'magenta'),
    'sec': ('blue', 'cyan')
}
PLOTS=True
if PLOTS:
    for test_or_train in ['test','train']:
            for combination, n_profiles_and_name in combinations.items():
                n_profiles, name_of_files = n_profiles_and_name
                
                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_{test_or_train}.csv")
                
                print(df.columns[1])
                policies = df[df.columns[1]]
                
                expert_df = df[df['Policy'].str.contains('Expert', case=False, na=False)]
                print(expert_df['sec'])
                
                learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
                print(learned_df['sec'])

                for col in ['sus', 'eff', 'sec']:
                    
                    expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                    
                    expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
                
                # Separate the averages and standard deviations for learned_df
                for col in ['sus', 'eff', 'sec']:
                    learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                    learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])

                # EXPERT VS LEARNED
                # Plot and save each plot in separate PDF files
                for col in ['sus', 'eff', 'sec']:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bar_width = 0.35
                    index = np.arange(len(expert_df))
                    
                    bars1 = ax.bar(index, expert_df[f'{col}_mean'], bar_width, yerr=expert_df[f'{col}_std'] if PUT_OD_STD else None, capsize=5, label=f"{'Original profile individual'}", color=colors[col][0])
                    bars2 = ax.bar(index + bar_width, learned_df[f'{col}_mean'], bar_width, yerr=learned_df[f'{col}_std'] if PUT_OD_STD  else None, capsize=5, label='Learned profile agent', color=colors[col][1])
                    
                    ax.set_title(f'Expected route {FULL_NAME_VALUES[col].lower()} costs ({test_or_train} set)')
                    #ax.set_xlabel('Policy')
                    ax.set_ylabel('Value')
                    ax.set_xticks(index + bar_width / 2)
                    ax.set_xticklabels([])  # Remove the x-tick labels
                    ax.legend()
                    
                    # Add Policy labels under each bar
                    for i, label in enumerate(expert_df['Policy']):
                        ax.text(i-bar_width/2, -0.01, label.split('Expert ')[-1], ha='center', va='top', rotation=70, transform=ax.get_xaxis_transform())
                    for i, label in enumerate(learned_df['Policy']):
                        ax.text(i-bar_width/2 + bar_width, -0.01, label.split('_')[-1], ha='center', va='top', rotation=70, transform=ax.get_xaxis_transform())
                    
                    plt.tight_layout()
                    plt.savefig(f'results/value_system_identification/{name_of_files}_{col}_values_comparison_given_profile_{test_or_train}.pdf')
                    
                    #plt.show()
                    plt.close(fig)
    for society_or_expert in ['expert', 'society']:
        for test_or_train in ['test','train']:
            for combination, n_profiles_and_name in combinations.items():
                n_profiles, name_of_files = n_profiles_and_name
                
                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_{society_or_expert}_{test_or_train}.csv")
                
                print(df.columns[1])
                policies = df[df.columns[1]]
                
                expert_df = df[df['Policy'].str.contains('Expert' if society_or_expert == 'expert' else 'Society', case=False, na=False)]
                print(expert_df['sec'])
                
                learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
                print(learned_df['sec'])

                for col in ['sus', 'eff', 'sec']:
                    
                    expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                    
                    expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
                
                # Separate the averages and standard deviations for learned_df
                for col in ['sus', 'eff', 'sec']:
                    learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                    learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])

                # EXPERT VS LEARNED
                # Plot and save each plot in separate PDF files
                for col in ['sus', 'eff', 'sec']:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bar_width = 0.35
                    index = np.arange(len(expert_df))
                    
                    bars1 = ax.bar(index, expert_df[f'{col}_mean'], bar_width, yerr=expert_df[f'{col}_std'] if PUT_OD_STD else None, capsize=5, label=f"{'Original profile individual' if society_or_expert == 'expert' else 'Profiled society'}", color=colors[col][0])
                    bars2 = ax.bar(index + bar_width, learned_df[f'{col}_mean'], bar_width, yerr=learned_df[f'{col}_std'] if PUT_OD_STD  else None, capsize=5, label='Learned profile agent', color=colors[col][1])
                    
                    ax.set_title(f'Expected route {FULL_NAME_VALUES[col].lower()} costs ({test_or_train} set)')
                    #ax.set_xlabel('Policy')
                    ax.set_ylabel('Value')
                    ax.set_xticks(index + bar_width / 2)
                    ax.set_xticklabels([])  # Remove the x-tick labels
                    ax.legend()
                    
                    # Add Policy labels under each bar
                    for i, label in enumerate(expert_df['Policy']):
                        ax.text(i-bar_width/2, -0.01, label.split('Expert ')[-1] if society_or_expert == 'expert' else label.split('Society ')[-1], ha='center', va='top', rotation=70, transform=ax.get_xaxis_transform())
                    for i, label in enumerate(learned_df['Policy']):
                        ax.text(i-bar_width/2 + bar_width, -0.01, label.split('_')[-1], ha='center', va='top', rotation=70, transform=ax.get_xaxis_transform())
                    
                    plt.tight_layout()
                    plt.savefig(f'results/value_system_identification/{name_of_files}_{col}_values_comparison_{society_or_expert}_{test_or_train}.pdf')
                    
                    #plt.show()
                    plt.close(fig)




# SIMILITUDES TABLA:

colors = {
    'sus': ('darkgreen', 'green'),
    'eff': ('red', 'magenta'),
    'sec': ('blue', 'cyan')
}
similarities = {
    'agou',
    'jaccard',
    'visitation_count'
}

def swap_columns(df):
    columns = df.columns.tolist()
    columns[1], columns[2] = columns[2], columns[1]
    return df[columns]

for society_or_expert in ['expert', 'society']:
    for test_or_train in ['test','train']:
        for combination, n_profiles_and_name in combinations.items():
            for similarity in similarities:
                n_profiles, name_of_files = n_profiles_and_name
                
                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_similarities_{similarity}_learning_from_{society_or_expert}_{test_or_train}.csv")
                df_only_means = df.copy()
                print(df.columns[3:])

                for col in df.columns[3:]:
                    
                    df_only_means[f'{col}'] = df[col].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else x)
                
                df_only_means[df.columns[1]] = df[df.columns[1]].apply(lambda x: tuple([float(f'{a:.2f}') for a in literal_eval(x)]))
                expert_df_swapped = swap_columns(df_only_means).iloc[:, 1:]
                expert_df_swapped.to_latex(f'results/value_system_identification/latex/{name_of_files}_similarities_{similarity}_learning_from_{society_or_expert}_{test_or_train}.tex',index=False,float_format='%.2f')
                
                #header = " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:5])])
                header = " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:2])])
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Original Sus" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Originial Sec" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Original Eff" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Sus" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Sec" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Eff" + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{sus}$ sim." + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{sec}$ sim." + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{eff}$ sim." + "}}"
                #header = header + " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[2:5])])
                
                # Export to LaTeX

                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_statistics_learning_from_{society_or_expert}_{test_or_train}.csv")
                
                print(df.columns[1])
                policies = df[df.columns[1]]
                
                expert_df = df[df['Policy'].str.contains('Expert' if society_or_expert == 'expert' else 'Society', case=False, na=False)]
                print(expert_df['sec'])
                
                learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
                print(learned_df['sec'])

                for col in ['sus', 'eff', 'sec']:
                    
                    expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                    
                    expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
                
                # Separate the averages and standard deviations for learned_df
                for col in ['sus', 'eff', 'sec']:
                    learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                    learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])
                if test_or_train == 'test' and 'all' in name_of_files:
                    with open(f'results/value_system_identification/latex/PAPER_{name_of_files}_{test_or_train}_{society_or_expert}.tex', 'w') as f:
                        f.write(r'\begin{table*}[h!]' + '\n')
                        f.write(r'\centering' + '\n')
                        f.write(r'\begin{tabularx}{\textwidth}{' +  'll' + "|XXX|XXX" + '}' + '\n')
                        f.write(r'\hline' + '\n')
                        f.write(header + r' \\' + '\n')
                        f.write(r'\hline' + '\n')
                        for index, row in expert_df_swapped.iterrows():
                            row_data = " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[0:2]])
                            print(learned_df['sus_mean'].to_list()[index])
                            row_data = row_data + " & " + f"{expert_df['sus_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{expert_df['sec_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{expert_df['eff_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['sus_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['sec_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['eff_mean'].to_list()[index]:.3f}"
                            #row_data = row_data + " & " + " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[2:5]])
                            
                            f.write(row_data + r' \\' + '\n')
                        f.write(r'\hline' + '\n')
                        f.write(r'\end{tabularx}' + '\n')
                        #f.write(r'\caption{Results for value system identification, learning from' + (r' expert agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. \textit{' + f'{similarity}' + r'} similarities with different route datasets of the learned trajectories are shown. The first three columns are similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r', The next three show the similarity with routes taken by agents with pure profiles, each in terms of their target value. The last two show similarity in terms of the target profile with the routes taken by an agent with that profile, and by a society with that profile, respectively.}' + '\n')
                        f.write(r'\caption{Results for value system identification, learning from' + (r' individual agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. The expected value alignments of the routes sampled with the original profile and the learned profile are shown in the first 6 columns. The last three columns represent the \textit{' + f'{similarity}' + r'} similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r'  according to the three values.}' + '\n')
                        
                        f.write(r'\label{table:expert_df}' + '\n')
                        f.write(r'\end{table*}' + '\n')

            


for test_or_train in ['test','train']:
        for combination, n_profiles_and_name in combinations.items():
            for similarity in similarities:
                n_profiles, name_of_files = n_profiles_and_name
                
                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_similarities_{similarity}_for_unseen_profile_{test_or_train}.csv")
                df_only_means = df.copy()
                print(df.columns[3:])

                for col in df.columns[3:]:
                    
                    df_only_means[f'{col}'] = df[col].apply(lambda x: literal_eval(x)[0] if isinstance(x,str) else x)
                
                df_only_means[df.columns[1]] = df[df.columns[1]].apply(lambda x: tuple([float(f'{a:.2f}') for a in literal_eval(x)]))
                expert_df_swapped = swap_columns(df_only_means).iloc[:, 1:]
                expert_df_swapped.to_latex(f'results/value_system_identification/latex/{name_of_files}_similarities_{similarity}_given_profile_{test_or_train}.tex',index=False,float_format='%.2f')
                
                #header = " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:5])])
                header = " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[0:2])])
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Original Sus" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Originial Sec" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Original Eff" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Sus" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Sec" + "}}"
                header = header + " & " + r"\rotatebox{90}{\makecell{" + "Learned Eff" + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{sus}$ sim." + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{sec}$ sim." + "}}"
                #header = header + " & " + r"\rotatebox{90}{\makecell{" + r"$\textit{PS}_{eff}$ sim." + "}}"
                #header = header + " & ".join([r"\rotatebox{90}{\makecell{" + col.replace("_", r"\\") + "}}" if ic >= 2 else r"\makecell{" + col.replace("_", r"\\") + "}" for ic, col in enumerate(expert_df_swapped.columns[2:5])])
                
                # Export to LaTeX

                df = pd.read_csv(f"results/value_system_identification/{name_of_files}_statistics_for_unseen_profile_{test_or_train}.csv")
                
                print(df.columns[1])
                policies = df[df.columns[1]]
                
                expert_df = df[df['Policy'].str.contains('Expert', case=False, na=False)]
                print(expert_df['sec'])
                
                learned_df = df[df['Policy'].str.contains('learn', case=False, na=False)]
                print(learned_df['sec'])

                for col in ['sus', 'eff', 'sec']:
                    
                    expert_df[f'{col}_mean'] = expert_df[col].apply(lambda x: literal_eval(x)[0])
                    
                    expert_df[f'{col}_std'] = expert_df[col].apply(lambda x: literal_eval(x)[1])
                
                # Separate the averages and standard deviations for learned_df
                for col in ['sus', 'eff', 'sec']:
                    learned_df[f'{col}_mean'] = learned_df[col].apply(lambda x: literal_eval(x)[0])
                    learned_df[f'{col}_std'] = learned_df[col].apply(lambda x: literal_eval(x)[1])
                if test_or_train == 'test' and 'all' in name_of_files:
                    with open(f'results/value_system_identification/latex/PAPER_{name_of_files}_{test_or_train}_given_profile.tex', 'w') as f:
                        f.write(r'\begin{table*}[h!]' + '\n')
                        f.write(r'\centering' + '\n')
                        f.write(r'\begin{tabularx}{\textwidth}{' +  'll' + "|XXX|XXX" + '}' + '\n')
                        f.write(r'\hline' + '\n')
                        f.write(header + r' \\' + '\n')
                        f.write(r'\hline' + '\n')
                        for index, row in expert_df_swapped.iterrows():
                            row_data = " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[0:2]])
                            print(learned_df['sus_mean'].to_list()[index])
                            row_data = row_data + " & " + f"{expert_df['sus_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{expert_df['sec_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{expert_df['eff_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['sus_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['sec_mean'].to_list()[index]:.3f}"
                            row_data = row_data + " & " + f"{learned_df['eff_mean'].to_list()[index]:.3f}"
                            #row_data = row_data + " & " + " & ".join([f"{val:.2f}" if isinstance(val, float) else str(val) for val in row[2:5]])
                            
                            f.write(row_data + r' \\' + '\n')
                        f.write(r'\hline' + '\n')
                        f.write(r'\end{tabularx}' + '\n')
                        #f.write(r'\caption{Results for value system identification, learning from' + (r' expert agents with different profiles' if society_or_expert == 'expert' else r' different profiled societies') + r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. \textit{' + f'{similarity}' + r'} similarities with different route datasets of the learned trajectories are shown. The first three columns are similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r', The next three show the similarity with routes taken by agents with pure profiles, each in terms of their target value. The last two show similarity in terms of the target profile with the routes taken by an agent with that profile, and by a society with that profile, respectively.}' + '\n')
                        f.write(r'\caption{Results for value grounding learning, observing the similarity of' + r' individual agents with different profiles'+ r'. The learned profile represents a linear combination value system alignment function that the algorithm found the most coherent with observed behaviour. The expected value alignments of the routes sampled with the original profile and the learned profile are shown in the first 6 columns. The last three columns represent the \textit{' + f'{similarity}' + r'} similarities with the observed trajectories ' + f'(from the {test_or_train} set)' + r'  according to the three values.}' + '\n')
                        
                        f.write(r'\label{table:expert_df}' + '\n')
                        f.write(r'\end{table*}' + '\n')

            


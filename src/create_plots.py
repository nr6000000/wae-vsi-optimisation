import argparse
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def rename(s):
    new_s = '_'.join(s.split('_')[:-1])
    if new_s == 'ga_blx_arith':
        return 'ga_arith'
    else:
        return new_s

def create_plots(out):
    df = pd.read_csv(out / 'runs.csv')
    df = df.drop('method', axis=1)
    df['method'] = df['case'].map(lambda x: rename(x))
    # print(df)

    print(df[df['method'] == 'ga_blx']['best_value'].describe())

    df = df.replace({'method': {'ga_blx': 'Algorytm\nGenetyczny (BLX)', 'ga_arith': 'Algorytm\nGenetyczny (średnia)', 'de': 'Ewolucja\nRóżnicowa', 'random': 'Losowa'}})
    idx = df.groupby('method')['best_value'].idxmin()
    print(df.iloc[idx].to_string())


    plt.figure(figsize=(8, 6))

    # Boxplot
    sns.boxplot(data=df, x='method', y='best_value', showfliers=False)
    # plt.title('Wyniki osiągnięte przez metody')
    plt.ylabel('J(x)')
    plt.xlabel('Metoda')
    plt.show()

    df = pd.read_csv(out / 'histories.csv')
    df = df.drop('method', axis=1)
    df['method'] = df['case'].map(lambda x: rename(x))

    df = df.replace({'method': {'ga': 'Algorytm Genetyczny', 'de': 'Ewolucja Różnicowa', 'random': 'Losowa'}})
    df = df.rename({'method': 'Metoda'})

    # Convergence
    for method in ['ga_blx', 'ga_arith', 'de', 'random']:
        df_de = df[df['method'] == method][['case', 'evaluation', 'best_value']]
        df_de = df_de.replace({'method': {'ga': 'Algorytm Genetyczny', 'de': 'Ewolucja Różnicowa', 'random': 'Losowa'}})
        df_de = df_de.rename({'method': 'Metoda'})
        df_de = df_de[df_de['evaluation'] < 50]
        # df_de = df_de[df_de['best_value'] < 10]

        sns.lineplot(data=df_de, hue='case', x='evaluation', y='best_value')
        plt.title('')
        plt.ylabel('J(x)')
        plt.xlabel('Liczba ewaluacji')
        plt.yscale('log')
        plt.legend(title='Uruchomienie')
        plt.ylim(0, 10)
        plt.show()

    # Convergence all
    df_conv_all = df[['method', 'evaluation', 'best_value']]
    df_conv_all = df_conv_all[df_conv_all['evaluation'] < 50]
    # df_conv_all = df_conv_all[df_conv_all['method'] == 'ga_blx']

    df_conv_all = df_conv_all.groupby(['method', 'evaluation'], as_index=False).mean()
    print(df_conv_all.to_string())
    sns.lineplot(data=df_conv_all, hue='method', x='evaluation', y='best_value',
                 estimator=np.mean, errorbar=None)
    plt.title('')
    plt.ylabel('J(x)')
    plt.xlabel('Liczba ewaluacji')
    plt.legend(title='Metoda')
    plt.yscale('log')
    plt.ylim(0, 10)
    plt.show()

    df_conv_all = df[['method', 'evaluation', 'best_value']]
    df_conv_all = df_conv_all[df_conv_all['evaluation'] < 50]
    sns.lineplot(data=df_conv_all, 
                      hue='method', x='evaluation', y='best_value')
    plt.title('')
    plt.ylabel('J(x)')
    plt.xlabel('Liczba ewaluacji')
    plt.yscale('log')
    plt.ylim(0, 10)
    plt.legend(title='Metoda')
    plt.yscale('log')

    plt.show()
def main() -> None:
    parser = argparse.ArgumentParser(description="Beta runner for WAE VSI experiments through the MATLAB XML-RPC server.")
    parser.add_argument("--out", type=Path, default=Path("results/tiny"))

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    create_plots(args.out)

if __name__ == "__main__":
    main()
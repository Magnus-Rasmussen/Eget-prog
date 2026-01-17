"""
Utility functions for house price analysis
"""
import numpy as np
import pandas as pd
from dstapi import DstApi

def download_house_prices(table='EJ56', landsdele_codes=None, ejendom_codes=None):
    """Download house price data from DST"""
    if landsdele_codes is None:
        landsdele_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    if ejendom_codes is None:
        ejendom_codes = ['0111', '0801', '2103']
    
    api = DstApi(table)
    params = {
        'table': table, 'format': 'BULK', 'lang': 'da',
        'variables': [
            {'code': 'OMRÅDE', 'values': landsdele_codes},
            {'code': 'EJENDOMSKATE', 'values': ejendom_codes},
            {'code': 'TAL', 'values': ['100']},
            {'code': 'Tid', 'values': ['*']}
        ]
    }
    return api.get_data(params=params)

def clean_house_data(df):
    """Clean and prepare house price data"""
    df = df.replace('..', np.nan).dropna(subset=['INDHOLD'])
    df['INDHOLD'] = df['INDHOLD'].str.replace(',', '.').astype(float)
    df = df.rename(columns={
        'INDHOLD': 'Indeks', 
        'TID': 'Kvartal', 
        'OMRÅDE': 'Landsdel', 
        'EJENDOMSKATE': 'Ejendomskategori'
    })
    df['År'] = df['Kvartal'].str[:4].astype(int)
    df['Kvartal_nr'] = df['Kvartal'].str[-1].astype(int)
    return df.sort_values(['Landsdel', 'Ejendomskategori', 'År', 'Kvartal_nr']).reset_index(drop=True)

def filter_complete_series(df):
    """Keep only landsdele with complete time series from earliest quarter"""
    earliest_quarter = df['Kvartal'].min()
    landsdele_to_keep = [
        (l, e) for e in df['Ejendomskategori'].unique() 
        for l in df[df['Ejendomskategori'] == e]['Landsdel'].unique()
        if df[(df['Landsdel'] == l) & (df['Ejendomskategori'] == e)]['Kvartal'].min() == earliest_quarter
    ]
    return df[df.apply(lambda row: (row['Landsdel'], row['Ejendomskategori']) in landsdele_to_keep, axis=1)].reset_index(drop=True)

def reindex_to_base(df, index_col='Indeks', new_col='Indeks_1992'):
    """Reindex time series to first observation = 100"""
    df[new_col] = df.groupby(['Landsdel', 'Ejendomskategori'], group_keys=False)[index_col].apply(
        lambda x: (x / x.iloc[0]) * 100
    )
    return df

def download_cpi():
    """Download CPI data from DST"""
    api = DstApi('PRIS113')
    df_cpi = api.get_data(params={
        'table': 'PRIS113', 'format': 'BULK', 'lang': 'en',
        'variables': [
            {'code': 'TYPE', 'values': ['INDEKS']},
            {'code': 'Tid', 'values': ['*']}
        ]
    })
    df_cpi.columns = ['TYPE', 'TIME', 'CPI']
    df_cpi['CPI'] = df_cpi['CPI'].astype(float)
    df_cpi['Quarter_Period'] = pd.to_datetime(df_cpi['TIME'].str.replace('M', ''), format='%Y%m').dt.to_period('Q')
    return df_cpi.groupby('Quarter_Period')['CPI'].mean().reset_index()

def calculate_real_prices(df, df_cpi):
    """Calculate real (inflation-adjusted) house prices"""
    df['Quarter_Period'] = pd.PeriodIndex(df['Kvartal'].str.replace('K', 'Q'), freq='Q')
    df_real = df.merge(df_cpi, on='Quarter_Period', how='left')
    df_real['Real_Index'] = (df_real['Indeks'] / df_real['CPI']) * 100
    return reindex_to_base(df_real, index_col='Real_Index', new_col='Real_Index_1992')

def plot_time_series(df, index_col='Indeks_1992', title_suffix='', figsize=(12, 14)):
    """Plot time series for all property types and landsdele"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    for idx, prop_type in enumerate(df['Ejendomskategori'].unique()):
        ax = axes[idx]
        df_prop = df[df['Ejendomskategori'] == prop_type]
        
        for landsdel in sorted(df_prop['Landsdel'].unique()):
            df_plot = df_prop[df_prop['Landsdel'] == landsdel].sort_values(['År', 'Kvartal_nr'])
            ax.plot(df_plot['År'] + (df_plot['Kvartal_nr']-1)/4, df_plot[index_col], label=landsdel, alpha=0.7)
        
        ax.set_title(f'{prop_type}{title_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlabel('År')
        ax.set_ylabel('Prisindeks (1992Q1=100)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    plt.show()

def plot_ranking(df, index_col='Indeks_1992', figsize=(12, 12)):
    """Plot ranking bar chart of price growth by landsdel"""
    import matplotlib.pyplot as plt
    
    latest_data = df[df['Kvartal'] == df['Kvartal'].max()].copy()
    latest_data['Vækst (%)'] = latest_data[index_col] - 100
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    for idx, ejendom in enumerate(sorted(latest_data['Ejendomskategori'].unique())):
        df_rank = latest_data[latest_data['Ejendomskategori'] == ejendom].sort_values(index_col, ascending=True)
        ax = axes[idx]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_rank)))
        ax.barh(df_rank['Landsdel'], df_rank['Vækst (%)'], color=colors)
        ax.set_xlabel('Vækst siden 1992Q1 (%)', fontsize=10)
        ax.set_title(f'{ejendom} - Prisvækst per landsdel', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (landsdel, vækst) in enumerate(zip(df_rank['Landsdel'], df_rank['Vækst (%)'])):
            ax.text(vækst + 10, i, f'{vækst:.0f}%', va='center', fontsize=9)
    
    plt.suptitle(f'Ranking af landsdele efter prisvækst (1992Q1 til {df["Kvartal"].max()})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

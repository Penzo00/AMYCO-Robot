#%%
import sys
def tell_description(name_id=sys.argv[1]):
    import pandas as pd
    import re
    # Carica il file CSV
    df = pd.read_csv('name_descriptions.csv', delimiter='\t', on_bad_lines='skip').fillna("")
    
    def pulisci_testo(testo):
        # Sostituisce i caratteri \n con uno spazio, poi rimuove link, caratteri _ e []
        return re.sub(r'\s+', ' ', re.sub(r'[_\[\]]', '', re.sub(r'http\S+|www\S+', '', testo.replace('\\n', ' ')))).strip()
    
    # Trova l'indice della riga con il valore specifico di name_id (come stringa)
    start_index = df.index[df['name_id'] == name_id].tolist()
    
    if not start_index:
        print(f"Nessuna riga trovata con name_id = {name_id}")
        return
    start_index = start_index[0]
    
    # Lista per accumulare i valori da concatenare
    testo_concatenato = []

    # Inizia dalla riga successiva alla riga trovata
    for i in range(start_index, len(df)):
        # Prendi i valori delle celle `id` e `name_id` della riga successiva
        next_id_val = df.at[i + 1, 'id'] if i + 1 < len(df) else None
        next_name_id_val = df.at[i + 1, 'name_id'] if i + 1 < len(df) else None
        
        # Interrompi se `id` o `name_id` della riga successiva contengono un numero intero come stringa
        if (isinstance(next_id_val, str) and next_id_val.isdigit()) or \
           (isinstance(next_name_id_val, str) and next_name_id_val.isdigit()):
            break

        # Aggiungi alla lista i valori della riga corrente, escluso `name_id`
        testo_concatenato.extend(map(str, df.iloc[i].drop('name_id').values))
    
    # Unisci i testi e puliscili
    testo_finale = pulisci_testo(' '.join(testo_concatenato))
    return testo_finale[3:]
tell_description()